// src/main.rs

//! This merged example provides three modes:
//!
//! 1. Running as a FUSE filesystem (the synthetic thumb drive).
//!    Usage: synthetic_thumb <mountpoint>
//!
//! 2. Running with a PySide6 GUI for settings.
//!    Usage: synthetic_thumb gui
//!
//! 3. Running a simulation that uses multi‑threading, statistical range
//!    compression, and a dual‑waveform timelock loop.
//!    Usage: synthetic_thumb simulate

use std::collections::HashMap;
use std::env;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use fuse::{Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty, ReplyOpen, ReplyWrite, Request};
use libc::{ENOENT, S_IFDIR, S_IFREG};
use pyo3::prelude::*;
use std::f64::consts::PI;

// A short TTL for cached attributes.
const TTL: Duration = Duration::from_secs(1);

/// File metadata structure.
#[derive(Clone)]
struct FileMeta {
    mode: u32,
    size: u64,
    atime: SystemTime,
    mtime: SystemTime,
    ctime: SystemTime,
    nlink: u32,
}

/// The synthetic thumb drive filesystem.
/// It keeps two hash maps (protected by Mutex):
///  - one for file metadata
///  - one for file data (stored in compressed form)
struct SyntheticThumbDrive {
    files: Mutex<HashMap<String, FileMeta>>,
    data: Mutex<HashMap<String, Vec<u8>>>, // compressed data
}

impl SyntheticThumbDrive {
    fn new() -> Self {
        let now = SystemTime::now();
        let mut files = HashMap::new();
        files.insert(
            "/".to_string(),
            FileMeta {
                mode: S_IFDIR as u32 | 0o755,
                size: 0,
                atime: now,
                mtime: now,
                ctime: now,
                nlink: 2,
            },
        );
        SyntheticThumbDrive {
            files: Mutex::new(files),
            data: Mutex::new(HashMap::new()),
        }
    }

    // --- Custom compression methods ---
    //
    // Our “geometric folding” compressor works as follows:
    // 1. Pad the input bytes to form a square matrix.
    // 2. Take the top‐left quadrant of half the size and replace each element
    //    by the average of the corresponding 4 symmetric values.
    // 3. Compress the resulting folded quadrant using zlib.
    //
    // Decompression “unfolds” by duplicating the quadrant into the four quadrants.
    //

    fn compress(data: &[u8]) -> Vec<u8> {
        let len = data.len();
        let size = (len as f64).sqrt().ceil() as usize;
        let total = size * size;
        let mut matrix = vec![0u8; total];
        matrix[..len].copy_from_slice(data);

        // Use half-size for the folded quadrant.
        let half = size / 2;
        let mut folded = vec![0u8; half * half];
        for i in 0..half {
            for j in 0..half {
                let a = matrix[i * size + j] as u32;
                let b = matrix[(size - 1 - i) * size + j] as u32;
                let c = matrix[i * size + (size - 1 - j)] as u32;
                let d = matrix[(size - 1 - i) * size + (size - 1 - j)] as u32;
                folded[i * half + j] = ((a + b + c + d) / 4) as u8;
            }
        }
        // Compress the folded quadrant.
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&folded).unwrap();
        encoder.finish().unwrap()
    }

    fn decompress(comp_data: &[u8]) -> Vec<u8> {
        // Decompress the folded quadrant.
        let mut decoder = ZlibDecoder::new(comp_data);
        let mut folded = Vec::new();
        decoder.read_to_end(&mut folded).unwrap();
        let half = (folded.len() as f64).sqrt() as usize;
        // Reconstruct a square matrix by duplicating the quadrant.
        let full_size = half * 2;
        let mut full = vec![0u8; full_size * full_size];
        for i in 0..half {
            for j in 0..half {
                let val = folded[i * half + j];
                full[i * full_size + j] = val;
                full[(full_size - 1 - i) * full_size + j] = val;
                full[i * full_size + (full_size - 1 - j)] = val;
                full[(full_size - 1 - i) * full_size + (full_size - 1 - j)] = val;
            }
        }
        full
    }
}

// --- Implement the FUSE filesystem operations ---
impl Filesystem for SyntheticThumbDrive {
    fn getattr(&mut self, _req: &Request<'_>, path: &OsStr, reply: ReplyAttr) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        let files = self.files.lock().unwrap();
        if let Some(meta) = files.get(&p) {
            let kind = if meta.mode & S_IFDIR as u32 != 0 {
                fuse::FileType::Directory
            } else {
                fuse::FileType::RegularFile
            };
            let attr = fuse::FileAttr {
                ino: 1,
                size: meta.size,
                blocks: 1,
                atime: meta.atime,
                mtime: meta.mtime,
                ctime: meta.ctime,
                crtime: meta.ctime,
                kind,
                perm: meta.mode & 0o777,
                nlink: meta.nlink,
                uid: unsafe { libc::getuid() },
                gid: unsafe { libc::getgid() },
                rdev: 0,
                flags: 0,
            };
            reply.attr(&TTL, &attr);
        } else {
            reply.error(ENOENT);
        }
    }

    fn readdir(
        &mut self,
        _req: &Request<'_>,
        path: &OsStr,
        _fh: u64,
        offset: i64,
        mut reply: ReplyDirectory,
    ) {
        let p = path.to_str().unwrap_or("");
        if p != "/" && p != "" {
            reply.error(ENOENT);
            return;
        }
        let files = self.files.lock().unwrap();
        let mut entries = vec![
            (1, fuse::FileType::Directory, "."),
            (1, fuse::FileType::Directory, ".."),
        ];
        for (name, meta) in files.iter() {
            if name == "/" {
                continue;
            }
            let fname = name.trim_start_matches("/");
            let kind = if meta.mode & S_IFDIR as u32 != 0 {
                fuse::FileType::Directory
            } else {
                fuse::FileType::RegularFile
            };
            entries.push((1, kind, fname));
        }
        for (i, entry) in entries.into_iter().enumerate().skip(offset as usize) {
            reply.add(entry.0, (i + 1) as i64, entry.1, entry.2);
        }
        reply.ok();
    }

    fn create(
        &mut self,
        _req: &Request<'_>,
        path: &OsStr,
        mode: u32,
        _flags: i32,
        reply: ReplyCreate,
    ) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        let now = SystemTime::now();
        let meta = FileMeta {
            mode: S_IFREG as u32 | mode,
            size: 0,
            atime: now,
            mtime: now,
            ctime: now,
            nlink: 1,
        };
        self.files.lock().unwrap().insert(p.clone(), meta);
        self.data.lock().unwrap().insert(p.clone(), Vec::new());
        let attr = fuse::FileAttr {
            ino: 1,
            size: 0,
            blocks: 1,
            atime: now,
            mtime: now,
            ctime: now,
            crtime: now,
            kind: fuse::FileType::RegularFile,
            perm: mode,
            nlink: 1,
            uid: unsafe { libc::getuid() },
            gid: unsafe { libc::getgid() },
            rdev: 0,
            flags: 0,
        };
        reply.created(&TTL, &attr, 0, 0, 0);
    }

    fn open(&mut self, _req: &Request<'_>, path: &OsStr, _flags: i32, reply: ReplyOpen) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        if self.files.lock().unwrap().contains_key(&p) {
            reply.opened(0, 0);
        } else {
            reply.error(ENOENT);
        }
    }

    fn read(&mut self, _req: &Request<'_>, path: &OsStr, size: u32, offset: i64, reply: ReplyData) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        let data_lock = self.data.lock().unwrap();
        if let Some(comp) = data_lock.get(&p) {
            let decompressed = SyntheticThumbDrive::decompress(comp);
            let end = ((offset as usize) + (size as usize)).min(decompressed.len());
            reply.data(&decompressed[offset as usize..end]);
        } else {
            reply.error(ENOENT);
        }
    }

    fn write(
        &mut self,
        _req: &Request<'_>,
        path: &OsStr,
        buf: &[u8],
        offset: i64,
        reply: ReplyWrite,
    ) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        let mut data_lock = self.data.lock().unwrap();
        let mut meta_lock = self.files.lock().unwrap();
        let mut decompressed = if let Some(comp) = data_lock.get(&p) {
            SyntheticThumbDrive::decompress(comp)
        } else {
            Vec::new()
        };
        let off = offset as usize;
        if decompressed.len() < off + buf.len() {
            decompressed.resize(off + buf.len(), 0);
        }
        decompressed[off..off + buf.len()].copy_from_slice(buf);
        let comp_new = SyntheticThumbDrive::compress(&decompressed);
        data_lock.insert(p.clone(), comp_new);
        if let Some(meta) = meta_lock.get_mut(&p) {
            meta.size = decompressed.len() as u64;
            meta.mtime = SystemTime::now();
        }
        reply.written(buf.len() as u32);
    }

    fn truncate(&mut self, _req: &Request<'_>, path: &OsStr, size: u64, reply: ReplyEmpty) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        let mut data_lock = self.data.lock().unwrap();
        let mut meta_lock = self.files.lock().unwrap();
        let mut decompressed = if let Some(comp) = data_lock.get(&p) {
            SyntheticThumbDrive::decompress(comp)
        } else {
            Vec::new()
        };
        decompressed.resize(size as usize, 0);
        let comp_new = SyntheticThumbDrive::compress(&decompressed);
        data_lock.insert(p.clone(), comp_new);
        if let Some(meta) = meta_lock.get_mut(&p) {
            meta.size = size;
            meta.mtime = SystemTime::now();
        }
        reply.ok();
    }

    fn unlink(&mut self, _req: &Request<'_>, path: &OsStr, reply: ReplyEmpty) {
        let p = format!("/{}", path.to_str().unwrap_or(""));
        self.files.lock().unwrap().remove(&p);
        self.data.lock().unwrap().remove(&p);
        reply.ok();
    }
}

// --- Embedded PySide6 GUI ---
//
// When run with the argument "gui", we launch a Python interpreter
// that runs a very simple PySide6 settings window. (For this to work, you
// must have Python 3 and PySide6 installed.)
fn launch_gui() -> PyResult<()> {
    Python::with_gil(|py| {
        // The Python code for our GUI.
        let gui_code = r#"
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox

class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synthetic Thumb Drive Settings")
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("Mountpoint:"))
        self.mount_edit = QLineEdit("/mnt/synthetic_thumb")
        self.layout.addWidget(self.mount_edit)
        self.auto_checkbox = QCheckBox("Auto-mount on start")
        self.layout.addWidget(self.auto_checkbox)
        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.clicked.connect(self.apply_settings)
        self.layout.addWidget(self.apply_btn)
        self.setLayout(self.layout)

    def apply_settings(self):
        mountpoint = self.mount_edit.text()
        auto_mount = self.auto_checkbox.isChecked()
        print("Settings applied:", mountpoint, auto_mount)
        # Here you might send these settings back to the Rust backend.
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SettingsWindow()
    window.show()
    sys.exit(app.exec())
"#;
        // Execute the GUI code.
        py.run(gui_code, None, None)?;
        Ok(())
    })
}

// --- Waveform and Vector Processing Simulation ---
//
// The following functions demonstrate multi‑threading,
// statistical range compression, vector serialization, and a dual‑waveform
// timelock “analog” loop that adjusts a time dilation factor based on
// sine vs. sawtooth waveforms.

/// Generate analog-like waveforms (sine and sawtooth) on a separate thread.
///
/// Every cycle it computes:
///   - a sine value: sin(2πft)
///   - a sawtooth value: a linear ramp from –1 to 1 each period
///   - an “offset” computed from the difference (and its inverse phase)
///
/// The values are sent over a channel.
fn generate_waveforms(sender: mpsc::Sender<(f64, f64, f64)>) {
    let start = Instant::now();
    let frequency = 1.0; // 1 Hz waveforms
    loop {
        let t = start.elapsed().as_secs_f64();
        let sin_val = (2.0 * PI * frequency * t).sin();
        let saw_val = 2.0 * ((t * frequency) - (t * frequency).floor()) - 1.0;
        let inv_phase = -sin_val;
        // Compute an "offset" as a function of phase difference.
        let offset = (sin_val - saw_val + inv_phase) / 3.0;
        if sender.send((sin_val, saw_val, offset)).is_err() {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

/// Serialize a vector of f64 values by first performing a statistical range
/// compression (normalizing to 0..255) and then converting to a Vec<u8>.
fn serialize_vector(vec: &[f64]) -> Vec<u8> {
    let min = vec.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    vec.iter()
        .map(|&v| {
            if range > 0.0 {
                (((v - min) / range) * 255.0).round() as u8
            } else {
                0
            }
        })
        .collect()
}

/// Process (serialize) a large vector by splitting it into chunks and
/// processing them in parallel using threads.
fn process_vector(vec: Vec<f64>) -> Vec<u8> {
    let chunk_size = 1000;
    let chunks: Vec<&[f64]> = vec.chunks(chunk_size).collect();
    let (tx, rx) = mpsc::channel();
    let mut handles = Vec::new();

    for chunk in chunks {
        let tx = tx.clone();
        let chunk_vec = chunk.to_vec();
        let handle = thread::spawn(move || {
            let serialized = serialize_vector(&chunk_vec);
            tx.send(serialized).expect("Failed to send serialized chunk");
        });
        handles.push(handle);
    }
    drop(tx);
    let mut result = Vec::new();
    for serialized_chunk in rx {
        result.extend(serialized_chunk);
    }
    for handle in handles {
        handle.join().unwrap();
    }
    result
}

/// A dual‑waveform timelock “analog” processing loop structure.
///
/// This structure gathers waveform offset values, adjusts a time dilation
/// factor based on the phase relationship of sine and sawtooth, and computes
/// a “future offset” based on historical data.
struct DualWaveformTimelock {
    offsets: Vec<f64>,
    time_dilation: f64,
}

impl DualWaveformTimelock {
    fn new() -> Self {
        Self {
            offsets: Vec::new(),
            time_dilation: 1.0,
        }
    }

    /// Process new waveform values. Adjust the time dilation based on
    /// whether the sine value exceeds the sawtooth value.
    fn process(&mut self, sin_val: f64, saw_val: f64, offset: f64) {
        if sin_val > saw_val {
            self.time_dilation *= 1.01;
        } else {
            self.time_dilation *= 0.99;
        }
        self.offsets.push(offset * self.time_dilation);
        if self.offsets.len() > 100 {
            self.offsets.remove(0);
        }
    }

    /// Compute a future offset prediction (here as a simple average).
    fn future_offset(&self) -> f64 {
        if self.offsets.is_empty() {
            0.0
        } else {
            self.offsets.iter().sum::<f64>() / (self.offsets.len() as f64)
        }
    }
}

/// Run the waveform simulation and vector processing demo.
fn run_simulation() {
    let (wave_tx, wave_rx) = mpsc::channel::<(f64, f64, f64)>();

    let waveform_thread = thread::spawn(move || {
        generate_waveforms(wave_tx);
    });

    let vector_thread = thread::spawn(|| {
        let large_vec: Vec<f64> = (0..10_000)
            .map(|x| ((x as f64) * 0.001).sin() * 100.0)
            .collect();
        let serialized = process_vector(large_vec);
        println!("Serialized vector length: {}", serialized.len());
    });

    let mut timelock = DualWaveformTimelock::new();
    for (sin_val, saw_val, offset) in wave_rx {
        timelock.process(sin_val, saw_val, offset);
        println!(
            "sin: {:>6.3}  saw: {:>6.3}  offset: {:>6.3}  future_offset: {:>6.3}  dilation: {:>6.3}",
            sin_val,
            saw_val,
            offset,
            timelock.future_offset(),
            timelock.time_dilation
        );
        let sleep_ms = (100.0 * timelock.time_dilation) as u64;
        thread::sleep(Duration::from_millis(sleep_ms));
    }

    let _ = vector_thread.join();
    let _ = waveform_thread.join();
}

// --- Main entry point ---
//
// This function parses the command line arguments and chooses one of three modes:
//  • fuse <mountpoint>
//  • gui
//  • simulate
fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  {} <mountpoint>   # Run as FUSE filesystem", args[0]);
        eprintln!("  {} gui            # Launch PySide6 GUI", args[0]);
        eprintln!("  {} simulate       # Run waveform simulation", args[0]);
        return;
    }
    match args[1].as_str() {
        "gui" => {
            if let Err(e) = launch_gui() {
                eprintln!("Failed to launch GUI: {:?}", e);
            }
        }
        "simulate" => {
            run_simulation();
        }
        mountpoint => {
            if !Path::new(mountpoint).exists() {
                eprintln!("Mountpoint '{}' does not exist.", mountpoint);
                return;
            }
            let fs = SyntheticThumbDrive::new();
            // This call blocks.
            fuse::mount(fs, mountpoint, &[]).unwrap();
        }
    }
}
