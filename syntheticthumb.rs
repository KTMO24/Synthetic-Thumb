// src/main.rs

//! This example provides two modes:
//! 1. Running as a FUSE filesystem (the synthetic thumb drive)
//!    Usage: synthetic_thumb <mountpoint>
//! 2. Running with a PySide6 GUI for settings:
//!    Usage: synthetic_thumb gui

use std::collections::HashMap;
use std::env;
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use fuse::{Filesystem, ReplyAttr, ReplyCreate, ReplyData, ReplyDirectory, ReplyEmpty, ReplyOpen, ReplyWrite, Request};
use libc::{ENOENT, S_IFDIR, S_IFREG};
use pyo3::prelude::*;

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

fn main() {
    let args: Vec<String> = env::args().collect();
    // If the argument "gui" is given, run the PySide6 GUI.
    if args.len() > 1 && args[1] == "gui" {
        if let Err(e) = launch_gui() {
            eprintln!("Failed to launch GUI: {:?}", e);
        }
        return;
    }

    // Otherwise, assume FUSE mount mode.
    if args.len() < 2 {
        eprintln!("Usage: {} <mountpoint>  OR  {} gui", args[0], args[0]);
        return;
    }
    let mountpoint = &args[1];
    if !Path::new(mountpoint).exists() {
        eprintln!("Mountpoint '{}' does not exist.", mountpoint);
        return;
    }
    let fs = SyntheticThumbDrive::new();
    // Mount the filesystem; note that this call blocks.
    fuse::mount(fs, &mountpoint, &[]).unwrap();
}
