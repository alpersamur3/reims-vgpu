//! The C library calls and constants the harness uses, in one file per OS with the same names.

#[cfg(any(target_os = "linux", target_os = "android"))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use linux::*;
#[cfg(target_os = "macos")]
pub use macos::*;
