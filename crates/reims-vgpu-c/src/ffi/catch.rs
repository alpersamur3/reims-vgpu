//! The catch around a call of a host function ([`super::host_call`]), in one file per OS with the
//! same function: `catch(f)` runs `f` and answers its value, or `Err(())` for what the host threw.

// Windows takes Linux's catch: std's `catch_unwind` (a Rust unwind stops at the frame; a foreign
// exception through an `extern "C-unwind"` host function ends the process there).
#[cfg(any(target_os = "linux", windows))]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

#[cfg(any(target_os = "linux", windows))]
pub(super) use linux::catch;
#[cfg(target_os = "macos")]
pub(super) use macos::catch;
