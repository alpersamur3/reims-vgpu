//! `Threads` on Windows: `std::thread`s, detached, with the stack and name core asks for. std
//! names a thread with `SetThreadDescription`, which keeps the whole name, so nothing is cut as
//! on Linux. No scheduler hints are given yet (Linux's `sched_setattr` utilization clamps have no
//! Windows counterpart this port needs).
//!
//! No Objective-C runs here, so the reason macOS starts plain pthreads does not apply. A panic on
//! one of these threads is the library's, and its panic hook aborts the process before anything
//! unwinds.

use std::sync::Arc;

use reims_vgpu_core::platform::os::{Threads, TimerClock};
use reims_vgpu_core::raise::death::{DeathId, DeathKind, fatal};

use super::clock::WindowsClock;

/// A detached thread named `name` with `stack_size` bytes of stack that runs `body`. A refused
/// thread ends the process.
fn spawn_thread(name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
    let made = std::thread::Builder::new()
        .name(name.to_owned())
        .stack_size(stack_size)
        .spawn(body);
    if made.is_err() {
        fatal(DeathId::Platform {
            kind: DeathKind::Abort,
            call: "CreateThread",
        })
    }
}

/// The process's threads, and its clock for timers.
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsThreads;

impl WindowsThreads {
    /// The process's Threads.
    pub fn new() -> WindowsThreads {
        WindowsThreads
    }

    /// Threads for a display window's: the same here, as no thread gets hints.
    pub fn without_hints() -> WindowsThreads {
        WindowsThreads
    }
}

impl Threads for WindowsThreads {
    /// A FIFO thread.
    fn spawn(&self, name: &'static str, stack_size: usize, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(name, stack_size, body);
    }

    /// A worker named after its queue, with a GCD worker's 512 KiB of stack, as on macOS.
    fn spawn_worker(&self, label: &'static str, body: Box<dyn FnOnce() + Send>) {
        spawn_thread(label, reims_vgpu_core::rt::queue::WORKER_STACK, body);
    }

    fn handle(&self) -> Arc<dyn Threads> {
        Arc::new(*self)
    }

    fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        Some(Arc::new(WindowsClock::new()))
    }
}
