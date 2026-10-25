//! `Clock` over Windows's clocks, as macOS's answers them.
//!
//! The monotonic clock is the interrupt time (`QueryInterruptTimePrecise`, 100 ns units): like
//! `CLOCK_BOOTTIME` on Linux and `CLOCK_MONOTONIC_RAW` on macOS, it counts the time the host
//! sleeps. The wall clock is `GetSystemTimePreciseAsFileTime`, moved from 1601 to the Unix epoch.
//! Timed waits are Linux's: a condvar's timed wait for the time left, then the deadline read on
//! its own clock.

use std::time::Duration;

use reims_vgpu_core::platform::os::{Clock, Deadline, MonoTime, TimedOut, WallTime};
use reims_vgpu_core::sync::{Condvar, MutexGuard};
use windows_sys::Win32::Foundation::FILETIME;
use windows_sys::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime;
use windows_sys::Win32::System::WindowsProgramming::QueryInterruptTimePrecise;

/// 100 ns intervals from 1601-01-01 to 1970-01-01.
const UNIX_EPOCH_IN_FILETIME: u64 = 116_444_736_000_000_000;

/// The interrupt time, in nanoseconds.
fn interrupt_nanos() -> u64 {
    let mut ticks = 0u64;
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { QueryInterruptTimePrecise(&mut ticks) };
    ticks.saturating_mul(100)
}

/// The system time, in nanoseconds since the Unix epoch.
fn wall_nanos() -> u64 {
    let mut ft = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { GetSystemTimePreciseAsFileTime(&mut ft) };
    let ticks = (u64::from(ft.dwHighDateTime) << 32) | u64::from(ft.dwLowDateTime);
    ticks
        .saturating_sub(UNIX_EPOCH_IN_FILETIME)
        .saturating_mul(100)
}

/// The host's clocks and timed waits.
#[derive(Debug, Default)]
pub struct WindowsClock;

impl WindowsClock {
    /// The process's Clock.
    pub fn new() -> WindowsClock {
        WindowsClock
    }

    /// How long until `deadline`, on its own clock; zero once reached.
    fn remaining(&self, deadline: Deadline) -> Duration {
        match deadline {
            Deadline::Mono(t) => t.saturating_duration_since(self.now_mono()),
            Deadline::Wall(t) => t.saturating_duration_since(self.now_wall()),
        }
    }
}

impl Clock for WindowsClock {
    /// The interrupt time, which counts a host sleep as macOS's monotonic clock does.
    fn now_mono(&self) -> MonoTime {
        MonoTime::from_nanos(interrupt_nanos())
    }

    /// The precise system time.
    fn now_wall(&self) -> WallTime {
        WallTime::from_nanos(wall_nanos())
    }

    /// One wait, as libc++'s is on macOS: a notify or a spurious wakeup ends it; the answer is
    /// whether the deadline has passed afterwards. A deadline already reached answers timeout
    /// without releasing the lock.
    fn wait_until<T>(
        &self,
        cv: &Condvar,
        g: &mut MutexGuard<'_, T>,
        deadline: Deadline,
    ) -> TimedOut {
        let d = self.remaining(deadline);
        if d.is_zero() {
            return TimedOut(true);
        }
        let _ = cv.wait_for(g, d);
        TimedOut(deadline.reached(self))
    }

    /// One untimed wait; the caller loops.
    fn wait<T>(&self, cv: &Condvar, g: &mut MutexGuard<'_, T>) {
        cv.wait(g);
    }

    /// Sleeps until `deadline` is reached on its own clock.
    fn sleep_until(&self, deadline: Deadline) {
        loop {
            let d = self.remaining(deadline);
            if d.is_zero() {
                return;
            }
            std::thread::sleep(d);
        }
    }
}
