//! Deferred views of the guest's pages (ASoftware's Android port): views the GPU reaches without
//! the kernel mapping them, so that a view of scattered pages costs one kernel mapping, not one per
//! run (`vm.max_map_count`, 65,530 on Android, which no app may raise; `super::budget`).
//!
//! - **What it is.** A deferred view is a reservation with no access (one mapping), whose runs the
//!   backing table records as the files behind its bytes without mapping them
//!   ([`super::files::record_deferred`]). The runtime is given its bytes as gathered memory
//!   (`rt::HostMemory::with_gather`, [`gather_source`]): it asks where the pages lie, and binds a
//!   sparse buffer over its chunks of the guest's RAM, QEMU's one mapping of its file, with no
//!   mapping of the view's own.
//! - **The CPU's bytes.** The CPU reaches the view's bytes at its address, as ever: its first touch
//!   of a page faults, and the fault handler maps the window of [`WINDOW`] bytes around it from
//!   the table's files (`materialize`), the same pages, then lets the access run again. At most
//!   [`MAX_PAGES`] pages of all views are mapped so at once: past that the oldest window goes back
//!   to no access (one `mmap` of no access over it, which the kernel joins to the reservation),
//!   and a later touch maps it again. A thread whose window went while it used it faults again
//!   and has it mapped again: the bytes are the same file pages either way.
//! - **Which views.** A view whose every byte is the guest's RAM ([`super::files::guest_place`]):
//!   the aliases of the guest's ranges a heap, a replace or a surface's buffer is made over
//!   (`crate::gpu::backing::gather`), and an IOSurface over several ranges of them
//!   (`super::surface`). A view with zeros of its own past the ranges' pages stays as it was.
//!   `REIMS_VGPU_DEFERRED_VIEWS=0` turns them off, `=1` on; on by default on Android only.
//! - **Pinned.** Where the runtime is to import a deferred view's bytes at their address after all
//!   (the source answers `GatherAnswer::Pointer`: a device that gathers nothing, or a refusal of
//!   its chunks), the whole view is mapped for good, as an eager view is.
//! - **The fault handler** takes `SIGSEGV` for addresses inside a deferred view whose page is not
//!   mapped; every other fault, and one inside a mapped page (a write to a read-only view, which
//!   kills the process as before), goes to the handler installed before it, or the default.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, KernReturn};
use reims_vgpu_core::sync::Mutex;

use super::files;
use super::vm;

/// The bytes a CPU touch maps of a deferred view at once, from a multiple of it in the view.
pub(super) const WINDOW: u64 = 128 << 10;

/// The most pages of host-page size the CPU's touches keep mapped across every deferred view
/// (module docs): `REIMS_VGPU_DEFERRED_PAGES`, else 16,384 (64 MiB of 4 KiB pages), each a
/// kernel mapping at most.
const MAX_PAGES: u64 = 16_384;

/// One deferred view: its end, and whether it is mapped for good (module docs).
#[derive(Debug)]
struct View {
    end: u64,
    pinned: bool,
}

/// The deferred views and the windows the CPU's touches mapped, oldest first.
#[derive(Debug, Default)]
struct State {
    views: BTreeMap<u64, View>,
    /// The mapped windows by their first byte: their length.
    windows: BTreeMap<u64, u64>,
    /// The mapped windows' first bytes, oldest first (a window may be named again after it went:
    /// skipped where `windows` lacks it).
    order: VecDeque<u64>,
    /// The pages the mapped windows hold.
    pages: u64,
}

impl State {
    /// The view holding byte `at`: its first byte and itself.
    fn view_at(&self, at: u64) -> Option<(u64, &View)> {
        (self.views.range(..=at).next_back())
            .filter(|(_, view)| at < view.end)
            .map(|(&first, view)| (first, view))
    }

    /// The view holding every byte of `first..end`, by its first byte.
    fn view_holding(&self, first: u64, end: u64) -> Option<u64> {
        self.view_at(first)
            .filter(|(_, view)| end <= view.end)
            .map(|(at, _)| at)
    }

    /// Unmaps the windows meeting `first..end` back to no access, and forgets them.
    fn unmap_windows(&mut self, first: u64, end: u64) {
        let meeting: Vec<(u64, u64)> = (self.windows.range(..end).rev())
            .take_while(|&(&at, &length)| at + length > first)
            .map(|(&at, &length)| (at, length))
            .collect();
        for (at, length) in meeting {
            self.drop_window(at, length);
        }
    }

    /// Puts the window `at` of `length` bytes back to no access, deferred again in the table.
    fn drop_window(&mut self, at: u64, length: u64) {
        if self.windows.remove(&at).is_none() {
            return;
        }
        self.pages -= length / vm::page_size();
        // Back to the reservation: anonymous, no access, which the kernel joins to its neighbours.
        // SAFETY: a window of a deferred view this module mapped; nothing of the library's holds
        // its address but the view, whose CPU users fault again and have it mapped again.
        let unmapped = unsafe {
            libc::mmap(
                vm::ptr(at),
                length.to_usize(),
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if unmapped == libc::MAP_FAILED {
            // Left mapped: the same pages, one more mapping than counted.
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "a deferred view's window of {length:#x} bytes at {at:#x} stays mapped: {}",
                std::io::Error::last_os_error()
            );
            return;
        }
        files::set_deferred(at, length, true);
    }

    /// Makes room for `pages` more mapped pages, dropping the oldest windows past the most
    /// ([`max_pages`]).
    fn make_room(&mut self, pages: u64) {
        let most = max_pages();
        while self.pages + pages > most {
            let Some(at) = self.order.pop_front() else {
                return;
            };
            if let Some(&length) = self.windows.get(&at) {
                self.drop_window(at, length);
            }
        }
    }
}

/// The process's deferred views: a leaf lock but for the table's, which it takes under it.
static STATE: Mutex<State> = Mutex::new(State {
    views: BTreeMap::new(),
    windows: BTreeMap::new(),
    order: VecDeque::new(),
    pages: 0,
});

/// The lowest first byte and highest end of any deferred view yet: a fault outside them is never
/// this module's, decided with no lock and no thread-local (a fault of code that runs with a
/// thread pointer not the host's, QEMU's native EL0, goes on untouched).
static LOW: AtomicU64 = AtomicU64::new(u64::MAX);
static HIGH: AtomicU64 = AtomicU64::new(0);

/// Counts for the reports ([`report`]).
static FAULTS: AtomicU64 = AtomicU64::new(0);
static VIEWS_MADE: AtomicU64 = AtomicU64::new(0);
static PINNED: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// A test's deferred views, on the thread that sets it, whatever the default.
    pub(super) static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether deferred views are on (module docs).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if TEST_ON.with(std::cell::Cell::get) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("REIMS_VGPU_DEFERRED_VIEWS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => cfg!(target_os = "android"),
    })
}

/// The most mapped pages ([`MAX_PAGES`]).
fn max_pages() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("REIMS_VGPU_DEFERRED_PAGES")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&pages| pages >= WINDOW / vm::page_size())
            .unwrap_or(MAX_PAGES)
    })
}

/// A fresh deferred view of `length` bytes, whole host pages: a reservation with no access, which
/// remaps into record their runs unmapped ([`remap`]). `Err` where the host gives no address
/// space.
pub(crate) fn allocate(length: u64) -> Result<HostAddr, KernReturn> {
    install_handler();
    // No access, as a dropped window is mapped again (`State::drop_window`), so the kernel joins
    // the two.
    // SAFETY: a fresh private reservation anywhere.
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length.to_usize(),
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if mapped == libc::MAP_FAILED {
        return Err(KernReturn(3));
    }
    let at = mapped.addr().to_u64();
    files::forget(at, length);
    STATE.lock().views.insert(
        at,
        View {
            end: at + length,
            pinned: false,
        },
    );
    LOW.fetch_min(at, Ordering::AcqRel);
    HIGH.fetch_max(at + length, Ordering::AcqRel);
    VIEWS_MADE.fetch_add(1, Ordering::Relaxed);
    report();
    Ok(HostAddr::from_addr(at.to_usize()))
}

/// Whether `length` bytes from `first` lie in one deferred view.
pub(crate) fn holds(first: u64, length: u64) -> bool {
    if first < LOW.load(Ordering::Acquire) {
        return false;
    }
    let end = first.saturating_add(length.max(1));
    STATE.lock().view_holding(first, end).is_some()
}

/// Records `pieces`, the source's pages, over `length` bytes from `target`, inside a deferred view
/// that is not pinned, as unmapped runs: `true`, where it did. The windows meeting them go first.
pub(crate) fn remap(target: u64, length: u64, pieces: &[files::Piece]) -> bool {
    let mut state = STATE.lock();
    let end = target.saturating_add(length);
    let Some(view) = state.view_holding(target, end) else {
        return false;
    };
    if state.views.get(&view).is_some_and(|v| v.pinned) {
        return false;
    }
    state.unmap_windows(target, end);
    for piece in pieces {
        files::record_deferred(
            target + piece.at,
            piece.length,
            piece.backing.clone(),
            piece.prot,
        );
    }
    true
}

/// Sets the protection of `length` bytes from `first`, inside a deferred view, to `bits`: in the
/// table, and of what of them the kernel maps. `true`, where they lie in one.
pub(crate) fn protect(first: u64, length: u64, bits: i32) -> bool {
    let state = STATE.lock();
    let end = first.saturating_add(length);
    let Some(view) = state.view_holding(first, end) else {
        return false;
    };
    files::protect(first, length, bits);
    let mapped: Vec<(u64, u64)> = if state.views.get(&view).is_some_and(|v| v.pinned) {
        vec![(first, end)]
    } else {
        (state.windows.range(..end).rev())
            .take_while(|&(&at, &len)| at + len > first)
            .map(|(&at, &len)| (at.max(first), (at + len).min(end)))
            .collect()
    };
    for (from, to) in mapped {
        // SAFETY: mapped pages of a deferred view, whose protection the caller sets.
        unsafe { libc::mprotect(vm::ptr(from), (to - from).to_usize(), bits) };
    }
    true
}

/// Forgets what of the deferred views lies in `length` bytes from `first`, which the caller then
/// unmaps or maps over: their windows, and the views or the parts of them there.
pub(crate) fn release(first: u64, length: u64) {
    if first.saturating_add(length) <= LOW.load(Ordering::Acquire) {
        return;
    }
    let end = first.saturating_add(length);
    let mut state = STATE.lock();
    let meeting: Vec<(u64, u64, bool)> = (state.views.range(..end).rev())
        .take_while(|(_, view)| view.end > first)
        .map(|(&at, view)| (at, view.end, view.pinned))
        .collect();
    if meeting.is_empty() {
        return;
    }
    // The windows go with the views: no `mmap` over them, which the caller unmaps or maps over.
    let windows: Vec<(u64, u64)> = (state.windows.range(..end).rev())
        .take_while(|&(&at, &len)| at + len > first)
        .map(|(&at, &len)| (at, len))
        .collect();
    for (at, len) in windows {
        state.windows.remove(&at);
        state.pages -= len / vm::page_size();
        if at < first || at + len > end {
            // A window cut by the release: what of it stays is mapped still, counted as a view's
            // run; the table keeps it so.
            let (keep_first, keep_end) = (at.max(end), at + len);
            if at < first {
                files::set_deferred(at, first - at, false);
            }
            if keep_first < keep_end {
                files::set_deferred(keep_first, keep_end - keep_first, false);
            }
        }
    }
    for (at, view_end, pinned) in meeting {
        state.views.remove(&at);
        if at < first {
            state.views.insert(
                at,
                View {
                    end: first,
                    pinned,
                },
            );
        }
        if end < view_end {
            state.views.insert(
                end,
                View {
                    end: view_end,
                    pinned,
                },
            );
        }
    }
}

/// Maps the whole deferred view holding `length` bytes from `first` for good (module docs), its
/// windows included: `true` where it is mapped now, `false` where a map was refused.
fn pin(first: u64, length: u64) -> bool {
    let mut state = STATE.lock();
    let end = first.saturating_add(length);
    let Some(view_first) = state.view_holding(first, end) else {
        // No deferred view: mapped as it is.
        return true;
    };
    let Some(view) = state.views.get(&view_first) else {
        return true;
    };
    if view.pinned {
        return true;
    }
    let view_end = view.end;
    state.unmap_windows(view_first, view_end);
    if !map_from_table(view_first, view_end - view_first) {
        return false;
    }
    if let Some(view) = state.views.get_mut(&view_first) {
        view.pinned = true;
    }
    PINNED.fetch_add(1, Ordering::Relaxed);
    log::info!(
        target: "reims_vgpu_vulkan::os",
        "a deferred view of {:#x} bytes at {view_first:#x} is mapped for good: the runtime \
         imports it at its address",
        view_end - view_first
    );
    true
}

/// Maps `length` bytes from `first`, a stretch of a deferred view, from the table's files at
/// their protections, and marks them mapped there: `false`, nothing marked, where a byte has no
/// file or the kernel refuses a map.
fn map_from_table(first: u64, length: u64) -> bool {
    let Some(pieces) = files::pieces(first, length) else {
        return false;
    };
    for piece in &pieces {
        if vm::file(Some(first + piece.at), piece.length, &piece.backing, piece.prot).is_err() {
            return false;
        }
    }
    files::set_deferred(first, length, false);
    true
}

/// The CPU's touch of byte `at`, which faulted: maps its window where it lies in a deferred view
/// and is not mapped yet (`true`, the access runs again), else `false`.
fn materialize(at: u64) -> bool {
    let mut state = STATE.lock();
    let Some((view_first, view)) = state.view_at(at) else {
        return false;
    };
    if view.pinned {
        return false;
    }
    let view_end = view.end;
    let first = view_first + (at - view_first) / WINDOW * WINDOW;
    let end = (first + WINDOW).min(view_end);
    if let Some((&window, &length)) = state.windows.range(..=at).next_back()
        && at < window + length
    {
        // Mapped already: another thread's touch mapped it after this one faulted, or the
        // access is one the mapping forbids (a write to a read-only view).
        return mapped_race(at);
    }
    // The window may meet windows mapped before a remap cut them: they go first.
    state.unmap_windows(first, end);
    let pages = (end - first) / vm::page_size();
    state.make_room(pages);
    if !map_from_table(first, end - first) {
        return false;
    }
    state.windows.insert(first, end - first);
    state.order.push_back(first);
    state.pages += pages;
    FAULTS.fetch_add(1, Ordering::Relaxed);
    LAST_RACE.with(|last| last.set(0));
    true
}

thread_local! {
    /// The address of this thread's last fault inside a mapped window ([`mapped_race`]).
    static LAST_RACE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// A fault at `at`, inside a window mapped now: the first time on this thread at that address, a
/// race with the thread that mapped it (`true`, the access runs again); the second, a fault of
/// the mapping's own (`false`).
fn mapped_race(at: u64) -> bool {
    LAST_RACE.with(|last| {
        if last.get() == at {
            last.set(0);
            false
        } else {
            last.set(at);
            true
        }
    })
}

/// The `SIGSEGV` action installed before this module's, which faults not its own go to.
static mut OLD: std::mem::MaybeUninit<libc::sigaction> = std::mem::MaybeUninit::uninit();

/// Installs the fault handler (module docs), once.
pub(super) fn install_handler() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        // SAFETY: a plain sigaction with this module's handler; the old action is written once,
        // here, before the handler can run, and read only by the handler.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault as *const () as usize;
            // On the faulting thread's own stack: a thread's alternate signal stack (Rust's, the
            // C library's) is a few KiB, too small for a window's mapping. Installed as the guest's
            // RAM is leased, before QEMU's vCPUs run, so QEMU's native-EL0 handler, which runs on
            // a stack of its own, comes before this one and hands it the host's faults alone.
            action.sa_flags = libc::SA_SIGINFO;
            libc::sigemptyset(&raw mut action.sa_mask);
            let old = (&raw mut OLD).cast::<libc::sigaction>();
            if libc::sigaction(libc::SIGSEGV, &action, old) != 0 {
                log::error!(
                    target: "reims_vgpu_vulkan::os",
                    "no fault handler for deferred views: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    });
}

/// The fault handler (module docs).
extern "C" fn on_fault(sig: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: the kernel's siginfo for a SIGSEGV, which carries the address.
    let at = unsafe { (*info).si_addr() }.addr().to_u64();
    if at >= LOW.load(Ordering::Acquire) && at < HIGH.load(Ordering::Acquire) && materialize(at) {
        return;
    }
    chain(sig, info, context);
}

/// Hands the fault to the action installed before this module's (module docs).
fn chain(sig: libc::c_int, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: OLD was written before the handler was installed and never since.
    let old = unsafe { (&raw const OLD).cast::<libc::sigaction>().read() };
    let handler = old.sa_sigaction;
    if handler == libc::SIG_DFL || handler == libc::SIG_IGN {
        // SAFETY: back to the default action; the access faults again and the kernel ends the
        // process there, as it would have.
        unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = libc::SIG_DFL;
            libc::sigaction(sig, &action, std::ptr::null_mut());
        }
        return;
    }
    if old.sa_flags & libc::SA_SIGINFO != 0 {
        // SAFETY: the previous handler, of the three-argument form its flags name.
        let f: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
            unsafe { std::mem::transmute(handler) };
        f(sig, info, context);
    } else {
        // SAFETY: the previous handler, of the one-argument form.
        let f: extern "C" fn(libc::c_int) = unsafe { std::mem::transmute(handler) };
        f(sig);
    }
}

/// The runtime's source for the bytes of a deferred view (module docs): where the guest's RAM
/// holds each of the requested pages; or, where a page lies elsewhere or misses the granule, or
/// the runtime asks for the window whole (`granule` its length: a device that gathers nothing),
/// the view mapped for good and its pointer.
pub(crate) fn gather_source() -> rt::GatherSource {
    std::sync::Arc::new(|request: rt::GatherRequest| {
        let first = request.start.as_ptr().addr().to_u64();
        let (length, granule) = (request.length, request.granule.max(1));
        let pointer = || {
            if pin(first, length) {
                Ok(rt::GatherAnswer::Pointer)
            } else {
                Err(std::io::Error::other("a deferred view could not be mapped"))
            }
        };
        if granule >= length {
            return pointer();
        }
        let Some(pieces) = files::pieces(first, length) else {
            return pointer();
        };
        let mut out: Vec<rt::GatherPiece> = Vec::with_capacity(pieces.len());
        for piece in &pieces {
            let Some((host, region_start, region_length)) =
                files::guest_place(&piece.backing, piece.length)
            else {
                return pointer();
            };
            let fits = piece.length.is_multiple_of(granule)
                && (host - region_start).is_multiple_of(granule);
            if !fits {
                return pointer();
            }
            // Pieces that follow on in the guest's RAM are one.
            match out.last_mut() {
                Some(last) if last.host + last.length == host && last.region_start == region_start => {
                    last.length += piece.length;
                }
                _ => out.push(rt::GatherPiece {
                    length: piece.length,
                    host,
                    region_start,
                    region_length,
                }),
            }
        }
        Ok(rt::GatherAnswer::Pieces(out))
    })
}

/// Whether every byte of `length` bytes from `first` is the guest's RAM, through the table: a view
/// of them can be deferred.
pub(crate) fn guest_backed(first: u64, length: u64) -> bool {
    files::pieces(first, length).is_some_and(|pieces| {
        pieces
            .iter()
            .all(|piece| files::guest_place(&piece.backing, piece.length).is_some())
    })
}

/// Logs the deferred views' and the runtime's gather counts, at most every 30 s.
fn report() {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    {
        let mut last = LAST.lock();
        if last.is_some_and(|at| at.elapsed() < Duration::from_secs(30)) {
            return;
        }
        *last = Some(Instant::now());
    }
    let (views, windows, pages) = {
        let state = STATE.lock();
        (state.views.len(), state.windows.len(), state.pages)
    };
    let c = rt::gather_counts();
    log::info!(
        target: "reims_vgpu_vulkan::os",
        "deferred views: {views} live ({} made, {} pinned), {windows} windows of {pages} pages \
         mapped by {} CPU faults; the library's mappings {}; gathered: {} buffers live of {} MiB \
         ({} made, {} binds in {} ms), {} chunks of {} MiB live ({} imported), {} pointer answers,          {} heap textures shadowed",
        VIEWS_MADE.load(Ordering::Relaxed),
        PINNED.load(Ordering::Relaxed),
        FAULTS.load(Ordering::Relaxed),
        files::mappings(),
        c.buffers_live,
        c.buffer_bytes_live >> 20,
        c.buffers,
        c.binds,
        c.bind_micros / 1000,
        c.chunks_live,
        c.chunk_bytes_live >> 20,
        c.chunks_imported,
        c.pointer_answers,
        c.heap_shadows,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::platform::os::vm::HostVm as _;

    fn read(at: u64) -> u8 {
        // SAFETY: test memory, readable once the handler maps it.
        unsafe { vm::ptr(at).cast::<u8>().read_volatile() }
    }

    fn write(at: u64, v: u8) {
        // SAFETY: test memory, writable once the handler maps it.
        unsafe { vm::ptr(at).cast::<u8>().write_volatile(v) }
    }

    /// A deferred view's runs take no mapping until the CPU touches them; a touch maps its
    /// window, the guest's pages both ways; the view goes as any range does
    #[test]
    fn a_deferred_view_maps_on_the_cpus_touch() {
        let v = super::super::LinuxVm::new();
        let p = vm::page_size();
        let pages = 2 * WINDOW / p;
        let src = vm::allocate_aliasable(pages * p).unwrap();
        let src = src.addr().to_u64();
        for i in 0..pages {
            write(src + i * p, i as u8);
        }
        let at = allocate(pages * p).unwrap().addr().to_u64();
        let end = at + pages * p;
        // In reverse order: a run each.
        for i in 0..pages {
            let from = HostAddr::from_addr((src + (pages - 1 - i) * p).to_usize());
            v.remap_shared_overwrite(HostAddr::from_addr((at + i * p).to_usize()), from, p)
                .unwrap();
        }
        assert_eq!(files::mappings_between(at, end), 0, "deferred: no mapping");
        assert_eq!(vm::tests::kernel_mappings(at, end), 1, "the reservation");
        assert_eq!(read(at + p), (pages - 2) as u8);
        assert_eq!(files::mappings_between(at, end), WINDOW / p, "the first window");
        write(at, 0xa5);
        assert_eq!(read(src + (pages - 1) * p), 0xa5, "the guest's page");
        // The window dropped: the reservation whole again, and a touch maps it again.
        STATE.lock().make_room(max_pages());
        assert_eq!(files::mappings_between(at, end), 0, "dropped");
        assert_eq!(vm::tests::kernel_mappings(at, end), 1, "joined again");
        assert_eq!(read(at), 0xa5);
        assert!(v.deallocate(HostAddr::from_addr(at.to_usize()), pages * p).is_success());
        assert!(!holds(at, 1));
        assert!(v.deallocate(HostAddr::from_addr(src.to_usize()), pages * p).is_success());
    }

    /// A buffer whose backing is replaced by scattered pages of the guest's RAM (a lease over a
    /// memfd, as QEMU's) lies in a deferred view the runtime gathers on the GPU: a GPU fill lands
    /// in the guest's pages in the ranges' order, the CPU reads it through the view, and the
    /// view's runs take no mapping but the windows the CPU touched. On Android (the phone's
    /// Turnip): lavapipe binds a sparse buffer only to memory of its own file, not to imported
    /// host memory, and the fill lands nowhere there.
    #[test]
    #[cfg(target_os = "android")]
    fn a_buffer_over_scattered_guest_pages_is_gathered() {
        use reims_vgpu_core::host::HostPtr;
        use reims_vgpu_core::platform::blit::{BlitCmd, BlitEncoder as _};
        use reims_vgpu_core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Buffer as _;
        use reims_vgpu_core::platform::types::{NsRange, ResourceAddressRange};
        use reims_vgpu_core::platform::values::{BufferDesc, OptionSet};
        use reims_vgpu_core::platform::{HostRange, Platform as _};
        use reims_vgpu_core::raise::OnFault;

        TEST_ON.with(|on| on.set(true));
        const GUEST: u64 = 64 << 20;
        const PIECE: u64 = 0x4000;
        const PIECES: u64 = 16;
        let fd = vm::memfd(GUEST);
        let backing = files::Backing {
            fd: std::sync::Arc::clone(&fd),
            offset: 0,
        };
        let guest = vm::file(None, GUEST, &backing, libc::PROT_READ | libc::PROT_WRITE).unwrap();
        let _lease = files::lease_guest_memory([crate::os::GuestRamFile {
            host: guest,
            length: GUEST,
            fd,
            offset: 0,
        }]);
        // Descending pieces, a chunk apart and more (the guest's page lists are LIFO).
        let piece = |i: u64| guest + GUEST - (i + 1) * (3 * PIECE + (i % 3) * (17 << 20) / 16);
        let p = crate::platform::tests::platform();
        let d = crate::gpu::device::tests::dev();
        let ranges: Vec<ResourceAddressRange> = (0..PIECES)
            .map(|i| ResourceAddressRange {
                address: piece(i),
                length: PIECE,
            })
            .collect();
        let ranges = p.new_address_range_array(&ranges).unwrap();
        let own = vm::allocate_aliasable(PIECES * PIECE).unwrap();
        // SAFETY: the pages stay mapped until the process ends.
        let contents = unsafe { HostRange::new(own, PIECES * PIECE) };
        let buffer = d
            .new_buffer(BufferDesc {
                length: PIECES * PIECE,
                options: vec![OptionSet::StorageMode(0)],
                pinned_gpu_address: None,
                contents: Some(contents),
                no_copy: true,
                deallocator: None,
            })
            .unwrap()
            .unwrap();
        buffer.detach_backing().unwrap();
        let before = rt::gather_counts();
        assert!(buffer.replace_backing(&ranges, false).unwrap());
        let at = buffer.contents().addr().to_u64();
        assert!(holds(at, PIECES * PIECE), "a deferred view");
        assert_eq!(files::mappings_between(at, at + PIECES * PIECE), 0);
        let q = d.new_command_queue().unwrap();
        let cb = q.command_buffer(OnFault::Return).unwrap().unwrap();
        let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
        let fill = BlitCmd::FillBuffer {
            buffer: &buffer,
            range: NsRange::new(PIECE, PIECES * PIECE - 2 * PIECE),
            value: 0x5c,
        };
        e.encode(OnFault::Return, fill).unwrap();
        e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        let after = rt::gather_counts();
        eprintln!("gather counts before {before:?} after {after:?}");
        assert!(
            after.buffers > before.buffers || after.pointer_answers > before.pointer_answers,
            "gathered, or imported at its pointer where the device gathers nothing"
        );
        // Where the fill landed in the guest's RAM, for the failure's message.
        let mut landed = Vec::new();
        let mut at_page = guest;
        while at_page < guest + GUEST {
            if read(at_page) == 0x5c {
                landed.push((at_page - guest) / PIECE);
            }
            at_page += PIECE;
        }
        eprintln!(
            "the fill's pages in the guest: {landed:?}; wanted {:?}",
            (1..PIECES - 1).map(|i| (piece(i) - guest) / PIECE).collect::<Vec<_>>()
        );
        for i in 0..PIECES {
            let want = if i == 0 || i == PIECES - 1 { 0 } else { 0x5c };
            let first = read(piece(i));
            let last = read(piece(i) + PIECE - 1);
            assert_eq!((first, last), (want, want), "the guest's piece {i}");
            assert_eq!(read(at + i * PIECE), want, "the view's piece {i}");
        }
        let _ = HostPtr::NULL;
        drop(buffer);
        TEST_ON.with(|on| on.set(false));
    }

    /// A texture placed in a heap over scattered pages of the guest's RAM is a sparse linear
    /// image bound to the chunks they lie in (or, where the device makes none, a shadow of the
    /// heap's bytes): its bytes round-trip, and its first row lies at the heap's first byte, in
    /// the guest's first page
    #[test]
    #[cfg(target_os = "android")]
    fn a_heap_texture_over_scattered_guest_pages_is_a_sparse_image() {
        use reims_vgpu_core::platform::Platform as _;
        use reims_vgpu_core::platform::device::GpuDevice as _;
        use reims_vgpu_core::platform::resource::Heap as _;
        use reims_vgpu_core::platform::types::ResourceAddressRange;
        use reims_vgpu_core::platform::values::{HeapDesc, OptionSet, TextureDesc};

        TEST_ON.with(|on| on.set(true));
        const GUEST: u64 = 64 << 20;
        const PIECE: u64 = 0x4000;
        const PIECES: u64 = 16;
        let fd = vm::memfd(GUEST);
        let backing = files::Backing {
            fd: std::sync::Arc::clone(&fd),
            offset: 0,
        };
        let guest = vm::file(None, GUEST, &backing, libc::PROT_READ | libc::PROT_WRITE).unwrap();
        let _lease = files::lease_guest_memory([crate::os::GuestRamFile {
            host: guest,
            length: GUEST,
            fd,
            offset: 0,
        }]);
        let piece = |i: u64| guest + GUEST - (i + 1) * (3 * PIECE + (i % 3) * (17 << 20) / 16);
        let p = crate::platform::tests::platform();
        let d = crate::gpu::device::tests::dev();
        let ranges: Vec<ResourceAddressRange> = (0..PIECES)
            .map(|i| ResourceAddressRange {
                address: piece(i),
                length: PIECE,
            })
            .collect();
        let ranges = p.new_address_range_array(&ranges).unwrap();
        let heap = d
            .new_heap(&HeapDesc {
                size: PIECES * PIECE,
                heap_type: 1,
                options: vec![OptionSet::StorageMode(0), OptionSet::HazardTrackingMode(0)],
                address_ranges: Some(&ranges),
                pinned_gpu_address: None,
            })
            .unwrap()
            .heap
            .unwrap();
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 128,
            height: 128,
            usage: 0x1,
            resource_options: Some(heap.resource_options()),
            ..TextureDesc::default()
        });
        let before = rt::gather_counts();
        let t = heap.new_texture(&desc, 0).unwrap().unwrap();
        let after = rt::gather_counts();
        eprintln!(
            "heap texture: sparse images {} -> {}, shadows {} -> {}",
            before.heap_sparse_images,
            after.heap_sparse_images,
            before.heap_shadows,
            after.heap_shadows
        );
        let bytes = crate::gpu::resource::tests::pattern(128, 128);
        crate::gpu::resource::tests::fill(&t, &bytes);
        assert_eq!(crate::gpu::resource::tests::read_back(&t), bytes, "round trip");
        let first: Vec<u8> = (0..4).map(|i| read(piece(0) + i)).collect();
        assert_eq!(first, bytes[..4].to_vec(), "the first texel in the guest's first page");
        assert!(
            after.heap_sparse_images > before.heap_sparse_images
                || after.heap_shadows > before.heap_shadows
        );
        drop(t);
        drop(heap);
        TEST_ON.with(|on| on.set(false));
    }
}
