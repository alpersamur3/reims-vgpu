//! Mach VM's calls (`HostVm`) over Win32 virtual memory: placeholders, private pages and
//! section views.
//!
//! Win32 cannot remap an address range's pages to another address as `mach_vm_remap` does, and
//! maps a view at a given address only into address space reserved as a **placeholder**
//! (`VirtualAlloc2` with `MEM_RESERVE_PLACEHOLDER`), which a view (`MapViewOfFile3` with
//! `MEM_REPLACE_PLACEHOLDER`) or private pages (`VirtualAlloc2` with the same flag) then replace
//! exactly, whole. So:
//!
//! - **Placement.** Every range this VM makes is a placeholder first, at or above 4 GiB
//!   (`MEM_ADDRESS_REQUIREMENTS::LowestStartingAddress`, as core's `GuestPtr` dies on any address
//!   below `ZERO_REGION_END`), replaced whole by private committed pages (`allocate_anywhere`:
//!   zero-filled, with no section behind them, so no view can be made of them, as of Linux's
//!   anonymous memory) or by a view of a fresh pagefile section of its own
//!   ([`allocate_aliasable`]: zero-filled, and a view can be made of it). The views made of the
//!   guest's RAM lie at placeholders too, so they are at or above 4 GiB as well.
//! - **Views** (`remap_shared_overwrite`) map the sections the backing table names for the
//!   source's bytes ([`super::files`]: the host's sections behind the guest's RAM, and the
//!   library's own). The target's bytes are given back to placeholders first ([`clear`]): private
//!   pages freed back to one with `MEM_PRESERVE_PLACEHOLDER`, which splits an allocation at any
//!   page, and a view unmapped whole (`UnmapViewOfFile2`, which unmaps nothing less) with the rest
//!   of it mapped again from its section. The placeholders are then carved into one per piece
//!   ([`isolate`]) and each piece's section mapped over its own. Views are page-granular, 4 KiB:
//!   a placeholder is replaced at any page, and a section mapped from any page offset (measured
//!   on Windows 11 24H2, `reims-win-tools/harness/probe-placeholders.c`: no 64 KiB rule applies to
//!   a placeholder's replacement). A piece the kernel refuses fails the call with nothing changed,
//!   as Mach's remap is all or nothing: the target is put back as it was ([`put_back`]). A source
//!   with a byte no section is behind fails with `KERN_INVALID_ADDRESS` before anything changes,
//!   as Mach fails for an unmapped source.
//! - **Fresh zeros** (`allocate_fixed_overwrite`) are private pages again: the pages given back
//!   to placeholders, coalesced into one, and committed read and write, zero-filled by Windows,
//!   so they show no section from then on.
//! - **Deallocation** (`deallocate`) gives the pages back to placeholders and releases each, so
//!   their address space is free, as after `munmap`: part of a range as well as the whole.
//!   Registered memory (the guest's RAM) is refused, unmapped by nothing.
//! - **Protection** (`protect_max`) is `VirtualProtect`, region by region, as a range may span
//!   the allocations the splits above made: Windows has no maximum protection, so a lowered
//!   protection can be raised again, where Mach refuses the raise; a forbidden access is an
//!   access violation, a death either way.
//! - Each call takes whole host pages (`GetSystemInfo`'s 4 KiB), from the page of its first byte
//!   to the page past its last, as Mach's do.
//!
//! The calls that map and unmap hold the table's mutation lock across their Win32 calls
//! ([`files::mutation`]), so no two threads carve the same placeholders at once; nothing of the
//! library runs under it. A Win32 refusal is answered as Mach would answer: no address space or
//! memory is `KERN_NO_SPACE`, a range not mapped `KERN_INVALID_ADDRESS`, a bad argument
//! `KERN_INVALID_ARGUMENT`, anything else `KERN_FAILURE` ([`refused`]).

use std::ffi::c_void;
use std::os::windows::io::{AsRawHandle as _, FromRawHandle as _, OwnedHandle};
use std::sync::Arc;

use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use windows_sys::Win32::Foundation::{
    ERROR_ACCESS_DENIED, ERROR_COMMITMENT_LIMIT, ERROR_INVALID_ADDRESS, ERROR_INVALID_PARAMETER,
    ERROR_NOT_ENOUGH_MEMORY, ERROR_OUTOFMEMORY, GetLastError, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_EXTENDED_PARAMETER_0, MEM_EXTENDED_PARAMETER_1, MEM_FREE, MEM_MAPPED,
    MEM_PRESERVE_PLACEHOLDER, MEM_PRIVATE, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY,
    PAGE_READWRITE, PAGE_WRITECOPY, SEC_COMMIT, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
    VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::files::{self, Backing, Piece};

/// `KERN_INVALID_ADDRESS`.
const INVALID_ADDRESS: KernReturn = KernReturn(1);
/// `KERN_NO_SPACE`.
const NO_SPACE: KernReturn = KernReturn(3);
/// `KERN_INVALID_ARGUMENT`.
const INVALID_ARGUMENT: KernReturn = KernReturn(4);
/// `KERN_FAILURE`.
const FAILURE: KernReturn = KernReturn(5);

/// The lowest address a range of this VM may start at: 4 GiB, core's `ZERO_REGION_END`.
pub(crate) const LOWEST: u64 = 0x1_0000_0000;

/// `VM_PROT_EXECUTE`.
const PROT_EXECUTE: i32 = 4;

/// `VirtualFree`'s `MEM_COALESCE_PLACEHOLDERS`: with `MEM_RELEASE`, merges the adjacent
/// placeholders that exactly fill the range into one. (`windows-sys` 0.61 lacks the name.)
const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;

/// The host's page size (`GetSystemInfo`'s `dwPageSize`, 4 KiB on x86-64).
pub(crate) fn page_size() -> u64 {
    static PAGE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PAGE.get_or_init(|| {
        // SAFETY: an all-zero SYSTEM_INFO is a valid value to fill.
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: a live out-parameter; the call cannot fail.
        unsafe { GetSystemInfo(&mut info) };
        let page = u64::from(info.dwPageSize);
        if page.is_power_of_two() { page } else { 0x1000 }
    })
}

/// The pages `len` bytes from `addr` lie in: the first page's address and the length in whole
/// pages; `None` where the end overflows.
fn pages(addr: HostAddr, len: u64) -> Option<(u64, u64)> {
    let page = page_size();
    let start = addr.addr().to_u64();
    let first = start & !(page - 1);
    let end = start.checked_add(len)?.checked_add(page - 1)? & !(page - 1);
    Some((first, end - first))
}

/// `first` as a pointer for a Win32 call.
pub(super) fn ptr(first: u64) -> *mut c_void {
    std::ptr::with_exposed_provenance_mut(first.to_usize())
}

/// A Mach answer for the Win32 error of a refused call, which `what` names and the debug log
/// shows: no memory or address space is `KERN_NO_SPACE`, a range not mapped
/// `KERN_INVALID_ADDRESS`, a bad argument `KERN_INVALID_ARGUMENT`, else `KERN_FAILURE`.
fn refused(what: &str) -> KernReturn {
    let error = std::io::Error::last_os_error();
    // SAFETY: a thread-local read; no memory of the caller's is touched.
    let code = unsafe { GetLastError() };
    log::debug!(target: "reims_vgpu_vulkan::os", "{what}: refused by Win32: {error}");
    match code {
        ERROR_NOT_ENOUGH_MEMORY | ERROR_OUTOFMEMORY | ERROR_COMMITMENT_LIMIT => NO_SPACE,
        ERROR_INVALID_ADDRESS => INVALID_ADDRESS,
        ERROR_INVALID_PARAMETER => INVALID_ARGUMENT,
        ERROR_ACCESS_DENIED => INVALID_ADDRESS,
        _ => FAILURE,
    }
}

/// The region `VirtualQuery` reports at `at`; `None` past the user address space.
fn query(at: u64) -> Option<MEMORY_BASIC_INFORMATION> {
    // SAFETY: an all-zero MEMORY_BASIC_INFORMATION is a valid value to fill.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a live out-parameter of the size given; any address may be queried.
    let n = unsafe {
        VirtualQuery(
            ptr(at),
            &mut info,
            std::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    (n != 0).then_some(info)
}

/// A region as `VirtualQuery` reports it: its first byte, its end, its state (`MEM_COMMIT`,
/// `MEM_RESERVE`, `MEM_FREE`), its type (`MEM_PRIVATE`, `MEM_MAPPED`, `MEM_IMAGE`, 0 where free)
/// and its protection.
#[derive(Clone, Copy, Debug)]
struct Region {
    first: u64,
    end: u64,
    state: u32,
    kind: u32,
    protect: PAGE_PROTECTION_FLAGS,
}

impl Region {
    /// Whether the region is a placeholder, or a reservation, which this VM never makes.
    fn placeholder(&self) -> bool {
        self.state == MEM_RESERVE
    }
}

/// The region holding `at`; `None` past the user address space.
fn region_at(at: u64) -> Option<Region> {
    let info = query(at)?;
    let first = info.BaseAddress.addr().to_u64();
    let end = first.checked_add(info.RegionSize.to_u64())?;
    (end > at).then_some(Region {
        first,
        end,
        state: info.State,
        kind: info.Type,
        protect: info.Protect,
    })
}

/// The regions `first..end` lies in, each clipped to it, in order; `None` where the range runs
/// past the user address space.
fn regions(first: u64, end: u64) -> Option<Vec<Region>> {
    let mut out = Vec::new();
    let mut at = first;
    while at < end {
        let region = region_at(at)?;
        out.push(Region {
            first: region.first.max(first),
            end: region.end.min(end),
            ..region
        });
        at = region.end;
    }
    Some(out)
}

/// Whether a page protection reads.
pub(super) fn reads(protect: PAGE_PROTECTION_FLAGS) -> bool {
    protect & PAGE_GUARD == 0
        && protect
            & (PAGE_READONLY
                | PAGE_READWRITE
                | PAGE_WRITECOPY
                | PAGE_EXECUTE_READ
                | PAGE_EXECUTE_READWRITE
                | PAGE_EXECUTE_WRITECOPY)
            != 0
}

/// Whether every page of `length` bytes from `first` is committed with a protection `ok` takes.
fn every_page(first: u64, length: u64, ok: impl Fn(PAGE_PROTECTION_FLAGS) -> bool) -> bool {
    let end = first.saturating_add(length);
    regions(first, end).is_some_and(|regions| {
        regions
            .iter()
            .all(|r| r.state == MEM_COMMIT && ok(r.protect))
    })
}

/// The Win32 protection for Mach's `prot`.
fn protection(prot: VmProt) -> PAGE_PROTECTION_FLAGS {
    let raw = prot.raw();
    let read = raw & VmProt::READ.raw() != 0;
    let write = raw & VmProt::WRITE.raw() != 0;
    let execute = raw & PROT_EXECUTE != 0;
    match (read || write || execute, write, execute) {
        (false, _, _) => PAGE_NOACCESS,
        (true, true, true) => PAGE_EXECUTE_READWRITE,
        (true, false, true) => PAGE_EXECUTE_READ,
        (true, true, false) => PAGE_READWRITE,
        (true, false, false) => PAGE_READONLY,
    }
}

/// The current process's pseudo-handle.
fn this_process() -> *mut c_void {
    // SAFETY: the call reads and writes no memory of the caller's.
    unsafe { GetCurrentProcess() }
}

/// A fresh placeholder of `length` bytes, whole pages, at or above 4 GiB (module docs): its
/// first byte.
fn reserve_placeholder(length: u64) -> Result<u64, KernReturn> {
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: ptr(LOWEST),
        HighestEndingAddress: std::ptr::null_mut(),
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER {
        Anonymous1: MEM_EXTENDED_PARAMETER_0 {
            _bitfield: MemExtendedParameterAddressRequirements as u64,
        },
        Anonymous2: MEM_EXTENDED_PARAMETER_1 {
            Pointer: (&raw mut requirements).cast(),
        },
    };
    // SAFETY: the current process's pseudo-handle, no base address, one extended parameter
    // that points at a live address requirement for the call's length.
    let at = unsafe {
        VirtualAlloc2(
            this_process(),
            std::ptr::null(),
            length.to_usize(),
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            &mut parameter,
            1,
        )
    };
    if at.is_null() {
        return Err(refused("a placeholder"));
    }
    Ok(at.addr().to_u64())
}

/// Replaces the placeholder that is exactly `first..first+length` with private committed pages
/// at `protect`, zero-filled.
fn commit_private(first: u64, length: u64, protect: PAGE_PROTECTION_FLAGS) -> Result<(), KernReturn> {
    // SAFETY: the base and length of a placeholder of this process, which only private pages
    // then take.
    let at = unsafe {
        VirtualAlloc2(
            this_process(),
            ptr(first),
            length.to_usize(),
            MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
            protect,
            std::ptr::null_mut(),
            0,
        )
    };
    if at.is_null() {
        return Err(refused("private pages over a placeholder"));
    }
    Ok(())
}

/// Frees `first..first+length`, pages of a private allocation this VM made or of a placeholder,
/// back to a placeholder of its own, splitting what holds them at either end.
fn free_to_placeholder(first: u64, length: u64) -> Result<(), KernReturn> {
    // SAFETY: pages of this process that only this VM uses; their contents go.
    let done = unsafe {
        VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        )
    };
    if done == 0 {
        return Err(refused("freeing pages back to a placeholder"));
    }
    Ok(())
}

/// Merges the adjacent placeholders that exactly fill `first..first+length` into one.
fn coalesce(first: u64, length: u64) -> bool {
    // SAFETY: placeholders of this process, which only this VM uses.
    unsafe {
        VirtualFree(
            ptr(first),
            length.to_usize(),
            MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
        ) != 0
    }
}

/// Releases the placeholder or allocation whose base is `first`: its address space is free.
fn release(first: u64) -> bool {
    // SAFETY: the base of an allocation of this process that nothing uses any more.
    unsafe { VirtualFree(ptr(first), 0, MEM_RELEASE) != 0 }
}

/// Maps `length` bytes of `backing` over the placeholder that is exactly `first..first+length`,
/// read and write, and lists the view.
fn map_view(first: u64, length: u64, backing: &Backing) -> Result<(), KernReturn> {
    // SAFETY: an open section handle, the current process, and the base and length of a
    // placeholder of this process; no extended parameters.
    let view = unsafe {
        MapViewOfFile3(
            backing.section.as_raw_handle(),
            this_process(),
            ptr(first),
            backing.offset,
            length.to_usize(),
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value.is_null() {
        return Err(refused("a view of a section"));
    }
    files::record_view(first, length, backing.clone());
    Ok(())
}

/// Unmaps the view listed at `first`, whole, leaving its placeholder, and unlists it: the view,
/// as listed; `None` for no view there, or one Win32 keeps (logged).
fn unmap_view(first: u64) -> Option<files::View> {
    let view = files::view_at(first)?;
    // SAFETY: the base of a view this VM mapped, which nothing uses once its bytes are taken.
    let done = unsafe {
        UnmapViewOfFile2(
            this_process(),
            MEMORY_MAPPED_VIEW_ADDRESS {
                Value: ptr(view.first),
            },
            MEM_PRESERVE_PLACEHOLDER,
        )
    };
    if done == 0 {
        log::error!(
            target: "reims_vgpu_vulkan::os",
            "the view of {:#x} bytes at {:#x} was not unmapped: {}",
            view.length,
            view.first,
            std::io::Error::last_os_error()
        );
        return None;
    }
    files::forget_view(view.first);
    Some(view)
}

/// Sets `protect` on `first..first+length`, region by region (module docs).
fn protect_pages(first: u64, length: u64, protect: PAGE_PROTECTION_FLAGS) -> Result<(), KernReturn> {
    let end = first.saturating_add(length);
    let Some(regions) = regions(first, end) else {
        return Err(INVALID_ADDRESS);
    };
    for region in regions {
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        // SAFETY: committed pages of this process and a live out-parameter.
        let done = unsafe {
            VirtualProtect(
                ptr(region.first),
                (region.end - region.first).to_usize(),
                protect,
                &mut old,
            )
        };
        if done == 0 {
            return Err(refused("a protection"));
        }
    }
    Ok(())
}

/// Maps `length` bytes of `backing` again over the placeholder `first..first+length`, the rest
/// of a view unmapped whole to take part of it: read and write, then each run's own protection
/// where the table has it set apart. Where Win32 refuses, the bytes are forgotten by the table
/// and stay a placeholder (an access there is a death), logged.
fn map_rest(first: u64, length: u64, backing: &Backing) {
    if let Err(kr) = map_view(first, length, backing) {
        log::error!(
            target: "reims_vgpu_vulkan::os",
            "the rest of a view, {length:#x} bytes at {first:#x}, was not mapped again \
             (kern_return {}); the range is forgotten",
            kr.0
        );
        files::forget(first, length);
        return;
    }
    for (at, len, piece) in files::run_pieces(first, length) {
        if let Some(piece) = piece
            && piece.prot != PAGE_READWRITE
        {
            let _ = protect_pages(at, len, piece.prot);
        }
    }
}

/// Gives the pages `first..end` back to placeholders, keeping every byte outside them as it is:
/// private pages are freed back to a placeholder, a view that meets the range is unmapped whole
/// and what of it lies outside the range mapped again from its section ([`map_rest`]). A
/// placeholder stays one. A free page is `KERN_INVALID_ADDRESS` unless `free_ok`; a view the
/// table does not list (the host's RAM) is `KERN_INVALID_ADDRESS`. Under the mutation lock.
fn clear(first: u64, end: u64, free_ok: bool) -> Result<(), KernReturn> {
    let mut at = first;
    while at < end {
        let Some(region) = region_at(at) else {
            return Err(INVALID_ADDRESS);
        };
        let (from, to) = (region.first.max(first), region.end.min(end));
        match (region.state, region.kind) {
            (MEM_FREE, _) => {
                if !free_ok {
                    return Err(INVALID_ADDRESS);
                }
            }
            (MEM_RESERVE, _) => {}
            (_, MEM_PRIVATE) => free_to_placeholder(from, to - from)?,
            (_, MEM_MAPPED) => {
                let Some(view) = unmap_view(at) else {
                    return Err(INVALID_ADDRESS);
                };
                let view_end = view.first + view.length;
                let (cut_first, cut_end) = (view.first.max(first), view_end.min(end));
                if cut_first > view.first || cut_end < view_end {
                    free_to_placeholder(cut_first, cut_end - cut_first)?;
                }
                if cut_first > view.first {
                    map_rest(view.first, cut_first - view.first, &view.backing);
                }
                if cut_end < view_end {
                    let mut rest = view.backing.clone();
                    rest.offset += cut_end - view.first;
                    map_rest(cut_end, view_end - cut_end, &rest);
                }
                at = view_end.max(to);
                continue;
            }
            _ => return Err(INVALID_ADDRESS),
        }
        at = to;
    }
    Ok(())
}

/// Makes `first..first+length`, placeholders now, exactly one placeholder: the placeholders
/// holding it coalesced where it spans more than one, and what of them lies outside it split
/// off.
fn isolate(first: u64, length: u64) -> Result<(), KernReturn> {
    let end = first + length;
    let Some(regions) = regions(first, end) else {
        return Err(INVALID_ADDRESS);
    };
    if regions.iter().any(|r| !r.placeholder()) {
        return Err(INVALID_ADDRESS);
    }
    let (Some(head), Some(tail)) = (regions.first(), regions.last()) else {
        return Ok(());
    };
    // The whole placeholders, unclipped, that hold the range's first and last pages.
    let span_first = region_at(head.first).map_or(first, |r| r.first);
    let span_end = region_at(tail.end - 1).map_or(end, |r| r.end);
    if regions.len() > 1 && !coalesce(span_first, span_end - span_first) {
        return Err(refused("coalescing placeholders"));
    }
    if span_first < first || end < span_end {
        free_to_placeholder(first, length)?;
    }
    Ok(())
}

/// Puts `length` bytes from `first`, placeholders now after a remap that failed part way, back to
/// what they showed before it, so the failed remap changes nothing: the table's runs mapped
/// again from their sections at the protection each was set to (the remap records nothing until
/// every piece is mapped), and fresh private pages with no access over bytes of no section.
///
/// Bytes of no section in a remap's target are a reservation: a mapping task's, which has no
/// access, or a fresh range its caller frees after the failure (core's range mappings, an alias,
/// a surface's view), so the protection of those pages never shows, and their old bytes, zeros
/// the caller never read, are not kept ([`crate::remap`]). A put-back Win32 refuses too leaves
/// the range forgotten by the table, logged.
fn put_back(first: u64, length: u64) {
    for (at, len, piece) in files::run_pieces(first, length) {
        let restored = match &piece {
            Some(piece) => {
                isolate(at, len).is_ok()
                    && map_view(at, len, &piece.backing).is_ok()
                    && (piece.prot == PAGE_READWRITE || protect_pages(at, len, piece.prot).is_ok())
            }
            None => isolate(at, len).is_ok() && commit_private(at, len, PAGE_NOACCESS).is_ok(),
        };
        if !restored {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "a failed remap's put-back of {len:#x} bytes at {at:#x} was refused: {}; the \
                 range is forgotten",
                std::io::Error::last_os_error()
            );
            files::forget(at, len);
        }
    }
}

/// A pagefile section of `length` bytes of zeros (`CreateFileMappingW` on no file,
/// `SEC_COMMIT`), which views of the library's own memory are made of; `KERN_NO_SPACE` where
/// Windows gives none.
pub(crate) fn section(length: u64) -> Result<Arc<OwnedHandle>, KernReturn> {
    let (high, low) = (
        u32::try_from(length >> 32).unwrap_or(u32::MAX),
        u32::try_from(length & 0xffff_ffff).unwrap_or(u32::MAX),
    );
    // SAFETY: no file (the pagefile), default security, no name; the handle is owned from here
    // on.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE | SEC_COMMIT,
            high,
            low,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return Err(refused("a pagefile section"));
    }
    // SAFETY: a handle just made, owned once.
    Ok(Arc::new(unsafe { OwnedHandle::from_raw_handle(handle) }))
}

/// A fresh range of `len` bytes, rounded up to whole pages, of zeros, that a view can be made of:
/// a view of a pagefile section of its own over a placeholder, recorded in the backing table.
/// Freed by `deallocate`, as any other range. The block of zeros a detached object reads, and
/// the tests' stand-in for guest memory, are such ranges.
pub(crate) fn allocate_aliasable(len: u64) -> Result<HostAddr, KernReturn> {
    let Some((_, length)) = pages(HostAddr::NULL, len) else {
        return Err(NO_SPACE);
    };
    if length == 0 {
        return Ok(HostAddr::NULL);
    }
    let backing = Backing {
        section: section(length)?,
        offset: 0,
    };
    let _mutation = files::mutation();
    let at = reserve_placeholder(length)?;
    if let Err(kr) = map_view(at, length, &backing) {
        release(at);
        return Err(kr);
    }
    files::record(at, length, backing, PAGE_READWRITE);
    Ok(HostAddr::from_addr(at.to_usize()))
}

/// Memory of the library's own that a view can be made of and that only its holder frees: a
/// view of a new pagefile section over a placeholder of its own, read and write, at or above
/// 4 GiB, registered in the backing table ([`files::Registration`]) rather than recorded as a
/// run, so that `deallocate` refuses it, as on Linux. An IOSurface stand-in's memory.
#[derive(Debug)]
pub(crate) struct OwnMemory {
    first: u64,
    registration: Option<files::Registration>,
}

impl OwnMemory {
    /// `len` bytes, rounded up to whole pages, of zeros; `None` for none, or where Windows gives
    /// no section or no address space.
    pub(crate) fn new(len: u64) -> Option<OwnMemory> {
        let (_, length) = pages(HostAddr::NULL, len)?;
        if length == 0 {
            return None;
        }
        let backing = Backing {
            section: section(length).ok()?,
            offset: 0,
        };
        let _mutation = files::mutation();
        let first = reserve_placeholder(length).ok()?;
        if map_view(first, length, &backing).is_err() {
            release(first);
            return None;
        }
        Some(OwnMemory {
            first,
            registration: Some(files::Registration::new(first, length, backing)),
        })
    }

    /// The first byte.
    pub(crate) fn first(&self) -> u64 {
        self.first
    }
}

impl Drop for OwnMemory {
    /// Drops the registration, then the view and the placeholder under it. The section goes
    /// with the last view or run of it: a view another range made of this memory keeps showing
    /// its pages.
    fn drop(&mut self) {
        drop(self.registration.take());
        let _mutation = files::mutation();
        if unmap_view(self.first).is_some() {
            release(self.first);
        }
    }
}

/// Mach VM's calls on the current process (module docs).
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsVm;

impl WindowsVm {
    /// The process's VM.
    pub fn new() -> WindowsVm {
        WindowsVm
    }

    /// Whether every page of the `len` bytes at `addr` is committed with read access, as
    /// `VirtualQuery` reports them (a view's pages and private pages alike). A `len` of 0 is
    /// `true`.
    pub fn readable(&self, addr: HostAddr, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let Some((first, length)) = pages(addr, len) else {
            return false;
        };
        every_page(first, length, reads)
    }
}

impl HostVm for WindowsVm {
    fn page_size(&self) -> u64 {
        page_size()
    }

    fn page_shift(&self) -> u32 {
        page_size().trailing_zeros()
    }

    /// A fresh range of `len` bytes, whole pages, of private pages, zero-filled, read and write,
    /// at or above 4 GiB, over a placeholder of its own (module docs); no view can be made of
    /// it. A length of 0 answers address 0, as Mach's does.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let Some((_, length)) = pages(HostAddr::NULL, len) else {
            return Err(NO_SPACE);
        };
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
        let at = reserve_placeholder(length)?;
        if let Err(kr) = commit_private(at, length, PAGE_READWRITE) {
            release(at);
            return Err(kr);
        }
        files::forget(at, length);
        Ok(HostAddr::from_addr(at.to_usize()))
    }

    /// Fresh private pages, zero-filled, read and write, over the pages of the range, which show
    /// no section from then on (module docs).
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let (first, length) = pages(addr, len).ok_or(NO_SPACE)?;
        if length == 0 {
            return Ok(());
        }
        let _mutation = files::mutation();
        clear(first, first + length, false)?;
        isolate(first, length)?;
        commit_private(first, length, PAGE_READWRITE)?;
        files::forget(first, length);
        Ok(())
    }

    /// The source's pages, from the page of its first byte, mapped from their sections over
    /// `dst`'s pages, each with the protection the source has there (module docs).
    fn remap_shared_overwrite(
        &self,
        dst: HostAddr,
        src: HostAddr,
        len: u64,
    ) -> Result<HostAddr, KernReturn> {
        let mask = page_size() - 1;
        let into = src.addr().to_u64() & mask;
        let (target, length) = pages(dst, len.saturating_add(into)).ok_or(INVALID_ARGUMENT)?;
        let (source, _) = pages(src, len).ok_or(INVALID_ADDRESS)?;
        if length == 0 {
            return Ok(dst);
        }
        let Some(pieces) = files::pieces(source, length) else {
            log::debug!(
                target: "reims_vgpu_vulkan::os",
                "remap of {length:#x} bytes at {source:#x}: no section behind them"
            );
            return Err(INVALID_ADDRESS);
        };
        let _mutation = files::mutation();
        // The target given back to placeholders, then one placeholder per piece, each mapped.
        if let Err(kr) = clear(target, target + length, false) {
            put_back(target, length);
            return Err(kr);
        }
        let mapped = |piece: &Piece| -> Result<(), KernReturn> {
            let at = target + piece.at;
            isolate(at, piece.length)?;
            map_view(at, piece.length, &piece.backing)?;
            if piece.prot != PAGE_READWRITE {
                protect_pages(at, piece.length, piece.prot)?;
            }
            Ok(())
        };
        for piece in &pieces {
            if let Err(kr) = mapped(piece) {
                put_back(target, length);
                return Err(kr);
            }
        }
        for piece in pieces {
            files::record(target + piece.at, piece.length, piece.backing, piece.prot);
        }
        Ok(dst)
    }

    /// `VirtualProtect` of the range's pages, which must all be committed (module docs).
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        let (first, length) = pages(addr, len).ok_or(INVALID_ARGUMENT)?;
        if length == 0 {
            return Ok(());
        }
        if !every_page(first, length, |_| true) {
            return Err(INVALID_ADDRESS);
        }
        let protect = protection(prot);
        protect_pages(first, length, protect)?;
        files::protect(first, length, protect);
        Ok(())
    }

    /// The range's pages given back to placeholders and released, so their address space is
    /// free (module docs). A range of memory the library registered and does not own (the
    /// guest's RAM) is refused, unmapped by nothing.
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        let Some((first, length)) = pages(addr, len) else {
            return INVALID_ARGUMENT;
        };
        if length == 0 {
            return KernReturn::SUCCESS;
        }
        if files::meets_registration(first, length) {
            log::error!(
                target: "reims_vgpu_vulkan::os",
                "deallocate of {length:#x} bytes at {first:#x}: registered memory, not unmapped"
            );
            return INVALID_ARGUMENT;
        }
        let end = first + length;
        let _mutation = files::mutation();
        files::forget(first, length);
        if let Err(kr) = clear(first, end, true) {
            return kr;
        }
        // Each placeholder in the range, split where one runs past it, released.
        let Some(regions) = regions(first, end) else {
            return INVALID_ADDRESS;
        };
        let mut answer = KernReturn::SUCCESS;
        for region in regions.iter().filter(|r| r.placeholder()) {
            let whole = region_at(region.first).map_or(*region, |r| r);
            let split = whole.first < region.first || region.end < whole.end;
            if split && free_to_placeholder(region.first, region.end - region.first).is_err() {
                answer = INVALID_ADDRESS;
                continue;
            }
            if !release(region.first) {
                answer = refused("releasing a placeholder");
            }
        }
        answer
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The host's page, which the tests lay their pages out by.
    pub(crate) const PAGE: u64 = 0x1000;

    /// Whether the page at `at` is committed, for tests.
    pub(crate) fn mapped(at: HostAddr) -> bool {
        every_page(at.addr().to_u64() & !(PAGE - 1), PAGE, |_| true)
    }

    /// Whether a page protection writes.
    fn writes(protect: PAGE_PROTECTION_FLAGS) -> bool {
        protect & PAGE_GUARD == 0
            && protect
                & (PAGE_READWRITE
                    | PAGE_WRITECOPY
                    | PAGE_EXECUTE_READWRITE
                    | PAGE_EXECUTE_WRITECOPY)
                != 0
    }

    /// Whether `len` bytes from `at` are committed writable, for tests.
    pub(crate) fn writable(at: HostAddr, len: u64) -> bool {
        pages(at, len).is_some_and(|(first, length)| every_page(first, length, writes))
    }

    /// Whether the page at `at` is free: no allocation, no placeholder.
    fn free(at: HostAddr) -> bool {
        region_at(at.addr().to_u64()).is_some_and(|r| r.state == MEM_FREE)
    }

    /// Whether the page at `at` is a view of a section (`MEM_MAPPED`).
    fn is_view(at: HostAddr) -> bool {
        region_at(at.addr().to_u64()).is_some_and(|r| r.state == MEM_COMMIT && r.kind == MEM_MAPPED)
    }

    fn vm() -> WindowsVm {
        WindowsVm::new()
    }

    fn page() -> u64 {
        page_size()
    }

    fn read(a: HostAddr, off: u64) -> u8 {
        // SAFETY: test memory this test mapped readable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().read_volatile() }
    }

    fn write(a: HostAddr, off: u64, v: u8) {
        // SAFETY: test memory this test mapped writable.
        unsafe { a.wrapping_add(off).as_ptr::<u8>().write_volatile(v) }
    }

    /// The guest's RAM as the host maps it: `length` bytes of a pagefile section, from `offset`
    /// into it, as a view at or above 4 GiB; the section and the view's first byte.
    fn guest_ram(length: u64, offset: u64) -> (Arc<OwnedHandle>, HostAddr) {
        let sec = section(offset + length).unwrap();
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: ptr(LOWEST),
            HighestEndingAddress: std::ptr::null_mut(),
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER {
            Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                _bitfield: MemExtendedParameterAddressRequirements as u64,
            },
            Anonymous2: MEM_EXTENDED_PARAMETER_1 {
                Pointer: (&raw mut requirements).cast(),
            },
        };
        // SAFETY: an open section, the current process, no base address, one live address
        // requirement.
        let view = unsafe {
            MapViewOfFile3(
                sec.as_raw_handle(),
                this_process(),
                std::ptr::null(),
                offset,
                length.to_usize(),
                0,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        };
        assert!(!view.Value.is_null());
        (sec, HostAddr::from_addr(view.Value.addr()))
    }

    /// Unmaps a view [`guest_ram`] made.
    fn unmap_guest_ram(at: HostAddr) {
        // SAFETY: a view this test mapped, which nothing uses any more.
        let done = unsafe {
            UnmapViewOfFile2(
                this_process(),
                MEMORY_MAPPED_VIEW_ADDRESS { Value: at.as_ptr() },
                0,
            )
        };
        assert_ne!(done, 0);
    }

    /// The host page is 4 KiB on x86-64 Windows, its shift 12
    #[test]
    fn the_page_is_the_hosts() {
        let v = vm();
        assert_eq!(v.page_size(), 1 << v.page_shift());
        assert_eq!((v.page_size(), v.page_shift()), (4096, 12));
    }

    /// An allocation anywhere is page-aligned, at or above 4 GiB and zero-filled, and no view
    /// can be made of it; one of 0 bytes is address 0, as Mach's is; one the address space
    /// cannot hold is refused; a deallocated one is free again
    #[test]
    fn an_allocation_is_zeros_no_view_is_made_of() {
        let v = vm();
        let len = 2 * page();
        let a = v.allocate_anywhere(len).unwrap();
        assert_eq!(a.addr().to_u64() % page(), 0);
        assert!(a.addr().to_u64() >= LOWEST);
        assert_eq!((read(a, 0), read(a, len - 1)), (0, 0));
        write(a, 7, 0xa5);
        let dst = v.allocate_anywhere(len).unwrap();
        assert_eq!(v.remap_shared_overwrite(dst, a, len), Err(INVALID_ADDRESS));
        assert_eq!(read(dst, 0), 0, "nothing changed");
        assert_eq!(v.allocate_anywhere(0), Ok(HostAddr::NULL));
        assert!(
            !v.allocate_anywhere(u64::MAX - page())
                .unwrap_err()
                .is_success()
        );
        assert!(v.deallocate(a, len).is_success());
        assert!(v.deallocate(dst, len).is_success());
        assert!(!mapped(a) && free(a) && free(dst));
    }

    /// The guest's RAM, a view of the host's section the lease registers: a view of its pages
    /// made over a fresh range shows them both ways, from the page of the source's first byte,
    /// and is a section view at or above 4 GiB; the range is free again after its deallocation,
    /// its placeholder gone; without the lease the remap is refused
    #[test]
    fn a_view_of_the_guests_ram_shows_its_pages_both_ways() {
        let v = vm();
        let p = page();
        let (sec, ram) = guest_ram(8 * p, 4 * p);
        write(ram, 4 * p + 5, 0x11);
        let dst = v.allocate_anywhere(4 * p).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(dst, ram.wrapping_add(4 * p), 4 * p),
            Err(INVALID_ADDRESS),
            "no lease, no view"
        );
        let lease = super::super::lease_guest_memory([crate::os::GuestRamFile {
            host: ram.addr().to_u64(),
            length: 8 * p,
            fd: Arc::clone(&sec),
            offset: 4 * p,
        }]);
        assert_eq!(
            v.remap_shared_overwrite(dst, ram.wrapping_add(4 * p + 0x10), 4 * p - 0x10),
            Ok(dst)
        );
        assert!(is_view(dst) && dst.addr().to_u64() >= LOWEST);
        assert_eq!(read(dst, 5), 0x11, "seen through the view");
        write(dst, p + 9, 0x22);
        assert_eq!(read(ram, 5 * p + 9), 0x22, "written through the view");
        write(ram, 7 * p + 1, 0x33);
        assert_eq!(read(dst, 3 * p + 1), 0x33, "written through the RAM");
        assert!(files::pieces(dst.addr().to_u64(), 4 * p).is_some());
        assert!(v.deallocate(dst, 4 * p).is_success());
        assert!(free(dst), "the placeholder went with the view");
        assert!(files::pieces(dst.addr().to_u64(), 1).is_none());
        assert_eq!(v.deallocate(ram, p), INVALID_ARGUMENT, "registered memory");
        assert!(mapped(ram));
        drop(lease);
        unmap_guest_ram(ram);
    }

    /// A fixed overwrite of an aliased run puts private zeros back over it, which show the
    /// source's pages no more and are no view; the rest of the view keeps showing them
    #[test]
    fn a_fixed_overwrite_of_a_view_puts_private_zeros_back() {
        let v = vm();
        let p = page();
        let src = allocate_aliasable(4 * p).unwrap();
        for i in 0..4 {
            write(src, i * p, 0x40 + u8::try_from(i).unwrap());
        }
        let at = v.allocate_anywhere(4 * p).unwrap();
        v.remap_shared_overwrite(at, src, 4 * p).unwrap();
        assert_eq!(files::views_between(at.addr().to_u64(), at.addr().to_u64() + 4 * p), 1);
        v.allocate_fixed_overwrite(at.wrapping_add(p + 1), 1).unwrap();
        assert_eq!(
            (read(at, 0), read(at, p), read(at, 2 * p), read(at, 3 * p)),
            (0x40, 0, 0x42, 0x43)
        );
        assert!(is_view(at) && !is_view(at.wrapping_add(p)) && is_view(at.wrapping_add(2 * p)));
        assert_eq!(files::views_between(at.addr().to_u64(), at.addr().to_u64() + 4 * p), 2);
        write(at, p + 2, 0x7e);
        assert_eq!(read(src, p + 2), 0x41, "private now: the write stays");
        write(src, 3 * p + 3, 0x77);
        assert_eq!(read(at, 3 * p + 3), 0x77, "the rest still shows the pages");
        assert!(files::pieces(at.addr().to_u64() + p, p).is_none());
        assert!(files::pieces(at.addr().to_u64() + 2 * p, 2 * p).is_some());
        assert!(v.deallocate(at, 4 * p).is_success());
        assert!(v.deallocate(src, 4 * p).is_success());
        assert!(free(at) && free(src));
    }

    /// A view shows its source's pages both ways, from the page of the source's first byte; a
    /// view of the view shows them too, and keeps showing them after the first view is unmapped
    #[test]
    fn a_view_of_a_view_outlasts_the_first() {
        let v = vm();
        let len = 4 * page();
        let src = allocate_aliasable(len).unwrap();
        let first = v.allocate_anywhere(len).unwrap();
        let second = v.allocate_anywhere(2 * page()).unwrap();
        write(src, page() + 5, 0x11);
        assert_eq!(
            v.remap_shared_overwrite(first, src.wrapping_add(0x10), len - 0x10),
            Ok(first)
        );
        assert_eq!(read(first, page() + 5), 0x11);
        write(first, 9, 0x33);
        assert_eq!(read(src, 9), 0x33);
        v.remap_shared_overwrite(second, first.wrapping_add(page()), 2 * page())
            .unwrap();
        assert_eq!(read(second, 5), 0x11);
        assert!(v.deallocate(first, len).is_success());
        write(src, page() + 6, 0x22);
        assert_eq!(read(second, 6), 0x22);
        write(second, page(), 0x44);
        assert_eq!(read(src, 2 * page()), 0x44);
        assert!(v.deallocate(second, 2 * page()).is_success());
        assert!(v.deallocate(src, len).is_success());
    }

    /// A view over another view replaces it; an overwrite or an unmap of part of a view leaves
    /// the rest showing its pages; a source with a byte of no section maps nothing
    #[test]
    fn views_are_overwritten_and_unmapped_in_part() {
        let v = vm();
        let (a, b) = (
            allocate_aliasable(2 * page()).unwrap(),
            allocate_aliasable(2 * page()).unwrap(),
        );
        write(a, 0, 1);
        write(a, page(), 2);
        write(b, 0, 3);
        let view = v.allocate_anywhere(2 * page()).unwrap();
        v.remap_shared_overwrite(view, a, 2 * page()).unwrap();
        v.remap_shared_overwrite(view, b, page()).unwrap();
        assert_eq!((read(view, 0), read(view, page())), (3, 2));
        v.allocate_fixed_overwrite(view.wrapping_add(page() + 1), 1)
            .unwrap();
        assert_eq!((read(view, 0), read(view, page())), (3, 0));
        let copy = v.allocate_anywhere(2 * page()).unwrap();
        assert_eq!(
            v.remap_shared_overwrite(copy, view, 2 * page()),
            Err(INVALID_ADDRESS)
        );
        assert_eq!(read(copy, 0), 0, "nothing was mapped");
        v.remap_shared_overwrite(copy, view, page()).unwrap();
        assert_eq!(read(copy, 0), 3);
        // Part of a view unmapped: the rest shows its pages, and the part is free.
        assert!(v.deallocate(view, page()).is_success());
        assert!(free(view) && mapped(view.wrapping_add(page())));
        assert_eq!(read(view, page()), 0);
        for (at, len) in [(view.wrapping_add(page()), 1), (copy, 2), (a, 2), (b, 2)] {
            assert!(v.deallocate(at, len * page()).is_success());
        }
    }

    /// A view of pages in and out of order over one range is one Win32 view per run of them,
    /// and a protect inside a view cuts no view, only a run; an unmap of a page inside a view
    /// leaves two views
    #[test]
    fn the_table_lists_the_views_win32_holds() {
        let v = vm();
        let p = page();
        let src = allocate_aliasable(8 * p).unwrap();
        let at = v.allocate_anywhere(8 * p).unwrap();
        for i in 0..8 {
            write(src, i * p, u8::try_from(i).unwrap());
        }
        // Pages 0-3 in order, then 6 and 7, then 4 and 5: three runs, three views.
        for (to, from) in [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 6),
            (5, 7),
            (6, 4),
            (7, 5),
        ] {
            v.remap_shared_overwrite(at.wrapping_add(to * p), src.wrapping_add(from * p), p)
                .unwrap();
        }
        let (first, end) = (at.addr().to_u64(), at.addr().to_u64() + 8 * p);
        assert_eq!(files::views_between(first, end), 8, "each remap is a view of its own");
        let shown: Vec<u8> = (0..8).map(|i| read(at, i * p)).collect();
        assert_eq!(shown, [0, 1, 2, 3, 6, 7, 4, 5]);
        v.protect_max(at.wrapping_add(p), p, VmProt::READ).unwrap();
        assert!(!writable(at.wrapping_add(p), p) && writable(at.wrapping_add(2 * p), p));
        assert_eq!(files::views_between(first, end), 8, "a protect cuts no view");
        v.protect_max(at.wrapping_add(p), p, VmProt::READ | VmProt::WRITE)
            .unwrap();
        assert!(writable(at, 8 * p));
        assert!(v.deallocate(at.wrapping_add(2 * p), p).is_success());
        assert_eq!(files::views_between(first, end), 7);
        assert!(free(at.wrapping_add(2 * p)));
        assert_eq!(read(at, 3 * p), 3, "the page after the unmap still shows its source");
        assert!(v.deallocate(at, 2 * p).is_success());
        assert!(v.deallocate(at.wrapping_add(3 * p), 5 * p).is_success());
        assert_eq!(files::views_between(first, end), 0);
        assert!(v.deallocate(src, 8 * p).is_success());
    }

    /// A remap whose source lies in one view mapped over part of a range, then a remap over
    /// the middle of that view: the view is unmapped whole and its two ends mapped again, each
    /// at the protection it had, showing the same pages
    #[test]
    fn a_remap_into_the_middle_of_a_view_keeps_its_ends() {
        let v = vm();
        let p = page();
        let src = allocate_aliasable(4 * p).unwrap();
        let other = allocate_aliasable(p).unwrap();
        for i in 0..4 {
            write(src, i * p, 0x10 + u8::try_from(i).unwrap());
        }
        write(other, 0, 0x99);
        let at = v.allocate_anywhere(6 * p).unwrap();
        v.remap_shared_overwrite(at.wrapping_add(p), src, 4 * p).unwrap();
        v.protect_max(at.wrapping_add(4 * p), p, VmProt::READ).unwrap();
        v.remap_shared_overwrite(at.wrapping_add(2 * p), other, p).unwrap();
        let shown: Vec<u8> = (1..5).map(|i| read(at, i * p)).collect();
        assert_eq!(shown, [0x10, 0x99, 0x12, 0x13]);
        assert!(!writable(at.wrapping_add(4 * p), p), "the end keeps its protection");
        assert!(writable(at.wrapping_add(p), p) && writable(at.wrapping_add(3 * p), p));
        write(src, 3 * p + 1, 0x55);
        assert_eq!(read(at, 4 * p + 1), 0x55, "the end still shows its pages");
        assert_eq!(files::views_between(at.addr().to_u64(), at.addr().to_u64() + 6 * p), 3);
        assert_eq!(read(at, 0), 0, "the private pages around stay");
        write(at, 5 * p, 0x66);
        assert_eq!(read(at, 5 * p), 0x66);
        for (range, len) in [(at, 6), (src, 4), (other, 1)] {
            assert!(v.deallocate(range, len * p).is_success());
        }
    }

    /// The protection set is what the pages read by; a range of no access is not readable; a
    /// freed range is neither readable nor protectable
    #[test]
    fn readable_follows_the_protection() {
        let v = vm();
        let p = page();
        let a = allocate_aliasable(2 * p).unwrap();
        let anon = v.allocate_anywhere(2 * p).unwrap();
        assert!(v.readable(a, 2 * p) && v.readable(anon, 2 * p) && v.readable(a, 0));
        v.protect_max(a.wrapping_add(p), p, VmProt::NONE).unwrap();
        v.protect_max(anon.wrapping_add(p), p, VmProt::NONE)
            .unwrap();
        assert!(v.readable(a, p) && !v.readable(a, 2 * p));
        assert!(v.readable(anon, p) && !v.readable(anon, 2 * p));
        v.protect_max(a, 2 * p, VmProt::READ).unwrap();
        assert!(v.readable(a, 2 * p));
        assert!(v.deallocate(anon, 2 * p).is_success());
        assert!(!v.readable(anon, 1));
        assert_eq!(v.protect_max(anon, p, VmProt::READ), Err(INVALID_ADDRESS));
        assert!(v.deallocate(a, 2 * p).is_success());
    }

    /// Deallocating nothing succeeds, a wrapping range is an invalid argument, and registered
    /// memory is never unmapped
    #[test]
    fn deallocate_answers_as_mach_and_keeps_registered_memory() {
        let v = vm();
        let a = v.allocate_anywhere(page()).unwrap();
        assert_eq!(v.deallocate(a, 0), KernReturn::SUCCESS);
        assert_eq!(
            v.deallocate(HostAddr::from_addr(usize::MAX & !0xfff), 2 * page()),
            INVALID_ARGUMENT
        );
        let (sec, ram) = guest_ram(page(), 0);
        let held = files::Registration::new(
            ram.addr().to_u64(),
            page(),
            Backing {
                section: sec,
                offset: 0,
            },
        );
        assert_eq!(v.deallocate(ram, page()), INVALID_ARGUMENT);
        assert!(mapped(ram));
        drop(held);
        unmap_guest_ram(ram);
        assert!(v.deallocate(a, page()).is_success());
    }

    /// A remap whose second piece Win32 refuses changes nothing, as Mach's: its first piece is
    /// put back as the target showed it, a view's section there, and private pages with no
    /// access over a reservation's bytes of no section; the table answers as before
    #[test]
    fn a_remap_that_fails_part_way_changes_nothing() {
        let v = vm();
        let p = page();
        // The source: a page of a section of 0x11s, then a registration of 0x22s over a
        // section whose handle is closed before the remap, which Win32 refuses to map.
        let src = v.allocate_anywhere(2 * p).unwrap();
        let first = src.addr().to_u64();
        let ones = allocate_aliasable(p).unwrap();
        write(ones, 0, 0x11);
        v.remap_shared_overwrite(src, ones, p).unwrap();
        let (twos_sec, twos) = guest_ram(p, 0);
        write(twos, 0, 0x22);
        let dead = {
            // SAFETY: a handle this test made and closes here, kept only as a value Win32
            // knows no section by.
            let raw = unsafe { OwnedHandle::from_raw_handle(twos_sec.as_raw_handle()) };
            drop(twos_sec);
            // The section object lives on through its view; its handle value is closed.
            Arc::new(raw)
        };
        let dead_value = dead.as_raw_handle();
        // The closed handle, as a backing: forgotten before the OwnedHandle closes it again.
        std::mem::forget(dead);
        // SAFETY: the handle value closed above, wrapped only to be named by a backing; it is
        // leaked below, never closed twice.
        let closed = Arc::new(unsafe { OwnedHandle::from_raw_handle(dead_value) });
        let held = files::Registration::new(
            first + p,
            p,
            Backing {
                section: Arc::clone(&closed),
                offset: 0,
            },
        );
        v.remap_shared_overwrite(src.wrapping_add(p), twos, p)
            .expect_err("the closed handle maps nothing");
        assert_eq!(files::pieces(first, 2 * p).map(|ps| ps.len()), Some(2));

        // A target whose first page is a view of a section of 0x33s.
        let view = v.allocate_anywhere(2 * p).unwrap();
        let threes = allocate_aliasable(p).unwrap();
        write(threes, 0, 0x33);
        v.remap_shared_overwrite(view, threes, p).unwrap();
        assert!(v.remap_shared_overwrite(view, src, 2 * p).is_err());
        assert_eq!(read(view, 0), 0x33, "the view's section is back");
        write(view, 1, 0x34);
        assert_eq!(read(threes, 1), 0x34, "the same pages");
        let shown = files::pieces(view.addr().to_u64(), p).unwrap();
        assert_eq!((shown.len(), shown[0].prot), (1, PAGE_READWRITE));
        assert!(files::pieces(view.addr().to_u64() + p, p).is_none());
        assert!(mapped(view.wrapping_add(p)) && !v.readable(view.wrapping_add(p), p));

        // A target of a reservation's bytes of no section.
        let reserved = v.allocate_anywhere(2 * p).unwrap();
        assert!(v.remap_shared_overwrite(reserved, src, 2 * p).is_err());
        assert!(files::pieces(reserved.addr().to_u64(), p).is_none());
        assert!(
            mapped(reserved) && !v.readable(reserved, p),
            "zeros, no access"
        );
        assert!(v.readable(reserved.wrapping_add(p), p), "untouched");

        drop(held);
        std::mem::forget(closed);
        unmap_guest_ram(twos);
        for (at, len) in [(src, 2), (view, 2), (threes, 1), (reserved, 2), (ones, 1)] {
            assert!(v.deallocate(at, len * p).is_success(), "{at:?}");
        }
    }
}
