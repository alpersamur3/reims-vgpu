//! The memory map's files on Windows: a range's backing is a section (`CreateFileMappingW`),
//! whose handle the host passes in this ABI's `int32_t` descriptor (`RvgMemoryRangeFile::fd`),
//! checked by mapping a view of it and duplicated for the map.
//!
//! **The descriptor is the handle.** A Win32 kernel handle's value always fits in 32 bits (the
//! kernel hands out handles below 2^32 so that a 32-bit process can hold them, and a 64-bit
//! process gets the same values; MS, "Interprocess Communication Between 32-bit and 64-bit
//! Applications"), so the host gives `(int32_t)(intptr_t)section` and the library widens it
//! back with sign extension, as the OS does for a handle a 32-bit process stored. -1 keeps
//! meaning "no file", as on unix; a section's handle is never -1 (`INVALID_HANDLE_VALUE`) nor 0.
//!
//! **The range's view.** The host's `host_address` must be a view of this very section from
//! `offset` (`MapViewOfFile3`, at or above 4 GiB), as on unix it must be a shared mapping of the
//! file: the library's own views of the range's pages are made of the section
//! (`reims_vgpu_vulkan::os::windows`'s `files` and `vm`), and they show the same pages only
//! when the host's view is of it too.

use std::num::NonZeroU64;
use std::os::windows::io::{FromRawHandle, OwnedHandle};

use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
use windows_sys::Win32::System::Memory::{MapViewOfFile3, PAGE_READONLY, UnmapViewOfFile2};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

use super::{RvgMemoryRange, RvgMemoryRangeFile};

/// The handle the descriptor `fd` (0 or more) names: its value widened back to the process's
/// pointer width with sign extension (module docs).
fn handle(fd: i32) -> HANDLE {
    std::ptr::with_exposed_provenance_mut(i64::from(fd) as usize)
}

/// Why the file `f` (an `fd` of 0 or more) of range `r` is refused, on Windows: a handle that
/// is not a section's or whose section holds fewer than `offset + physical_length` bytes (a
/// view of that span cannot be mapped, the `fstat` of a section), or an `offset` that is not a
/// multiple of `page`, in that order. A section of 0 bytes cannot be made, so a range of 0
/// bytes is checked by a one-page view, which any section holds.
pub(super) fn file_contents_refusal(
    r: &RvgMemoryRange,
    f: &RvgMemoryRangeFile,
    page: NonZeroU64,
) -> Option<String> {
    if f.fd == 0 {
        return Some(format!("has fd {}, which names no section", f.fd));
    }
    let Some(end) = f.offset.checked_add(r.physical_length) else {
        return Some(format!(
            "has offset {:#x} and length {:#x}, past the end of any section",
            f.offset, r.physical_length
        ));
    };
    if f.offset % page != 0 {
        return Some(format!(
            "has offset {:#x}, not a multiple of the page size {page:#x}",
            f.offset
        ));
    }
    let probe = r.physical_length.max(page.get());
    let Ok(length) = usize::try_from(probe) else {
        return Some(format!("has length {probe:#x}, past the address space"));
    };
    // SAFETY: a handle value the host passed, the current process, no base address (anywhere),
    // a read-only view of `length` bytes from `offset`, no extended parameters. A handle that
    // names no section, or a section shorter than the span, is a failure the call returns.
    let view = unsafe {
        MapViewOfFile3(
            handle(f.fd),
            GetCurrentProcess(),
            std::ptr::null(),
            f.offset,
            length,
            0,
            PAGE_READONLY,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value.is_null() {
        let e = std::io::Error::last_os_error();
        return Some(format!(
            "has fd {:#x}, which no view of {probe:#x} bytes from offset {:#x} can be made of \
             (not a section's handle, or one holding fewer than {end:#x} bytes): {e}",
            f.fd, f.offset
        ));
    }
    // SAFETY: the view just mapped, unmapped whole and once.
    unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, 0) };
    None
}

/// The host's page size on Windows (`GetSystemInfo`'s `dwPageSize`): 4 KiB on x86-64.
pub(super) fn host_page_size() -> NonZeroU64 {
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    // SAFETY: an all-zero SYSTEM_INFO is a valid value to fill.
    let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
    // SAFETY: a live out-parameter; the call cannot fail.
    unsafe { GetSystemInfo(&mut info) };
    NonZeroU64::new(u64::from(info.dwPageSize)).unwrap_or(NonZeroU64::MIN)
}

/// A duplicate of the section handle `fd` names (`DuplicateHandle`, same access, not
/// inherited), which the map holds.
pub(super) fn duplicate(fd: i32) -> std::io::Result<OwnedHandle> {
    let mut new: HANDLE = std::ptr::null_mut();
    // SAFETY: the current process's pseudo-handle on both sides, a handle value the host passed
    // (one that names nothing is an error the call returns), a live out-parameter.
    let done = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            handle(fd),
            GetCurrentProcess(),
            &mut new,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    };
    if done == 0 || new.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `new` is a handle this call opened, owned by nothing else.
    Ok(unsafe { OwnedHandle::from_raw_handle(new) })
}
