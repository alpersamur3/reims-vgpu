//! The memory map's files on unix: a range's file is a descriptor the host passes
//! (`RvgMemoryRangeFile::fd`), checked with `fstat` and duplicated close-on-exec for the map.

use std::mem::MaybeUninit;
use std::num::NonZeroU64;
use std::os::fd::{FromRawFd, OwnedFd};

use super::{RvgMemoryRange, RvgMemoryRangeFile};

/// Why the file `f` (an `fd` of 0 or more) of range `r` is refused, on unix: a file `fstat`
/// cannot read, one holding fewer than `offset + physical_length` bytes, or an `offset` that is
/// not a multiple of `page`, in that order.
pub(super) fn file_contents_refusal(
    r: &RvgMemoryRange,
    f: &RvgMemoryRangeFile,
    page: NonZeroU64,
) -> Option<String> {
    let size = match file_size(f.fd) {
        Ok(size) => size,
        Err(e) => return Some(format!("has fd {}, which fstat cannot read: {e}", f.fd)),
    };
    if f.offset
        .checked_add(r.physical_length)
        .is_none_or(|end| size < end)
    {
        return Some(format!(
            "holds {size:#x} bytes, fewer than its offset {:#x} and the range's {:#x}",
            f.offset, r.physical_length
        ));
    }
    if f.offset % page != 0 {
        return Some(format!(
            "has offset {:#x}, not a multiple of the page size {page:#x}",
            f.offset
        ));
    }
    None
}

/// The host's page size (`sysconf(_SC_PAGESIZE)`, which does not fail): 16 KiB on arm64 macOS,
/// 4 KiB on x86-64.
pub(super) fn host_page_size() -> NonZeroU64 {
    // SAFETY: a query of a constant; it reads and writes no memory of the caller's.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page)
        .ok()
        .and_then(NonZeroU64::new)
        .unwrap_or(NonZeroU64::MIN)
}

/// The size of the file `fd` names (`fstat`).
fn file_size(fd: i32) -> std::io::Result<u64> {
    let mut st = MaybeUninit::<libc::stat>::zeroed();
    // SAFETY: `st` is a writable `struct stat`; an `fd` that names no open file is an error the
    // call returns.
    if unsafe { libc::fstat(fd, st.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `fstat` succeeded, so it filled `st`; it was zeroed before.
    let st = unsafe { st.assume_init() };
    Ok(u64::try_from(st.st_size).unwrap_or(0))
}

/// A duplicate of `fd`, close-on-exec (`fcntl(F_DUPFD_CLOEXEC)`), which the map holds.
pub(super) fn duplicate(fd: i32) -> std::io::Result<OwnedFd> {
    // SAFETY: `fcntl` reads and writes no memory of the caller's; an `fd` that names no open
    // file is an error the call returns.
    let new = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if new < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `new` is a descriptor this call opened, owned by nothing else.
    Ok(unsafe { OwnedFd::from_raw_fd(new) })
}
