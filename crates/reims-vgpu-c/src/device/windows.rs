//! The memory map's files on Windows: none yet. A range's backing there is a section
//! (`CreateFileMappingW`), whose handle this ABI's `int32_t` descriptor cannot carry, so every
//! descriptor is refused and a host gives -1.

use std::num::NonZeroU64;
use std::os::windows::io::OwnedHandle;

use super::{RvgMemoryRange, RvgMemoryRangeFile};

/// Why the file `f` (an `fd` of 0 or more) of range `r` is refused, on Windows: always, as a
/// range's backing there is a section handle (`CreateFileMappingW`), which this ABI's `int32_t`
/// descriptor cannot carry yet. A host gives -1 until the ABI names a handle.
pub(super) fn file_contents_refusal(
    _r: &RvgMemoryRange,
    f: &RvgMemoryRangeFile,
    _page: NonZeroU64,
) -> Option<String> {
    Some(format!(
        "has fd {}, but Windows takes no memory files yet: give -1",
        f.fd
    ))
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

/// A duplicate of `fd`: none on Windows, where a range carries no file yet
/// ([`file_contents_refusal`] refuses every `fd` before this is reached).
pub(super) fn duplicate(_fd: i32) -> std::io::Result<OwnedHandle> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}
