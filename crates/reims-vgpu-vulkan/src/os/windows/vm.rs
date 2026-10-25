//! Mach VM's calls (`HostVm`) over Win32 virtual memory.
//!
//! - **Placement.** Every range this VM makes lies at or above 4 GiB: `VirtualAlloc2` with an
//!   address requirement (`MEM_ADDRESS_REQUIREMENTS::LowestStartingAddress`), as core's
//!   `GuestPtr` dies on any address below `ZERO_REGION_END` (0x1_0000_0000), and it is built
//!   over the addresses this VM answers too, not only over the guest's RAM.
//! - **Zero fill and overwrite.** A fresh range is committed read and write, which Windows
//!   zero-fills. `allocate_fixed_overwrite` decommits and commits again, which zero-fills too;
//!   it works inside ranges this VM made, not inside a section's view.
//! - **Protection.** Windows has no maximum protection: `protect_max` sets the current one,
//!   which a later call may raise again, where Mach refuses the raise.
//! - **Deallocation.** A whole range this VM made is released; part of one is decommitted, so
//!   its address space stays reserved until the rest goes (Windows releases only whole
//!   allocations without placeholders).
//! - **Views.** Not made yet: `remap_shared_overwrite` answers `KERN_INVALID_ADDRESS`. A shared
//!   alias of the guest's pages needs those pages to be a section's (the host's
//!   `CreateFileMappingW` RAM) and a placeholder to map a view of it into (`MapViewOfFile3` with
//!   `MEM_REPLACE_PLACEHOLDER`), the next step of the port.

use std::ffi::c_void;
use std::sync::Once;

use reims_vgpu_core::num::{ToU64, ToUsize};
use reims_vgpu_core::platform::os::vm::{HostAddr, HostVm, KernReturn, VmProt};
use windows_sys::Win32::System::Memory::{
    MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT, MEM_EXTENDED_PARAMETER,
    MEM_EXTENDED_PARAMETER_0, MEM_EXTENDED_PARAMETER_1, MEM_FREE, MEM_RELEASE, MEM_RESERVE,
    MEMORY_BASIC_INFORMATION, MemExtendedParameterAddressRequirements, PAGE_EXECUTE_READ,
    PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS,
    PAGE_PROTECTION_FLAGS, PAGE_READONLY, PAGE_READWRITE, PAGE_WRITECOPY, VirtualAlloc,
    VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// `KERN_INVALID_ADDRESS`.
const INVALID_ADDRESS: KernReturn = KernReturn(1);
/// `KERN_NO_SPACE`.
const NO_SPACE: KernReturn = KernReturn(3);
/// `KERN_INVALID_ARGUMENT`.
const INVALID_ARGUMENT: KernReturn = KernReturn(4);

/// The lowest address a range of this VM may start at: 4 GiB, core's `ZERO_REGION_END`.
pub(crate) const LOWEST: u64 = 0x1_0000_0000;

/// `VM_PROT_EXECUTE`.
const PROT_EXECUTE: i32 = 4;

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
fn ptr(first: u64) -> *mut c_void {
    std::ptr::with_exposed_provenance_mut(first.to_usize())
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

/// Whether a page protection reads.
fn reads(protect: PAGE_PROTECTION_FLAGS) -> bool {
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
    let mut at = first;
    while at < end {
        let Some(info) = query(at) else {
            return false;
        };
        if info.State != MEM_COMMIT || !ok(info.Protect) {
            return false;
        }
        let next = (info.BaseAddress.addr().to_u64()).saturating_add(info.RegionSize.to_u64());
        if next <= at {
            return false;
        }
        at = next;
    }
    true
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

/// Mach VM's calls on the current process.
#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsVm;

impl WindowsVm {
    /// The process's VM.
    pub fn new() -> WindowsVm {
        WindowsVm
    }

    /// Whether every page of the `len` bytes at `addr` is committed with read access. A `len` of
    /// 0 is `true`.
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

    /// A fresh committed range of `len` bytes, whole pages, zero-filled, at or above 4 GiB.
    fn allocate_anywhere(&self, len: u64) -> Result<HostAddr, KernReturn> {
        let Some((_, length)) = pages(HostAddr::NULL, len) else {
            return Err(NO_SPACE);
        };
        if length == 0 {
            return Ok(HostAddr::NULL);
        }
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
                GetCurrentProcess(),
                std::ptr::null(),
                length.to_usize(),
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        };
        if at.is_null() {
            return Err(NO_SPACE);
        }
        Ok(HostAddr::from_addr(at.addr()))
    }

    /// `addr..addr+len` zero-filled again: decommitted and committed read and write, inside a
    /// range this VM made.
    fn allocate_fixed_overwrite(&self, addr: HostAddr, len: u64) -> Result<(), KernReturn> {
        let Some((first, length)) = pages(addr, len) else {
            return Err(INVALID_ARGUMENT);
        };
        if length == 0 {
            return Ok(());
        }
        let Some(info) = query(first) else {
            return Err(INVALID_ADDRESS);
        };
        if info.State == MEM_FREE {
            return Err(NO_SPACE);
        }
        // The whole span, whatever its first page's state: decommitting pages already
        // decommitted is no error, and a committed page anywhere in it must lose its contents.
        // SAFETY: pages of one allocation of this process; decommitting drops their contents.
        let decommitted = unsafe { VirtualFree(ptr(first), length.to_usize(), MEM_DECOMMIT) };
        if decommitted == 0 {
            return Err(NO_SPACE);
        }
        // SAFETY: reserved pages of this process, committed read and write (zero-filled).
        let made =
            unsafe { VirtualAlloc(ptr(first), length.to_usize(), MEM_COMMIT, PAGE_READWRITE) };
        if made.is_null() {
            Err(NO_SPACE)
        } else {
            Ok(())
        }
    }

    /// Refused: views are not made yet (module docs).
    fn remap_shared_overwrite(
        &self,
        _dst: HostAddr,
        _src: HostAddr,
        _len: u64,
    ) -> Result<HostAddr, KernReturn> {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            log::warn!(
                target: "reims_vgpu_vulkan::os",
                "no shared views on Windows yet: section-backed guest RAM and placeholder views \
                 (MapViewOfFile3) are not in place, so remap_shared_overwrite is refused"
            );
        });
        Err(INVALID_ADDRESS)
    }

    /// The current protection of `addr..addr+len` set to `prot` (Windows keeps no maximum).
    fn protect_max(&self, addr: HostAddr, len: u64, prot: VmProt) -> Result<(), KernReturn> {
        let Some((first, length)) = pages(addr, len) else {
            return Err(INVALID_ARGUMENT);
        };
        if length == 0 {
            return Ok(());
        }
        if !every_page(first, length, |_| true) {
            return Err(INVALID_ADDRESS);
        }
        let mut old: PAGE_PROTECTION_FLAGS = 0;
        // SAFETY: committed pages of this process (checked above) and a live out-parameter.
        let done =
            unsafe { VirtualProtect(ptr(first), length.to_usize(), protection(prot), &mut old) };
        if done == 0 {
            Err(INVALID_ADDRESS)
        } else {
            Ok(())
        }
    }

    /// Releases a whole range this VM made; decommits part of one (module docs).
    fn deallocate(&self, addr: HostAddr, len: u64) -> KernReturn {
        let Some((first, length)) = pages(addr, len) else {
            return INVALID_ARGUMENT;
        };
        if length == 0 {
            return KernReturn::SUCCESS;
        }
        let Some(info) = query(first) else {
            return INVALID_ADDRESS;
        };
        if info.State == MEM_FREE {
            return INVALID_ADDRESS;
        }
        let base = info.AllocationBase.addr().to_u64();
        let whole = base == first && allocation_length(base) == length;
        // SAFETY: pages of this process: the allocation's base for a release, else a span of
        // its pages to decommit.
        let done = unsafe {
            if whole {
                VirtualFree(ptr(first), 0, MEM_RELEASE)
            } else {
                VirtualFree(ptr(first), length.to_usize(), MEM_DECOMMIT)
            }
        };
        if done == 0 {
            INVALID_ADDRESS
        } else {
            KernReturn::SUCCESS
        }
    }
}

/// The length of the allocation that starts at `base`: its regions, in order, while they share
/// its allocation base.
fn allocation_length(base: u64) -> u64 {
    let mut at = base;
    while let Some(info) = query(at) {
        if info.AllocationBase.addr().to_u64() != base || info.State == MEM_FREE {
            break;
        }
        let next = info
            .BaseAddress
            .addr()
            .to_u64()
            .saturating_add(info.RegionSize.to_u64());
        if next <= at {
            break;
        }
        at = next;
    }
    at - base
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
}
