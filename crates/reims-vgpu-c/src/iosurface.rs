//! The IOSurface device: vmapple's second MMIO region, [`RvgIOSurfaceDeviceConfig`], and the
//! `rvg_iosurface_device_*` functions.

use std::ffi::c_void;
use std::sync::Arc;

use reims_vgpu::{IoSurfaceDevice, IoSurfaceDeviceConfig, RaiseInterrupt};

use crate::device::{RvgMemoryMap, memory_map};
use crate::ffi;
use crate::handle;
use crate::sized::{self, HostStruct};
use crate::status::{RVG_ERR_INVALID_ARGUMENT, RVG_OK, RvgStatus, status};
use crate::table::{RvgDestroyFn, Table, WithTable};
use crate::{Plat, null_handle};

/// `raise_interrupt(ctx, vector)` of the IOSurface device.
///
/// Thread: the IOSurface device's own serial worker, one call at a time.
/// May call: every function of this header except rvg_iosurface_device_release on its own
/// device, which ends the process. Without it, the IOSurface device's first interrupt ends
/// the process.
pub type RvgIOSurfaceRaiseInterruptFn =
    Option<unsafe extern "C-unwind" fn(ctx: *mut c_void, vector: u32)>;

/// The IOSurface device's callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
#[expect(
    missing_docs,
    reason = "the fields' names say what they hold, as the header shows them"
)]
pub struct RvgIOSurfaceCallbacks {
    /// sizeof(RvgIOSurfaceCallbacks) as the host was compiled.
    pub struct_size: u32,
    pub ctx: *mut c_void,
    /// Runs once after the IOSurface device's end, when its raise runs no more.
    pub destroy: RvgDestroyFn,
    pub raise_interrupt: RvgIOSurfaceRaiseInterruptFn,
}

// SAFETY: repr(C), a u32 first; a pointer and optional function pointers.
unsafe impl HostStruct for RvgIOSurfaceCallbacks {}

/// The IOSurface device's config.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RvgIOSurfaceDeviceConfig {
    /// sizeof(RvgIOSurfaceDeviceConfig) as the host was compiled.
    pub struct_size: u32,
    /// Required.
    pub memory_map: *const RvgMemoryMap,
    /// NULL is a table with every callback absent.
    pub callbacks: *const RvgIOSurfaceCallbacks,
}

// SAFETY: repr(C), a u32 first; pointers.
unsafe impl HostStruct for RvgIOSurfaceDeviceConfig {}

/// The IOSurface device: vmapple's second MMIO region.
pub struct RvgIOSurfaceDevice {
    pub(crate) device: WithTable<IoSurfaceDevice<Plat>>,
}

/// The IOSurface device behind a handle, or the process's end for NULL.
///
/// # Safety
/// `device` is NULL or a live IOSurface device handle.
unsafe fn live<'a>(
    device: *const RvgIOSurfaceDevice,
    call: &'static str,
) -> &'a RvgIOSurfaceDevice {
    // SAFETY: the caller's contract.
    unsafe { handle::get(device) }.unwrap_or_else(|| null_handle(call))
}

/// A new IOSurface device, the process's latest (a device with `use_iosurface_mapper` finds
/// it at its first use). Refusals, checked in this order: RVG_ERR_INVALID_ARGUMENT,
/// RVG_ERR_UNSUPPORTED, RVG_ERR_NO_MEMORY_MAP, RVG_ERR_MEMORY_MAP. Calls back: never.
///
/// # Safety
/// `config` is NULL or a live config; `device` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_iosurface_device_create(
    config: *const RvgIOSurfaceDeviceConfig,
    device: *mut *mut RvgIOSurfaceDevice,
) -> RvgStatus {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        let c = match unsafe { sized::read(config) } {
            Ok(Some(c)) => c,
            Ok(None) => return RVG_ERR_INVALID_ARGUMENT,
            Err(s) => return s,
        };
        // SAFETY: the config's contract: NULL or a live table.
        let callbacks = match unsafe { sized::read_or_zero(c.callbacks) } {
            Ok(t) => t,
            Err(s) => return s,
        };
        // From here the table is the library's: a refusal ends it before it returns.
        let table = Table::new(callbacks.ctx, callbacks.destroy);
        let refuse = |s: RvgStatus| {
            table.end();
            s
        };
        if device.is_null() {
            return refuse(RVG_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: the config's contract: NULL or a live map.
        let memory_map = match unsafe { memory_map(c.memory_map, std::ptr::null()) } {
            Ok(m) => m,
            Err(s) => return refuse(s),
        };
        let raise_interrupt = callbacks.raise_interrupt.map(|f| {
            let t = table.clone();
            let r: RaiseInterrupt = Arc::new(move |vector| {
                // SAFETY: the host's function over its ctx.
                ffi::host_call("raise_interrupt", || unsafe { f(t.ctx(), vector) });
            });
            r
        });
        let config = IoSurfaceDeviceConfig {
            memory_map,
            raise_interrupt,
        };
        match IoSurfaceDevice::with_platform(<Plat as reims_vgpu::ApiPlatform>::process(), config) {
            Ok(d) => {
                let h = handle::new(RvgIOSurfaceDevice {
                    device: WithTable::new(d, table),
                });
                // SAFETY: checked non-NULL; the caller's contract.
                unsafe { device.write(h) };
                RVG_OK
            }
            Err(e) => refuse(status(e)),
        }
    })
}

/// Drops the host's reference; the last one is the IOSurface device's teardown, which waits for
/// its thread. Inside its own raise_interrupt it ends the process.
///
/// # Safety
/// `device` is NULL or a live IOSurface device handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_iosurface_device_release(device: *mut RvgIOSurfaceDevice) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { handle::release(device) }
    })
}

/// A 64-bit read at `offset` in the IOSurface device's registers (0x1000 to 0x1030); an offset
/// that names no register reads all ones.
///
/// # Safety
/// `device` is a live IOSurface device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_iosurface_device_mmio_read(
    device: *mut RvgIOSurfaceDevice,
    offset: u64,
) -> u64 {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_iosurface_device_mmio_read") }
            .device
            .mmio_read(offset)
    })
}

/// A 64-bit write at `offset` in the IOSurface device's registers; other offsets are ignored.
/// May block on the IOSurface device's thread.
///
/// # Safety
/// `device` is a live IOSurface device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_iosurface_device_mmio_write(
    device: *mut RvgIOSurfaceDevice,
    offset: u64,
    value: u64,
) {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_iosurface_device_mmio_write") }
            .device
            .mmio_write(offset, value);
    })
}

/// After a saved state's load, before the guest runs: maps again every surface whose guest
/// descriptor is marked mapped. The guest's memory comes back with its mappings, while a reset
/// device has none; write the registers again first (the read counter, the descriptor enable,
/// the descriptor base, then the ring). Returns how many surfaces the device holds after it. May
/// block on the IOSurface device's thread.
///
/// # Safety
/// `device` is a live IOSurface device handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rvg_iosurface_device_remap_surfaces(
    device: *mut RvgIOSurfaceDevice,
) -> u32 {
    ffi::export(|| {
        // SAFETY: the caller's contract.
        unsafe { live(device, "rvg_iosurface_device_remap_surfaces") }
            .device
            .remap_surfaces()
    })
}

#[cfg(test)]
mod tests;
