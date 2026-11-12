//! The OS side of the idle views (ASoftware's Android port): what the library's views hold of
//! the kernel's mappings, read for the census, and the call that gives an idle view's mappings
//! back. Which views are idle, and the census itself, are `crate::gpu::idle`'s; this module is
//! what only Linux knows.
//!
//! - **Idle views** are views nothing needs mapped but their address: the alias a detached
//!   buffer or heap keeps for its `contents` ([`crate::gpu::backing`]). With deferred views off
//!   (the old path, which maps every run of a view at once and so costs one mapping per run of
//!   scattered 16 KiB pages), such a view holds all its mappings until its object is replaced or
//!   dropped, however long that is, and the budget ([`super::budget`]) only admits or ends the
//!   process. [`give_back`] puts one reservation with no access in their place, which the CPU's
//!   touch maps a window of again ([`super::deferred::make_lazy`]).
//! - **The switch.** `REIMS_VGPU_IDLE_VIEWS`: unset, on where the old path runs on Android;
//!   `1`, on wherever deferred views are off; `0`, off, which is the old path as it was. Where
//!   deferred views are on there is nothing to give back that they have not deferred already.
//! - **The census** (`crate::gpu::idle`) runs wherever deferred views are off, switch or no
//!   switch: it is how to tell, on a device, which class of views holds the mappings as their
//!   count grows.

use reims_vgpu_core::num::ToU64 as _;

use super::{budget, deferred, files};

/// The mappings the library holds, and how, as the views census prints them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Counts {
    /// The mappings the views hold, as the budget counts them.
    pub(crate) held: u64,
    /// `vm.max_map_count` as last read.
    pub(crate) limit: u64,
    /// The mappings the views may hold under it.
    pub(crate) budget: u64,
    /// Of `held`, the import twins' and the stand-ins' memory's.
    pub(crate) twins: u64,
    pub(crate) stand_ins: u64,
    /// Of `held`, the backing table's runs' (every view and every range a view can be made of).
    pub(crate) table: u64,
    /// The lazy views, and the windows and pages the CPU's touches have mapped of them.
    pub(crate) lazy_views: u64,
    pub(crate) windows: u64,
    pub(crate) pages: u64,
}

impl Counts {
    /// Nothing counted.
    pub(crate) const NONE: Counts = Counts {
        held: 0,
        limit: 0,
        budget: 0,
        twins: 0,
        stand_ins: 0,
        table: 0,
        lazy_views: 0,
        windows: 0,
        pages: 0,
    };
}

/// The counts now.
pub(crate) fn counts() -> Counts {
    let counted = budget::counted();
    let (lazy_views, windows, pages) = deferred::lazy_census();
    Counts {
        held: counted.held,
        limit: counted.limit,
        budget: counted.budget,
        twins: counted.twins,
        stand_ins: counted.stand_ins,
        table: files::mappings(),
        lazy_views: lazy_views.to_u64(),
        windows: windows.to_u64(),
        pages,
    }
}

/// The mappings of the backing table's runs outside `except` (sorted, apart), by the maker of
/// each run, the most first.
pub(crate) fn mappings_by_maker(except: &[(u64, u64)]) -> Vec<(&'static str, u64)> {
    files::mappings_by_maker_except(except)
}

/// The mappings the backing table's runs in each of `ranges` (a first byte and an end) take.
pub(crate) fn mappings_in(ranges: &[(u64, u64)]) -> Vec<u64> {
    files::mappings_in(ranges)
}

#[cfg(test)]
thread_local! {
    /// A test's idle views, on the thread that sets it, whatever the default.
    pub(crate) static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether idle views are given back (module docs).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if TEST_ON.with(std::cell::Cell::get) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| match std::env::var("REIMS_VGPU_IDLE_VIEWS").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => cfg!(target_os = "android"),
    });
    on && !deferred::enabled()
}

/// Whether the views census runs: where deferred views are off, which have a census of their own.
pub(crate) fn census_on() -> bool {
    !deferred::enabled()
}

/// Gives the view of `length` bytes from `first` back to the kernel, its address kept
/// ([`deferred::make_lazy`]): `true` where it holds one mapping, not its runs, from now on.
pub(crate) fn give_back(first: u64, length: u64) -> bool {
    deferred::make_lazy(first, length)
}
