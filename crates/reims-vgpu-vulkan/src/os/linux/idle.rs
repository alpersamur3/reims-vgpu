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
//! - **Core's own mappings** of the guest's pages (object lists, the task memory behind a host
//!   address, the pages a no-copy buffer is made over) take a kernel mapping for each run of
//!   scattered pages, as an alias does, and grow with what the guest keeps open: 46,000 of them
//!   at one point on the phone, 6,000 when the screen closed. Those only the CPU uses are
//!   deferred views ([`lazy_core`]); the pages the GPU runtime imports by pointer are not, unless
//!   named.
//! - **IOSurface views** over the guest's pages (`REIMS_VGPU_LAZY_MAPPINGS=surfaces`, on by default
//!   where deferred views are off; `lists` alone turns them off) are the largest class on the
//!   phone once the guest's screens are heavy: a surface
//!   over several ranges has a view of its own, one mapping per run of scattered pages, for as
//!   long as the guest keeps the surface (22,000 of 25,000 mappings at one sample). Such a view
//!   is made lazy from the start ([`lazy_surfaces`], `super::surface`): a reservation, mapped in
//!   whole only while a runtime import of its pointer lives (the pin of [`super::deferred`]) and
//!   given back when the last hold of the import goes and the budget has no room or a few
//!   seconds pass ([`give_back_surfaces`]).
//! - **The census** (`crate::gpu::idle`) runs wherever deferred views are off, switch or no
//!   switch: it is how to tell, on a device, which class of views holds the mappings as their
//!   count grows.

use reims_vgpu_core::num::ToU64 as _;

use super::{budget, deferred, files, surface};

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
    /// The lazy IOSurface views alive, and the times one was given back.
    pub(crate) lazy_surfaces: u64,
    pub(crate) surfaces_given: u64,
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
        lazy_surfaces: 0,
        surfaces_given: 0,
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
        lazy_surfaces: surface::lazy_alive(),
        surfaces_given: surface::given_back(),
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

/// Whether the mappings core makes for `maker` (its scope, `reims_vgpu_core::memory::mapping::
/// maker`) are deferred views: a reservation with no access whose runs the table records
/// unmapped, and whose 128 KiB windows the CPU's touch maps from the table (the fault handler of
/// [`super::deferred`], windows bounded by its page limit), instead of a kernel mapping for each
/// run of scattered pages.
///
/// Only where idle views are on ([`enabled`]: the old path, deferred views off) and only for
/// makers whose pointer the CPU alone uses. `REIMS_VGPU_LAZY_MAPPINGS` names them, comma
/// separated: `lists` (object lists, the default), `addresses` (the task memory a command reads
/// or writes through a host address for the length of the command), `buffers` (the pages a
/// no-copy buffer is made over, which the GPU runtime imports: off unless named, because on a
/// driver that cannot gather the import maps the whole view for good, as an eager mapping
/// would be, [`super::deferred::gather_source`]); `none` or `0` for none. `surfaces` is
/// [`lazy_surfaces`]'s.
pub(crate) fn lazy_core(maker: u8) -> bool {
    use reims_vgpu_core::memory::mapping::maker as m;
    #[cfg(test)]
    if TEST_ON.with(std::cell::Cell::get) {
        return maker == m::OBJECT_LIST;
    }
    if !enabled() {
        return false;
    }
    let named = named();
    match maker {
        m::OBJECT_LIST => named.lists,
        m::TASK_ADDRESS => named.addresses,
        m::BUFFER_OVER_PAGES => named.buffers,
        _ => false,
    }
}

/// The makers `REIMS_VGPU_LAZY_MAPPINGS` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Named {
    lists: bool,
    addresses: bool,
    buffers: bool,
    surfaces: bool,
}

impl Named {
    /// What no variable says: object lists and surface views. Every reader sits behind
    /// [`enabled`], which is only on where deferred views are off (the compatibility path), so
    /// where they are on the default stays the object lists alone: that path's census showed
    /// the surface views running out of the kernel's mappings (33,128 held, fatal) while
    /// `lists,surfaces` peaked at 5,731. A value of the variable names what it says and
    /// nothing more, so `lists` alone turns the surface views off again.
    const DEFAULT: Named = Named {
        lists: true,
        addresses: false,
        buffers: false,
        surfaces: true,
    };

    /// What `value` names; a name it does not know names nothing.
    fn parse(value: &str) -> Named {
        let has = |what: &str| value.split(',').any(|n| n.trim() == what);
        Named {
            lists: has("lists"),
            addresses: has("addresses"),
            buffers: has("buffers"),
            surfaces: has("surfaces"),
        }
    }
}

/// The makers named, read once.
fn named() -> Named {
    static NAMED: std::sync::OnceLock<Named> = std::sync::OnceLock::new();
    *NAMED.get_or_init(|| match std::env::var("REIMS_VGPU_LAZY_MAPPINGS") {
        Err(_) => Named::DEFAULT,
        Ok(value) => Named::parse(&value),
    })
}

#[cfg(test)]
thread_local! {
    /// A test's lazy surface views, on the thread that sets it, whatever the default.
    pub(crate) static TEST_SURFACES: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the view of an IOSurface over the guest's pages is a lazy view from the start
/// (module docs): where idle views are on (so deferred views are off) and
/// `REIMS_VGPU_LAZY_MAPPINGS` is unset or names `surfaces` (`lists,surfaces` for both). On by
/// default there, [`Named::DEFAULT`]; `lists` alone turns it off.
pub(crate) fn lazy_surfaces() -> bool {
    #[cfg(test)]
    if TEST_SURFACES.with(std::cell::Cell::get) {
        return true;
    }
    enabled() && named().surfaces
}

/// Whether the views census runs: where deferred views are off, which have a census of their own.
pub(crate) fn census_on() -> bool {
    !deferred::enabled()
}

/// Gives back the lazy surface views no import holds, those idle for `older_than` at least (any,
/// with `None`), until `enough`: how many ([`surface::give_back_idle`]).
pub(crate) fn give_back_surfaces(
    older_than: Option<std::time::Duration>,
    enough: &dyn Fn() -> bool,
) -> u64 {
    surface::give_back_idle(older_than, enough)
}

/// Gives the view of `length` bytes from `first` back to the kernel, its address kept
/// ([`deferred::make_lazy`]): `true` where it holds one mapping, not its runs, from now on.
pub(crate) fn give_back(first: u64, length: u64) -> bool {
    deferred::make_lazy(first, length)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Surface views are named like the other makers: on by default (where idle views are on at
    /// all), off by a value that does not name them, on by `surfaces` alone or beside the rest,
    /// and not by a name that only looks like it
    #[test]
    fn surfaces_are_named_by_the_switch() {
        assert!(Named::DEFAULT.surfaces && Named::DEFAULT.lists);
        assert!(!Named::DEFAULT.addresses && !Named::DEFAULT.buffers);
        let lists = Named::parse("lists");
        assert!(
            lists.lists && !lists.surfaces,
            "an explicit value wins: lists alone is the old default"
        );
        let only = Named::parse("surfaces");
        assert!(only.surfaces && !only.lists, "naming replaces the default");
        let both = Named::parse("lists, surfaces");
        assert!(both.surfaces && both.lists && !both.buffers && !both.addresses);
        let all = Named::parse("lists,addresses,buffers,surfaces");
        assert!(all.surfaces && all.lists && all.addresses && all.buffers);
        for none in ["none", "0", "", "surface", "Surfaces", "surfaces2"] {
            assert_eq!(
                Named::parse(none),
                Named {
                    lists: false,
                    addresses: false,
                    buffers: false,
                    surfaces: false
                },
                "{none:?}"
            );
        }
    }
}
