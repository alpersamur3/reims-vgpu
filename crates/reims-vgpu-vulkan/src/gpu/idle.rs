//! Which of the library's views of the guest's memory nobody needs mapped, and the census of the
//! kernel mappings the views hold (ASoftware's Android port, with deferred views off: the old
//! path, where a view of N scattered 16 KiB pages costs N mappings and Android allows a process
//! 65,530 of them, `crate::os`'s `budget`).
//!
//! - **The classes.** Each [`Alias`](super::backing::Alias) carries a [`Tag`] with its class:
//!   *live* (attached to an object the runtime may import it for, or made for a surface),
//!   *kept* (the alias a detached buffer or heap keeps only so that its `contents` stays a valid
//!   pointer, [`super::backing::Backing::kept_contents`]), *lazy* (a kept alias given back to the
//!   kernel, below), and *runtime only* (its object is gone or replaced it, and the runtime's
//!   keep-alive holds the rest of its life: a heap whose cover buffer is parked in core's
//!   `KEPT_COVERS` is such an alias until the runtime lets go of the heap).
//! - **Idle views.** A kept alias is idle once the runtime holds no import of it: the object has
//!   the only reference left (`Arc::strong_count`; an import holds the alias in its keep-alive,
//!   [`super::backing::keep_alive`], until the runtime freed the import, after the detach, which
//!   it does when the command buffers that used the pages completed). Nothing can take a new
//!   reference to an alias once its object holds it (only the call that makes it clones it, for
//!   its import), so the count cannot rise again. Such a view is given back
//!   ([`crate::os::give_back_view`]): its mappings become one reservation whose pages the CPU's
//!   touch maps again from the backing table, so `contents` reads the same bytes at the same
//!   address. Only kept aliases are: a live alias may be imported, and the driver's pin of
//!   pages under a mapping is not for this module to move.
//! - **Dropped heaps.** The exception is the alias of a heap that dropped
//!   ([`heap_dropped`]): its runtime heap lives on while core parks its cover buffer
//!   (`KEPT_COVERS`, until the device tears its tasks down), and the runtime's keep-alive holds the
//!   alias with every mapping for the whole run (307 covers held about 18,000 mappings on the
//!   phone). Nothing reads such a heap's `contents` (core keeps the cover for Metal's numbering
//!   alone), and the guest deleted its pages' use, so after a few seconds, for commands still in
//!   flight, the alias is made lazy, not freed: the address the runtime's import was made at
//!   stays mapped (one reservation), the same file pages show through it when touched, and the
//!   driver's pin of pages, if it has one, is of the same file pages. That last is an
//!   assumption about the driver this repository cannot check.
//!   Giving back is on unless `REIMS_VGPU_IDLE_VIEWS=0` (on Android; `1` elsewhere,
//!   `crate::os`'s `idle`). It happens a few seconds after the detach once the runtime let go
//!   ([`tick`]), and at once for the oldest first where the budget would end the process
//!   ([`give_back_for`]).
//! - **IOSurface views** (`REIMS_VGPU_LAZY_MAPPINGS=surfaces`) are given back by the same
//!   passes ([`crate::os::give_back_surface_views`]): a lazy surface view whose runtime imports
//!   were all freed. The census names them by "lazy surfaces", their count and how many times
//!   one was given back, and the mappings of the ones mapped now remain in "other by maker"
//!   as `surfaces`.
//! - **The census** logs one line every 30 s (at the first view made or object detached after
//!   them) with the budget's count and the mappings of each class, so that a device's log says
//!   which class holds them as the count grows. The line starts with [`CENSUS`].

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) use real::*;

#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub(crate) use none::*;

/// The prefix of the census line.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) const CENSUS: &str = "reims-vgpu: views census:";

/// No idle views on this OS: the tags count nothing and nothing is given back.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
mod none {
    use std::sync::Arc;

    use super::super::backing::Alias;

    /// A view's class: none here.
    #[derive(Debug, Default)]
    pub(crate) struct Tag;

    pub(crate) fn made(_first: u64, _length: u64) -> Arc<Tag> {
        Arc::new(Tag)
    }

    pub(crate) fn dropped(_first: u64) {}

    pub(crate) fn detached(_alias: &Arc<Alias>) {}

    pub(crate) fn released(_alias: &Arc<Alias>) {}

    pub(crate) fn heap_dropped(_alias: &Arc<Alias>) {}

    pub(crate) fn tick() {}
}

#[cfg(any(target_os = "linux", target_os = "android"))]
mod real {
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
    use std::sync::{Arc, OnceLock, Weak};
    use std::time::{Duration, Instant};

    use reims_vgpu_core::num::ToU64 as _;
    use reims_vgpu_core::sync::Mutex;

    use super::super::backing::Alias;
    use super::CENSUS;
    use crate::os;

    /// How often the census is logged.
    const CENSUS_EVERY: Duration = Duration::from_secs(30);

    /// How often idle views are looked for, and how long an alias is kept before it is given
    /// back: a detached object whose `contents` is read soon after is not worth a fault.
    const SWEEP_EVERY: Duration = Duration::from_secs(5);
    const IDLE_AFTER: Duration = Duration::from_secs(5);

    /// An alias's class (module docs).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Class {
        Live,
        Kept,
        Lazy,
        RuntimeOnly,
    }

    impl Class {
        fn bits(self) -> u8 {
            self as u8
        }

        fn of(bits: u8) -> Class {
            match bits {
                1 => Class::Kept,
                2 => Class::Lazy,
                3 => Class::RuntimeOnly,
                _ => Class::Live,
            }
        }
    }

    /// An alias's class, shared with the census's registry.
    #[derive(Debug, Default)]
    pub(crate) struct Tag {
        class: AtomicU8,
        /// Its object was a heap that dropped ([`heap_dropped`]).
        heap_gone: AtomicBool,
    }

    impl Tag {
        /// The class now.
        pub(crate) fn class(&self) -> Class {
            Class::of(self.class.load(Ordering::Acquire))
        }

        /// Moves the alias from class `from` to `to`: whether it was in `from`.
        fn moved(&self, from: Class, to: Class) -> bool {
            self.class
                .compare_exchange(from.bits(), to.bits(), Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        }

        /// The alias's object let go of it (released or replaced): the runtime's keep-alive holds
        /// the rest of its life, unless it is lazy already, which stays a lazy view until it
        /// drops.
        fn orphaned(&self) {
            let _ = self
                .class
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| {
                    matches!(Class::of(bits), Class::Live | Class::Kept)
                        .then_some(Class::RuntimeOnly.bits())
                });
        }
    }

    /// The aliases alive, by their first byte: their length and class.
    static REGISTRY: Mutex<BTreeMap<u64, (u64, Arc<Tag>)>> = Mutex::new(BTreeMap::new());

    /// The aliases of heaps that dropped, not given back yet, oldest first, and when each did.
    static ORPHANS: Mutex<VecDeque<(Weak<Alias>, Instant)>> = Mutex::new(VecDeque::new());

    /// The kept aliases not given back yet, oldest first, and when each was detached.
    static CANDIDATES: Mutex<VecDeque<(Weak<Alias>, Instant)>> = Mutex::new(VecDeque::new());

    /// Held while a pass looks for idle views: a second pass in the meantime would hold
    /// references of its own, and see an alias the first looks at as one the runtime still holds.
    static PASS: Mutex<()> = Mutex::new(());

    /// Views given back, in all and of them under the budget's pressure.
    static GIVEN_BACK: AtomicU64 = AtomicU64::new(0);
    static UNDER_PRESSURE: AtomicU64 = AtomicU64::new(0);

    /// When the next pass and the next census are due, in milliseconds since the first call.
    static NEXT_PASS: AtomicU64 = AtomicU64::new(0);
    static NEXT_CENSUS: AtomicU64 = AtomicU64::new(0);

    /// The class tag of an alias of `length` bytes from `first`, made now.
    pub(crate) fn made(first: u64, length: u64) -> Arc<Tag> {
        let tag = Arc::new(Tag::default());
        if os::views_census_on() {
            REGISTRY.lock().insert(first, (length, Arc::clone(&tag)));
        }
        tag
    }

    /// The alias from `first` is freed.
    pub(crate) fn dropped(first: u64) {
        REGISTRY.lock().remove(&first);
    }

    /// The object holding `alias` is detached: it is kept for its `contents` from now on, idle
    /// once the runtime lets go of it (module docs).
    pub(crate) fn detached(alias: &Arc<Alias>) {
        if alias.tag().moved(Class::Live, Class::Kept) && os::idle_views_enabled() {
            CANDIDATES
                .lock()
                .push_back((Arc::downgrade(alias), Instant::now()));
        }
    }

    /// The heap holding `alias` dropped. Its runtime heap may live on, retained by the cover
    /// buffer core parks until the device tears its tasks down (`KEPT_COVERS`), and with it the
    /// runtime's keep-alive of the alias, with all its mappings; nothing reads the heap's
    /// `contents` any more, and the guest deleted it, so the alias is given back (module docs)
    /// after a few seconds for the commands that may still use it to complete, rather than
    /// freed: the address stays valid for the runtime's import, and the bytes the same.
    pub(crate) fn heap_dropped(alias: &Arc<Alias>) {
        alias.tag().heap_gone.store(true, Ordering::Release);
        if os::idle_views_enabled() {
            ORPHANS
                .lock()
                .push_back((Arc::downgrade(alias), Instant::now()));
        }
    }

    /// The object holding `alias` is released, or replaced it.
    pub(crate) fn released(alias: &Arc<Alias>) {
        alias.tag().orphaned();
    }

    /// Looks for idle views and logs the census, each when due (module docs): called where views
    /// are made and objects detached, never where one drops, which may run on the runtime's
    /// threads in the middle of its own work.
    pub(crate) fn tick() {
        let now = milliseconds();
        if os::idle_views_enabled()
            && due(&NEXT_PASS, now, SWEEP_EVERY)
            && let Some(_pass) = PASS.try_lock()
        {
            give_back_where(Some(IDLE_AFTER), || false);
        }
        if os::views_census_on() && due(&NEXT_CENSUS, now, CENSUS_EVERY) {
            census();
        }
    }

    /// Milliseconds since the first call.
    fn milliseconds() -> u64 {
        static START: OnceLock<Instant> = OnceLock::new();
        u64::try_from(START.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Whether `next` has passed at `now`, taking the turn `every` on: one caller of several
    /// that find it so.
    fn due(next: &AtomicU64, now: u64, every: Duration) -> bool {
        let at = next.load(Ordering::Relaxed);
        let every = u64::try_from(every.as_millis()).unwrap_or(u64::MAX);
        now >= at
            && next
                .compare_exchange(at, now.saturating_add(every), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
    }

    /// Gives back kept aliases the runtime holds no import of, oldest first, those kept for
    /// `older_than` at least (any, with `None`), until `enough`: how many. The pass lock is the
    /// caller's.
    fn give_back_where(older_than: Option<Duration>, enough: impl Fn() -> bool) -> u64 {
        let ready: Vec<Arc<Alias>> = {
            let mut list = CANDIDATES.lock();
            list.retain(|(alias, _)| {
                alias
                    .upgrade()
                    .is_some_and(|alias| alias.tag().class() == Class::Kept)
            });
            list.iter()
                .filter(|(_, since)| older_than.is_none_or(|age| since.elapsed() >= age))
                .filter_map(|(alias, _)| alias.upgrade())
                .collect()
        };
        let orphans: Vec<Arc<Alias>> = {
            let mut list = ORPHANS.lock();
            list.retain(|(alias, _)| {
                alias
                    .upgrade()
                    .is_some_and(|alias| alias.tag().class() == Class::RuntimeOnly)
            });
            list.iter()
                .filter(|(_, since)| older_than.is_none_or(|age| since.elapsed() >= age))
                .filter_map(|(alias, _)| alias.upgrade())
                .collect()
        };
        let mut given = 0;
        for alias in &orphans {
            if enough() {
                break;
            }
            if give_back_orphan(alias) {
                given += 1;
            }
        }
        for alias in &ready {
            if enough() {
                break;
            }
            if give_back_alias(alias) {
                given += 1;
            }
        }
        // The lazy IOSurface views whose imports the runtime freed (`crate::os`'s `surface`).
        if !enough() {
            given += os::give_back_surface_views(older_than, &enough);
        }
        GIVEN_BACK.fetch_add(given, Ordering::Relaxed);
        given
    }

    /// Gives `alias` back where it is kept and nothing but its object and the caller's one
    /// reference in `give_back_where` hold it. The count is read before the class: the object
    /// sets the class to runtime-only before it lets go of its reference
    /// ([`released`]), so a count of two that is the runtime's and the caller's is read
    /// with the class already moved.
    fn give_back_alias(alias: &Arc<Alias>) -> bool {
        const OBJECT_AND_CALLER: usize = 2;
        if Arc::strong_count(alias) != OBJECT_AND_CALLER || alias.tag().class() != Class::Kept {
            return false;
        }
        let (first, length) = alias.span();
        if !os::give_back_view(first, length) {
            return false;
        }
        alias.tag().moved(Class::Kept, Class::Lazy);
        true
    }

    /// Gives back the alias of a heap that dropped, whatever holds it: only the runtime's import
    /// of it does, of a heap nothing uses but commands already submitted. Its object's `contents`
    /// is gone, so the CPU never touches it; a pinned import keeps the same file pages.
    fn give_back_orphan(alias: &Arc<Alias>) -> bool {
        let tag = alias.tag();
        if !tag.heap_gone.load(Ordering::Acquire) || tag.class() != Class::RuntimeOnly {
            return false;
        }
        let (first, length) = alias.span();
        if !os::give_back_view(first, length) {
            return false;
        }
        tag.moved(Class::RuntimeOnly, Class::Lazy);
        true
    }

    /// Gives idle views back, the oldest first, until `enough` holds (the budget has room):
    /// whether it does. Nothing is given back where the idle views are off.
    pub(crate) fn give_back_for(enough: impl Fn() -> bool) -> bool {
        if !os::idle_views_enabled() {
            return false;
        }
        let given = {
            let _pass = PASS.lock();
            give_back_where(None, &enough)
        };
        if given > 0 {
            UNDER_PRESSURE.fetch_add(given, Ordering::Relaxed);
            log::info!(
                target: "reims_vgpu_vulkan::os",
                "{given} idle views given back to make room under the budget of mappings"
            );
        }
        enough()
    }

    /// The views of one class: how many, and the kernel mappings they hold.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Share {
        pub(crate) views: u64,
        pub(crate) mappings: u64,
    }

    /// The census (module docs).
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub(crate) struct Census {
        pub(crate) counts: Option<os::ViewCounts>,
        pub(crate) live: Share,
        pub(crate) kept: Share,
        pub(crate) lazy: Share,
        pub(crate) runtime_only: Share,
        pub(crate) covers: u64,
        pub(crate) given_back: u64,
        pub(crate) under_pressure: u64,
        /// The table's runs that are no alias's, by who made them, the most first.
        pub(crate) makers: Vec<(&'static str, u64)>,
    }

    impl Census {
        /// Counts an alias of class `class` holding `mappings`.
        pub(crate) fn add(&mut self, class: Class, mappings: u64) {
            let share = match class {
                Class::Live => &mut self.live,
                Class::Kept => &mut self.kept,
                Class::Lazy => &mut self.lazy,
                Class::RuntimeOnly => &mut self.runtime_only,
            };
            share.views += 1;
            share.mappings += mappings;
        }

        /// The aliases in all.
        fn aliases(&self) -> Share {
            let all = [self.live, self.kept, self.lazy, self.runtime_only];
            Share {
                views: all.iter().map(|s| s.views).sum(),
                mappings: all.iter().map(|s| s.mappings).sum(),
            }
        }

        /// The line (module docs): the budget's count against its limit, then each class's views
        /// and mappings, then the rest of what the table and the budget count.
        pub(crate) fn line(&self) -> String {
            let counts = self.counts.unwrap_or(os::ViewCounts::NONE);
            let all = self.aliases();
            // The table's runs that are no alias's: ranges a view can be made of, surfaces' views.
            let other = counts.table.saturating_sub(all.mappings);
            format!(
                "{CENSUS} held {} of budget {} (vm.max_map_count {}); aliases {} holding {} \
                 [live {}/{}, kept for contents {}/{}, runtime only {}/{}, lazy {}/{}]; lazy \
                 windows {} of {} pages; kept covers {}; twins {}; stand-ins {}; other table \
                 runs {}; lazy surfaces {} ({} given back); given back {} ({} under pressure);                  other by maker [{}]",
                counts.held,
                counts.budget,
                counts.limit,
                all.views,
                all.mappings,
                self.live.views,
                self.live.mappings,
                self.kept.views,
                self.kept.mappings,
                self.runtime_only.views,
                self.runtime_only.mappings,
                self.lazy.views,
                self.lazy.mappings,
                counts.windows,
                counts.pages,
                self.covers,
                counts.twins,
                counts.stand_ins,
                other,
                counts.lazy_surfaces,
                counts.surfaces_given,
                self.given_back,
                self.under_pressure,
                (self.makers.iter())
                    .map(|(name, held)| format!("{name} {held}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            )
        }
    }

    /// The census now.
    pub(crate) fn census_now() -> Census {
        let entries: Vec<(u64, u64, Class)> = REGISTRY
            .lock()
            .iter()
            .map(|(&first, (length, tag))| (first, *length, tag.class()))
            .collect();
        let ranges: Vec<(u64, u64)> = entries
            .iter()
            .map(|&(first, length, _)| (first, first + length))
            .collect();
        let mappings = os::view_mappings(&ranges);
        let mut census = Census {
            counts: Some(os::view_counts()),
            covers: reims_vgpu_core::resources::heap::detach::kept_covers().to_u64(),
            given_back: GIVEN_BACK.load(Ordering::Relaxed),
            under_pressure: UNDER_PRESSURE.load(Ordering::Relaxed),
            makers: os::view_makers(&ranges),
            ..Census::default()
        };
        for (&(_, _, class), &held) in entries.iter().zip(&mappings) {
            census.add(class, held);
        }
        census
    }

    /// Logs the census.
    fn census() {
        log::info!(target: "reims_vgpu_vulkan::os", "{}", census_now().line());
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The line starts with the fixed prefix and names the budget's count against its limit,
        /// each class's views and mappings, and the table's runs that no alias holds
        #[test]
        fn the_census_line_names_each_class() {
            let mut census = Census {
                counts: Some(os::ViewCounts {
                    held: 40_000,
                    limit: 65_530,
                    budget: 49_146,
                    twins: 30,
                    stand_ins: 4,
                    table: 39_900,
                    lazy_views: 1,
                    windows: 3,
                    pages: 96,
                    lazy_surfaces: 12,
                    surfaces_given: 4,
                }),
                covers: 7,
                given_back: 5,
                under_pressure: 2,
                makers: vec![("buffers over task pages", 800), ("object lists", 100)],
                ..Census::default()
            };
            census.add(Class::Live, 900);
            census.add(Class::Live, 100);
            census.add(Class::Kept, 30_000);
            census.add(Class::RuntimeOnly, 8_000);
            census.add(Class::Lazy, 0);
            let line = census.line();
            assert!(line.starts_with("reims-vgpu: views census:"), "{line}");
            for part in [
                "held 40000 of budget 49146 (vm.max_map_count 65530)",
                "aliases 5 holding 39000",
                "live 2/1000",
                "kept for contents 1/30000",
                "runtime only 1/8000",
                "lazy 1/0",
                "lazy windows 3 of 96 pages",
                "kept covers 7",
                "twins 30",
                "stand-ins 4",
                "other table runs 900",
                "lazy surfaces 12 (4 given back)",
                "given back 5 (2 under pressure)",
                "other by maker [buffers over task pages 800, object lists 100]",
            ] {
                assert!(line.contains(part), "{part}: {line}");
            }
        }

        /// A tag moves between classes only from the class it is in; an object that lets go of
        /// its alias makes it the runtime's, but not a lazy one
        #[test]
        fn a_tag_moves_between_classes() {
            let tag = Tag::default();
            assert_eq!(tag.class(), Class::Live);
            assert!(!tag.moved(Class::Kept, Class::Lazy));
            assert!(tag.moved(Class::Live, Class::Kept));
            assert!(tag.moved(Class::Kept, Class::Lazy));
            tag.orphaned();
            assert_eq!(tag.class(), Class::Lazy);
            let other = Tag::default();
            assert!(other.moved(Class::Live, Class::Kept));
            other.orphaned();
            assert_eq!(other.class(), Class::RuntimeOnly);
        }

        /// A turn is taken by one of the callers that find it due, and is not due again before
        /// its interval
        #[test]
        fn a_turn_is_taken_once() {
            let next = AtomicU64::new(0);
            let every = Duration::from_secs(30);
            assert!(due(&next, 100, every));
            assert!(!due(&next, 100, every));
            assert!(!due(&next, 30_099, every));
            assert!(due(&next, 30_100, every));
        }
    }
}
