//! The guest display's tests.

#![allow(clippy::float_cmp, clippy::field_reassign_with_default)]

use super::present::{FOURCC_BGRA, FOURCC_RGBA, MTL_PIXEL_FORMAT_BGRA8_UNORM, PageColorMatrix};
use super::*;
use crate::device::DeviceCore;
use crate::device::test_support::{Opts, TestDevice, device};
use crate::display::descriptor::DisplayDescriptor;
use crate::display::host_calls::CursorGlyph;
use crate::display::{ColorMatrix, DisplayCoord};
use crate::fifo::wire::Records;
use crate::platform::Handle;
use crate::raise::death::DeathSite;
use crate::rt::Hold;
use crate::testkit::calllog::Arg;
use crate::testkit::capture::catch_death;
use crate::testkit::host::{FakeHostQueue, QueueMode};
use crate::testkit::platform::FakePlatform;
use crate::testkit::script::Step;
use std::time::Duration;

type Fp = FakePlatform;

use crate::display::tests::display_lines;

const PAGE: u32 = 7;

/// A guest write a test plays against the shared page.
type GuestWrite = Box<dyn FnMut(GuestPage)>;

thread_local! {
    /// A guest write the test plays between `getDirtyColorMatrix`'s two index stores.
    static BETWEEN_INDEX_STORES: std::cell::RefCell<Option<GuestWrite>> =
        const { std::cell::RefCell::new(None) };
}

/// Called by `dirty_color_matrix_locked` between its stores of the read and write indices:
/// where a guest write racing the host lands.
pub(super) fn between_color_matrix_index_stores(page: GuestPage) {
    BETWEEN_INDEX_STORES.with(|h| {
        if let Some(f) = h.borrow_mut().as_mut() {
            f(page);
        }
    });
}

/// A guest display over a real 16 KiB page, a recorder device, and the process's host log.
struct Fx {
    p: Arc<Fp>,
    /// The guest display's device, with its root task and a kernel task (task 0, local): held for
    /// the fixture's life (its drop shuts the device down).
    _td: TestDevice,
    host: Arc<DeviceCore<Fp>>,
    guest_display: Arc<GuestDisplay<Fp>>,
}

/// A device with its root task and a kernel task (task 0, local, over guest page 0x55).
fn guest_display_device() -> TestDevice {
    let td = device(Opts::default()).with_root_page();
    td.task_root_node(0x55, 0x4000);
    {
        {
            let hold = Hold::NONE;
            td.device
                .create_task_id(hold, 0, 0x55, 0x4000, false)
                .unwrap();
        }
    }
    td
}

/// The device as a display host.
fn host_of(td: &TestDevice) -> Arc<DeviceCore<Fp>> {
    Arc::clone(&td.device)
}

/// Guest RAM at guest-physical `pa`, as the host sees it now.
fn guest_bytes(td: &TestDevice, pa: u64, n: usize) -> Vec<u8> {
    td.platform.fake_vm().sync();
    let mut v = vec![0u8; n];
    assert!(td.host.ram().read(pa, &mut v));
    v
}

/// A guest display raise's kind.
fn kind(e: &Raise) -> RaiseKind {
    match e.id() {
        RaiseId::GuestDisplay(k) => k,
        other => panic!("not a guest display raise: {other}"),
    }
}

/// A raise's detail (its reason).
fn reason(e: &Raise) -> &str {
    e.detail().unwrap_or_default()
}

/// Where an uncaught raise of `k` dies.
fn uncaught(k: RaiseKind) -> DeathSite {
    DeathSite {
        kind: crate::raise::death::DeathKind::UnhandledRaise,
        file: k.id(),
        line: 0,
    }
}

/// The death the tests' own page accesses name (never reached: the page is mapped).
const ANY: DeathKind = DeathKind::PageWhilePaused;

/// The state machine, as the guest display's methods run it under `guest_display_mutex`.
fn run_state_machine(on: OnFault, guest_display: &GuestDisplay<Fp>) -> RaiseResult<()> {
    let mut l = guest_display.guest_display_mutex.lock();
    let keep = guest_display.client_core();
    guest_display.state_machine(on, &mut l, keep.as_ref())
}

/// The mode list sync under `guest_display_mutex`.
fn sync_mode_list(guest_display: &GuestDisplay<Fp>) -> bool {
    let mut l = guest_display.guest_display_mutex.lock();
    guest_display.sync_mode_list(&mut l)
}

/// The settings sync under `guest_display_mutex`.
fn sync_settings(guest_display: &GuestDisplay<Fp>) -> bool {
    let l = guest_display.guest_display_mutex.lock();
    let keep = guest_display.client_core();
    guest_display.sync_settings(&l, keep.as_ref())
}

/// The gamma table's store under `guest_display_mutex`.
fn set_gamma_table(guest_display: &GuestDisplay<Fp>, gamma: GammaTableArgs) -> RaiseResult<()> {
    let l = guest_display.guest_display_mutex.lock();
    guest_display.store_gamma_table(&l, gamma)
}

/// The dirty colour matrix under `guest_display_mutex`, read out before the lock goes.
fn get_dirty_color_matrix(guest_display: &GuestDisplay<Fp>) -> Option<[[f32; 4]; 3]> {
    let l = guest_display.guest_display_mutex.lock();
    guest_display.take_dirty_color_matrix(&l).map(|m| {
        let mut out = [[0f32; 4]; 3];
        for (row, rv) in (0u8..).zip(out.iter_mut()) {
            for (col, cv) in (0u8..).zip(rv.iter_mut()) {
                *cv = m.at(row, col);
            }
        }
        out
    })
}

/// The mapped present from the retained surface on, in its own pool and holder: the part a test
/// reaches without an IOSurface host device.
#[expect(
    clippy::too_many_arguments,
    reason = "the present's arguments, and the hold"
)]
fn present_mapped_with_surface(
    hold: Hold<'_>,
    guest_display: &GuestDisplay<Fp>,
    l: &mut GuestDisplayLocked<Fp>,
    client: Option<&Arc<DisplayCore<Fp>>>,
    task: KernelTaskRef<Fp>,
    io_surface: Option<<Fp as Platform>::Surface>,
    surface: u64,
    gamma: GammaTableArgs,
) -> RaiseResult<()> {
    crate::rt::pool::around(&*guest_display.platform, || {
        crate::rt::Held::raising(|_| {
            guest_display.present_mapped_retained(hold, l, client, task, io_surface, gamma)
        })
    })?;
    guest_display.presented_mapped_surface(l, surface);
    Ok(())
}

/// The page, as the lock-free readers reach it (dies while paused).
fn shared_state(guest_display: &GuestDisplay<Fp>) -> GuestPage {
    guest_display.page_unlocked().0
}

impl Fx {
    fn with_port(port: u64) -> Fx {
        let td = guest_display_device();
        let p = td.platform.clone();
        let host = host_of(&td);
        let guest_display =
            GuestDisplay::new(p.clone(), Arc::downgrade(&host), port, PAGE).unwrap();
        p.call_log().clear();
        Fx {
            p,
            _td: td,
            host,
            guest_display,
        }
    }

    /// The guest display's page, where the guest display maps it.
    fn page(&self) -> GuestPtr {
        shared_state(&self.guest_display).base()
    }

    /// `n` bytes of the page at `off`.
    fn read_vec(&self, off: u64, n: usize) -> Vec<u8> {
        let mut v = vec![0u8; n];
        self.page().copy_to(off, &mut v, ANY);
        v
    }

    fn new() -> Fx {
        Fx::with_port(0)
    }

    fn r8(&self, off: u64) -> u8 {
        self.page().read_u8(off, ANY)
    }
    fn w8(&self, off: u64, v: u8) {
        self.page().write_u8(off, v, ANY);
    }
    fn r16(&self, off: u64) -> u16 {
        self.page().read_u16(off, ANY)
    }
    fn w16(&self, off: u64, v: u16) {
        self.page().write_u16(off, v, ANY);
    }
    fn r32(&self, off: u64) -> u32 {
        self.page().read_u32(off, ANY)
    }
    fn w32(&self, off: u64, v: u32) {
        self.page().write_u32(off, v, ANY);
    }
    fn r64(&self, off: u64) -> u64 {
        self.page().read_u64(off, ANY)
    }
    fn w64(&self, off: u64, v: u64) {
        self.page().write_u64(off, v, ANY);
    }
    fn rf32(&self, off: u64) -> f32 {
        f32::from_bits(self.r32(off))
    }
    fn wf32(&self, off: u64, v: f32) {
        self.w32(off, v.to_bits());
    }

    fn sels(&self) -> Vec<&'static str> {
        self.p.call_log().sels()
    }
    fn count(&self, sel: &str) -> usize {
        self.p.call_log().count(sel)
    }
    fn clear(&self) {
        self.p.call_log().clear();
    }
    /// The device's pokes: its `raiseInterrupt` host calls.
    fn pokes(&self) -> usize {
        self.count("raiseInterrupt")
    }

    /// A display attached through `Display::new` (the recorder device accepts the port),
    /// then linked to the guest display as the device's attach does.
    fn display(&self, desc: DisplayDescriptor) -> Arc<Display<Fp>> {
        Display::new(&self.p, self.host.clone(), &desc, 0, 0x1234)
            .unwrap()
            .unwrap()
    }

    fn attach(&self, d: &Arc<Display<Fp>>) {
        self.guest_display
            .set_client(OnFault::Return, Some(Arc::downgrade(d)))
            .unwrap();
    }
}

fn mode(w: u16, h: u16, hz: f64) -> DisplayMode {
    DisplayMode {
        size_in_pixels: DisplayCoord { x: w, y: h },
        refresh_rate_in_hz: hz,
        supports_edr: false,
        supports_10bpc: false,
    }
}

/// A descriptor with a queue that runs jobs at once.
fn queued(q: &Arc<FakeHostQueue>) -> DisplayDescriptor {
    let queue: Arc<dyn HostQueue> = q.clone();
    DisplayDescriptor {
        queue: Some(queue),
        ..DisplayDescriptor::default()
    }
}

mod init {
    use super::*;

    /// Maps page `PAGE` for one guest page, writes port (truncated), field1c, the diagonal of the
    /// matrix at the guest's write index, and dirty; logs its creation.
    #[test]
    fn writes_the_page() {
        let logs = crate::testkit::logs::capture();
        let td = guest_display_device();
        let p = td.platform.clone();
        let host = host_of(&td);
        let pa = u64::from(PAGE) * crate::guest::records::APV_PAGE_SIZE;
        // A guest write index of 5, and garbage off the diagonal.
        assert!(td.host.ram().write(pa + sp::COLOR_MATRIX_WRITE_INDEX, &[5]));
        assert!(
            td.host
                .ram()
                .write_u32(pa + sp::color_matrix(5, 0, 1), 0xdead_beef)
        );
        td.platform.fake_vm().sync();
        let guest_display =
            GuestDisplay::new(p.clone(), Arc::downgrade(&host), 0x1_0003, PAGE).unwrap();
        let ram = |off: u64, n: usize| guest_bytes(&td, pa + off, n);
        let u32_at = |off: u64| u32::from_le_bytes(ram(off, 4).try_into().unwrap());
        assert_eq!(ram(sp::PORT, 2), vec![3, 0]);
        assert_eq!(u32_at(sp::FIELD_1C), 1024);
        for i in 0..3u8 {
            assert_eq!(u32_at(sp::color_matrix(5, i, i)), 1.0f32.to_bits());
        }
        assert_eq!(u32_at(sp::color_matrix(5, 0, 1)), 0xdead_beef);
        assert_eq!(u32_at(sp::color_matrix(0, 0, 0)), 0);
        assert_eq!(ram(sp::COLOR_MATRIX_DIRTY, 1), vec![1]);
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Info,
                "guest display on port 65539 created".to_owned()
            )]
        );
        assert_eq!(guest_display.port, 0x1_0003);
    }

    /// A nil mapping (a page outside guest RAM) throws after a warning, naming the site its
    /// death would.
    #[test]
    fn nil_mapping_throws() {
        let td = guest_display_device();
        let p = td.platform.clone();
        let host = host_of(&td);
        {
            let logs = crate::testkit::logs::capture();
            {
                let e = GuestDisplay::new(p.clone(), Arc::downgrade(&host), 2, 0x9000)
                    .err()
                    .unwrap();
                assert_eq!(kind(&e), RaiseKind::MapPage);
                assert_eq!(
                    reason(&e),
                    "guest display on port 2: shared state page 0x9000 could not be mapped"
                );
                assert_eq!(
                    display_lines(&logs),
                    vec![(
                    log::Level::Warn,
                    "guest display on port 2 refused: its shared state page 0x9000 could not be \
                     mapped"
                        .to_owned()
                )]
                );
            }
        }
        let r = GuestDisplay::new(p.clone(), Arc::downgrade(&host), 2, 0x9000);
        assert_eq!(
            r.err().map(|e| e.site()),
            Some(uncaught(RaiseKind::MapPage))
        );
    }

    /// A device already gone maps nothing: the same throw.
    #[test]
    fn gone_device_throws() {
        let p = Arc::new(FakePlatform::new());
        let host: Weak<DeviceCore<Fp>> = {
            let other = device(Opts::default());
            Arc::downgrade(&host_of(&other))
        };
        assert_eq!(host.strong_count(), 0);
        let e = GuestDisplay::new(p, host, 0, 1).err().unwrap();
        assert_eq!(kind(&e), RaiseKind::MapPage);
    }
}

#[test]
fn shared_state_asserts_unpaused() {
    let fx = Fx::new();
    let _ = shared_state(&fx.guest_display);
    fx.guest_display.pause();
    let r = catch_death(|| shared_state(&fx.guest_display));
    assert_eq!(r.err(), Some(DeathKind::PageWhilePaused.site()));
}

mod set_client {
    use super::*;

    /// Sets the client (both copies), runs the machine; a second client asserts.
    #[test]
    fn sets_and_asserts_one_at_a_time() {
        let fx = Fx::new();
        let d = fx.display(DisplayDescriptor::default());
        fx.attach(&d);
        assert!(fx.guest_display.client().is_some());
        assert!(
            fx.guest_display
                .client()
                .unwrap()
                .ptr_eq(&Arc::downgrade(&d))
        );
        // No modes: state 0 stays 0, no interrupt.
        assert_eq!(fx.r32(sp::STATE), 0);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 0);
        let r = catch_death(|| {
            fx.guest_display
                .set_client(OnFault::Die, Some(Arc::downgrade(&d)))
        });
        assert_eq!(r.err(), Some(DeathKind::ClientAlreadySet.site()));
        fx.guest_display.set_client(OnFault::Die, None).unwrap();
        assert!(!fx.guest_display.client().is_some());
        assert!(fx.guest_display.client().is_none());
        fx.guest_display.set_client(OnFault::Die, None).unwrap();
    }
}

#[test]
fn set_mode_list_connects_and_consumes() {
    let fx = Fx::new();
    let d = fx.display(DisplayDescriptor::default());
    fx.attach(&d);
    fx.guest_display
        .set_mode_list(OnFault::Die, Arc::from([mode(1920, 1080, 60.0)]))
        .unwrap();
    // Consumed by the connect.
    assert!(fx.guest_display.mode_list().is_empty());
    assert_eq!(fx.r32(sp::STATE), 1);
    assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 1);
    // While paused the machine does nothing: the list is kept for the unpause.
    fx.guest_display.pause();
    fx.guest_display
        .set_mode_list(OnFault::Die, Arc::from([mode(800, 600, 60.0)]))
        .unwrap();
    assert_eq!(fx.guest_display.mode_list().len(), 1);
}

/// A present holds the guest display's mutex and then takes the display's. A `set_mode_list`
/// meanwhile waits for the guest display's mutex without holding the display's, so the present goes
/// on; the list reaches the guest display once the present lets go.
#[test]
#[allow(clippy::disallowed_methods)] // test scaffolding: a second thread, real-time waits
fn set_mode_list_while_a_present_holds_the_guest_display() {
    let fx = Fx::new();
    let d = fx.display(DisplayDescriptor::default());
    fx.attach(&d);
    // Linked both ways, as the device's attach links them.
    d.set_guest_display(OnFault::Return, Some(Arc::downgrade(&fx.guest_display)))
        .unwrap();
    let held = fx.guest_display.guest_display_mutex.lock();
    std::thread::scope(|s| {
        let h = s.spawn(|| d.set_mode_list(&[mode(1024, 768, 60.0)]).is_ok());
        // The list is stored under the display's mutex before the guest display's is asked for.
        while d.mode_list().is_empty() {
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(20));
        // As a present does: the display's mutex, under the guest display's.
        let g = d.display_mutex.try_lock_for(Duration::from_secs(2));
        assert!(g.is_some(), "set_mode_list held the display's mutex");
        drop(g);
        drop(held);
        assert!(h.join().unwrap());
    });
    assert_eq!(fx.r32(sp::STATE), 1);
    assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 1);
    d.set_guest_display(OnFault::Return, None).unwrap();
}

/// The connect/ack machine.
mod state_machine {
    use super::*;

    fn connected(fx: &Fx) -> Arc<Display<Fp>> {
        let d = fx.display(DisplayDescriptor::default());
        fx.attach(&d);
        fx.guest_display
            .set_mode_list(OnFault::Die, Arc::from([mode(1024, 768, 60.0)]))
            .unwrap();
        d
    }

    /// State 0 without a client or modes: nothing but `_didRestore = NO`.
    #[test]
    fn state0_idle() {
        let fx = Fx::new();
        fx.guest_display.guest_display_mutex.lock().did_restore = true;
        run_state_machine(OnFault::Die, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::STATE), 0);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 0);
        assert!(!fx.guest_display.guest_display_mutex.lock().did_restore);
        assert_eq!(fx.pokes(), 0);
    }

    /// State 0 with both: settings, modes, protection, colour space, ack+1 (wrapping),
    /// pending, interrupt 4 (poked only when enabled), then state 1.
    #[test]
    fn state0_connects() {
        let fx = Fx::new();
        fx.w32(sp::EXPECTED_ACK, 0xffff_ffff);
        fx.w32(sp::ENABLED_INTERRUPTS, 4);
        fx.p.script()
            .push("enable_protected_content", Step::value(true));
        let d = fx.display(DisplayDescriptor::default());
        d.protection_options.store(0x30, Ordering::SeqCst);
        fx.attach(&d);
        fx.clear();
        fx.guest_display
            .set_mode_list(OnFault::Die, Arc::from([mode(1024, 768, 60.0)]))
            .unwrap();
        assert_eq!(fx.r32(sp::SERIAL_NUM), 0x1234);
        assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 1);
        assert_eq!(fx.r64(sp::PROTECTION_OPTIONS), 0x30);
        assert_eq!(fx.r32(sp::EXPECTED_ACK), 0);
        assert_eq!(fx.r8(sp::ACK_PENDING), 1);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 4);
        assert_eq!(fx.r32(sp::STATE), 1);
        assert_eq!(fx.pokes(), 1);
        // The colour space is dirty from the connect.
        assert!(
            fx.guest_display
                .guest_display_mutex
                .lock()
                .guest_color_space_dirty
        );
    }

    /// State 1 with `ackPending == 1`: state 1 rewritten, no interrupt, `_didRestore` kept, and
    /// a debug line.
    #[test]
    fn state1_pending_returns() {
        let fx = Fx::new();
        let logs = crate::testkit::logs::capture();
        fx.w32(sp::STATE, 1);
        fx.w8(sp::ACK_PENDING, 1);
        fx.guest_display.guest_display_mutex.lock().did_restore = true;
        run_state_machine(OnFault::Die, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::STATE), 1);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 0);
        assert!(fx.guest_display.guest_display_mutex.lock().did_restore);
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Debug,
                "guest display on port 0: state 1 with an ack still pending, left as it is"
                    .to_owned()
            )]
        );
    }

    /// State 1 tests `== 1`, so a pending byte of 3 moves on to 2, and then (no client)
    /// disconnects.
    #[test]
    fn state1_pending_byte_3_moves_on() {
        let fx = Fx::new();
        fx.w32(sp::STATE, 1);
        fx.w8(sp::ACK_PENDING, 3);
        fx.w16(sp::MODE_LIST_COUNT, 5);
        run_state_machine(OnFault::Die, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::STATE), 0);
        assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 0);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 8);
    }

    /// State 2 with a client, a page count and nothing new: only `state = 2`.
    #[test]
    fn state2_quiet() {
        let fx = Fx::new();
        let d = connected(&fx);
        fx.guest_display.ack(fx.r32(sp::EXPECTED_ACK)).unwrap();
        assert_eq!(fx.r32(sp::STATE), 2);
        fx.w32(sp::RAISED_INTERRUPTS, 0);
        let ack = fx.r32(sp::EXPECTED_ACK);
        run_state_machine(OnFault::Return, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::STATE), 2);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 0);
        assert_eq!(fx.r32(sp::EXPECTED_ACK), ack);
        drop(d);
    }

    /// State 2 with a new mode list: update (12), state 1.
    #[test]
    fn state2_update() {
        let fx = Fx::new();
        let d = connected(&fx);
        fx.guest_display.ack(fx.r32(sp::EXPECTED_ACK)).unwrap();
        fx.w32(sp::RAISED_INTERRUPTS, 0);
        fx.guest_display
            .set_mode_list(OnFault::Return, Arc::from([mode(800, 600, 60.0)]))
            .unwrap();
        assert_eq!(fx.r32(sp::STATE), 1);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 12);
        assert_eq!(fx.r16(sp::mode_entry(0)), 800);
        drop(d);
    }

    /// State 2, client gone: count zeroed, disconnect (8), state 0.
    #[test]
    fn state2_detach_disconnects() {
        let fx = Fx::new();
        let d = connected(&fx);
        fx.guest_display.ack(fx.r32(sp::EXPECTED_ACK)).unwrap();
        fx.w32(sp::RAISED_INTERRUPTS, 0);
        fx.guest_display.set_client(OnFault::Return, None).unwrap();
        assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 0);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 8);
        assert_eq!(fx.r32(sp::STATE), 0);
        drop(d);
    }

    /// State 2 with a client but a guest-zeroed count and no new list: disconnect.
    #[test]
    fn state2_zero_count_disconnects() {
        let fx = Fx::new();
        let d = connected(&fx);
        fx.guest_display.ack(fx.r32(sp::EXPECTED_ACK)).unwrap();
        fx.w16(sp::MODE_LIST_COUNT, 0);
        fx.w32(sp::RAISED_INTERRUPTS, 0);
        run_state_machine(OnFault::Return, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::STATE), 0);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 8);
        drop(d);
    }

    /// A bogus state (7, 0xffffffff) with a client and the page's count: resend (12).
    #[test]
    fn bogus_state_resends() {
        for bogus in [7u32, 0xffff_ffff] {
            let fx = Fx::new();
            let d = connected(&fx);
            fx.w32(sp::STATE, bogus);
            fx.w32(sp::RAISED_INTERRUPTS, 0);
            let ack = fx.r32(sp::EXPECTED_ACK);
            let logs = crate::testkit::logs::capture();
            run_state_machine(OnFault::Die, &fx.guest_display).unwrap();
            assert_eq!(fx.r32(sp::STATE), 1);
            assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 12);
            assert_eq!(fx.r32(sp::EXPECTED_ACK), ack.wrapping_add(1));
            let warnings: Vec<_> = display_lines(&logs)
                .into_iter()
                .filter(|(level, _)| *level == log::Level::Warn)
                .collect();
            assert_eq!(
                warnings,
                vec![(
                    log::Level::Warn,
                    format!(
                        "guest display on port 0: state {bogus} is not a state of the protocol"
                    )
                )]
            );
            drop(logs);
            drop(d);
        }
    }

    /// A bogus state without a client disconnects.
    #[test]
    fn bogus_state_without_client_disconnects() {
        let fx = Fx::new();
        fx.w32(sp::STATE, 9);
        fx.w16(sp::MODE_LIST_COUNT, 2);
        run_state_machine(OnFault::Die, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::STATE), 0);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 8);
    }

    /// After a restore with the page's count, settings are re-tested; unchanged means no
    /// update, and `_didRestore` is cleared.
    #[test]
    fn did_restore_retests_settings() {
        let fx = Fx::new();
        let d = connected(&fx);
        fx.guest_display.ack(fx.r32(sp::EXPECTED_ACK)).unwrap();
        // A guest overwrite of the serial number shows up as a change.
        fx.w32(sp::SERIAL_NUM, 1);
        fx.w32(sp::RAISED_INTERRUPTS, 0);
        fx.guest_display.guest_display_mutex.lock().did_restore = true;
        run_state_machine(OnFault::Return, &fx.guest_display).unwrap();
        assert_eq!(fx.r32(sp::SERIAL_NUM), 0x1234);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 12);
        assert!(!fx.guest_display.guest_display_mutex.lock().did_restore);
        drop(d);
    }
}

mod ack {
    use super::*;

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.ack(0));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    /// Bit 0: a pending byte of 2 is "nothing pending".
    #[test]
    fn nothing_pending_throws() {
        let fx = Fx::new();
        fx.w8(sp::ACK_PENDING, 2);
        let e = fx.guest_display.ack(5).err().unwrap();
        assert_eq!(kind(&e), RaiseKind::AckNothingPending);
        assert_eq!(
            reason(&e),
            "guest display on port 0: ack 5 with none pending"
        );
    }

    #[test]
    fn wrong_ack_throws() {
        let fx = Fx::new();
        fx.w8(sp::ACK_PENDING, 3);
        fx.w32(sp::EXPECTED_ACK, 9);
        let e = fx.guest_display.ack(5).err().unwrap();
        assert_eq!(kind(&e), RaiseKind::AckUnexpected);
        assert_eq!(
            reason(&e),
            "guest display on port 0: ack 5, where 9 was expected"
        );
        assert_eq!(fx.r8(sp::ACK_PENDING), 3);
    }

    /// The checks run in no pool of their own: a raise leaves nothing pushed, and an accepted
    /// ack pushes none before the state machine.
    #[test]
    fn the_checks_open_no_pool() {
        use crate::rt::pool::test_native::{events, with_test_native};
        let fx = Fx::new();
        fx.w8(sp::ACK_PENDING, 1);
        fx.w32(sp::EXPECTED_ACK, 9);
        with_test_native(|| {
            assert!(fx.guest_display.ack(5).is_err());
            assert!(events().is_empty());
            fx.guest_display.ack(9).unwrap();
            let ev = events();
            assert!(ev.is_empty(), "{ev:?}");
        });
    }

    /// The right ack: lastAck, pending cleared, the machine runs (state 1 -> 2).
    #[test]
    fn accepted() {
        let fx = Fx::new();
        fx.w8(sp::ACK_PENDING, 1);
        fx.w32(sp::EXPECTED_ACK, 9);
        fx.w32(sp::STATE, 1);
        fx.w16(sp::MODE_LIST_COUNT, 1);
        fx.guest_display.ack(9).unwrap();
        assert_eq!(fx.r32(sp::LAST_ACK), 9);
        assert_eq!(fx.r8(sp::ACK_PENDING), 0);
        // No client: state 2 disconnects.
        assert_eq!(fx.r32(sp::STATE), 0);
    }
}

mod set_gamma_table {
    use super::*;

    fn g(count: u32, len: u64) -> GammaTableArgs {
        GammaTableArgs {
            virtual_offset: 0x100,
            mapped_length: len,
            entry_count: count,
            sum: 0xabc,
        }
    }

    #[test]
    fn records_a_valid_table() {
        let fx = Fx::new();
        set_gamma_table(&fx.guest_display, g(1024, 4096)).unwrap();
        assert_eq!(fx.r64(sp::GAMMA_TABLE_VIRTUAL_OFFSET), 0x100);
        assert_eq!(fx.r64(sp::GAMMA_TABLE_MAPPED_LENGTH), 4096);
        assert_eq!(fx.r32(sp::GAMMA_TABLE_ENTRY_COUNT), 1024);
        assert_eq!(fx.r32(sp::GAMMA_TABLE_SUM), 0xabc);
    }

    #[test]
    fn refuses_count_and_length() {
        let fx = Fx::new();
        let e = set_gamma_table(&fx.guest_display, g(1025, u64::MAX))
            .err()
            .unwrap();
        assert_eq!(kind(&e), RaiseKind::GammaTable);
        assert_eq!(
            reason(&e),
            "guest display on port 0: a gamma table of 1025 entries (at most 1024) in \
         0xffffffffffffffff mapped bytes"
        );
        let e = set_gamma_table(&fx.guest_display, g(3, 11)).err().unwrap();
        assert_eq!(kind(&e), RaiseKind::GammaTable);
        assert_eq!(fx.r32(sp::GAMMA_TABLE_ENTRY_COUNT), 0);
        set_gamma_table(&fx.guest_display, g(3, 12)).unwrap();
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| set_gamma_table(&fx.guest_display, g(1, 4)));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }
}

mod dirty_color_matrix {
    use super::*;

    #[test]
    fn clean_returns_none() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_DIRTY, 0);
        fx.w8(sp::COLOR_MATRIX_READ_INDEX, 9);
        assert_eq!(get_dirty_color_matrix(&fx.guest_display), None);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_READ_INDEX), 9);
    }

    /// Dirty: read = write, write = !write, dirty cleared, the read matrix returned.
    #[test]
    fn dirty_swaps() {
        let fx = Fx::new();
        // Init left the identity diagonal in matrix 0 and dirty set.
        let m = get_dirty_color_matrix(&fx.guest_display).unwrap();
        assert_eq!(m[0][0], 1.0);
        assert_eq!(m[1][1], 1.0);
        assert_eq!(m[0][1], 0.0);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_READ_INDEX), 0);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_WRITE_INDEX), 1);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_DIRTY), 0);
        // A write index of 7: read 7, write !7 = 0.
        fx.w8(sp::COLOR_MATRIX_WRITE_INDEX, 7);
        fx.w8(sp::COLOR_MATRIX_DIRTY, 1);
        fx.wf32(sp::color_matrix(7, 2, 3), 2.5);
        let m = get_dirty_color_matrix(&fx.guest_display).unwrap();
        assert_eq!(m[2][3], 2.5);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_READ_INDEX), 7);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_WRITE_INDEX), 0);
    }

    /// `ldrb w8,[x0,#0xa72]; strb w8,[x0,#0xa70]; cmp w8,#0; cset w8,eq; strb w8,[x0,#0xa72]`:
    /// the write index is loaded once, so a guest write of it after that load is overwritten
    /// with the negation of the value the read index got, and the page keeps write == !read.
    #[test]
    fn one_load_of_the_write_index() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_WRITE_INDEX, 0);
        fx.w8(sp::COLOR_MATRIX_DIRTY, 1);
        // The guest writes 5 to the write index after the host's load.
        BETWEEN_INDEX_STORES.with(|h| {
            *h.borrow_mut() = Some(Box::new(|page: GuestPage| {
                page.base()
                    .write_u8(sp::COLOR_MATRIX_WRITE_INDEX, 5, DeathKind::NullPage.site());
            }));
        });
        let m = get_dirty_color_matrix(&fx.guest_display);
        BETWEEN_INDEX_STORES.with(|h| *h.borrow_mut() = None);
        assert!(m.is_some());
        assert_eq!(fx.r8(sp::COLOR_MATRIX_READ_INDEX), 0);
        // !0, over the guest's 5; a second load would have stored !5 = 0 (read == write).
        assert_eq!(fx.r8(sp::COLOR_MATRIX_WRITE_INDEX), 1);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_DIRTY), 0);
    }

    /// Forced: the read index as it is, no swap, force cleared.
    #[test]
    fn forced_reads_without_swap() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_DIRTY, 0);
        fx.w8(sp::COLOR_MATRIX_FORCE_READ, 1);
        fx.w8(sp::COLOR_MATRIX_READ_INDEX, 1);
        fx.wf32(sp::color_matrix(1, 0, 0), 3.0);
        let m = get_dirty_color_matrix(&fx.guest_display).unwrap();
        assert_eq!(m[0][0], 3.0);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_FORCE_READ), 0);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_WRITE_INDEX), 0);
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| get_dirty_color_matrix(&fx.guest_display));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }
}

mod present_surface {
    use super::*;

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| {
            fx.guest_display
                .present_surface(1, GammaTableArgs::default())
        });
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    #[test]
    fn task_throw_propagates() {
        let fx = Fx::new();
        fx.p.script().push(
            "get_kernel_task",
            Step::raise("NSGenericException", "Invalid task"),
        );
        let hold = Hold::NONE;
        let r = fx
            .guest_display
            .present_surface(1, GammaTableArgs::default());
        hold.land(r.unwrap_err());
        assert_eq!(fx.r64(sp::PRESENT_COUNT), 0);
    }
}

mod present_mapped_surface {
    use super::*;

    fn present(
        hold: Hold<'_>,
        fx: &Fx,
        surface: Option<crate::testkit::surfaces::FakeSurface>,
    ) -> RaiseResult<()> {
        let task = fx.host.get_kernel_task(OnFault::Return).unwrap();
        let mut l = fx.guest_display.guest_display_mutex.lock();
        let keep = fx.guest_display.client_core();
        present_mapped_with_surface(
            hold,
            &fx.guest_display,
            &mut l,
            keep.as_ref(),
            task,
            surface,
            0x1_0000_0005,
            GammaTableArgs::default(),
        )
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| {
            fx.guest_display
                .present_mapped_surface(1, GammaTableArgs::default())
        });
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    #[test]
    fn task_throw_first() {
        let fx = Fx::new();
        fx.p.script().push(
            "get_kernel_task",
            Step::raise("NSGenericException", "Invalid task"),
        );
        let hold = Hold::NONE;
        let r = fx
            .guest_display
            .present_mapped_surface(5, GammaTableArgs::default());
        hold.land(r.unwrap_err());
        assert!(!fx.sels().contains(&"mapper_service"));
    }

    /// A BGRA surface: descriptor, texture over the surface; no client runs the completion;
    /// the page records the whole u64 mapping ID.
    #[test]
    fn presents_without_client() {
        let fx = Fx::new();
        let s = fx.p.fake_surfaces().make_simple(64, 32, FOURCC_BGRA, 4);
        fx.clear();
        let hold = Hold::NONE;
        present(hold, &fx, Some(s.clone_ref())).unwrap();
        let c =
            fx.p.call_log()
                .calls_to("texture2DDescriptorWithPixelFormat:width:height:mipmapped:");
        assert_eq!(
            c[0].args,
            vec![
                Arg::U64(MTL_PIXEL_FORMAT_BGRA8_UNORM),
                Arg::U64(64),
                Arg::U64(32),
                Arg::Bool(false)
            ]
        );
        assert_eq!(fx.count("newTextureWithDescriptor:iosurface:plane:"), 1);
        assert!(fx.p.call_log().contains_in_order(&[
            "IOSurfaceGetPixelFormat",
            "IOSurfaceGetWidth",
            "IOSurfaceGetHeight",
            "IOSurfaceGetBytesPerElement",
            "texture2DDescriptorWithPixelFormat:width:height:mipmapped:",
            "newTextureWithDescriptor:iosurface:plane:",
        ]));
        assert_eq!(fx.r64(sp::PRESENTED_MAPPING_ID), 0x1_0000_0005);
        assert_eq!(fx.r64(sp::PRESENT_COUNT), 1);
        assert_eq!(fx.r32(sp::RAISED_INTERRUPTS), 2);
        // The pool drained: only the test's reference is left.
        assert_eq!(Fk::retain_count(&s), 1);
    }

    use crate::testkit::objects::Fk;

    #[test]
    fn invalid_format_and_bytes_per_element() {
        let fx = Fx::new();
        let hold = Hold::NONE;
        let s = fx.p.fake_surfaces().make_simple(4, 4, 0x7979_7979, 4);
        let e = present(hold, &fx, Some(s)).err().unwrap();
        assert_eq!(kind(&e), RaiseKind::PixelFormat);
        hold.land(e);
        let s = fx.p.fake_surfaces().make_simple(4, 4, FOURCC_RGBA, 8);
        let e = present(hold, &fx, Some(s)).err().unwrap();
        assert_eq!(kind(&e), RaiseKind::BytesPerPixel);
        assert_eq!(
            reason(&e),
            "guest display on port 0: the surface has 8 bytes per pixel, not 4"
        );
        hold.land(e);
        assert_eq!(fx.r64(sp::PRESENT_COUNT), 0);
    }

    /// A nil texture asserts (613); Metal's own abort inside the call is a death there
    /// (testing.md 7.4).
    #[test]
    fn texture_nil_asserts_and_metal_abort() {
        let fx = Fx::new();
        {
            let hold = Hold::NONE;
            fx.p.script()
                .push("newTextureWithDescriptor:iosurface:plane:", Step::Nil);
            let s = fx.p.fake_surfaces().make_simple(4, 4, FOURCC_BGRA, 4);
            let r = catch_death(|| present(hold, &fx, Some(s)));
            assert_eq!(r.err(), Some(DeathKind::MappedTextureRefused.site()));
            let fx = Fx::new();
            fx.p.script()
                .push("newTextureWithDescriptor:iosurface:plane:", Step::crash());
            let s = fx.p.fake_surfaces().make_simple(20000, 4, FOURCC_BGRA, 4);
            let r = catch_death(|| present(hold, &fx, Some(s)));
            assert_eq!(r.err().unwrap().kind, crate::raise::death::DeathKind::Abort);
        }
    }

    /// A throw after the surface is taken carries it (and the texture) on with the raise, to
    /// the holder where the raise ends (on a host thread, the API entry's): they go at its end,
    /// not with the raise.
    #[test]
    fn a_throw_carries_the_surface_to_where_it_ends() {
        let fx = Fx::new();
        let outer = crate::rt::Held::new();
        let hold = Hold::NONE;
        let s = fx.p.fake_surfaces().make_simple(4, 4, FOURCC_BGRA, 4);
        fx.w32(sp::ENABLED_INTERRUPTS, 0);
        let task = fx.host.get_kernel_task(OnFault::Return).unwrap();
        let mut l = fx.guest_display.guest_display_mutex.lock();
        let keep = fx.guest_display.client_core();
        let r = present_mapped_with_surface(
            hold,
            &fx.guest_display,
            &mut l,
            keep.as_ref(),
            task,
            Some(s.clone_ref()),
            1,
            GammaTableArgs {
                virtual_offset: 0,
                mapped_length: 0,
                entry_count: 2000,
                sum: 0,
            },
        );
        drop(l);
        let raised = r.err().unwrap();
        assert_eq!(kind(&raised), RaiseKind::GammaTable);
        // Surface: the test's, the raise's, and the texture's it carries.
        assert_eq!(Fk::retain_count(&s), 3);
        Hold::of(&outer).land(raised);
        // The raise has ended: the outer holder holds them now.
        assert_eq!(Fk::retain_count(&s), 3);
        drop(outer);
        assert_eq!(Fk::retain_count(&s), 1);
    }

    /// With a presenting client: the guest colour space is handed over once.
    #[test]
    fn hands_the_color_space_dirty_flag() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let mut desc = queued(&q);
        desc.new_frame_event_handler = Some(Arc::new(|| {}));
        let d = fx.display(desc);
        fx.attach(&d);
        fx.guest_display
            .guest_display_mutex
            .lock()
            .guest_color_space_dirty = true;
        let hold = Hold::NONE;
        let s = fx.p.fake_surfaces().make_simple(4, 4, FOURCC_BGRA, 4);
        present(hold, &fx, Some(s)).unwrap();
        assert!(
            !fx.guest_display
                .guest_display_mutex
                .lock()
                .guest_color_space_dirty
        );
        assert_eq!(fx.r64(sp::PRESENT_COUNT), 1);
        drop(d);
    }
}

mod set_property {
    use super::*;

    fn words(n: u32) -> Vec<u8> {
        (0..n)
            .flat_map(|i| (i as f32 + 0.5).to_bits().to_le_bytes())
            .collect()
    }

    /// Property 0, 12 words: copied at the guest's write index, dirty set.
    #[test]
    fn color_matrix() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_WRITE_INDEX, 1);
        fx.w8(sp::COLOR_MATRIX_DIRTY, 0);
        let b = words(12);
        fx.guest_display
            .set_property(0, Records::new(&b, 12).unwrap())
            .unwrap();
        assert_eq!(fx.rf32(sp::color_matrix(1, 0, 0)), 0.5);
        assert_eq!(fx.rf32(sp::color_matrix(1, 2, 3)), 11.5);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_DIRTY), 1);
    }

    #[test]
    fn wrong_count_and_unknown_property_throw() {
        let fx = Fx::new();
        let b = words(11);
        let e = fx
            .guest_display
            .set_property(0, Records::new(&b, 11).unwrap())
            .err()
            .unwrap();
        assert_eq!(kind(&e), RaiseKind::PropertyWordCount);
        assert_eq!(
            reason(&e),
            "guest display on port 0: a colour matrix of 11 words, not 12"
        );
        let e = fx
            .guest_display
            .set_property(-1, Records::empty())
            .err()
            .unwrap();
        assert_eq!(kind(&e), RaiseKind::UnknownProperty);
        assert_eq!(
            reason(&e),
            "guest display on port 0: unknown property 4294967295"
        );
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.set_property(0, Records::empty()));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }
}

mod compositor_parameters {
    use super::*;

    /// Nothing is dispatched, with or without a client.
    #[test]
    fn dispatches_nothing() {
        let fx = Fx::new();
        fx.guest_display
            .compositor_parameters(1, DisplayCoord::default(), 2);
        let q = FakeHostQueue::new(QueueMode::Manual);
        let d = fx.display(queued(&q));
        fx.attach(&d);
        fx.guest_display
            .compositor_parameters(9, DisplayCoord { x: 3, y: 4 }, 2);
        assert_eq!(q.pending(), 0);
        drop(d);
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| {
            fx.guest_display
                .compositor_parameters(0, DisplayCoord::default(), 0)
        });
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }
}

mod guest_color_space {
    use super::*;

    /// Length 0: nil space, dirty; no task call.
    #[test]
    fn zero_length_clears() {
        let fx = Fx::new();
        fx.guest_display
            .set_guest_icc_profile_length(0, 0x40)
            .unwrap();
        assert_eq!(fx.r64(sp::GUEST_ICC_PROFILE_VIRTUAL_OFFSET), 0x40);
        assert_eq!(fx.r32(sp::GUEST_ICC_PROFILE_LENGTH), 0);
        assert!(fx.guest_display.guest_color_space().is_none());
        assert!(
            fx.guest_display
                .guest_display_mutex
                .lock()
                .guest_color_space_dirty
        );
        assert!(!fx.sels().contains(&"data_from_virtual_offset"));
    }
}

mod sleep_state {
    use super::*;

    /// The raw byte is stored; the block sleeps only on 1.
    #[test]
    fn raw_byte_and_block() {
        let fx = Fx::new();
        fx.guest_display.sleep_state(3);
        assert_eq!(fx.r8(sp::ASLEEP), 3);
        assert_eq!(fx.guest_display.guest_display_mutex.lock().asleep, 3);
        let q = FakeHostQueue::new(QueueMode::Inline);
        let d = fx.display(queued(&q));
        fx.attach(&d);
        let logs = crate::testkit::logs::capture();
        fx.guest_display.sleep_state(1);
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Info,
                "guest display on port 0: sleep, unhandled".to_owned()
            )]
        );
        fx.guest_display.sleep_state(3);
        assert_eq!(
            display_lines(&logs),
            vec![(
                log::Level::Info,
                "guest display on port 0: wake, unhandled".to_owned()
            )]
        );
        assert_eq!(fx.r8(sp::ASLEEP), 3);
        drop(d);
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.sleep_state(1));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }
}

mod cursor_glyph {
    use super::*;

    fn args(w: u16, h: u16, stride: u64, len: u64, sum: u32) -> CursorGlyphArgs {
        CursorGlyphArgs {
            virtual_offset: 0x2000,
            mapped_length: len,
            stride,
            width: w,
            height: h,
            hot_spot_x: 3,
            hot_spot_y: 4,
            sum,
        }
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.cursor_glyph(args(1, 1, 4, 4, 0)));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    /// The size checks in order, each before the task.
    #[test]
    fn refusals() {
        let fx = Fx::new();
        for a in [
            args(0, 1, 4, 4, 0),
            args(257, 1, 4096, 4096, 0),
            args(1, 0, 4, 4, 0),
            args(1, 257, 4, 4096, 0),
            args(1, 3, u64::MAX, u64::MAX, 0),
            args(256, 1, 0, 1023, 0),
            args(2, 2, 8, 15, 0),
        ] {
            let e = fx.guest_display.cursor_glyph(a).err().unwrap();
            assert_eq!(kind(&e), RaiseKind::CursorParameters, "{a:?}");
        }
        let e = fx
            .guest_display
            .cursor_glyph(args(2, 2, 8, 15, 0))
            .err()
            .unwrap();
        assert_eq!(
            reason(&e),
            "guest display on port 0: cursor of 2x2 (at most 256 a side), stride 0x8, in 0xf mapped \
         bytes is not valid"
        );
        assert_eq!(fx.count("get_kernel_task"), 0);
        assert_eq!(fx.r64(sp::CURSOR_GLYPH_MAPPED_LENGTH), 0);
    }

    /// A mapped length past the largest local task raises before any block is made or any task
    /// is asked, with a task or without
    #[test]
    fn glyph_mapped_length_past_any_task_faults() {
        use crate::device::tasks::MAX_LOCAL_TASK_LENGTH;
        let fx = Fx::new();
        let hold = Hold::NONE;
        let mut buffer = [0u32; 1];
        for len in [MAX_LOCAL_TASK_LENGTH + 1, 1 << 62, u64::MAX] {
            let e = fx
                .guest_display
                .copy_guest_glyph(hold, None, args(1, 1, 4, len, 0), &mut buffer)
                .err()
                .unwrap();
            assert_eq!(kind(&e), RaiseKind::CursorBuffer, "{len:#x}");
        }
        // Without a task a glyph within the bound is all zeros.
        assert!(
            fx.guest_display
                .copy_guest_glyph(hold, None, args(1, 1, 4, 4, 0), &mut buffer)
                .is_ok()
        );
        assert_eq!(buffer, [0]);
    }

    /// A host image for the maker tests: its retains and releases, and the pixels the copy writes
    /// into its buffer.
    struct MadeImage {
        log: Arc<Mutex<Vec<String>>>,
        pixels: Mutex<Vec<u32>>,
        bitmap: bool,
    }

    struct ImageRef(Arc<Mutex<Vec<String>>>, &'static str);

    impl Drop for ImageRef {
        fn drop(&mut self) {
            self.0.lock().push(format!("release {}", self.1));
        }
    }

    impl crate::display::host_calls::HostObject for MadeImage {
        fn retain(&self) -> crate::display::host_calls::HostRetained {
            self.log.lock().push("retain".into());
            Box::new(ImageRef(self.log.clone(), "block"))
        }
    }

    impl crate::display::host_calls::GlyphImage for MadeImage {
        fn write_bitmap_data(
            &self,
            hold: Hold<'_>,
            write: &mut dyn FnMut(Hold<'_>, Option<&mut [u32]>) -> RaiseResult<()>,
        ) -> RaiseResult<()> {
            self.log.lock().push("bitmapData".into());
            if !self.bitmap {
                return write(hold, None);
            }
            write(hold, Some(&mut self.pixels.lock()))
        }

        fn host_ptr(&self) -> crate::host::HostPtr {
            crate::host::HostPtr::from_addr(0x1234)
        }
    }

    /// A maker answering `Some` (with or without a bitmap) or nil, logging each call.
    fn maker(
        log: &Arc<Mutex<Vec<String>>>,
        answer: Option<bool>,
    ) -> crate::display::host_calls::GlyphImageMaker {
        let log = log.clone();
        Arc::new(move |w, h| {
            log.lock().push(format!("make {w}x{h}"));
            let bitmap = answer?;
            let image: crate::display::host_calls::GlyphImageRef = Arc::new(MadeImage {
                log: log.clone(),
                pixels: Mutex::new(vec![0; usize::from(w) * usize::from(h)]),
                bitmap,
            });
            let made: crate::display::host_calls::HostRetained =
                Box::new(ImageRef(log.clone(), "made"));
            Some((image, made))
        })
    }

    /// A nil image raises before the copy, and nothing is dispatched or recorded.
    #[test]
    fn v1_image_refusals() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut desc = queued(&q);
        desc.cursor_glyph_image_maker = Some(maker(&log, None));
        desc.cursor_glyph_handler = Some(Arc::new(|_: &CursorGlyph, _| ()));
        let d = fx.display(desc);
        fx.attach(&d);
        let hold = Hold::NONE;
        let before = fx.count("copy_from_virtual_offset");
        let e = fx
            .guest_display
            .cursor_glyph(args(1, 1, 4, 4, 0))
            .err()
            .unwrap();
        assert_eq!(reason(&e), "no cursor image made from the glyph");
        assert_eq!(fx.count("copy_from_virtual_offset"), before);
        assert_eq!(*log.lock(), ["make 1x1"]);
        assert_eq!(q.run_pending(), 0);
        drop(d);
        hold.land(e);
    }

    /// `assert(buffer)`: an image with no bitmap dies.
    #[test]
    fn v1_image_without_a_bitmap_dies() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let mut desc = queued(&q);
        desc.cursor_glyph_image_maker = Some(maker(&log, Some(false)));
        desc.cursor_glyph_handler = Some(Arc::new(|_: &CursorGlyph, _| ()));
        let d = fx.display(desc);
        fx.attach(&d);
        let r = catch_death(|| fx.guest_display.cursor_glyph(args(1, 1, 4, 4, 0)));
        assert_eq!(r.err(), Some(DeathKind::GlyphBufferNull.site()));
        drop(d);
    }
}

mod cursor_show {
    use super::*;

    /// The raw byte to the page and to the handler.
    #[test]
    fn raw_byte() {
        let fx = Fx::new();
        fx.guest_display.cursor_show(2);
        assert_eq!(fx.r8(sp::CURSOR_SHOW), 2);
        let q = FakeHostQueue::new(QueueMode::Inline);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        let mut desc = queued(&q);
        desc.cursor_show_handler = Some(Arc::new(move |b| s.lock().push(b)));
        let d = fx.display(desc);
        fx.attach(&d);
        fx.guest_display.cursor_show(0xff);
        assert_eq!(*seen.lock(), vec![0xff]);
        assert_eq!(fx.r8(sp::CURSOR_SHOW), 0xff);
        drop(d);
    }

    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.cursor_show(1));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    /// Blocks that re-read `_client` at execution (`cursorShow`, `cursorMove`, glyph, compositor
    /// parameters) call a nil block if the client detached in between: crash
    #[test]
    fn detached_before_the_block_dies() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let mut desc = queued(&q);
        desc.cursor_show_handler = Some(Arc::new(|_| {}));
        let d = fx.display(desc);
        fx.attach(&d);
        fx.guest_display.cursor_show(1);
        fx.guest_display.set_client(OnFault::Die, None).unwrap();
        let r = catch_death(|| q.run_pending());
        assert_eq!(r.err(), Some(DeathKind::ShowHandlerGone.site()));
        drop(d);
    }
}

mod doorbell {
    use super::*;

    /// A doorbell while paused (testing.md 7.4).
    #[test]
    fn paused_dies() {
        let fx = Fx::new();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.doorbell());
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    /// No client: no page access.
    #[test]
    fn no_client_nothing() {
        let fx = Fx::new();
        fx.w8(sp::GUEST_CURSOR_SHOW, 1);
        fx.w8(sp::COMPOSITOR_SUPPORTS_LIVE_RESIZE, 1);
        fx.guest_display.doorbell();
        assert_eq!(fx.r8(sp::CURSOR_SHOW), 0);
        assert!(!fx.guest_display.compositor_supports_live_resize());
    }

    /// Both blocks re-read the client.
    #[test]
    fn detached_before_the_blocks_die() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let mut desc = queued(&q);
        desc.cursor_move_handler = Some(Arc::new(|| {}));
        let d = fx.display(desc);
        fx.attach(&d);
        fx.guest_display.doorbell();
        fx.guest_display.set_client(OnFault::Die, None).unwrap();
        let r = catch_death(|| q.run_pending());
        assert_eq!(r.err(), Some(DeathKind::MoveHandlerGone.site()));
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let mut desc = queued(&q);
        desc.cursor_show_handler = Some(Arc::new(|_| {}));
        let d2 = fx.display(desc);
        fx.attach(&d2);
        fx.w8(sp::GUEST_CURSOR_SHOW, 1);
        fx.guest_display.doorbell();
        fx.guest_display.set_client(OnFault::Die, None).unwrap();
        let r = catch_death(|| q.run_pending());
        assert_eq!(r.err(), Some(DeathKind::ShowHandlerGone.site()));
        drop((d, d2));
    }
}

#[test]
fn cursor_position_unpacks_the_word() {
    let fx = Fx::new();
    fx.w32(sp::CURSOR_POSITION, 0x0002_0001);
    assert_eq!(
        fx.guest_display.cursor_position(),
        DisplayCoord { x: 1, y: 2 }
    );
    fx.guest_display.pause();
    let r = catch_death(|| fx.guest_display.cursor_position());
    assert_eq!(r.err(), Some(DeathKind::PageWhilePaused.site()));
}

mod update_mode_list {
    use super::*;

    fn load(fx: &Fx, modes: &[DisplayMode]) {
        fx.guest_display.guest_display_mutex.lock().mode_list = Some(Arc::from(modes));
    }

    #[test]
    fn asserts() {
        let fx = Fx::new();
        let r = catch_death(|| sync_mode_list(&fx.guest_display));
        assert_eq!(r.err(), Some(DeathKind::NoModeList.site()));
        load(&fx, &[mode(1, 1, 1.0)]);
        fx.guest_display.guest_display_mutex.lock().paused = true;
        let r = catch_death(|| sync_mode_list(&fx.guest_display));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
        fx.guest_display.guest_display_mutex.lock().paused = false;
        // A guest count of 129 (testing.md 7.4).
        fx.w16(sp::MODE_LIST_COUNT, 129);
        let r = catch_death(|| sync_mode_list(&fx.guest_display));
        assert_eq!(r.err(), Some(DeathKind::ModeCountTooLarge.site()));
    }

    /// Fields written as they differ; the flags byte keeps its other bits; the 16.16
    /// refresh floors (NaN to 0); unchanged returns NO; consumed.
    #[test]
    fn writes_and_consumes() {
        let fx = Fx::new();
        let e = sp::mode_entry(0);
        fx.w8(e + sp::MODE_FLAGS, 0xfc);
        let mut m = mode(1920, 1080, 59.94);
        m.supports_10bpc = true;
        load(&fx, &[m, mode(2, 3, f64::NAN)]);
        assert!(sync_mode_list(&fx.guest_display));
        assert!(fx.guest_display.mode_list().is_empty());
        assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 2);
        assert_eq!(fx.r16(e + sp::MODE_WIDTH), 1920);
        assert_eq!(fx.r16(e + sp::MODE_HEIGHT), 1080);
        assert_eq!(
            fx.r32(e + sp::MODE_REFRESH_RATE),
            (59.94f64 * 65536.0 + 0.5).floor() as u32
        );
        assert_eq!(fx.r8(e + sp::MODE_FLAGS), 0xfe);
        assert_eq!(fx.r32(sp::mode_entry(1) + sp::MODE_REFRESH_RATE), 0);
        load(&fx, &[m, mode(2, 3, f64::NAN)]);
        assert!(!sync_mode_list(&fx.guest_display));
        // EDR set, 10bpc cleared.
        m.supports_10bpc = false;
        m.supports_edr = true;
        load(&fx, &[m]);
        assert!(sync_mode_list(&fx.guest_display));
        assert_eq!(fx.r8(e + sp::MODE_FLAGS), 0xfd);
        assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 1);
        // Entries past the count are left.
        assert_eq!(fx.r16(sp::mode_entry(1) + sp::MODE_WIDTH), 2);
    }

    /// At most 128 entries.
    #[test]
    fn caps_at_128() {
        let fx = Fx::new();
        let modes: Vec<_> = (0..130u16).map(|i| mode(i, i, 60.0)).collect();
        load(&fx, &modes);
        assert!(sync_mode_list(&fx.guest_display));
        assert_eq!(fx.r16(sp::MODE_LIST_COUNT), 128);
        assert_eq!(fx.r16(sp::mode_entry(127)), 127);
        // Entry 128 would be the colour matrices.
        assert_eq!(fx.r32(sp::color_matrix(0, 0, 0)), 1.0f32.to_bits());
    }
}

mod update_settings {
    use super::*;

    #[test]
    fn asserts() {
        let fx = Fx::new();
        let r = catch_death(|| sync_settings(&fx.guest_display));
        assert_eq!(r.err(), Some(DeathKind::SettingsWithoutClient.site()));
        let d = fx.display(DisplayDescriptor::default());
        fx.attach(&d);
        fx.guest_display.guest_display_mutex.lock().paused = true;
        let r = catch_death(|| sync_settings(&fx.guest_display));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
        fx.guest_display.guest_display_mutex.lock().paused = false;
        drop(d);
    }

    /// Every field compared and stored; a second pass reports no change.
    #[test]
    fn writes_every_field() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Inline);
        let mut desc = queued(&q);
        desc.name = Some("Screen".into());
        desc.display_uuid = Some("ABCD".into());
        desc.size_in_millimeters = (300.4, -3.7);
        desc.color_space = crate::display::descriptor::DEFAULT_COLOR_SPACE;
        desc.origin = DisplayCoord { x: 5, y: 6 };
        desc.scale_factor = 2;
        desc.config_epoch = 0x1_0000_0001;
        desc.connection_type = -2;
        desc.min_nits = 0.5;
        desc.max_nits = 600.0;
        desc.max_sdr_nits = 400.0;
        desc.cursor_show_handler = Some(Arc::new(|_| {}));
        desc.cursor_move_handler = Some(Arc::new(|| {}));
        desc.cursor_glyph_handler = Some(Arc::new(|_: &CursorGlyph, _| {}));
        let d = fx.display(desc);
        fx.attach(&d);
        assert!(sync_settings(&fx.guest_display));
        assert_eq!(fx.r32(sp::SERIAL_NUM), 0x1234);
        assert_eq!(fx.rf32(sp::WIDTH_IN_MILLIMETERS), 300.4f64 as f32);
        // (u16)(i32)(-3.7 + 0.5) = 0xfffd; 300.9 -> 300.
        assert_eq!(fx.r16(sp::ROUNDED_WIDTH_IN_MILLIMETERS), 300);
        assert_eq!(fx.r16(sp::ROUNDED_HEIGHT_IN_MILLIMETERS), 0xfffd);
        assert_eq!(fx.rf32(sp::COLOR_SPACE), 0.64);
        assert_eq!(fx.rf32(sp::COLOR_SPACE + 28), 0.329_023);
        assert_eq!(fx.r16(sp::ORIGIN), 5);
        assert_eq!(fx.r16(sp::ORIGIN + 2), 6);
        assert_eq!(fx.r8(sp::SCALE_FACTOR), 2);
        assert_eq!(fx.rf32(sp::MIN_NITS), 0.5);
        assert_eq!(fx.rf32(sp::MAX_NITS), 600.0);
        assert_eq!(fx.rf32(sp::MAX_SDR_NITS), 400.0);
        assert_eq!(fx.r64(sp::CONFIG_EPOCH), 0x1_0000_0001);
        assert_eq!(fx.r32(sp::CONNECTION_TYPE), 0xffff_fffe);
        assert_eq!(fx.read_vec(sp::NAME, 7), b"Screen\0");
        assert_eq!(fx.read_vec(sp::DISPLAY_UUID, 5), b"ABCD\0");
        assert_eq!(fx.r16(sp::CURSOR_GLYPH_MAX_WIDTH), 256);
        assert_eq!(fx.r16(sp::CURSOR_GLYPH_MAX_HEIGHT), 256);
        assert_eq!(fx.r32(sp::CURSOR_HANDLERS), 3);
        assert!(!sync_settings(&fx.guest_display));
        drop(d);
    }

    /// A name of 14 bytes fails and leaves zeros; 13 fit. strncmp stops at a shared NUL; a
    /// 39-byte UUID fills the field with no NUL.
    #[test]
    fn names_and_uuids() {
        let fx = Fx::new();
        let mut desc = DisplayDescriptor::default();
        desc.name = Some("ABCDEFGHIJKLMN".into());
        desc.display_uuid = Some("0123456789012345678901234567890123456789".into());
        let d = fx.display(desc);
        fx.w8(sp::NAME, b'x');
        fx.attach(&d);
        sync_settings(&fx.guest_display);
        assert_eq!(fx.read_vec(sp::NAME, 14), vec![0; 14]);
        // 40 bytes do not fit 40: nothing, so the page gets zeros.
        assert_eq!(fx.r8(sp::DISPLAY_UUID), 0);
        fx.guest_display.set_client(OnFault::Die, None).unwrap();
        // The serial number is reserved while the first display lives.
        drop(d);
        let mut desc = DisplayDescriptor::default();
        desc.name = Some("ABCDEFGHIJKLM".into());
        desc.display_uuid = Some("012345678901234567890123456789012345678".into());
        let d2 = fx.display(desc);
        fx.attach(&d2);
        sync_settings(&fx.guest_display);
        assert_eq!(fx.read_vec(sp::NAME, 14), b"ABCDEFGHIJKLM\0");
        assert_eq!(
            fx.read_vec(sp::DISPLAY_UUID, 39),
            b"012345678901234567890123456789012345678"
        );
        // Bytes after a shared NUL are not compared.
        fx.w8(sp::NAME + 13, 0);
        assert!(!sync_settings(&fx.guest_display));
        drop(d2);
    }

    /// A NaN width differs every time.
    #[test]
    fn nan_always_differs() {
        let fx = Fx::new();
        let mut desc = DisplayDescriptor::default();
        desc.size_in_millimeters = (f64::NAN, 1.0);
        let d = fx.display(desc);
        fx.attach(&d);
        assert!(sync_settings(&fx.guest_display));
        assert!(sync_settings(&fx.guest_display));
        assert_eq!(fx.r16(sp::ROUNDED_WIDTH_IN_MILLIMETERS), 0);
        drop(d);
    }
}

mod restore {
    use super::*;

    #[test]
    fn asserts() {
        let fx = Fx::new();
        let r = catch_death(|| fx.guest_display.finish_restore(Hold::NONE));
        assert_eq!(r.err(), Some(DeathKind::FinishRestoreUnexpected.site()));
        fx.guest_display.will_restore();
        let r = catch_death(|| fx.guest_display.will_restore());
        assert_eq!(r.err(), Some(DeathKind::WillRestoreTwice.site()));
        let fx = Fx::new();
        fx.guest_display.will_restore();
        fx.guest_display.pause();
        let r = catch_death(|| fx.guest_display.finish_restore(Hold::NONE));
        assert_eq!(r.err(), Some(DeathKind::CommandWhilePaused.site()));
    }

    /// `asleep == 3`: bit 0 set skips the present; `== 1` fails: no re-sleep.
    #[test]
    fn asleep_3_does_neither() {
        let fx = Fx::new();
        fx.guest_display.will_restore();
        fx.w64(sp::PRESENT_COUNT, 1);
        fx.w8(sp::ASLEEP, 3);
        fx.guest_display.finish_restore(Hold::NONE);
        assert!(!fx.sels().contains(&"mapper_service"));
        assert_eq!(fx.guest_display.guest_display_mutex.lock().asleep, 0);
    }
}

mod color_matrix_late_read {
    use super::*;

    /// `ldp s0, s1, [m]` and `ldp s2, s3, [m, #8]` read m[0][0..1] and m[0][2..3] of the page
    /// each as one 64-bit access, the low half the lower element: a guest's 64-bit store to row
    /// 0 is seen whole or not at all, never one element of each.
    #[test]
    fn row_zero_pairs_are_single_loads() {
        use std::sync::atomic::AtomicBool;
        let fx = Fx::new();
        let page = fx.guest_display.page_unlocked();
        let m = PageColorMatrix { page, index: 2 };
        let bits = |lo: f32, hi: f32| (u64::from(hi.to_bits()) << 32) | u64::from(lo.to_bits());
        fx.w64(sp::color_matrix(2, 0, 0), bits(1.0, 2.0));
        fx.w64(sp::color_matrix(2, 0, 2), bits(3.0, 4.0));
        assert_eq!(m.pair(0, 0), (1.0, 2.0));
        assert_eq!(m.pair(0, 2), (3.0, 4.0));
        assert_eq!((m.at(0, 1), m.at(0, 3)), (2.0, 4.0));
        // The guest rewrites the pair with one 64-bit store, over and over, until the reads
        // are done (or have failed: the guard stops the writer either way).
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        let stop = AtomicBool::new(false);
        page.w64(sp::color_matrix(2, 0, 0), bits(1.5, 1.5));
        let torn = std::thread::scope(|sc| {
            sc.spawn(|| {
                let mut v = 0u32;
                while !stop.load(Ordering::SeqCst) {
                    v = v.wrapping_add(1);
                    let f = f32::from_bits(0x3f80_0000 | (v & 0x7f_ffff));
                    page.w64(sp::color_matrix(2, 0, 0), bits(f, f));
                }
            });
            let _stop = Stop(&stop);
            (0..200_000)
                .map(|_| m.pair(0, 0))
                .find(|(a, b)| a.to_bits() != b.to_bits())
        });
        assert_eq!(torn, None, "a torn pair");
    }
}

/// On 4 KiB guest pages the page is 4 KiB: matrices 0-30 lie inside it, matrix 31's first two rows end at its
/// end, and every access past it ends the process there, before it is made. Each test sets only
/// bytes inside the page.
#[cfg(apv_guest_4k)]
mod color_matrix_outside_page {
    use super::*;

    const SITE: DeathKind = DeathKind::ColorMatrixOutsidePage;

    /// Init at write index 31: the stores to [0][0] (0xfe0) and [1][1] (0xff4) land, the one to
    /// [2][2] (0x1008) dies, and dirty is never set.
    #[test]
    fn init_dies_at_the_store_past_the_page() {
        let td = guest_display_device();
        let p = td.platform.clone();
        let host = host_of(&td);
        let pa = u64::from(PAGE) * crate::guest::records::APV_PAGE_SIZE;
        assert!(
            td.host
                .ram()
                .write(pa + sp::COLOR_MATRIX_WRITE_INDEX, &[31])
        );
        td.platform.fake_vm().sync();
        let r = catch_death(|| GuestDisplay::new(p.clone(), Arc::downgrade(&host), 0, PAGE));
        assert_eq!(r.err(), Some(SITE.site()));
        let u32_at =
            |off: u64| u32::from_le_bytes(guest_bytes(&td, pa + off, 4).try_into().unwrap());
        assert_eq!(u32_at(sp::color_matrix(31, 0, 0)), 1.0f32.to_bits());
        assert_eq!(u32_at(sp::color_matrix(31, 1, 1)), 1.0f32.to_bits());
        assert_eq!(guest_bytes(&td, pa + sp::COLOR_MATRIX_DIRTY, 1), vec![0]);
    }

    /// Init at write index 32: its first store is past the page, so nothing is stored.
    #[test]
    fn init_past_the_page_stores_nothing() {
        let td = guest_display_device();
        let p = td.platform.clone();
        let host = host_of(&td);
        let pa = u64::from(PAGE) * crate::guest::records::APV_PAGE_SIZE;
        assert!(
            td.host
                .ram()
                .write(pa + sp::COLOR_MATRIX_WRITE_INDEX, &[32])
        );
        td.platform.fake_vm().sync();
        let r = catch_death(|| GuestDisplay::new(p.clone(), Arc::downgrade(&host), 0, PAGE));
        assert_eq!(r.err(), Some(SITE.site()));
        assert_eq!(guest_bytes(&td, pa + sp::COLOR_MATRIX_DIRTY, 1), vec![0]);
    }

    /// Init at write index 30, the last whole matrix: the diagonal as at any other index.
    #[test]
    fn init_at_the_last_whole_matrix_writes_it() {
        let td = guest_display_device();
        let p = td.platform.clone();
        let host = host_of(&td);
        let pa = u64::from(PAGE) * crate::guest::records::APV_PAGE_SIZE;
        assert!(
            td.host
                .ram()
                .write(pa + sp::COLOR_MATRIX_WRITE_INDEX, &[30])
        );
        td.platform.fake_vm().sync();
        let _guest_display = GuestDisplay::new(p.clone(), Arc::downgrade(&host), 0, PAGE).unwrap();
        let u32_at =
            |off: u64| u32::from_le_bytes(guest_bytes(&td, pa + off, 4).try_into().unwrap());
        for i in 0..3u8 {
            assert_eq!(u32_at(sp::color_matrix(30, i, i)), 1.0f32.to_bits());
        }
        assert_eq!(guest_bytes(&td, pa + sp::COLOR_MATRIX_DIRTY, 1), vec![1]);
    }

    /// Property 0 at write index 31: the matrix leaves the page, so it dies before any of its
    /// 48 bytes is written, and dirty stays as it was.
    #[test]
    fn property_write_dies_before_writing() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_WRITE_INDEX, 31);
        fx.w8(sp::COLOR_MATRIX_DIRTY, 0);
        let b: Vec<u8> = (0..12u32)
            .flat_map(|i| (i as f32 + 0.5).to_bits().to_le_bytes())
            .collect();
        let r = catch_death(|| {
            fx.guest_display
                .set_property(0, Records::new(&b, 12).unwrap())
        });
        assert_eq!(r.err(), Some(SITE.site()));
        assert_eq!(fx.read_vec(sp::color_matrix(31, 0, 0), 32), vec![0; 32]);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_DIRTY), 0);
    }

    /// Property 0 at write index 30 lands whole.
    #[test]
    fn property_write_at_the_last_whole_matrix_lands() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_WRITE_INDEX, 30);
        let b: Vec<u8> = (0..12u32)
            .flat_map(|i| (i as f32 + 0.5).to_bits().to_le_bytes())
            .collect();
        fx.guest_display
            .set_property(0, Records::new(&b, 12).unwrap())
            .unwrap();
        assert_eq!(fx.rf32(sp::color_matrix(30, 0, 0)), 0.5);
        assert_eq!(fx.rf32(sp::color_matrix(30, 2, 3)), 11.5);
    }

    /// A dirty matrix at write index 31: the indices flip as at any index, rows 0 and 1 read,
    /// and the first read of row 2 dies.
    #[test]
    fn read_dies_at_the_first_access_past_the_page() {
        let fx = Fx::new();
        fx.w8(sp::COLOR_MATRIX_WRITE_INDEX, 31);
        fx.w8(sp::COLOR_MATRIX_DIRTY, 1);
        let l = fx.guest_display.guest_display_mutex.lock();
        let m = fx.guest_display.take_dirty_color_matrix(&l).unwrap();
        assert_eq!(fx.r8(sp::COLOR_MATRIX_READ_INDEX), 31);
        assert_eq!(fx.r8(sp::COLOR_MATRIX_WRITE_INDEX), 0);
        fx.wf32(sp::color_matrix(31, 1, 3), 7.0);
        assert_eq!(m.at(1, 3), 7.0);
        assert_eq!(m.pair(1, 2), (0.0, 7.0));
        assert_eq!(catch_death(|| m.at(2, 0)).err(), Some(SITE.site()));
        assert_eq!(catch_death(|| m.pair(2, 0)).err(), Some(SITE.site()));
        drop(l);
    }
}

/// The blocks reach the client unretained and call the host's handler: a handler that drops the
/// display's last reference runs the whole display's end inside the call, so by the time the
/// release returns the display's Metal objects (the buffers, the current frame over the guest's
/// IOSurface, the pipeline cache) are released, before the device and the property buffer.
mod handler_releases_the_display {
    use super::*;
    use crate::testkit::objects::{Fk, kind};

    /// A display on `q` whose handler (`set` installs `release`) drops the host's last
    /// reference, then records the references to its position buffer, released first, and to its
    /// property buffer, released last (the test's own and, if not released, the display's).
    fn releasing(
        fx: &Fx,
        q: &Arc<FakeHostQueue>,
        set: impl FnOnce(&mut DisplayDescriptor, Arc<dyn Fn() + Send + Sync>),
    ) -> Arc<Mutex<Vec<String>>> {
        type Held = Arc<Mutex<Option<Fk<kind::Buffer>>>>;
        let slot: Arc<Mutex<Option<Arc<Display<Fp>>>>> = Arc::default();
        let (pos, props): (Held, Held) = Default::default();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (s2, pos2, props2, seen2) = (slot.clone(), pos.clone(), props.clone(), seen.clone());
        let release: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            drop(s2.lock().take());
            let refs = |held: &Held| held.lock().as_ref().map_or(0, Fk::retain_count);
            seen2
                .lock()
                .push(format!("pos={} props={}", refs(&pos2), refs(&props2)));
        });
        let mut desc = queued(q);
        set(&mut desc, release);
        let d = fx.display(desc);
        {
            let g = d.display_mutex.lock();
            *pos.lock() = g.pos_buf.as_ref().map(Handle::clone_ref);
            *props.lock() = g.frag_shader_property_buf.as_ref().map(Handle::clone_ref);
        }
        fx.attach(&d);
        *slot.lock() = Some(d);
        seen
    }

    #[test]
    fn cursor_show_handler() {
        let fx = Fx::new();
        let q = FakeHostQueue::new(QueueMode::Manual);
        let seen = releasing(&fx, &q, |d, release| {
            d.cursor_show_handler = Some(Arc::new(move |_| release()));
        });
        fx.guest_display.cursor_show(1);
        q.run_pending();
        assert_eq!(*seen.lock(), vec!["pos=1 props=1"]);
    }
}
