//! The IOSurface host device: the guest's IOSurface mapper, the second device behind
//! `apple-gfx-mmio`.
//!
//! The guest shares surfaces with the host through structures in its own memory that it names
//! by MMIO ([`regs`]): a ring of 16-byte commands, a one-page table of descriptor pages, and the
//! descriptor pages, whose descriptors are in the first form (0x200 bytes) or the second (0x400)
//! as the enable register chooses ([`descriptor`]). A write of the ring's write counter is the
//! doorbell ([`ring`]): the device's queue runs every command from the read counter up to the
//! write counter, maps, unmaps or tests the surface each names ([`table`], with the page walk of
//! [`walk`]), stores the command's status back into the ring and raises interrupt 0. [`device`]
//! is the host's object: its creation, the process's last device, the surface lookup the
//! graphics device makes, and its teardown.
//!
//! # Ordering contracts
//!
//! - **The ring runs one command at a time, in ring order.** A command's ring read, its work and
//!   its status store are made under the queue state's lock; the read counter's increment and
//!   interrupt 0 follow, outside every lock, once per command (`ring::tests::runs_the_ring`).
//! - **Configuration writes wait for the ring.** The ring base, ring size, descriptor base and
//!   enable writes run on the device's queue through a `sync` from the writing thread, so each
//!   waits for a running batch; the write and read counters are stored on the caller's thread
//!   (`regs::tests::mmio::write_counter_wakes`). Register reads are unsynchronised seq_cst loads.
//! - **The teardown never waits for a command.** The host's last release unmaps the ring, the
//!   table and the descriptor pages at once on the releasing thread, then waits only for the
//!   queue; a command still running faults at its next access through what was unmapped
//!   (`device::tests::teardown_during_a_command_faults_its_status_store`).
//! - **Unmap is Busy while anything else holds the surface**: an unmap needs the IOSurface's
//!   retain count to be 1 (`table::tests::unmap::busy_then_released`).
//! - **The last device created wins**, process-wide, while it lives
//!   (`device::tests::creation::defaults_and_the_last_device_wins`).

pub mod descriptor;
mod device;
mod regs;
mod ring;
mod table;
#[cfg(test)]
mod testing;
mod walk;

use std::fmt;

use crate::raise::death::DeathSite;

pub use descriptor::{
    IoSurfaceDeviceBaseDescriptor, IoSurfaceDeviceBaseDescriptor2, IoSurfaceDeviceDescriptor,
    IoSurfaceDeviceDescriptor2, IoSurfaceDevicePlaneDescriptor, IoSurfaceDevicePlaneDescriptor2,
    element_dimension, upgrade_descriptor,
};
pub use device::{HostDeviceCore, HostDeviceHandle, IoSurfaceHostDevice, IoSurfaceRefusal};

/// The IOSurface device's deaths. Each access death is the device's fault at one load or store
/// through a pointer the guest made NULL or the teardown unmapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// The newly mapped ring's first word, read for its debug line.
    RingPeek,
    /// A command read from the ring.
    RingRead,
    /// A command's status stored back into the ring.
    StatusStore,
    /// A descriptor page's entry read from the descriptor table.
    TableRead,
    /// A descriptor copied out of its page.
    DescriptorRead,
    /// A descriptor's mapped bit set after its surface is made.
    MappedBitSet,
    /// A descriptor's mapped bit cleared by an unmap.
    MappedBitClear,
    /// A page-table entry read by the page walk.
    PageTableRead,
    /// The test pattern written into a surface.
    PatternWrite,
    /// A descriptor page index past the device's page slots (a guest page above 16 KiB, which
    /// no build has).
    PageIndex,
    /// Interrupt 0 raised with no interrupt block.
    NoInterrupt,
    /// Interrupt 0 raised through the block the teardown released, once the host freed it.
    FreedInterrupt,
    /// The descriptor's `mapMemory` or `unmapMemory`, which QEMU declares and the SDK's class does
    /// not answer: the unrecognized message is uncaught.
    UnrecognizedSelector,
}

impl DeathKind {
    /// The dotted id the death prints.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::RingPeek => "iosurface.ring.peek",
            DeathKind::RingRead => "iosurface.ring.read",
            DeathKind::StatusStore => "iosurface.ring.status_store",
            DeathKind::TableRead => "iosurface.table.read",
            DeathKind::DescriptorRead => "iosurface.descriptor.read",
            DeathKind::MappedBitSet => "iosurface.descriptor.mapped_set",
            DeathKind::MappedBitClear => "iosurface.descriptor.mapped_clear",
            DeathKind::PageTableRead => "iosurface.walk.read",
            DeathKind::PatternWrite => "iosurface.test.write",
            DeathKind::PageIndex => "iosurface.table.page_index",
            DeathKind::NoInterrupt => "iosurface.interrupt.nil",
            DeathKind::FreedInterrupt => "iosurface.interrupt.freed",
            DeathKind::UnrecognizedSelector => "iosurface.descriptor.unrecognized_selector",
        }
    }

    /// The site `testkit::capture` records and the stderr line names: the kind of fault, and the id.
    pub const fn site(self) -> DeathSite {
        use crate::raise::death::DeathKind as Fault;
        let kind = match self {
            DeathKind::PageIndex => Fault::OutOfBounds,
            DeathKind::NoInterrupt => Fault::NilBlock,
            DeathKind::UnrecognizedSelector => Fault::UncaughtObjc,
            _ => Fault::NullDeref,
        };
        DeathSite {
            kind,
            file: self.id(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}
