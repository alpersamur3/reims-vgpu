//! Draws, ICB executes and tile dispatches.
//!
//! - Every handler reads its whole record first, so a short record raises before anything
//!   else (`tests::draw_primitives64`).
//! - The buffer lookups come first, in the record's order, then the pipeline check, then the
//!   buffers found (the indirect buffer before the index or control-point one), then the
//!   primitive type; the lookups precede the pipeline check (`tests::draw_indexed_primitives64`).
//! - Only a primitive type of 10 or more fails, patches have no type check, and the index type is
//!   never checked (`tests::primitive_types_under_ten_pass`).
//! - A patch index buffer reference of 0 is no buffer, and is not looked up
//!   (`tests::draw_patches64`).
//! - Tile dispatches check the pipeline only; 0xa2 reads the whole record of 0xa3 and ignores
//!   its render target index, which 0xa3 passes as a `u32` (D41)
//!   (`tests::dispatch_threads_per_tile_in_region_with_index`).

use crate::deserializer::records::{
    CmdDispatchThreadsPerTile, CmdDispatchThreadsPerTileInRegionWithIndex,
    CmdDrawIndexedInstancedBasePrimitives16, CmdDrawIndexedInstancedBasePrimitives16_2,
    CmdDrawIndexedInstancedBasePrimitives64, CmdDrawIndexedInstancedBasePrimitives64_2,
    CmdDrawIndexedInstancedPrimitives16, CmdDrawIndexedInstancedPrimitives64,
    CmdDrawIndexedPatches16, CmdDrawIndexedPatches64, CmdDrawIndexedPatchesIndirect,
    CmdDrawIndexedPrimitives16, CmdDrawIndexedPrimitives64, CmdDrawIndexedPrimitivesIndirect,
    CmdDrawInstancedBasePrimitives16, CmdDrawInstancedBasePrimitives64,
    CmdDrawInstancedPrimitives16, CmdDrawInstancedPrimitives64, CmdDrawPatches16, CmdDrawPatches64,
    CmdDrawPatchesIndirect, CmdDrawPrimitives16, CmdDrawPrimitives64, CmdDrawPrimitivesIndirect,
    CmdExecuteCommandsInBuffer, CmdExecuteCommandsInBufferRanged,
};
use crate::deserializer::render::{RaiseKind, RenderDecoder};
use crate::memory::cursor::Cursor;
use crate::platform::Platform;
use crate::platform::icb::IndirectCommandBuffer as _;
use crate::platform::render::RenderCmd;
use crate::platform::types::{MtlRegion, MtlSize, NsRange};
use crate::raise::{OnFault, Raise, RaiseResult};
use crate::rt::Hold;

/// Only `type >= 10` fails, so primitive types 5-9 pass validation; the index type is never
/// validated.
pub(super) fn check_primitive_type(primitive_type: u64) -> RaiseResult<()> {
    if primitive_type >= 10 {
        crate::raise!(
            RaiseKind::PrimitiveType,
            format_args!("primitive type {primitive_type} is not valid")
        );
    }
    Ok(())
}

impl<P: Platform> RenderDecoder<P> {
    /// A draw needs a pipeline state.
    fn check_draw_pipeline(&self) -> RaiseResult<()> {
        // Ours: or a state the platform refused, whose commands are skipped (`send`).
        if self.pipeline.is_none() && !self.pipeline_refused {
            crate::raise!(
                RaiseKind::DrawWithoutPipeline,
                "draw without a pipeline state"
            );
        }
        Ok(())
    }

    /// A tile dispatch needs a pipeline state.
    fn check_dispatch_pipeline(&self) -> RaiseResult<()> {
        // Ours: or a state the platform refused, whose commands are skipped (`send`).
        if self.pipeline.is_none() && !self.pipeline_refused {
            crate::raise!(
                RaiseKind::DispatchWithoutPipeline,
                "tile dispatch without a pipeline state"
            );
        }
        Ok(())
    }

    /// The patch index buffer of a patch draw: none for reference 0, which is not looked up.
    fn patch_index_buffer(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<Option<P::Buffer>> {
        if reference != 0 {
            self.buffer(hold, reference)
        } else {
            Ok(None)
        }
    }

    /// The indexed draws' checks: the index buffer's lookup, the pipeline, the index buffer
    /// found, then the primitive type. The lookup precedes the pipeline check, and the index type
    /// is never checked.
    fn indexed_draw_prefix(
        &self,
        hold: Hold<'_>,
        index_buffer_reference: u32,
        primitive_type: u64,
    ) -> RaiseResult<P::Buffer> {
        let index_buffer = self.buffer(hold, index_buffer_reference)?;
        self.check_draw_pipeline()?;
        let Some(index_buffer) = index_buffer else {
            crate::raise!(
                RaiseKind::NoIndexBuffer,
                "indexed draw without an index buffer"
            );
        };
        check_primitive_type(primitive_type)?;
        Ok(index_buffer)
    }

    /// 0x0.
    pub(super) fn draw_primitives64(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawPrimitives64 = c.read_or_raise(OnFault::Return)?;
        self.check_draw_pipeline()?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: command.vertex_start,
            vertex_count: command.vertex_count,
        })
    }

    /// 0x1.
    pub(super) fn draw_primitives16(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawPrimitives16 = c.read_or_raise(OnFault::Return)?;
        self.check_draw_pipeline()?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitives {
            primitive_type,
            vertex_start: u64::from(command.vertex_start),
            vertex_count: u64::from(command.vertex_count),
        })
    }

    /// 0x2.
    pub(super) fn draw_instanced_primitives64(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawInstancedPrimitives64 = c.read_or_raise(OnFault::Return)?;
        self.check_draw_pipeline()?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitivesInstances {
            primitive_type,
            vertex_start: command.vertex_start,
            vertex_count: command.vertex_count,
            instance_count: command.instance_count,
        })
    }

    /// 0x3.
    pub(super) fn draw_instanced_primitives16(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawInstancedPrimitives16 = c.read_or_raise(OnFault::Return)?;
        self.check_draw_pipeline()?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitivesInstances {
            primitive_type,
            vertex_start: u64::from(command.vertex_start),
            vertex_count: u64::from(command.vertex_count),
            instance_count: u64::from(command.instance_count),
        })
    }

    /// 0x4.
    pub(super) fn draw_instanced_base_primitives64(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawInstancedBasePrimitives64 = c.read_or_raise(OnFault::Return)?;
        self.check_draw_pipeline()?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitivesBase {
            primitive_type,
            vertex_start: command.vertex_start,
            vertex_count: command.vertex_count,
            instance_count: command.instance_count,
            base_instance: command.base_instance,
        })
    }

    /// 0x5.
    pub(super) fn draw_instanced_base_primitives16(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawInstancedBasePrimitives16 = c.read_or_raise(OnFault::Return)?;
        self.check_draw_pipeline()?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitivesBase {
            primitive_type,
            vertex_start: u64::from(command.vertex_start),
            vertex_count: u64::from(command.vertex_count),
            instance_count: u64::from(command.instance_count),
            base_instance: u64::from(command.base_instance),
        })
    }

    /// 0x6.
    pub(super) fn draw_indexed_primitives64(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedPrimitives64 = c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count: command.index_count,
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: command.index_buffer_offset,
        })
    }

    /// 0x7.
    pub(super) fn draw_indexed_primitives16(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedPrimitives16 = c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitives {
            primitive_type,
            index_count: u64::from(command.index_count),
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: u64::from(command.index_buffer_offset),
        })
    }

    /// 0x8.
    pub(super) fn draw_indexed_instanced_primitives64(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedInstancedPrimitives64 = c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitivesInstances {
            primitive_type,
            index_count: command.index_count,
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: command.index_buffer_offset,
            instance_count: command.instance_count,
        })
    }

    /// 0x9.
    pub(super) fn draw_indexed_instanced_primitives16(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedInstancedPrimitives16 = c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitivesInstances {
            primitive_type,
            index_count: u64::from(command.index_count),
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: u64::from(command.index_buffer_offset),
            instance_count: u64::from(command.instance_count),
        })
    }

    /// 0xa: the `u64` base vertex goes as `NSInteger`.
    pub(super) fn draw_indexed_instanced_base_primitives64(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedInstancedBasePrimitives64 = c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type,
            index_count: command.index_count,
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: command.index_buffer_offset,
            instance_count: command.instance_count,
            base_vertex: command.base_vertex.cast_signed(),
            base_instance: command.base_instance,
        })
    }

    /// 0xb.
    pub(super) fn draw_indexed_instanced_base_primitives16(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedInstancedBasePrimitives16 = c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        // 0x0b's `u16` base vertex is zero-extended, where 0x1d sign-extends its `i16` one.
        let base_vertex = crate::num::zero_extend_16(command.base_vertex);
        self.send(RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type,
            index_count: u64::from(command.index_count),
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: u64::from(command.index_buffer_offset),
            instance_count: u64::from(command.instance_count),
            base_vertex,
            base_instance: u64::from(command.base_instance),
        })
    }

    /// 0x1c.
    pub(super) fn draw_indexed_instanced_base_primitives64_2(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedInstancedBasePrimitives64_2 =
            c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type,
            index_count: command.index_count,
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: command.index_buffer_offset,
            instance_count: command.instance_count,
            base_vertex: command.base_vertex,
            base_instance: command.base_instance,
        })
    }

    /// 0x1d.
    pub(super) fn draw_indexed_instanced_base_primitives16_2(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedInstancedBasePrimitives16_2 =
            c.read_or_raise(OnFault::Return)?;
        let primitive_type = u64::from(command.primitive_type);
        let index_buffer =
            self.indexed_draw_prefix(hold, command.index_buffer_reference, primitive_type)?;
        // 0x1d's `i16` base vertex is sign-extended, where 0x0b zero-extends its `u16` one.
        let base_vertex = crate::num::sign_extend_16(command.base_vertex);
        self.send(RenderCmd::DrawIndexedPrimitivesBase {
            primitive_type,
            index_count: u64::from(command.index_count),
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: u64::from(command.index_buffer_offset),
            instance_count: u64::from(command.instance_count),
            base_vertex,
            base_instance: u64::from(command.base_instance),
        })
    }

    /// 0xc. Patches have no type check.
    pub(super) fn draw_patches64(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawPatches64 = c.read_or_raise(OnFault::Return)?;
        let patch_index_buffer =
            self.patch_index_buffer(hold, command.patch_index_buffer_reference)?;
        self.check_draw_pipeline()?;
        self.send(RenderCmd::DrawPatches {
            number_of_patch_control_points: u64::from(command.number_of_patch_control_points),
            patch_start: command.patch_start,
            patch_count: command.patch_count,
            patch_index_buffer: patch_index_buffer.as_ref(),
            patch_index_buffer_offset: command.patch_index_buffer_offset,
            instance_count: command.instance_count,
            base_instance: command.base_instance,
        })
    }

    /// 0xd.
    pub(super) fn draw_patches16(&mut self, hold: Hold<'_>, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDrawPatches16 = c.read_or_raise(OnFault::Return)?;
        let patch_index_buffer =
            self.patch_index_buffer(hold, command.patch_index_buffer_reference)?;
        self.check_draw_pipeline()?;
        self.send(RenderCmd::DrawPatches {
            number_of_patch_control_points: u64::from(command.number_of_patch_control_points),
            patch_start: u64::from(command.patch_start),
            patch_count: u64::from(command.patch_count),
            patch_index_buffer: patch_index_buffer.as_ref(),
            patch_index_buffer_offset: u64::from(command.patch_index_buffer_offset),
            instance_count: u64::from(command.instance_count),
            base_instance: u64::from(command.base_instance),
        })
    }

    /// The control-point index buffer an indexed patch draw found.
    fn control_point_index_buffer(buffer: Option<P::Buffer>) -> RaiseResult<P::Buffer> {
        match buffer {
            Some(buffer) => Ok(buffer),
            None => Err(Raise::new(RaiseKind::NoControlPointIndexBuffer)
                .with_detail("indexed patch draw without a control point index buffer")),
        }
    }

    /// The indirect buffer an indirect draw found.
    fn indirect_buffer(buffer: Option<P::Buffer>) -> RaiseResult<P::Buffer> {
        match buffer {
            Some(buffer) => Ok(buffer),
            None => Err(Raise::new(RaiseKind::NoIndirectBuffer)
                .with_detail("indirect draw without an indirect buffer")),
        }
    }

    /// 0xe.
    pub(super) fn draw_indexed_patches64(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedPatches64 = c.read_or_raise(OnFault::Return)?;
        let patch_index_buffer =
            self.patch_index_buffer(hold, command.patch_index_buffer_reference)?;
        let control_point_index_buffer =
            self.buffer(hold, command.control_point_index_buffer_reference)?;
        self.check_draw_pipeline()?;
        let control_point_index_buffer =
            Self::control_point_index_buffer(control_point_index_buffer)?;
        self.send(RenderCmd::DrawIndexedPatches {
            number_of_patch_control_points: u64::from(command.number_of_patch_control_points),
            patch_start: command.patch_start,
            patch_count: command.patch_count,
            patch_index_buffer: patch_index_buffer.as_ref(),
            patch_index_buffer_offset: command.patch_index_buffer_offset,
            control_point_index_buffer: &control_point_index_buffer,
            control_point_index_buffer_offset: command.control_point_index_buffer_offset,
            instance_count: command.instance_count,
            base_instance: command.base_instance,
        })
    }

    /// 0xf.
    pub(super) fn draw_indexed_patches16(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedPatches16 = c.read_or_raise(OnFault::Return)?;
        let patch_index_buffer =
            self.patch_index_buffer(hold, command.patch_index_buffer_reference)?;
        let control_point_index_buffer =
            self.buffer(hold, command.control_point_index_buffer_reference)?;
        self.check_draw_pipeline()?;
        let control_point_index_buffer =
            Self::control_point_index_buffer(control_point_index_buffer)?;
        self.send(RenderCmd::DrawIndexedPatches {
            number_of_patch_control_points: u64::from(command.number_of_patch_control_points),
            patch_start: u64::from(command.patch_start),
            patch_count: u64::from(command.patch_count),
            patch_index_buffer: patch_index_buffer.as_ref(),
            patch_index_buffer_offset: u64::from(command.patch_index_buffer_offset),
            control_point_index_buffer: &control_point_index_buffer,
            control_point_index_buffer_offset: u64::from(command.control_point_index_buffer_offset),
            instance_count: u64::from(command.instance_count),
            base_instance: u64::from(command.base_instance),
        })
    }

    /// 0x10.
    pub(super) fn draw_primitives_indirect(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawPrimitivesIndirect = c.read_or_raise(OnFault::Return)?;
        let indirect_buffer = self.buffer(hold, command.indirect_buffer_reference)?;
        self.check_draw_pipeline()?;
        let indirect_buffer = Self::indirect_buffer(indirect_buffer)?;
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawPrimitivesIndirect {
            primitive_type,
            indirect_buffer: &indirect_buffer,
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }

    /// 0x11: the index buffer is looked up first, the indirect buffer checked first.
    pub(super) fn draw_indexed_primitives_indirect(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedPrimitivesIndirect = c.read_or_raise(OnFault::Return)?;
        let index_buffer = self.buffer(hold, command.index_buffer_reference)?;
        let indirect_buffer = self.buffer(hold, command.indirect_buffer_reference)?;
        self.check_draw_pipeline()?;
        let indirect_buffer = Self::indirect_buffer(indirect_buffer)?;
        let Some(index_buffer) = index_buffer else {
            crate::raise!(
                RaiseKind::NoIndexBuffer,
                "indexed draw without an index buffer"
            );
        };
        let primitive_type = u64::from(command.primitive_type);
        check_primitive_type(primitive_type)?;
        self.send(RenderCmd::DrawIndexedPrimitivesIndirect {
            primitive_type,
            index_type: u64::from(command.index_type),
            index_buffer: &index_buffer,
            index_buffer_offset: command.index_buffer_offset,
            indirect_buffer: &indirect_buffer,
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }

    /// 0x12.
    pub(super) fn draw_patches_indirect(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawPatchesIndirect = c.read_or_raise(OnFault::Return)?;
        let patch_index_buffer =
            self.patch_index_buffer(hold, command.patch_index_buffer_reference)?;
        let indirect_buffer = self.buffer(hold, command.indirect_buffer_reference)?;
        self.check_draw_pipeline()?;
        let indirect_buffer = Self::indirect_buffer(indirect_buffer)?;
        self.send(RenderCmd::DrawPatchesIndirect {
            number_of_patch_control_points: u64::from(command.number_of_patch_control_points),
            patch_index_buffer: patch_index_buffer.as_ref(),
            patch_index_buffer_offset: command.patch_index_buffer_offset,
            indirect_buffer: &indirect_buffer,
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }

    /// 0x13: the indirect buffer is checked before the control-point one.
    pub(super) fn draw_indexed_patches_indirect(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDrawIndexedPatchesIndirect = c.read_or_raise(OnFault::Return)?;
        let patch_index_buffer =
            self.patch_index_buffer(hold, command.patch_index_buffer_reference)?;
        let control_point_index_buffer =
            self.buffer(hold, command.control_point_index_buffer_reference)?;
        let indirect_buffer = self.buffer(hold, command.indirect_buffer_reference)?;
        self.check_draw_pipeline()?;
        let indirect_buffer = Self::indirect_buffer(indirect_buffer)?;
        let control_point_index_buffer =
            Self::control_point_index_buffer(control_point_index_buffer)?;
        self.send(RenderCmd::DrawIndexedPatchesIndirect {
            number_of_patch_control_points: u64::from(command.number_of_patch_control_points),
            patch_index_buffer: patch_index_buffer.as_ref(),
            patch_index_buffer_offset: command.patch_index_buffer_offset,
            control_point_index_buffer: &control_point_index_buffer,
            control_point_index_buffer_offset: command.control_point_index_buffer_offset,
            indirect_buffer: &indirect_buffer,
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }

    /// The ICB `reference` names, or the raise that names the reference.
    fn icb(&self, hold: Hold<'_>, reference: u32) -> RaiseResult<P::Icb> {
        match self
            .deserializer()
            .object_delegate()
            .get_indirect_command_buffer(hold, reference)?
        {
            Some(icb) => Ok(icb),
            None => Err(Raise::new(RaiseKind::NoIcb).with_detail(format_args!(
                "no indirect command buffer for reference {reference}"
            ))),
        }
    }

    /// 0x15: the range must end within the ICB's size (asked first) without overflowing.
    pub(super) fn execute_commands_in_buffer_ranged(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdExecuteCommandsInBufferRanged = c.read_or_raise(OnFault::Return)?;
        let icb = self.icb(hold, command.indirect_command_buffer_reference)?;
        let range = NsRange::new(command.range_location, command.range_length);
        let size = icb.size();
        let within = range
            .location
            .checked_add(range.length)
            .is_some_and(|end| end <= size);
        if !within {
            crate::raise!(
                RaiseKind::IcbRange,
                format_args!(
                    "range {}+{} runs past the indirect command buffer's {size} commands",
                    range.location, range.length
                )
            );
        }
        self.send(RenderCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: range,
        })
    }

    /// 0x14: the range comes from a buffer, which the GPU reads.
    pub(super) fn execute_commands_in_buffer(
        &mut self,
        hold: Hold<'_>,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdExecuteCommandsInBuffer = c.read_or_raise(OnFault::Return)?;
        let icb_reference = command.indirect_command_buffer_reference;
        let icb = self.icb(hold, icb_reference)?;
        let Some(indirect_range_buffer) =
            self.buffer(hold, command.indirect_range_buffer_reference)?
        else {
            crate::raise!(
                RaiseKind::NoIndirectRangeBuffer,
                format_args!(
                    "no indirect range buffer for reference {}",
                    command.indirect_range_buffer_reference
                )
            );
        };
        self.send(RenderCmd::ExecuteCommandsIndirect {
            indirect_commandbuffer: &icb,
            indirect_range_buffer: &indirect_range_buffer,
            indirect_buffer_offset: command.indirect_buffer_offset,
        })
    }

    /// 0x9b.
    pub(super) fn dispatch_threads_per_tile(&mut self, c: &mut Cursor) -> RaiseResult<()> {
        let command: CmdDispatchThreadsPerTile = c.read_or_raise(OnFault::Return)?;
        self.check_dispatch_pipeline()?;
        self.send(RenderCmd::DispatchThreadsPerTile {
            threads_per_tile: MtlSize::from(command.threads_per_tile),
        })
    }

    /// 0xa2: reads 0xa3's whole record (0x4c bytes) and ignores the index.
    pub(super) fn dispatch_threads_per_tile_in_region(
        &mut self,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDispatchThreadsPerTileInRegionWithIndex =
            c.read_or_raise(OnFault::Return)?;
        self.check_dispatch_pipeline()?;
        self.send(RenderCmd::DispatchThreadsPerTileInRegion {
            threads_per_tile: MtlSize::from(command.threads_per_tile),
            region: MtlRegion::new(command.region_origin.into(), command.region_size.into()),
        })
    }

    /// 0xa3: the render target index goes as the record's `u32` (D41).
    pub(super) fn dispatch_threads_per_tile_in_region_with_index(
        &mut self,
        c: &mut Cursor,
    ) -> RaiseResult<()> {
        let command: CmdDispatchThreadsPerTileInRegionWithIndex =
            c.read_or_raise(OnFault::Return)?;
        self.check_dispatch_pipeline()?;
        self.send(RenderCmd::DispatchThreadsPerTileInRegionIndex {
            threads_per_tile: MtlSize::from(command.threads_per_tile),
            region: MtlRegion::new(command.region_origin.into(), command.region_size.into()),
            render_target_array_index: command.render_target_array_index,
        })
    }
}

/// Shared test scaffolding for the render decoder's handlers (`draw` and `state`).
#[cfg(test)]
pub(super) mod test_support {

    use std::fmt::Debug;
    use std::sync::Arc;

    use zerocopy::{Immutable, IntoBytes};

    use super::RenderDecoder;
    use crate::deserializer::Deserializer;
    use crate::deserializer::test_lookups::{Lookups, set_caps};
    use crate::host::HostPtr;
    use crate::memory::cursor::Cursor;
    use crate::platform::Handle;
    use crate::platform::caps::DeviceCaps;
    use crate::raise::{RaiseId, RaiseResult};
    use crate::testkit::calllog::{Arg, Call};
    use crate::testkit::objects::{Fk, Kind, Model, kind};
    use crate::testkit::platform::FakePlatform;

    /// A render decoder over a fake platform, its lookups served by a resource manager.
    pub struct Rig {
        pub p: Arc<FakePlatform>,
        pub seams: Arc<Lookups>,
        pub dec: RenderDecoder<FakePlatform>,
        pub encoder: Fk<kind::RenderEncoder>,
        /// The deserializer the decoder reaches weakly: the caller's reference.
        _deserializer: Arc<Deserializer<FakePlatform>>,
    }

    impl Rig {
        /// An open encoder, no pipeline, the highest colour index 0.
        pub fn new() -> Rig {
            let p = Arc::new(FakePlatform::new());
            let seams = Arc::new(Lookups::new(&p));
            let deserializer = Deserializer::new(p.clone(), Some(p.device()), seams.lookup());
            let command_buffer = p.world().new_obj::<kind::CommandBuffer>(Model::None);
            let encoder = p.world().new_obj::<kind::RenderEncoder>(Model::None);
            let dec = RenderDecoder {
                deserializer: Arc::downgrade(&deserializer),
                encoder: Some(encoder.clone_ref()),
                command_buffer: Some(command_buffer),
                pipeline: None,
                max_color_attachment_index: 0,
            };
            p.call_log().clear();
            Rig {
                p,
                seams,
                dec,
                encoder,
                _deserializer: deserializer,
            }
        }

        /// The decoder with no encoder.
        pub fn without_encoder() -> Rig {
            let mut r = Rig::new();
            r.dec.encoder = None;
            r
        }

        /// Sets the pipeline state to a fresh pipeline.
        pub fn with_pipeline(mut self) -> Rig {
            let pipeline = self.p.world().new_obj::<kind::RenderPipeline>(Model::None);
            self.dec.pipeline = Some(pipeline);
            self
        }

        /// A new fake object of kind `K`, provided to the lookups under `reference`.
        pub fn provide<K: Kind>(&self, reference: u32) -> Fk<K> {
            let h = self.p.world().new_obj::<K>(Model::None);
            self.seams.provide(reference, &h);
            h
        }

        /// Every ordered call so far: lookups and the Metal calls whose position is observable.
        pub fn calls(&self) -> Vec<Call> {
            self.p.call_log().calls()
        }

        /// The ordered calls' selectors, lookups included (`ManagerLookup` rows by method name).
        pub fn sels(&self) -> Vec<&'static str> {
            self.p.call_log().sels()
        }

        /// Sets the device's capabilities: its defaults, changed by `f`.
        pub fn set_caps(&self, f: impl FnOnce(&mut DeviceCaps)) {
            set_caps(&self.p, f);
        }

        /// The calls to the encoder, as (selector, args).
        pub fn encoder_calls(&self) -> Vec<(&'static str, Vec<Arg>)> {
            self.p
                .call_log()
                .calls_on(self.encoder.id())
                .into_iter()
                .map(|c| (c.sel, c.args))
                .collect()
        }
    }

    /// A cursor over `bytes` (leaked host memory: the test's lifetime).
    pub fn cursor(bytes: &[u8]) -> Cursor {
        let mem: &'static mut [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
        let len = mem.len() as u64;
        Cursor::with_pointer(Arc::new(()), HostPtr::from_ptr(mem.as_mut_ptr()), len)
    }

    /// A record's bytes.
    pub fn rec<T: IntoBytes + Immutable>(t: &T) -> Vec<u8> {
        t.as_bytes().to_vec()
    }

    /// Asserts `r` raised `id` with `detail`.
    pub fn assert_raise<T: Debug>(r: RaiseResult<T>, id: impl Into<RaiseId>, detail: &str) {
        let e = r.expect_err("expected a raise");
        assert_eq!(e.id(), id.into(), "{e}");
        assert_eq!(e.detail(), Some(detail), "{e}");
    }

    /// Asserts `r` is the cursor's refused read (a short record).
    pub fn assert_failed_read<T: Debug>(r: RaiseResult<T>) {
        let e = r.expect_err("expected a raise");
        let detail = e.detail().unwrap_or_default();
        assert!(detail.starts_with("record not readable: "), "{e}");
        assert_eq!(
            e.id(),
            RaiseId::Memory(crate::memory::RaiseKind::CursorReadFailed),
            "{e}"
        );
    }

    /// The object id of a handle, as the call log records an object argument.
    pub fn obj<H: Handle>(h: &H) -> Arg {
        Arg::obj(h)
    }
}

#[cfg(test)]
mod tests {

    use super::test_support::*;
    use super::*;
    use crate::deserializer::records::{MtlOriginRecord, MtlSizeRecord};
    use crate::raise::death::DeathKind;
    use crate::raise::death::DeathSite;

    use crate::testkit::calllog::Arg;

    use crate::testkit::script::Step;

    const PIPE: &str = "draw without a pipeline state";
    const TPIPE: &str = "tile dispatch without a pipeline state";

    #[test]
    fn primitive_types_under_ten_pass() {
        // 0-9 pass, 5-9 included.
        for t in 0..10 {
            check_primitive_type(t).unwrap();
        }
        assert_raise(
            check_primitive_type(10),
            RaiseKind::PrimitiveType,
            "primitive type 10 is not valid",
        );
        assert_raise(
            check_primitive_type(u64::MAX),
            RaiseKind::PrimitiveType,
            "primitive type 18446744073709551615 is not valid",
        );
    }

    /// A raise names its own site.
    #[test]
    fn a_raise_names_its_site() {
        let d = check_primitive_type(10).unwrap_err().site();
        assert_eq!(
            d,
            DeathSite {
                kind: DeathKind::UnhandledRaise,
                file: "deserializer.render.draw.primitive_type",
                line: 0
            }
        );
    }

    // ------------------------------------------------------------------ non-indexed draws

    #[test]
    fn draw_primitives64() {
        let cmd = CmdDrawPrimitives64 {
            primitive_type: 3,
            vertex_start: 0x1_0000_0001,
            vertex_count: 7,
        };
        // No pipeline: pipe, nothing sent.
        {
            let mut r = Rig::new();
            assert_raise(
                r.dec.draw_primitives64(&mut cursor(&rec(&cmd))),
                RaiseKind::DrawWithoutPipeline,
                PIPE,
            );
            assert!(r.encoder_calls().is_empty());
            // Pipeline, bad type: type check after pipe.
            let mut r = Rig::new().with_pipeline();
            let bad = CmdDrawPrimitives64 {
                primitive_type: 10,
                ..cmd
            };
            assert_raise(
                r.dec.draw_primitives64(&mut cursor(&rec(&bad))),
                RaiseKind::PrimitiveType,
                "primitive type 10 is not valid",
            );
            assert!(r.encoder_calls().is_empty());
            // Short record: Failed to read before the pipeline check.
            let mut r = Rig::new();
            assert_failed_read(r.dec.draw_primitives64(&mut cursor(&rec(&cmd)[..0x13])));
            // The draw.
            let mut r = Rig::new().with_pipeline();
            r.dec.draw_primitives64(&mut cursor(&rec(&cmd))).unwrap();
            assert_eq!(
                r.encoder_calls(),
                vec![(
                    "drawPrimitives:vertexStart:vertexCount:",
                    vec![Arg::U64(3), Arg::U64(0x1_0000_0001), Arg::U64(7)]
                )]
            );
            // A nil encoder: the send is a no-op.
            let mut r = Rig::without_encoder().with_pipeline();
            r.dec.draw_primitives64(&mut cursor(&rec(&cmd))).unwrap();
            assert!(r.calls().is_empty());
        };
    }

    #[test]
    fn draw_primitives16() {
        let cmd = CmdDrawPrimitives16 {
            primitive_type: 9,
            vertex_start: 0xffff,
            vertex_count: 2,
        };
        let mut r = Rig::new();
        assert_raise(
            r.dec.draw_primitives16(&mut cursor(&rec(&cmd))),
            RaiseKind::DrawWithoutPipeline,
            PIPE,
        );
        let mut r = Rig::new().with_pipeline();
        let bad = CmdDrawPrimitives16 {
            primitive_type: 0x8000_0000,
            ..cmd
        };
        assert_raise(
            r.dec.draw_primitives16(&mut cursor(&rec(&bad))),
            RaiseKind::PrimitiveType,
            "primitive type 2147483648 is not valid",
        );
        let mut r = Rig::new();
        assert_failed_read(r.dec.draw_primitives16(&mut cursor(&rec(&cmd)[..7])));
        // Type 9 passes; u16s widen unsigned.
        let mut r = Rig::new().with_pipeline();
        r.dec.draw_primitives16(&mut cursor(&rec(&cmd))).unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "drawPrimitives:vertexStart:vertexCount:",
                vec![Arg::U64(9), Arg::U64(0xffff), Arg::U64(2)]
            )]
        );
    }

    #[test]
    fn draw_instanced_primitives64() {
        let cmd = CmdDrawInstancedPrimitives64 {
            vertex_start: 1,
            vertex_count: 2,
            instance_count: 3,
            primitive_type: 4,
            unknown1a: [0; 2],
        };
        let mut r = Rig::new();
        assert_raise(
            r.dec.draw_instanced_primitives64(&mut cursor(&rec(&cmd))),
            RaiseKind::DrawWithoutPipeline,
            PIPE,
        );
        let mut r = Rig::new().with_pipeline();
        let bad = CmdDrawInstancedPrimitives64 {
            primitive_type: 11,
            ..cmd
        };
        assert_raise(
            r.dec.draw_instanced_primitives64(&mut cursor(&rec(&bad))),
            RaiseKind::PrimitiveType,
            "primitive type 11 is not valid",
        );
        let mut r = Rig::new();
        assert_failed_read(
            r.dec
                .draw_instanced_primitives64(&mut cursor(&rec(&cmd)[..0x1b])),
        );
        let mut r = Rig::new().with_pipeline();
        r.dec
            .draw_instanced_primitives64(&mut cursor(&rec(&cmd)))
            .unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "drawPrimitives:vertexStart:vertexCount:instanceCount:",
                vec![Arg::U64(4), Arg::U64(1), Arg::U64(2), Arg::U64(3)]
            )]
        );
    }

    #[test]
    fn draw_instanced_primitives16() {
        let cmd = CmdDrawInstancedPrimitives16 {
            vertex_start: 0xfffe,
            vertex_count: 2,
            instance_count: 3,
            primitive_type: 1,
        };
        let mut r = Rig::new();
        assert_raise(
            r.dec.draw_instanced_primitives16(&mut cursor(&rec(&cmd))),
            RaiseKind::DrawWithoutPipeline,
            PIPE,
        );
        let mut r = Rig::new().with_pipeline();
        let bad = CmdDrawInstancedPrimitives16 {
            primitive_type: 10,
            ..cmd
        };
        assert_raise(
            r.dec.draw_instanced_primitives16(&mut cursor(&rec(&bad))),
            RaiseKind::PrimitiveType,
            "primitive type 10 is not valid",
        );
        let mut r = Rig::new();
        assert_failed_read(
            r.dec
                .draw_instanced_primitives16(&mut cursor(&rec(&cmd)[..7])),
        );
        let mut r = Rig::new().with_pipeline();
        r.dec
            .draw_instanced_primitives16(&mut cursor(&rec(&cmd)))
            .unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "drawPrimitives:vertexStart:vertexCount:instanceCount:",
                vec![Arg::U64(1), Arg::U64(0xfffe), Arg::U64(2), Arg::U64(3)]
            )]
        );
    }

    #[test]
    fn draw_instanced_base_primitives64() {
        let cmd = CmdDrawInstancedBasePrimitives64 {
            vertex_start: 1,
            vertex_count: 2,
            instance_count: 3,
            base_instance: 4,
            primitive_type: 5,
            unknown22: [0; 2],
        };
        let mut r = Rig::new();
        assert_raise(
            r.dec
                .draw_instanced_base_primitives64(&mut cursor(&rec(&cmd))),
            RaiseKind::DrawWithoutPipeline,
            PIPE,
        );
        let mut r = Rig::new().with_pipeline();
        let bad = CmdDrawInstancedBasePrimitives64 {
            primitive_type: 10,
            ..cmd
        };
        assert_raise(
            r.dec
                .draw_instanced_base_primitives64(&mut cursor(&rec(&bad))),
            RaiseKind::PrimitiveType,
            "primitive type 10 is not valid",
        );
        let mut r = Rig::new();
        assert_failed_read(
            r.dec
                .draw_instanced_base_primitives64(&mut cursor(&rec(&cmd)[..0x23])),
        );
        let mut r = Rig::new().with_pipeline();
        r.dec
            .draw_instanced_base_primitives64(&mut cursor(&rec(&cmd)))
            .unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                vec![
                    Arg::U64(5),
                    Arg::U64(1),
                    Arg::U64(2),
                    Arg::U64(3),
                    Arg::U64(4)
                ]
            )]
        );
    }

    #[test]
    fn draw_instanced_base_primitives16() {
        let cmd = CmdDrawInstancedBasePrimitives16 {
            vertex_start: 1,
            vertex_count: 2,
            instance_count: 3,
            base_instance: 0xffff,
            primitive_type: 6,
            unknown0a: [0; 2],
        };
        let mut r = Rig::new();
        assert_raise(
            r.dec
                .draw_instanced_base_primitives16(&mut cursor(&rec(&cmd))),
            RaiseKind::DrawWithoutPipeline,
            PIPE,
        );
        let mut r = Rig::new().with_pipeline();
        let bad = CmdDrawInstancedBasePrimitives16 {
            primitive_type: 10,
            ..cmd
        };
        assert_raise(
            r.dec
                .draw_instanced_base_primitives16(&mut cursor(&rec(&bad))),
            RaiseKind::PrimitiveType,
            "primitive type 10 is not valid",
        );
        let mut r = Rig::new();
        assert_failed_read(
            r.dec
                .draw_instanced_base_primitives16(&mut cursor(&rec(&cmd)[..0xb])),
        );
        let mut r = Rig::new().with_pipeline();
        r.dec
            .draw_instanced_base_primitives16(&mut cursor(&rec(&cmd)))
            .unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
                vec![
                    Arg::U64(6),
                    Arg::U64(1),
                    Arg::U64(2),
                    Arg::U64(3),
                    Arg::U64(0xffff)
                ]
            )]
        );
    }

    // ------------------------------------------------------------------ indexed draws

    // ------------------------------------------------------------------ patches

    // ------------------------------------------------------------------ indirect draws

    // ------------------------------------------------------------------ ICB executes

    // ------------------------------------------------------------------ tile dispatches

    #[test]
    fn dispatch_threads_per_tile() {
        let cmd = CmdDispatchThreadsPerTile {
            threads_per_tile: MtlSizeRecord {
                width: 1,
                height: 2,
                depth: 3,
            },
        };
        let f = RenderDecoder::dispatch_threads_per_tile;
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&rec(&cmd)[..0x17])));
        let mut r = Rig::new();
        assert_raise(
            f(&mut r.dec, &mut cursor(&rec(&cmd))),
            RaiseKind::DispatchWithoutPipeline,
            TPIPE,
        );
        let mut r = Rig::new().with_pipeline();
        f(&mut r.dec, &mut cursor(&rec(&cmd))).unwrap();
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "dispatchThreadsPerTile:",
                vec![Arg::val(&MtlSize::new(1, 2, 3))]
            )]
        );
    }

    fn region_cmd() -> CmdDispatchThreadsPerTileInRegionWithIndex {
        CmdDispatchThreadsPerTileInRegionWithIndex {
            threads_per_tile: MtlSizeRecord {
                width: 1,
                height: 2,
                depth: 3,
            },
            region_origin: MtlOriginRecord { x: 4, y: 5, z: 6 },
            region_size: MtlSizeRecord {
                width: 7,
                height: 8,
                depth: 9,
            },
            render_target_array_index: 10,
        }
    }

    #[test]
    fn dispatch_threads_per_tile_in_region() {
        let cmd = region_cmd();
        let f = RenderDecoder::dispatch_threads_per_tile_in_region;
        // The whole 0x4c-byte record is read, index included.
        {
            let mut r = Rig::new().with_pipeline();
            assert_failed_read(f(&mut r.dec, &mut cursor(&rec(&cmd)[..0x4b])));
            let mut r = Rig::new();
            assert_raise(
                f(&mut r.dec, &mut cursor(&rec(&cmd))),
                RaiseKind::DispatchWithoutPipeline,
                TPIPE,
            );
            let mut r = Rig::new().with_pipeline();
            f(&mut r.dec, &mut cursor(&rec(&cmd))).unwrap();
            let region = MtlRegion::new(
                crate::platform::types::MtlOrigin::new(4, 5, 6),
                MtlSize::new(7, 8, 9),
            );
            assert_eq!(
                r.encoder_calls(),
                vec![(
                    "dispatchThreadsPerTile:inRegion:",
                    vec![Arg::val(&MtlSize::new(1, 2, 3)), Arg::val(&region)]
                )]
            );
        };
    }

    #[test]
    fn dispatch_threads_per_tile_in_region_with_index() {
        let cmd = region_cmd();
        let f = RenderDecoder::dispatch_threads_per_tile_in_region_with_index;
        let mut r = Rig::new();
        assert_failed_read(f(&mut r.dec, &mut cursor(&rec(&cmd)[..0x4b])));
        let mut r = Rig::new();
        assert_raise(
            f(&mut r.dec, &mut cursor(&rec(&cmd))),
            RaiseKind::DispatchWithoutPipeline,
            TPIPE,
        );
        // An SPI the encoder lacks raises, as testing.md 6.16 describes: the Metal exception
        // propagates.
        let mut r = Rig::new().with_pipeline();
        r.p.script().push(
            "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
            Step::unrecognized("dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:"),
        );
        assert!(f(&mut r.dec, &mut cursor(&rec(&cmd))).is_err());
        let mut r = Rig::new().with_pipeline();
        f(&mut r.dec, &mut cursor(&rec(&cmd))).unwrap();
        let region = MtlRegion::new(
            crate::platform::types::MtlOrigin::new(4, 5, 6),
            MtlSize::new(7, 8, 9),
        );
        assert_eq!(
            r.encoder_calls(),
            vec![(
                "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
                vec![
                    Arg::val(&MtlSize::new(1, 2, 3)),
                    Arg::val(&region),
                    Arg::U32(10)
                ]
            )]
        );
    }
}
