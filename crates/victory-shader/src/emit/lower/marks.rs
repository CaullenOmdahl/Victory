//! GPU shader write-marks: marks the 64 KiB pages touched by shader buffer
//! stores, atomics, and global writes into the host's write-mark bitmap
//! (`u0`, [`super::super::WRITE_MARKS_SPACE`]) using wave-level coalescing and
//! pre-load checks to avoid redundant bus-crossing atomics.

use super::super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::Lower;
use crate::native::RootType;
use crate::resource::Source;

/// Resolved title-memory target for a buffer write instruction, prepared
/// before `begin_guarded` so any control-flow descriptor slot phis are built
/// safely in the enclosing IR block.
pub(super) enum WriteTarget {
    /// Shader uses address-space pointers (`made_buffer`): `(w0, w1, raw_byte_off)`.
    ByAddress(V, V, V),
    /// Root UAV descriptor: root parameter `slot` plus byte offset (computed inside guard if needed).
    Root(u8),
    /// Descriptor heap / table buffer: view-heap `slot` index `V`.
    Heap(V),
    /// Cannot statically/dynamically resolve (never happens for valid buffers).
    None,
}

impl<'a> Lower<'a> {
    /// Resolves how to find the 64-bit title base address for a `Fields::Buffer`
    /// write through `src`. When `by_addr` is `Some((w0, w1, raw_off))`, uses that
    /// directly without re-reading the heap slot.
    pub(super) fn resolve_write_target(
        &mut self,
        src: &Source,
        by_addr: Option<(V, V, V)>,
    ) -> WriteTarget {
        if !super::super::write_marks() {
            return WriteTarget::None;
        }
        if let Some((lo, hi, off)) = by_addr {
            return WriteTarget::ByAddress(lo, hi, off);
        }
        if let Source::Built { address, .. } = src
            && let Some(rp) = self.x.native.layout.get(address.param)
            && rp.root == RootType::Descriptor
        {
            return WriteTarget::Root(rp.slot);
        }
        match self.heap_slot(src, false) {
            Ok(slot) => WriteTarget::Heap(slot),
            Err(_) => WriteTarget::None,
        }
    }

    /// Host user-data dwords `(lo, hi)` carrying the 64-bit title base address
    /// of root UAV parameter `slot`.
    pub(super) fn root_addr_dwords(&mut self, slot: u8) -> (V, V) {
        if let Some(&(_, lo, hi)) = self.root_addrs.iter().find(|r| r.0 == slot) {
            return (lo, hi);
        }
        // Start above MEMORY (0), GROUPS (1), THREADS (2), and all possible LIMIT
        // slots (1..=layout.len()).
        let base_nth = (self.x.native.layout.len() as u32).max(2) + 1;
        let nth = base_nth + 2 * self.root_addrs.len() as u32;
        let lo = self.host_data(nth, RootType::ROOT_ADDR_LO, u32::from(slot));
        let hi = self.host_data(nth + 1, RootType::ROOT_ADDR_HI, u32::from(slot));
        self.root_addrs.push((slot, lo, hi));
        (lo, hi)
    }

    /// Emits `mark_address_write` for a `WriteTarget` inside an active-lane guard.
    /// `untyped_byte_off` is used for `Root` and untyped `Heap` buffers; for `typed`
    /// heap buffers `elem_index` is multiplied by the runtime stride stored at `slot`.
    pub(super) fn emit_buffer_write_mark(
        &mut self,
        target: WriteTarget,
        untyped_byte_off: V,
        elem_index: Option<V>,
        write_bytes: u32,
    ) {
        if !super::super::write_marks() {
            return;
        }
        match target {
            WriteTarget::ByAddress(lo, hi, raw_off) => {
                let (addr_lo, addr_hi) = self.add_byte_offset(lo, hi, raw_off);
                self.mark_address_write(addr_lo, addr_hi, write_bytes);
            }
            WriteTarget::Root(slot) => {
                let (base_lo, base_hi) = self.root_addr_dwords(slot);
                self.mark_guarded_base(base_lo, base_hi, untyped_byte_off, write_bytes);
            }
            WriteTarget::Heap(slot) => {
                let byte_off = if let Some(idx) = elem_index {
                    let stride = self.slot_dword(slot, 0);
                    let c = self.cur;
                    self.s.m.bin(c, Bin::Mul, idx, stride)
                } else {
                    untyped_byte_off
                };
                let base_lo = self.slot_dword(slot, 2 * super::super::OWN_SLOTS);
                let base_hi = self.slot_dword(slot, 3 * super::super::OWN_SLOTS);
                self.mark_guarded_base(base_lo, base_hi, byte_off, write_bytes);
            }
            WriteTarget::None => {}
        }
    }

    /// Marks the 64 KiB page(s) touched by `write_bytes` at 64-bit address `(lo, hi)`.
    pub(super) fn mark_address_write(&mut self, lo: V, hi: V, write_bytes: u32) {
        if !super::super::write_marks() {
            return;
        }
        // Pixel helper lanes run only for screen-space derivatives and do not store.
        let merge_helper = if self.s.pixel {
            let c = self.cur;
            let helper = self
                .s
                .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
            let one = self.s.m.bool(true);
            let active = self.s.m.bin(c, Bin::Xor, helper, one);
            let go = self.s.m.new_block();
            let done = self.s.m.new_block();
            self.s.m.terminate(c, Term::CondBr(active, go, done));
            self.cur = go;
            Some(done)
        } else {
            None
        };

        let first_page = self.page_of_addr(lo, hi);
        self.emit_mark_page(first_page);

        if write_bytes > 1 {
            let delta = self.s.m.i32(write_bytes - 1);
            let (last_lo, last_hi) = self.add_byte_offset(lo, hi, delta);
            let last_page = self.page_of_addr(last_lo, last_hi);
            let c = self.cur;
            let straddles = self.s.m.cmp(c, pred::NE, last_page, first_page);
            let second = self.s.m.new_block();
            let after = self.s.m.new_block();
            self.s.m.terminate(c, Term::CondBr(straddles, second, after));
            self.cur = second;
            self.emit_mark_page(last_page);
            let c2 = self.cur;
            self.s.m.terminate(c2, Term::Br(after));
            self.cur = after;
        }

        if let Some(done) = merge_helper {
            let c = self.cur;
            self.s.m.terminate(c, Term::Br(done));
            self.cur = done;
        }
    }

    fn add_byte_offset(&mut self, lo: V, hi: V, byte_off: V) -> (V, V) {
        let c = self.cur;
        let sum_lo = self.s.m.bin(c, Bin::Add, lo, byte_off);
        let wrapped = self.s.m.cmp(c, pred::IULT, sum_lo, lo);
        let carry = self.s.m.cast(c, Cast::ZExt, wrapped, self.s.t.i32);
        let sum_hi = self.s.m.bin(c, Bin::Add, hi, carry);
        (sum_lo, sum_hi)
    }

    fn page_of_addr(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let k16 = self.s.m.i32(16);
        let k255 = self.s.m.i32(0xFF);
        let low_part = self.s.m.bin(c, Bin::LShr, lo, k16);
        let hi_masked = self.s.m.bin(c, Bin::And, hi, k255);
        let hi_shifted = self.s.m.bin(c, Bin::Shl, hi_masked, k16);
        self.s.m.bin(c, Bin::Or, hi_shifted, low_part)
    }

    /// Guards `(base_lo | base_hi) != 0` and marks `(base_lo, base_hi) + byte_off`.
    pub(super) fn mark_guarded_base(
        &mut self,
        base_lo: V,
        base_hi: V,
        byte_off: V,
        write_bytes: u32,
    ) {
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let any = self.s.m.bin(c, Bin::Or, base_lo, base_hi);
        let valid = self.s.m.cmp(c, pred::NE, any, zero);
        let go = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s.m.terminate(c, Term::CondBr(valid, go, done));
        self.cur = go;
        let (addr_lo, addr_hi) = self.add_byte_offset(base_lo, base_hi, byte_off);
        self.mark_address_write(addr_lo, addr_hi, write_bytes);
        let c2 = self.cur;
        self.s.m.terminate(c2, Term::Br(done));
        self.cur = done;
    }

    fn emit_mark_page(&mut self, page: V) {
        let marks = self.s.write_marks_handle();
        let c = self.cur;
        let (i32t, i1t, iu) = (self.s.t.i32, self.s.t.i1, self.s.m.undef(self.s.t.i32));
        let (k5, k31, k2, k1) = (
            self.s.m.i32(5),
            self.s.m.i32(31),
            self.s.m.i32(2),
            self.s.m.i32(1),
        );
        let word_idx = self.s.m.bin(c, Bin::LShr, page, k5);
        let word_byte = self.s.m.bin(c, Bin::Shl, word_idx, k2);
        let bit_pos = self.s.m.bin(c, Bin::And, page, k31);
        let bit_mask = self.s.m.bin(c, Bin::Shl, k1, bit_pos);

        let lead_word = self.s.op(
            c,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[word_idx],
        );
        let same = self.s.m.cmp(c, pred::EQ, word_idx, lead_word);
        let all_same = self
            .s
            .op(c, "waveAllTrue", Attr::NoUnwind, i1t, 114, &[same]);

        let coal = self.s.m.new_block();
        let coal_first = self.s.m.new_block();
        let div = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s.m.terminate(c, Term::CondBr(all_same, coal, div));

        self.cur = coal;
        let or_op = self.s.m.i8(1);
        let combined = self.s.op(
            coal,
            "waveActiveBit.i32",
            Attr::NoUnwind,
            i32t,
            120,
            &[bit_mask, or_op],
        );
        let is_first = self
            .s
            .op(coal, "waveIsFirstLane", Attr::NoUnwind, i1t, 110, &[]);
        self.s.m.terminate(coal, Term::CondBr(is_first, coal_first, done));

        self.cur = coal_first;
        self.emit_load_or_atomic_or(marks, word_byte, combined, done, iu);

        self.cur = div;
        self.emit_load_or_atomic_or(marks, word_byte, bit_mask, done, iu);

        self.cur = done;
    }

    fn emit_load_or_atomic_or(
        &mut self,
        marks: V,
        word_byte: V,
        mask: V,
        done: usize,
        iu: V,
    ) {
        let c = self.cur;
        let (i32t, ret) = (self.s.t.i32, self.s.t.res_ret);
        let (m1, align) = (self.s.m.i8(1), self.s.m.i32(4));
        let loaded = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            ret,
            139,
            &[marks, word_byte, iu, m1, align],
        );
        let cur_f = self.s.m.extract(c, loaded, 0);
        let cur_i = self.as_int(cur_f);
        let present = self.s.m.bin(c, Bin::And, mask, cur_i);
        let needs = self.s.m.cmp(c, pred::NE, present, mask);
        let atomic_bb = self.s.m.new_block();
        self.s.m.terminate(c, Term::CondBr(needs, atomic_bb, done));

        self.cur = atomic_bb;
        let atomic_or = self.s.m.i32(2);
        self.s.op(
            atomic_bb,
            "atomicBinOp.i32",
            Attr::NoUnwind,
            i32t,
            78,
            &[marks, atomic_or, word_byte, iu, iu, mask],
        );
        self.s.m.terminate(atomic_bb, Term::Br(done));
    }
}

#[cfg(test)]
mod tests {
    use crate::emit::lower::{Parts, lower};
    use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};

    fn test_lower(words: &[u32], layout: Vec<RootParam>, used: Vec<UsedResource>) -> crate::emit::lower::Lowered {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
        .unwrap()
    }

    #[test]
    fn heap_buffer_store_emits_wave_coalesced_write_marks() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_02AA, // v_mov_b32 v1, 42
            0xE070_1000,
            0x8002_0100, // buffer_store_dword v1, v0, s[8:11], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        crate::emit::with_write_marks_enabled(|| {
            let l = test_lower(
                words,
                vec![
                    RootParam {
                        root: RootType::Constants,
                        table_offset_dwords: 0,
                        sgprs: 4,
                        base_register: 0,
                        user_data: 0,
                        slot: 0,
                    },
                    RootParam {
                        root: RootType::Table,
                        table_offset_dwords: 0,
                        sgprs: 1,
                        base_register: 0,
                        user_data: 4,
                        slot: 1,
                    },
                ],
                vec![UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                }],
            );
            assert!(l.write_marks, "heap UAV store must emit write_marks");
            assert!(l.stride_table, "heap UAV store must bind runtime table for base address");
            assert!(l.wave_ops, "write_marks must enable WAVE_OPS flag");
            #[cfg(windows)]
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                let mut dxil = l.dxil.clone();
                v.validate(&mut dxil).expect("heap write-marks DXIL must validate");
            }
        });
    }

    #[test]
    fn root_uav_store_emits_root_addr_user_data_and_write_marks() {
        let words: &[u32] = &[
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0301, // s_mov_b32 s5, s1
            0xBE86_03C1, // s_mov_b32 s6, -1
            0xBE87_03FF,
            0x1101_4FAC, // s_mov_b32 s7, 0x11014fac
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0287, // v_mov_b32 v1, 7
            0xE070_1000,
            0x8001_0100, // buffer_store_dword v1, v0, s[4:7], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        crate::emit::with_write_marks_enabled(|| {
            let l = test_lower(
                words,
                vec![RootParam {
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    sgprs: 2,
                    base_register: 0,
                    user_data: 0,
                    slot: 3,
                }],
                vec![UsedResource {
                    kind: ResourceType::Uav,
                    root: RootType::Descriptor,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 3,
                }],
            );
            assert!(l.write_marks, "root UAV store must emit write_marks");
            assert!(l.user_data.iter().any(|u| u.root == RootType::ROOT_ADDR_LO && u.dword == 3));
            assert!(l.user_data.iter().any(|u| u.root == RootType::ROOT_ADDR_HI && u.dword == 3));
            #[cfg(windows)]
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                let mut dxil = l.dxil.clone();
                v.validate(&mut dxil).expect("root UAV write-marks DXIL must validate");
            }
        });
    }
}
