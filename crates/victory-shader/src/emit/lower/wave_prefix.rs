use super::Lower;
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::wave::Op;

impl Lower<'_> {
    /// Lowers floating-point wave prefix sums (`AddF`, `AddF16`) and non-sum
    /// wave prefix reductions (`MinU`, `MaxU`, `MinI`, `MaxI`, `MinF`, `MaxF`,
    /// `And`, `Or`, `Xor`) via 6-round active-lane pointer jumping.
    pub(super) fn lower_wave_prefix(&mut self, op: Op, arg: V, id_i32: V, inclusive: bool) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let lane = self.lane_index();
        let active = self.s.m.bool(true);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[active]);
        let b0 = self.s.m.extract(c, ballot, 0);
        let b1 = self.s.m.extract(c, ballot, 1);

        let k0 = self.s.m.i32(0);
        let k1 = self.s.m.i32(1);
        let k31 = self.s.m.i32(31);
        let k32 = self.s.m.i32(32);
        let k63 = self.s.m.i32(63);
        let all_ones = self.s.m.i32(u32::MAX);

        let is_hi = self.s.m.cmp(c, pred::IUGE, lane, k32);
        let shift_lo = self.s.m.select(c, is_hi, k0, lane);
        let lo_pow = self.s.m.bin(c, Bin::Shl, k1, shift_lo);
        let raw_lo = self.s.m.bin(c, Bin::Sub, lo_pow, k1);
        let lo_mask = self.s.m.select(c, is_hi, all_ones, raw_lo);

        let lane_m32 = self.s.m.bin(c, Bin::Sub, lane, k32);
        let shift_hi = self.s.m.select(c, is_hi, lane_m32, k0);
        let hi_pow = self.s.m.bin(c, Bin::Shl, k1, shift_hi);
        let raw_hi = self.s.m.bin(c, Bin::Sub, hi_pow, k1);
        let hi_mask = self.s.m.select(c, is_hi, raw_hi, k0);

        let lo_below = self.s.m.bin(c, Bin::And, b0, lo_mask);
        let hi_below = self.s.m.bin(c, Bin::And, b1, hi_mask);
        let hi_nz = self.s.m.cmp(c, pred::NE, hi_below, k0);
        let lo_nz = self.s.m.cmp(c, pred::NE, lo_below, k0);
        let mut has_prev = self.s.m.bin(c, Bin::Or, hi_nz, lo_nz);

        let hi_fb = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[hi_below]);
        let lo_fb = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[lo_below]);
        let hi_idx = self.s.m.bin(c, Bin::Sub, k63, hi_fb);
        let lo_idx = self.s.m.bin(c, Bin::Sub, k31, lo_fb);
        let best_idx = self.s.m.select(c, hi_nz, hi_idx, lo_idx);
        // Point inactive-predecessor lanes at themselves so waveReadLaneAt only reads active lanes.
        let mut prev = self.s.m.select(c, has_prev, best_idx, lane);

        let orig_prev = prev;
        let orig_has_prev = has_prev;

        let float = matches!(op, Op::AddF | Op::AddF16 | Op::MinF | Op::MaxF);
        let mut acc = if float { self.as_int(arg) } else { arg };

        for _ in 0..6 {
            let prev_val = self.read_lane(acc, prev);
            let prev_prev = self.read_lane(prev, prev);
            let has_i32 = self.s.m.select(c, has_prev, k1, k0);
            let prev_has_i32 = self.read_lane(has_i32, prev);

            let combined = match op {
                Op::AddF | Op::AddF16 => {
                    let fa = self.as_float(prev_val);
                    let fb = self.as_float(acc);
                    let sum = self.s.m.bin(c, Bin::Add, fa, fb);
                    self.as_int(sum)
                }
                Op::MinF => {
                    let fa = self.as_float(prev_val);
                    let fb = self.as_float(acc);
                    let res = self.max_min(false, fa, fb);
                    self.as_int(res)
                }
                Op::MaxF => {
                    let fa = self.as_float(prev_val);
                    let fb = self.as_float(acc);
                    let res = self.max_min(true, fa, fb);
                    self.as_int(res)
                }
                Op::MinU => self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[prev_val, acc]),
                Op::MaxU => self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 39, &[prev_val, acc]),
                Op::MinI => self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 38, &[prev_val, acc]),
                Op::MaxI => self.s.op(c, "binary.i32", Attr::ReadNone, i32t, 37, &[prev_val, acc]),
                Op::And => self.s.m.bin(c, Bin::And, prev_val, acc),
                Op::Or => self.s.m.bin(c, Bin::Or, prev_val, acc),
                Op::Xor => self.s.m.bin(c, Bin::Xor, prev_val, acc),
                Op::AddU => self.s.m.bin(c, Bin::Add, prev_val, acc),
            };
            acc = self.s.m.select(c, has_prev, combined, acc);
            let prev_has = self.s.m.cmp(c, pred::NE, prev_has_i32, k0);
            has_prev = self.s.m.bin(c, Bin::And, has_prev, prev_has);
            prev = self.s.m.select(c, has_prev, prev_prev, lane);
        }

        let res_i32 = if inclusive {
            acc
        } else {
            let excl = self.read_lane(acc, orig_prev);
            self.s.m.select(c, orig_has_prev, excl, id_i32)
        };
        if float { self.as_float(res_i32) } else { res_i32 }
    }
}

#[cfg(test)]
mod tests {
    use crate::emit::lower::{Parts, Lowered, lower};
    use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};

    fn prefix_program_words(vop2_opcode: u32, inclusive: bool, ident: Option<u32>) -> Vec<u32> {
        let dpp = (vop2_opcode << 25) | 0x0002_02FA;
        let e32 = (vop2_opcode << 25) | 0x0002_0701;
        let store_reg = if inclusive { 1 } else { 4 };
        let store_word1 = 0x8001_0002 | (store_reg << 8);
        let init_ident = ident.unwrap_or(0);
        let cnd_src0 = if ident.is_some() { 0x105 } else { 0x80 }; // v5 or inline 0
        let mov_ident = ident.map_or(vec![], |bits| vec![0x7E0A_02FF, bits]); // v_mov_b32 v5, lit
        let mov_row_clear = if let Some(bits) = ident {
            vec![0x7E06_02FF, bits, 0x7E08_02FF, bits]
        } else {
            vec![]
        };
        let mut w = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0301, // s_mov_b32 s5, s1
            0xBE86_03C1, // s_mov_b32 s6, -1
            0xBE87_03FF,
            0x1101_4FAC, // s_mov_b32 s7, 0x11014fac
            0xBE80_3DC1, // s_or_saveexec_b32 s0, -1
        ];
        w.extend(mov_ident);
        w.extend([
            0xD501_0001,
            0x0002_0000 | cnd_src0, // v_cndmask_b32_e64 v1, {0|v5}, v0, s0
            dpp,
            0xFF01_1101, // <op>_dpp v1, v1, v1 row_shr:1
            dpp,
            0xFF01_1201, // <op>_dpp v1, v1, v1 row_shr:2
            dpp,
            0xFF01_1401, // <op>_dpp v1, v1, v1 row_shr:4
            dpp,
            0xFF01_1801, // <op>_dpp v1, v1, v1 row_shr:8
            0xD778_0002,
            0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
        ]);
        if let Some(bits) = ident {
            w.extend([0x7E06_02FF, bits]); // v_mov_b32_e32 v3, ident
        } else {
            w.push(0x7E06_0280); // v_mov_b32_e32 v3, 0
        }
        w.extend([
            0x7E06_02FA,
            0xAF00_E402, // v_mov_b32_dpp v3, v2 quad_perm:[0,1,2,3] row_mask:0xa
            e32,         // <op>_e32 v1, v1, v3
        ]);
        if let Some(bits) = ident {
            w.extend([0x7E08_02FF, bits]); // v_mov_b32_e32 v4, ident
        } else {
            w.push(0x7E08_0280); // v_mov_b32_e32 v4, 0
        }
        let _ = (init_ident, mov_row_clear);
        w.extend([
            0x7E08_02FA,
            0xFF01_1101, // v_mov_b32_dpp v4, v1 row_shr:1
            0xD760_0001,
            0x0001_1F01, // v_readlane_b32 s1, v1, 15
            0xD761_0004,
            0x0001_2001, // v_writelane_b32 v4, s1, 16
            0xBEFE_0300, // s_mov_b32 exec_lo, s0
            0x7E04_0280, // v_mov_b32 v2, 0
            0xE070_1000,
            store_word1, // buffer_store_dword v{1|4}, v2, s[4:7], 0 offen
            0xBF81_0000, // s_endpgm
        ]);
        w
    }

    fn lower_prefix_cs(words: &[u32]) -> Lowered {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 3,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 3,
            }],
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
    fn float_sum_prefix_lowers_via_pointer_jumping_without_wave_prefix_op() {
        for inclusive in [false, true] {
            let l = lower_prefix_cs(&prefix_program_words(0x03, inclusive, None));
            assert!(l.wave_ops);
            assert!(l.wave_prefix_calls.0 >= 1, "expected waveActiveBallot");
            assert!(l.wave_prefix_calls.1 >= 18, "expected 6 pointer-jumping rounds");
            assert_eq!(l.wave_prefix_calls.2, 0, "must not emit wavePrefixOp.f32");
            #[cfg(windows)]
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                let mut dxil = l.dxil.clone();
                v.validate(&mut dxil).expect("AddF wave prefix DXIL must validate");
            }
        }
    }

    #[test]
    fn non_sum_wave_prefixes_lower_and_validate() {
        // v_max_u32 (0x14, ident=0), v_or_b32 (0x1c, ident=0), v_min_f32 (0x0f, ident=+inf)
        for (op, ident, expect_umax) in [
            (0x14u32, None, true),
            (0x1cu32, None, false),
            (0x0fu32, Some(0x7F80_0000), false),
        ] {
            for inclusive in [false, true] {
                let l = lower_prefix_cs(&prefix_program_words(op, inclusive, ident));
                assert!(l.wave_ops);
                assert!(l.wave_prefix_calls.0 >= 1, "expected waveActiveBallot");
                assert!(l.wave_prefix_calls.1 >= 18, "expected 6 pointer-jumping rounds");
                assert_eq!(l.wave_prefix_calls.2, 0, "must not emit wavePrefixOp.f32");
                if expect_umax {
                    assert!(l.binary_i32_opcodes.contains(&39));
                }
                #[cfg(windows)]
                if let Ok(mut v) = crate::emit::validator::Validator::load(
                    crate::emit::validator::Validator::default_path().as_deref(),
                ) {
                    let mut dxil = l.dxil.clone();
                    v.validate(&mut dxil).expect("non-sum wave prefix DXIL must validate");
                }
            }
        }
    }
}
