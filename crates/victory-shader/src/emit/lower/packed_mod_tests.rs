//! Regression tests verifying `Lower::packed16` fails closed on unmodeled
//! non-VOP3P modifiers (`sdwa`, `abs`, and `v_pk_fmac_f16` `neg`/`opsel`)
//! instead of silently ignoring them or negating only the low f16 half, while
//! keeping plain `v_pk_fmac_f16` and VOP3P `v_pk_fma_f16` modifiers intact.

use super::{Parts, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

fn lower_packed(inst_words: &[u32]) -> Result<super::Lowered, ShaderError> {
    let mut words = vec![
        0xBE8C_0302u32, // s_mov_b32 s12, s2
        0xBE8D_0380,    // s_mov_b32 s13, 0
        0xF408_0406,
        0xFA00_0000,    // s_load_dwordx4 s[16:19], s[12:13], 0x0
        0x7E02_0280,    // v_mov_b32 v1, 0
        0x7E04_0280,    // v_mov_b32 v2, 0
        0xBF8C_C07F,    // s_waitcnt lgkmcnt(0)
    ];
    words.extend_from_slice(inst_words);
    words.extend([
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 6,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 2,
            slot: 2,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 2,
            slot: 2,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
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
}

#[test]
fn packed16_rejects_unmodeled_fmac_and_sdwa_modifiers_while_keeping_valid_ops() {
    let plain_vop2 = (0x3Cu32 << 25) | (2 << 17) | (1 << 9) | 256;
    assert!(
        lower_packed(&[plain_vop2]).is_ok(),
        "plain VOP2 v_pk_fmac_f16 must lower"
    );

    let vop3_w0 = 0xD400_0000u32 | (0x13C << 16) | 2;
    let vop3_w1 = (257u32 << 9) | 256;
    assert!(
        lower_packed(&[vop3_w0, vop3_w1]).is_ok(),
        "plain VOP3 v_pk_fmac_f16 must lower"
    );

    let vop3p_w0 = 0xCC00_0000u32 | (0x0E << 16) | (1 << 8) | 2;
    let vop3p_w1 = (1u32 << 29) | (258 << 18) | (257 << 9) | 256;
    let ok = lower_packed(&[vop3p_w0, vop3p_w1])
        .expect("VOP3P v_pk_fma_f16 with neg/neg_hi must lower");
    let _ = ok;

    assert!(
        lower_packed(&[vop3_w0, vop3_w1 | (1 << 29)]).is_err(),
        "VOP3 v_pk_fmac_f16 with neg must be rejected"
    );
    assert!(
        lower_packed(&[vop3_w0 | (1 << 8), vop3_w1]).is_err(),
        "VOP3 v_pk_fmac_f16 with abs must be rejected"
    );
    assert!(
        lower_packed(&[vop3_w0 | (1 << 11), vop3_w1]).is_err(),
        "VOP3 v_pk_fmac_f16 with opsel must be rejected"
    );

    let sdwa_w0 = (0x3Cu32 << 25) | (2 << 17) | (1 << 9) | 0xF9;
    let sdwa_w1 = 0x0606_0600u32;
    assert!(
        lower_packed(&[sdwa_w0, sdwa_w1]).is_err(),
        "VOP2 v_pk_fmac_f16 with SDWA must be rejected"
    );
}

#[test]
fn min3_max3_med3_use_dxil_integer_min_max() {
    const I_MAX: i64 = 37;
    const I_MIN: i64 = 38;
    const U_MAX: i64 = 39;
    const U_MIN: i64 = 40;
    let cases: [(&str, u32, &[i64]); 6] = [
        ("v_min3_i32", 0x152, &[I_MIN, I_MIN]),
        ("v_min3_u32", 0x153, &[U_MIN, U_MIN]),
        ("v_max3_i32", 0x155, &[I_MAX, I_MAX]),
        ("v_max3_u32", 0x156, &[U_MAX, U_MAX]),
        ("v_med3_i32", 0x158, &[I_MAX, I_MAX, I_MIN, I_MIN]),
        ("v_med3_u32", 0x159, &[U_MAX, U_MAX, U_MIN, U_MIN]),
    ];
    let control = lower_packed(&[0xD578_0002, 0x040A_0301]).unwrap();
    for (name, opcode, expected) in cases {
        let lowered = lower_packed(&[
            0xD400_0002 | (opcode << 16), // VOP3 v2, v1, v1, v2 (writing v2)
            0x040A_0301,
        ])
        .unwrap();
        let mut opcodes = lowered.binary_i32_opcodes;
        opcodes.sort_unstable();
        let mut expected_opcodes = control.binary_i32_opcodes.clone();
        expected_opcodes.extend_from_slice(expected);
        expected_opcodes.sort_unstable();
        assert_eq!(
            (opcodes, lowered.icmps_and_selects),
            (expected_opcodes, control.icmps_and_selects),
            "{name}: dx.op.binary.i32 opcodes and (icmp, select) counts"
        );
    }
}

#[test]
fn dot4_lowers_to_dxil_dot4_add_packed_and_expands_constants() {
    let vop3p = |op: u32, clamp: u32, opsel: u32, opsel_hi: u32, s0: u32, s1: u32, s2: u32| -> [u32; 2] {
        [
            0xCC00_0000 | (op << 16) | (clamp << 15) | (opsel << 11) | 2,
            (opsel_hi << 27) | (s2 << 18) | (s1 << 9) | s0,
        ]
    };
    let runtime_prefix = [
        0x3402_0081u32, // v_lshlrev_b32 v1, 1, v0
        0x4A04_0082,    // v_add_nc_u32 v2, 2, v0
        0x4A06_0083,    // v_add_nc_u32 v3, 3, v0
    ];
    let with_runtime = |tail: &[u32]| {
        let mut w = runtime_prefix.to_vec();
        w.extend_from_slice(tail);
        lower_packed(&w).unwrap()
    };

    // Signed VOP3P v_dot4_i32_i8 (0x16) and unsigned v_dot4_u32_u8 (0x17) emit opcode 163 / 164.
    let dot4_i = with_runtime(&vop3p(0x16, 0, 0, 3, 256, 257, 259));
    assert_eq!(dot4_i.dot4_opcodes, vec![163]);

    let dot4_u = with_runtime(&vop3p(0x17, 0, 0, 3, 256, 257, 259));
    assert_eq!(dot4_u.dot4_opcodes, vec![164]);

    // VOP2 v_dot4c_i32_i8 (0x0D) with implicit runtime v2 accumulator emits opcode 163.
    let dot4c = with_runtime(&[(0x0Du32 << 25) | (2 << 17) | (1 << 9) | 256]);
    assert_eq!(dot4c.dot4_opcodes, vec![163]);

    // Clamped signed and unsigned forms keep emitting dot4AddPacked.i32.
    let dot4_i_clamp = with_runtime(&vop3p(0x16, 1, 0, 3, 256, 257, 259));
    assert_eq!(dot4_i_clamp.dot4_opcodes, vec![163]);
    let dot4_u_clamp = with_runtime(&vop3p(0x17, 1, 0, 3, 256, 257, 259));
    assert_eq!(dot4_u_clamp.dot4_opcodes, vec![164]);

    // Non-default opsel / opsel_hi on v_dot4_u32_u8 swaps halves cleanly.
    let dot4_opsel = with_runtime(&vop3p(0x17, 0, 1, 2, 256, 257, 259));
    assert_eq!(dot4_opsel.dot4_opcodes, vec![164]);

    // Constant multiplicand (`v1 = 0x0000_8000`) expands via scalar ops instead of dot4AddPacked.
    let mov_lit_v1 = [0x7E02_02FFu32, 0x0000_8000];
    let mut const_i = mov_lit_v1.to_vec();
    const_i.extend_from_slice(&vop3p(0x16, 0, 0, 3, 256, 257, 259));
    let lowered_const = with_runtime(&const_i);
    assert!(
        lowered_const.dot4_opcodes.is_empty(),
        "constant dot4 multiplicand must expand without dot4AddPacked"
    );
}
