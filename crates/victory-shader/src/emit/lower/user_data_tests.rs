use super::*;
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo, UsedResource};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let mut p = ir::build(&native.code, false, native.vgprs)?;
    crate::unspill::unspill(&mut p, &native.layout, native.stage);
    let res = resource::track(&p, &native.layout, native.stage)?;
    let io = io::raise(&p, &native)?;
    let st = structure::structure(&p)?;
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        psv: None,
        console_cf: false,
        per_lane_only: false,
    })
}

#[test]
fn user_data_cbv_rows_and_dwords_are_cached_in_entry_block() {
    let direct_words = [
        0x8008_0100u32, // s_add_u32 s8, s0, s1
        0x8008_0208,    // s_add_u32 s8, s8, s2
        0x8008_0308,    // s_add_u32 s8, s8, s3
        0xBF06_8000,    // s_cmp_eq_u32 s0, 0
        0xBF85_000B,    // s_cbranch_scc1 -> s_endpgm
        0x8008_0408,    // s_add_u32 s8, s8, s4
        0x8008_0508,    // s_add_u32 s8, s8, s5
        0x8008_0608,    // s_add_u32 s8, s8, s6
        0x8008_0708,    // s_add_u32 s8, s8, s7
        0x8008_0008,    // s_add_u32 s8, s8, s0 (repeat)
        0x8008_0108,    // s_add_u32 s8, s8, s1 (repeat)
        0x8008_0208,    // s_add_u32 s8, s8, s2 (repeat)
        0x8008_0308,    // s_add_u32 s8, s8, s3 (repeat)
        0x7E02_0208,    // v_mov_b32 v1, s8
        0xD834_0000,
        0x0000_0100,    // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let spilled_words = [
        0xF408_0100u32,
        0xFA00_0000,    // s_load_dwordx4 s[4:7], s[0:1], 0x0 (indices 16..20)
        0x8014_0504,    // s_add_u32 s20, s4, s5
        0x8014_0614,    // s_add_u32 s20, s20, s6
        0x8014_0714,    // s_add_u32 s20, s20, s7
        0xBF06_8004,    // s_cmp_eq_u32 s4, 0
        0xBF85_0013,    // s_cbranch_scc1 -> s_endpgm
        0xF408_0200,
        0xFA00_0010,    // s_load_dwordx4 s[8:11], s[0:1], 0x10 (indices 20..24)
        0xF408_0300,
        0xFA00_0000,    // s_load_dwordx4 s[12:15], s[0:1], 0x0 (repeat 16..20)
        0xF404_0400,
        0xFA00_0014,    // s_load_dwordx2 s[16:17], s[0:1], 0x14 (repeat 21..23)
        0x8014_0814,    // s_add_u32 s20, s20, s8
        0x8014_0914,    // s_add_u32 s20, s20, s9
        0x8014_0A14,    // s_add_u32 s20, s20, s10
        0x8014_0B14,    // s_add_u32 s20, s20, s11
        0x8014_0C14,    // s_add_u32 s20, s20, s12
        0x8014_0D14,    // s_add_u32 s20, s20, s13
        0x8014_0E14,    // s_add_u32 s20, s20, s14
        0x8014_0F14,    // s_add_u32 s20, s20, s15
        0x8014_1014,    // s_add_u32 s20, s20, s16
        0x8014_1114,    // s_add_u32 s20, s20, s17
        0x7E02_0214,    // v_mov_b32 v1, s20
        0xD834_0000,
        0x0000_0100,    // ds_write_b32 v0, v1
        S_ENDPGM,
    ];

    for (label, words, user_data, expected_rows) in [
        ("direct", &direct_words[..], 0u8, [0u32, 1]),
        ("spilled", &spilled_words[..], 16u8, [4u32, 5]),
    ] {
        let layout = vec![RootParam {
            sgprs: 8,
            ..root(RootType::Constants, user_data, 0)
        }];
        let emitted = lower_with(words, layout, vec![]).unwrap();
        assert_eq!(emitted.user_data.len(), 8, "{label}: 8 distinct dwords");

        let stats = emitted.cbuf_stats;
        assert!(
            stats.blocks >= 3,
            "{label}: expected multi-block shader, got {} blocks",
            stats.blocks
        );
        assert_eq!(
            stats.loads,
            vec![(0, expected_rows[0]), (0, expected_rows[1])],
            "{label}: expected 2 entry-block cbufferLoadLegacy.i32 calls"
        );
        assert_eq!(
            stats.extracts.len(),
            8,
            "{label}: expected 8 cached extractvalue ops, got {:?}",
            stats.extracts
        );
        assert!(
            stats.extracts.iter().all(|&(b, _)| b == 0),
            "{label}: all user-data extracts must live in entry block 0: {:?}",
            stats.extracts
        );
    }
}
