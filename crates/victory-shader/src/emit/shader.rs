//! A DXIL vertex, pixel, compute or mesh shader under construction: the module,
//! the `dx.op` declarations it calls, its signatures and bound resources,
//! and the metadata and container that wrap it up (SM 6.6, validator 1.6).

use super::bitcode::{Attr, Md, Module, Ty, V};
use super::container::{self, Binding, Psv, SigElement};

/// Where the shader reads its user data: `b0` in this space.
pub const USER_DATA_SPACE: u32 = 0;
/// Where immediate samplers are bound: `s<n>` in this space.
pub const SAMPLER_SPACE: u32 = 0;
/// Where root CBVs and SRVs are bound: `b<root parameter index>` and
/// `t<root parameter index>` in this space.
pub const ROOT_CBV_SPACE: u32 = 1;
/// Where the host's runtime stride table is bound: `t0` in this space
/// (one `u32` element stride per view-heap slot).
pub const STRIDE_SPACE: u32 = 2;

/// `ShaderFlags` bits (metadata) and their `SFI0` feature bits.
pub(crate) mod flags {
    pub const DOUBLES: u64 = 1 << 2;
    pub const RAW_BUFFERS: u64 = 1 << 4;
    /// 16-bit types are used; with [`NATIVE_LOW_PRECISION`], as native
    /// 16-bit ones (SM 6.2), not minimum precision.
    pub const LOW_PRECISION: u64 = 1 << 5;
    pub const DOUBLE_EXT: u64 = 1 << 6;
    pub const STENCIL_REF: u64 = 1 << 11;
    pub const TYPED_UAV_LOADS: u64 = 1 << 13;
    /// UAVs in a vertex, geometry or mesh program.
    pub const UAVS_EVERY_STAGE: u64 = 1 << 16;
    pub const WAVE_OPS: u64 = 1 << 19;
    pub const BARYCENTRICS: u64 = 1 << 22;
    pub const INT64: u64 = 1 << 20;
    pub const NATIVE_LOW_PRECISION: u64 = 1 << 23;
    pub const ATOMIC_TYPED64: u64 = 1 << 27;
    pub const ATOMIC_INT64_GROUP_SHARED: u64 = 1 << 28;
    pub const ATOMIC_HEAP64: u64 = 1 << 32;
    pub const RESOURCE_HEAP: u64 = 1 << 30;
    pub const SAMPLER_HEAP: u64 = 1 << 31;
}

pub(crate) struct Types {
    pub void: Ty,
    pub i1: Ty,
    pub i32: Ty,
    pub f32: Ty,
    pub handle: Ty,
    pub props: Ty,
    pub bind: Ty,
    pub cbuf_ret: Ty,
    pub res_ret: Ty,
}

/// The vertices a rectangle's geometry shader emits.
pub(crate) const RECTANGLE_VERTICES: u32 = 6;

/// What a mesh shader declares beyond its thread-group size.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mesh {
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// 1 lines, 2 triangles.
    pub topology: u8,
}

pub(crate) struct Shader {
    pub m: Module,
    /// Test builds: the property words of each handle annotation, in order.
    #[cfg(test)]
    pub annotations: Vec<(u32, u32)>,
    pub t: Types,
    pub pixel: bool,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    /// A mesh shader's per-primitive outputs.
    pub prims: Vec<SigElement>,
    /// A mesh shader (`threads` is its thread-group size).
    pub mesh: Option<Mesh>,
    /// A geometry shader from triangles to two-triangle strips
    /// (`rectangle`).
    pub rectangles: bool,
    /// Dwords of user data the shader reads (0: no user-data buffer).
    pub user_data: u32,
    /// Immediate samplers, bound as `s0..` in [`SAMPLER_SPACE`].
    pub samplers: u32,
    /// Bit n: immediate sampler n is a comparison sampler.
    pub cmp_samplers: u64,
    /// Root CBVs read, by root parameter index, with their handles.
    pub root_cbvs: Vec<(u8, V)>,
    /// Root SRVs (raw buffers) read, likewise.
    pub root_srvs: Vec<(u8, V)>,
    /// Where reads through a handle (a root view's) may start at most.
    pub limits: Vec<(V, V)>,
    /// Root UAVs (raw buffers) read or written, likewise.
    pub root_uavs: Vec<(u8, V)>,
    /// Whether they are globally coherent.
    pub root_uavs_shared: bool,
    /// Handle of the runtime stride table (`t0`, [`STRIDE_SPACE`]), if read.
    pub stride_table: Option<V>,
    pub flags: u64,
    /// Heap handles made with a non-uniform index.
    pub nonuniform_handles: u32,
    /// Sentinel null view slot to clamp `ResourceDescriptorHeap` indices to.
    view_clamp: Option<u32>,
    /// Sentinel null sampler slot to clamp `SamplerDescriptorHeap` indices to.
    sampler_clamp: Option<u32>,
    /// A compute shader with this thread-group size.
    pub threads: Option<[u32; 3]>,
    /// The wave size a compute shader requires (`WaveSize`).
    pub wave_size: Option<u32>,
    /// Thread-group shared memory, in dwords.
    pub lds_dwords: u32,
    lds: Option<V>,
    main: V,
    pub entry: usize,
}

impl Shader {
    pub fn new(pixel: bool) -> Self {
        let mut m = Module::default();
        let void = m.void();
        let i1 = m.int(1);
        let i8 = m.int(8);
        let i32 = m.int(32);
        let f32 = m.float();
        let i8p = m.ptr(i8);
        let handle = m.named_struct("dx.types.Handle", vec![i8p]);
        let props = m.named_struct("dx.types.ResourceProperties", vec![i32, i32]);
        let bind = m.named_struct("dx.types.ResBind", vec![i32, i32, i32, i8]);
        let cbuf_ret = m.named_struct("dx.types.CBufRet.i32", vec![i32, i32, i32, i32]);
        let res_ret = m.named_struct("dx.types.ResRet.f32", vec![f32, f32, f32, f32, i32]);
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        let main = m.function("main", main_ty, None, true);
        let entry = m.new_block();
        Shader {
            m,
            #[cfg(test)]
            annotations: Vec::new(),
            t: Types {
                void,
                i1,
                i32,
                f32,
                handle,
                props,
                bind,
                cbuf_ret,
                res_ret,
            },
            pixel,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh: None,
            rectangles: false,
            user_data: 0,
            samplers: 0,
            cmp_samplers: 0,
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            limits: Vec::new(),
            root_uavs: Vec::new(),
            root_uavs_shared: false,
            stride_table: None,
            flags: 0,
            nonuniform_handles: 0,
            view_clamp: super::active_clamp().0,
            sampler_clamp: super::active_clamp().1,
            threads: None,
            wave_size: None,
            lds_dwords: 0,
            lds: None,
            main,
            entry,
        }
    }

    /// The single LDS allocation in address space 3.
    pub fn lds(&mut self, wide: bool) -> V {
        if let Some(g) = self.lds {
            return g;
        }
        let arr = if wide {
            let i64t = self.m.int(64);
            self.m.array(self.lds_dwords.div_ceil(2), i64t)
        } else {
            self.m.array(self.lds_dwords, self.t.i32)
        };
        let g = self.m.global("lds", arr, 3);
        if wide {
            self.m.align_global_to_eight(g);
        }
        self.lds = Some(g);
        g
    }

    /// Calls `dx.op.<name>` (class and overload) with `opcode` and `args`.
    pub fn op(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        // A raw buffer load through a limited handle.
        const RAW_LOAD: u32 = 139;
        if opcode == RAW_LOAD
            && let Some(&(_, limit)) = self.limits.iter().find(|l| l.0 == args[0])
        {
            let i32t = self.t.i32;
            let at = self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[args[1], limit]);
            let mut args = args.to_vec();
            args[1] = at;
            return self.op_as_is(block, name, attr, ret, opcode, &args);
        }
        self.op_as_is(block, name, attr, ret, opcode, args)
    }

    fn op_as_is(
        &mut self,
        block: usize,
        name: &str,
        attr: Attr,
        ret: Ty,
        opcode: u32,
        args: &[V],
    ) -> V {
        let mut params = vec![self.t.i32];
        params.extend(args.iter().map(|&a| self.m.type_of(a)));
        let fty = self.m.func_ty(ret, params);
        self.m.ptr(fty);
        let f = self
            .m
            .function(&format!("dx.op.{name}"), fty, Some(attr), false);
        let mut all = vec![self.m.i32(opcode)];
        all.extend(args);
        self.m.call(block, f, all)
    }

    /// The handle of the user-data constant buffer, made in the entry block.
    pub fn user_data_handle(&mut self) -> V {
        let bind = [
            self.m.i32(0),
            self.m.i32(0),
            self.m.i32(USER_DATA_SPACE),
            self.m.i8(2), // CBV
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let block = self.entry;
        let (i, f) = (self.m.i32(0), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        // The size is known in `finish`, which fixes this placeholder.
        self.annotate(block, h, 13, u32::MAX)
    }

    /// The handle of the root CBV of root parameter `slot`, made in the
    /// entry block.
    pub fn root_cbv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_cbvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(2)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 13, 65536);
        self.root_cbvs.push((slot, h));
        h
    }

    /// The handle of the root SRV (a raw buffer) of root parameter `slot`.
    pub fn root_srv_handle(&mut self, slot: u8) -> V {
        if let Some(&(_, h)) = self.root_srvs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_srvs.push((slot, h));
        h
    }

    /// The handle of the root UAV (a raw buffer) of root parameter `slot`.
    pub fn root_uav_handle(&mut self, slot: u8, shared: bool) -> V {
        if let Some(&(_, h)) = self.root_uavs.iter().find(|r| r.0 == slot) {
            return h;
        }
        let r = self.m.i32(u32::from(slot));
        let bind = [r, r, self.m.i32(ROOT_CBV_SPACE), self.m.i8(1)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, r, f],
        );
        self.root_uavs_shared |= shared;
        let shared = if shared { 0x4000 } else { 0 };
        let h = self.annotate(block, h, 0x1000 | 11 | shared, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.root_uavs.push((slot, h));
        h
    }

    /// The handle of the host's runtime stride table (`t0` in
    /// [`STRIDE_SPACE`], a raw buffer of `u32` byte strides indexed by
    /// view-heap slot).
    pub fn stride_table_handle(&mut self) -> V {
        if let Some(h) = self.stride_table {
            return h;
        }
        let z = self.m.i32(0);
        let bind = [z, z, self.m.i32(STRIDE_SPACE), self.m.i8(0)];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let f = self.m.bool(false);
        let block = self.entry;
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, z, f],
        );
        let h = self.annotate(block, h, 11, 0);
        self.flags |= flags::RAW_BUFFERS;
        self.stride_table = Some(h);
        h
    }

    /// The handle of immediate sampler `n` (a comparison sampler when its
    /// bit in `cmp_samplers` is set).
    pub fn sampler_handle(&mut self, block: usize, n: u32) -> V {
        let bind = [
            self.m.i32(n),
            self.m.i32(n),
            self.m.i32(SAMPLER_SPACE),
            self.m.i8(3), // sampler
        ];
        let bind = self.m.aggregate(self.t.bind, &bind);
        let (i, f) = (self.m.i32(n), self.m.bool(false));
        let h = self.op(
            block,
            "createHandleFromBinding",
            Attr::ReadNone,
            self.t.handle,
            217,
            &[bind, i, f],
        );
        let cmp = if self.cmp_samplers >> n & 1 != 0 {
            0x8000
        } else {
            0
        };
        self.annotate(block, h, 14 | cmp, 0)
    }

    pub fn annotate(&mut self, block: usize, h: V, props0: u32, props1: u32) -> V {
        #[cfg(test)]
        self.annotations.push((props0, props1));
        let p = [self.m.i32(props0), self.m.i32(props1)];
        let p = self.m.aggregate(self.t.props, &p);
        self.op(
            block,
            "annotateHandle",
            Attr::ReadNone,
            self.t.handle,
            216,
            &[h, p],
        )
    }

    /// A descriptor heap handle, annotated with `props`; `nonuniform` when
    /// the index may differ between lanes (`NonUniformResourceIndex`).
    pub fn heap_handle(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        // An index past the bound heap faults the host GPU (e.g. AMD RDNA
        // `s_load_b128` off the end of a view or sampler heap); clamp both to
        // their permanent null sentinel slots.
        let limit = if sampler {
            self.sampler_clamp
        } else {
            self.view_clamp
        };
        let index = if let Some(last) = limit {
            let last = self.m.i32(last);
            let i32t = self.t.i32;
            self.op(block, "binary.i32", Attr::ReadNone, i32t, 40, &[index, last])
        } else {
            index
        };
        self.handle_at(block, index, sampler, nonuniform, props)
    }

    /// A handle of the heap slot at `index`, which is known to be in it.
    pub fn handle_at(
        &mut self,
        block: usize,
        index: V,
        sampler: bool,
        nonuniform: bool,
        props: (u32, u32),
    ) -> V {
        self.flags |= if sampler {
            flags::SAMPLER_HEAP
        } else {
            flags::RESOURCE_HEAP
        };
        self.nonuniform_handles += u32::from(nonuniform);
        let (s, n) = (self.m.bool(sampler), self.m.bool(nonuniform));
        let h = self.op(
            block,
            "createHandleFromHeap",
            Attr::ReadNone,
            self.t.handle,
            218,
            &[index, s, n],
        );
        self.annotate(block, h, props.0, props.1)
    }

    /// Aligns unused components in packed pixel-shader input registers to the
    /// interpolation mode used by active siblings in the same register.
    fn align_packed_pixel_inputs(&mut self) {
        if !self.pixel {
            return;
        }
        // Reading a per-vertex value with attributeAtVertex needs a
        // constant input, even when the source declaration was smooth.
        // D3D requires all components packed into the same register to
        // share a mode, including the ones the shader never reads.
        for index in 0..self.inputs.len() {
            let unused = &self.inputs[index];
            if unused.kind != 0 || unused.usage != 0 || unused.row.is_none() {
                continue;
            }
            let mut modes = self
                .inputs
                .iter()
                .filter(|e| e.row == unused.row && e.usage != 0)
                .map(|e| e.interp);
            if let Some(mode) = modes.next()
                && (mode == 1 || unused.comp == 9)
                && modes.all(|other| other == mode)
            {
                self.inputs[index].interp = mode;
            }
        }
    }

    /// The finished container.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_packed_pixel_inputs();
        let compute = self.threads.is_some() && self.mesh.is_none() && !self.rectangles;
        if !self.root_uavs.is_empty() && !self.pixel && !compute {
            self.flags |= flags::UAVS_EVERY_STAGE;
        }
        let ud_size = (self.user_data * 4).next_multiple_of(16);
        self.fix_user_data_size(ud_size);
        // The flags of what lowering made, as the module writes it: what only
        // unread values or never-taken edges used is not written.
        let uses = self.m.uses();
        for (flag, used) in [
            (flags::RESOURCE_HEAP, uses.heap),
            (flags::SAMPLER_HEAP, uses.sampler_heap),
            (flags::WAVE_OPS, uses.wave),
            (flags::INT64, uses.int64),
            (flags::TYPED_UAV_LOADS, uses.typed_uav_loads),
            (flags::ATOMIC_TYPED64, uses.atomic64_typed),
            (flags::ATOMIC_HEAP64, uses.atomic64_heap),
            (
                flags::LOW_PRECISION | flags::NATIVE_LOW_PRECISION,
                uses.native16,
            ),
        ] {
            self.flags = if used { self.flags | flag } else { self.flags & !flag };
        }
        self.m.native16 = uses.native16;
        let mut resources = None;
        let mut bindings = Vec::new();
        let i32c = |s: &mut Self, v: u32| {
            let c = s.m.i32(v);
            s.m.md_value(c)
        };
        if self.user_data > 0
            || self.samplers > 0
            || !self.root_cbvs.is_empty()
            || !self.root_srvs.is_empty()
            || !self.root_uavs.is_empty()
            || self.stride_table.is_some()
        {
            let empty = self.m.md_str("");
            let mut cbvs = None;
            let mut cbv_list = Vec::new();
            if self.user_data > 0 {
                let arr = self.m.array(self.user_data.next_multiple_of(4), self.t.i32);
                let ty = self.m.named_struct("UserData", vec![arr]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let ops = vec![
                    Some(i32c(&mut self, 0)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, USER_DATA_SPACE)),
                    Some(i32c(&mut self, 0)),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, ud_size)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: USER_DATA_SPACE,
                    register: 0,
                    dxil_kind: 13,
                });
            }
            let blob = self.m.array(16384, self.t.i32);
            let ty = self.m.named_struct("RootCbv", vec![blob]);
            let p = self.m.ptr(ty);
            let u = self.m.undef(p);
            for (slot, _) in self.root_cbvs.clone() {
                let ops = vec![
                    Some(i32c(&mut self, cbv_list.len() as u32)),
                    Some(self.m.md_value(u)),
                    Some(empty),
                    Some(i32c(&mut self, ROOT_CBV_SPACE)),
                    Some(i32c(&mut self, u32::from(slot))),
                    Some(i32c(&mut self, 1)),
                    Some(i32c(&mut self, 65536)),
                    None,
                ];
                cbv_list.push(Some(self.m.md_node(ops)));
                bindings.push(Binding {
                    kind: 2,
                    space: ROOT_CBV_SPACE,
                    register: u32::from(slot),
                    dxil_kind: 13,
                });
            }
            if !cbv_list.is_empty() {
                cbvs = Some(self.m.md_node(cbv_list));
            }
            let mut srvs = None;
            if !self.root_srvs.is_empty() || self.stride_table.is_some() {
                let mut specs: Vec<(u32, u32)> = self
                    .root_srvs
                    .iter()
                    .map(|&(slot, _)| (ROOT_CBV_SPACE, u32::from(slot)))
                    .collect();
                if self.stride_table.is_some() {
                    specs.push((STRIDE_SPACE, 0));
                }
                let ty = self
                    .m
                    .named_struct("struct.ByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for (n, (space, reg)) in specs.into_iter().enumerate() {
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, space)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(i32c(&mut self, 0)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 4,
                        space,
                        register: reg,
                        dxil_kind: 11,
                    });
                }
                srvs = Some(self.m.md_node(list));
            }
            let mut uavs = None;
            if !self.root_uavs.is_empty() {
                let shared = self.m.bool(self.root_uavs_shared);
                let shared_md = self.m.md_value(shared);
                let ty = self
                    .m
                    .named_struct("struct.RWByteAddressBuffer", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let f = self.m.bool(false);
                let f_md = self.m.md_value(f);
                let mut list = Vec::new();
                for (n, &(slot, _)) in self.root_uavs.clone().iter().enumerate() {
                    let reg = u32::from(slot);
                    let ops = vec![
                        Some(i32c(&mut self, n as u32)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, ROOT_CBV_SPACE)),
                        Some(i32c(&mut self, reg)),
                        Some(i32c(&mut self, 1)),
                        Some(i32c(&mut self, 11)),
                        Some(shared_md),
                        Some(f_md),
                        Some(f_md),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 7,
                        space: ROOT_CBV_SPACE,
                        register: reg,
                        dxil_kind: 11,
                    });
                }
                uavs = Some(self.m.md_node(list));
            }
            let mut samplers = None;
            if self.samplers > 0 {
                let ty = self.m.named_struct("struct.SamplerState", vec![self.t.i32]);
                let p = self.m.ptr(ty);
                let u = self.m.undef(p);
                let mut list = Vec::new();
                for n in 0..self.samplers {
                    let kind = (self.cmp_samplers >> n & 1) as u32;
                    let ops = vec![
                        Some(i32c(&mut self, n)),
                        Some(self.m.md_value(u)),
                        Some(empty),
                        Some(i32c(&mut self, SAMPLER_SPACE)),
                        Some(i32c(&mut self, n)),
                        Some(i32c(&mut self, 1)),
                        // Sampler kind: 1 comparison.
                        Some(i32c(&mut self, kind)),
                        None,
                    ];
                    list.push(Some(self.m.md_node(ops)));
                    bindings.push(Binding {
                        kind: 1,
                        space: SAMPLER_SPACE,
                        register: n,
                        dxil_kind: 14,
                    });
                }
                samplers = Some(self.m.md_node(list));
            }
            resources = Some(self.m.md_node(vec![srvs, uavs, cbvs, samplers]));
            // PSV0 lists them as the module does: CBVs, samplers, SRVs, UAVs.
            bindings.sort_by_key(|b| match b.kind {
                2 => 0,
                1 => 1,
                4 => 2,
                _ => 3,
            });
        }
        if let Some(r) = resources {
            self.m.named_md("dx.resources", vec![r]);
        }

        let graphics = self.threads.is_none() || self.mesh.is_some();
        if graphics {
            // shortcut: every output is declared to depend on every input; the
            // exact sets need a dependence pass. Drivers read this state.
            let scalars = |els: &[SigElement]| {
                els.iter()
                    .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                    .max()
                    .unwrap_or(0)
            };
            let (ni, no) = (scalars(&self.inputs), scalars(&self.outputs));
            let mut view = vec![ni, no];
            // Mesh shaders have no inputs: then the primitive outputs' count.
            if self.mesh.is_some() {
                view.push(scalars(&self.prims));
            }
            for _ in 0..ni {
                for k in 0..no.div_ceil(32) {
                    let bits = (no - 32 * k).min(32);
                    view.push(if bits == 32 {
                        u32::MAX
                    } else {
                        (1 << bits) - 1
                    });
                }
            }
            // Geometry shaders list four streams; ours use the first.
            if self.rectangles {
                view.extend([0; 3]);
            }
            let arr = self.m.array(view.len() as u32, self.t.i32);
            let elems: Vec<V> = view.iter().map(|&v| self.m.i32(v)).collect();
            let arr = self.m.aggregate(arr, &elems);
            let arr = self.m.md_value(arr);
            let view_node = self.m.md_node(vec![Some(arr)]);
            self.m.named_md("dx.viewIdState", vec![view_node]);
        }

        let sigs = if graphics {
            let ins = self.signature_md(self.inputs.clone());
            let outs = self.signature_md(self.outputs.clone());
            let prims = self.signature_md(self.prims.clone());
            Some(self.m.md_node(vec![ins, outs, prims]))
        } else {
            None
        };
        // A pixel shader with an SV_StencilRef output (as DXC collects it).
        if self.pixel && self.outputs.iter().any(|o| o.kind == 20) {
            self.flags |= flags::STENCIL_REF;
        }
        // Entry properties: shader flags (tag 0), thread-group size (4),
        // wave size (11).
        let mut props = Vec::new();
        if self.flags != 0 {
            props.push(i32c(&mut self, 0));
            let i64t = self.m.int(64);
            let f = self.m.int_const(i64t, self.flags as i64);
            props.push(self.m.md_value(f));
        }
        if let Some(threads) = self.threads {
            let t = threads.map(|n| Some(i32c(&mut self, n)));
            let t = self.m.md_node(t.to_vec());
            match self.mesh {
                // Mesh state (tag 9): threads, maximum vertices and
                // primitives, topology, payload bytes.
                Some(mesh) => {
                    props.push(i32c(&mut self, 9));
                    let state = [
                        mesh.max_vertices,
                        mesh.max_primitives,
                        mesh.topology.into(),
                        0,
                    ]
                    .map(|n| Some(i32c(&mut self, n)));
                    let mut node = vec![Some(t)];
                    node.extend(state);
                    props.push(self.m.md_node(node));
                }
                None => {
                    props.push(i32c(&mut self, 4));
                    props.push(t);
                }
            }
        }
        // Geometry state (tag 1): triangles in, six vertices at most, stream
        // 0, triangle strips out, one instance.
        if self.rectangles {
            props.push(i32c(&mut self, 1));
            let state = [3, RECTANGLE_VERTICES, 1, 5, 1].map(|n| Some(i32c(&mut self, n)));
            props.push(self.m.md_node(state.to_vec()));
        }
        if let Some(n) = self.wave_size {
            props.push(i32c(&mut self, 11));
            let n = Some(i32c(&mut self, n));
            props.push(self.m.md_node(vec![n]));
        }
        let props =
            (!props.is_empty()).then(|| self.m.md_node(props.into_iter().map(Some).collect()));
        let main = self.m.md_value(self.main);
        let name = self.m.md_str("main");
        let entry = self
            .m
            .md_node(vec![Some(main), Some(name), sigs, resources, props]);

        let ident = self.m.md_str("victory");
        let ident = self.m.md_node(vec![Some(ident)]);
        self.m.named_md("llvm.ident", vec![ident]);
        let one = i32c(&mut self, 1);
        let six = i32c(&mut self, 6);
        let version = self.m.md_node(vec![Some(one), Some(six)]);
        self.m.named_md("dx.version", vec![version]);
        self.m.named_md("dx.valver", vec![version]);
        // Program kinds: 0 pixel, 1 vertex, 2 geometry, 5 compute, 13 mesh.
        let (name, kind) = match (self.threads, self.pixel) {
            _ if self.mesh.is_some() => ("ms", 13u32),
            _ if self.rectangles => ("gs", 2),
            (Some(_), _) => ("cs", 5u32),
            (None, true) => ("ps", 0),
            (None, false) => ("vs", 1),
        };
        let name = self.m.md_str(name);
        let sm = self.m.md_node(vec![Some(name), Some(six), Some(six)]);
        self.m.named_md("dx.shaderModel", vec![sm]);
        self.m.named_md("dx.entryPoints", vec![entry]);

        self.m.single_latches();
        let bitcode = self.m.write();
        let mut dxil = Vec::with_capacity(24 + bitcode.len());
        dxil.extend((kind << 16 | 0x66).to_le_bytes());
        dxil.extend((((24 + bitcode.len()) / 4) as u32).to_le_bytes());
        dxil.extend(b"DXIL");
        dxil.extend(0x106u32.to_le_bytes());
        dxil.extend(16u32.to_le_bytes());
        dxil.extend((bitcode.len() as u32).to_le_bytes());
        dxil.extend(&bitcode);

        let mut features = 0u64;
        if self.flags & flags::UAVS_EVERY_STAGE != 0 {
            features |= 0x4;
        }
        if self.flags & flags::DOUBLES != 0 {
            features |= 0x1;
        }
        if self.flags & flags::DOUBLE_EXT != 0 {
            features |= 0x20;
        }
        if self.flags & flags::STENCIL_REF != 0 {
            features |= 0x200;
        }
        if self.flags & flags::TYPED_UAV_LOADS != 0 {
            features |= 0x800;
        }
        if self.flags & flags::WAVE_OPS != 0 {
            features |= 0x4000;
        }
        if self.flags & flags::INT64 != 0 {
            features |= 0x8000;
        }
        if self.flags & flags::NATIVE_LOW_PRECISION != 0 {
            features |= 0x4_0000;
        }
        if self.flags & flags::BARYCENTRICS != 0 {
            features |= 0x2_0000;
        }
        if self.flags & flags::RESOURCE_HEAP != 0 {
            features |= 0x200_0000;
        }
        if self.flags & flags::SAMPLER_HEAP != 0 {
            features |= 0x400_0000;
        }
        if self.flags & flags::ATOMIC_TYPED64 != 0 {
            features |= 0x40_0000;
        }
        if self.flags & flags::ATOMIC_INT64_GROUP_SHARED != 0 {
            features |= 0x80_0000;
        }
        if self.flags & flags::ATOMIC_HEAP64 != 0 {
            features |= 0x1000_0000;
        }
        let sfi0 = features.to_le_bytes();
        let isg1 = container::signature(&self.inputs, true);
        let osg1 = container::signature(&self.outputs, false);
        let flag = if self.pixel {
            self.outputs.iter().any(|o| (17..=19).contains(&o.kind))
        } else {
            self.outputs.iter().any(|o| o.kind == 3)
        };
        let psv0 = container::psv(&Psv {
            stage: kind as u8,
            flag: flag && self.threads.is_none(),
            inputs: &self.inputs,
            outputs: &self.outputs,
            prims: &self.prims,
            mesh: self.mesh.map(|m| (m, self.lds_dwords * 4)),
            rectangles: self.rectangles,
            bindings: &bindings,
            threads: self.threads.unwrap_or_default(),
            wave_size: self.wave_size,
        });
        let hash = [0u8; 20];
        let psg1 = container::signature(&self.prims, false);
        let mut parts: Vec<(&[u8; 4], &[u8])> =
            vec![(b"SFI0", &sfi0), (b"ISG1", &isg1), (b"OSG1", &osg1)];
        if self.mesh.is_some() {
            parts.push((b"PSG1", &psg1));
        }
        parts.extend([(b"PSV0", &psv0[..]), (b"HASH", &hash), (b"DXIL", &dxil)]);
        container::build(&parts)
    }

    /// The user-data buffer's annotation carries its size, known only now.
    fn fix_user_data_size(&mut self, size: u32) {
        let old = [self.m.i32(13), self.m.i32(u32::MAX)];
        let old = self.m.aggregate(self.t.props, &old);
        let new = [self.m.i32(13), self.m.i32(size)];
        let new = self.m.aggregate(self.t.props, &new);
        self.m.replace_const(old, new);
    }

    fn signature_md(&mut self, elements: Vec<SigElement>) -> Option<Md> {
        if elements.is_empty() {
            return None;
        }
        let mut list = Vec::new();
        for (id, e) in elements.iter().enumerate() {
            let c32 = |s: &mut Self, v: u32| {
                let c = s.m.i32(v);
                Some(s.m.md_value(c))
            };
            let c8 = |s: &mut Self, v: u8| {
                let c = s.m.i8(v);
                Some(s.m.md_value(c))
            };
            let idx = c32(self, e.index);
            let indices = Some(self.m.md_node(vec![idx]));
            let usage = if e.usage != 0 {
                let (tag, mask) = (c32(self, 3), c32(self, u32::from(e.usage)));
                Some(self.m.md_node(vec![tag, mask]))
            } else {
                None
            };
            let ops = vec![
                c32(self, id as u32),
                Some(self.m.md_str(&e.name)),
                c8(self, e.comp),
                c8(self, e.kind),
                indices,
                c8(self, e.interp),
                c32(self, 1),
                c8(self, e.cols),
                c32(self, e.row.unwrap_or(u32::MAX)),
                c8(self, if e.row.is_some() { e.col } else { u8::MAX }),
                usage,
            ];
            list.push(Some(self.m.md_node(ops)));
        }
        Some(self.m.md_node(list))
    }
}

#[cfg(test)]
mod tests {
    use super::super::bitcode::Term;
    use super::*;

    fn finish_with_root_uav(pixel: bool, threads: Option<[u32; 3]>, mesh: bool, with_uav: bool) -> u64 {
        let mut s = Shader::new(pixel);
        s.threads = threads;
        if mesh {
            s.mesh = Some(Mesh {
                max_vertices: 1,
                max_primitives: 1,
                topology: 2,
            });
        }
        if with_uav {
            let u = s.root_uav_handle(0, false);
            let b = s.entry;
            let (or, zero, one) = (s.m.i32(2), s.m.i32(0), s.m.i32(1));
            let iu = s.m.undef(s.t.i32);
            let i32t = s.t.i32;
            s.op(b, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &[u, or, zero, iu, iu, one]);
        }
        let b = s.entry;
        s.m.terminate(b, Term::Ret);
        let dxil = s.finish();
        // Read SFI0 feature flags chunk from the DXBC container.
        let count = u32::from_le_bytes(dxil[28..32].try_into().unwrap()) as usize;
        for i in 0..count {
            let off = u32::from_le_bytes(dxil[32 + i * 4..36 + i * 4].try_into().unwrap()) as usize;
            if &dxil[off..off + 4] == b"SFI0" {
                return u64::from_le_bytes(dxil[off + 8..off + 16].try_into().unwrap());
            }
        }
        panic!("missing SFI0 chunk");
    }

    #[test]
    fn uavs_outside_pixel_and_compute_declare_feature_bit() {
        // Vertex and mesh shaders with declared root UAVs set bit 0x4 (UAVS_EVERY_STAGE).
        assert_ne!(finish_with_root_uav(false, None, false, true) & 0x4, 0);
        assert_ne!(finish_with_root_uav(false, Some([32, 1, 1]), true, true) & 0x4, 0);
        // Pixel, compute, and UAV-less vertex shaders leave bit 0x4 clear.
        assert_eq!(finish_with_root_uav(true, None, false, true) & 0x4, 0);
        assert_eq!(finish_with_root_uav(false, Some([64, 1, 1]), false, true) & 0x4, 0);
        assert_eq!(finish_with_root_uav(false, None, false, false) & 0x4, 0);
    }

    #[test]
    fn unused_packed_input_matches_used_sibling_interp_mode() {
        use super::super::container::SigElement;
        let mut s = Shader::new(true);
        s.inputs = vec![
            SigElement {
                name: "TEXCOORD".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 2, // linear originally
                cols: 2,
                row: Some(0),
                col: 0,
                usage: 0, // unused
                half: false,
            },
            SigElement {
                name: "NORMAL".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 1, // constant for attributeAtVertex
                cols: 2,
                row: Some(0),
                col: 2,
                usage: 3, // used
                half: false,
            },
        ];
        let b = s.entry;
        s.m.terminate(b, Term::Ret);
        s.align_packed_pixel_inputs();
        assert_eq!(s.inputs[0].interp, 1);
        assert_eq!(s.inputs[1].interp, 1);
        let dxil = s.finish();
        assert!(!dxil.is_empty());
    }

    #[test]
    fn typed_uav_load_flags_follow_dce() {
        use super::super::bitcode::Term;
        use crate::dxbc::{Container, FourCC};

        for (texture, store, read) in [
            (false, false, false),
            (false, true, false),
            (false, true, true),
            (true, false, false),
            (true, true, false),
            (true, true, true),
        ] {
            let mut s = Shader::new(false);
            s.threads = Some([1, 1, 1]);
            if !read {
                s.flags |= flags::TYPED_UAV_LOADS;
            }
            let b = s.entry;
            let zero = s.m.i32(0);
            let iu = s.m.undef(s.t.i32);
            let kind = if texture { 2 } else { 10 };
            let h = s.heap_handle(b, zero, false, false, (kind | 0x1000, 0x409));
            let (load, load_op, load_args) = if texture {
                ("textureLoad.f32", 66, vec![h, iu, zero, zero, iu, iu, iu, iu])
            } else {
                ("bufferLoad.f32", 68, vec![h, zero, iu])
            };
            let r = s.op(b, load, Attr::ReadOnly, s.t.res_ret, load_op, &load_args);
            if store {
                let v = if read { s.m.extract(b, r, 0) } else { s.m.f32(0) };
                let mask = s.m.i8(0xF);
                let (name, op, args) = if texture {
                    ("textureStore.f32", 67, vec![h, zero, zero, iu, v, v, v, v, mask])
                } else {
                    ("bufferStore.f32", 69, vec![h, zero, iu, v, v, v, v, mask])
                };
                s.op(b, name, Attr::NoUnwind, s.t.void, op, &args);
            }
            s.m.terminate(b, Term::Ret);
            let case = format!("{load}, store={store}, read={read}");
            assert_eq!(s.m.written_calls(&format!("dx.op.{load}")), usize::from(read), "{case}");
            let dxil = s.finish();
            let container = Container::parse(&dxil).unwrap();
            let part = container.part(FourCC::SFI0).unwrap();
            let features = u64::from_le_bytes(part.data.try_into().unwrap());
            assert_eq!(features & 0x800 != 0, read, "{case}: typed UAV loads");
        }
    }

    #[test]
    fn a_64_bit_atomic_on_an_untaken_path_declares_no_atomic64_features() {
        use super::super::bitcode::Term;
        use crate::dxbc::{Container, FourCC};

        let mut s = Shader::new(false);
        s.threads = Some([1, 1, 1]);
        s.flags |= flags::ATOMIC_TYPED64 | flags::ATOMIC_HEAP64;
        let (entry, live, dead) = (s.entry, s.m.new_block(), s.m.new_block());
        let yes = s.m.bool(true);
        s.m.terminate(entry, Term::CondBr(yes, live, dead));
        s.m.terminate(live, Term::Ret);
        s.m.terminate(dead, Term::Ret);
        let zero = s.m.i32(0);
        let h = s.heap_handle(dead, zero, false, false, (2 | 0x1000, 0x105));
        let i64t = s.m.int(64);
        let (or, iu, one) = (s.m.i32(2), s.m.undef(s.t.i32), s.m.int_const(i64t, 1));
        s.op(dead, "atomicBinOp.i64", Attr::NoUnwind, i64t, 78, &[h, or, iu, iu, iu, one]);
        let dxil = s.finish();
        let container = Container::parse(&dxil).unwrap();
        let part = container.part(FourCC::SFI0).unwrap();
        let features = u64::from_le_bytes(part.data.try_into().unwrap());
        assert_eq!(features & 0x1040_0000, 0, "untaken atomic64 flags leaked: {features:#x}");
    }
}
