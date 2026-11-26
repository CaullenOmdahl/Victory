# VIctory fork for AMD Radeon & Linux/Proton (work in progress)

This fork ([CaullenOmdahl/Victory](https://github.com/CaullenOmdahl/Victory)) builds on [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) and [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (plus selected fixes from upstream PRs, `DexNom/VIctory`, and `fxceless/victory-fork`) to run **Grand Theft Auto VI (Xbox Series X 1.0.2.1)** on **Linux under Proton (`vkd3d-proton` + RADV)**.

**This repository contains no game files, code or assets.** You need your own
copy of the game. It is not affiliated with or endorsed by the game's
developer or publisher.

## Linux / Proton Status (`CaullenOmdahl/Victory`, 2026-11-26)

Tested on: **CachyOS Linux (`7.3.4-1-cachyos`), Proton (`proton-patched`, `vkd3d-proton` + Mesa RADV), AMD Radeon RX 6800 16 GB (RDNA2), Ryzen 9 5900X (12C/24T), 62 GiB RAM, Series X profile.**

| What | State on CachyOS + Proton (RX 6800) |
|------|-------------------------------------|
| Boot, menus | 22–30 fps |
| In-world rendering | Renders lit open world, characters, textures, water, sky, and geometry |
| Frame rate in the world | **2.1–4.8 fps** (`206.3–538.1 ms/frame` outdoors, up to **`22.9 fps` / `43.6 ms/frame`** indoors; Loops 105–107 added title frame-rate/resolution unlocks (`722c0e1`/`f63b4b1`), `victory_shader::SOURCE_HASH` cache keying, and offline 24T DXIL + Proton `victory warm --used` prewarming (`4.3 fps` outdoors / `15.2 fps` indoors); Loop 108 added `VICTORY_COPY_BACK_BESIDE` (`#3069`, **`4.8 fps` / `206.3 ms/frame`** at `~5,000` draws/frame, `21.1–24.5 ms/frame` copy-back); Loop 109 sharded `watch::JOURNALS` into 64 aligned `RwLock` stripes and deduplicated 4 KiB `GetWriteWatch` pages into 64 KiB pages (`4.52 avg_fps` overall, **`22.9 fps` / `43.6 ms/frame`** indoors, `0` GPUVM/page faults, crisp `frame-26124` 1080p capture); Loop 110 added opt-in `VICTORY_TEXTURE_BACK=1` (`texback.rs`, tiling shader-written ocean/displacement UAV textures back to title memory cleanly with `0` GPUVM/page faults); Loops 111–112 landed Batch G2 shader precision/correctness fixes (`#3159` directed f16 rounding, `#3155` negative-offset `CODE` descriptors, `#3127` cube-map subnormal/zero semantics, `#2160` near-unity `v_log_f32`), pre-warmed `25,694` DXIL shaders (`2,123` pipeline recipes), and held `0` GPUVM/page faults in the prison-window world run (`1,727` pipelines loaded, `frame-6067` clean); Loop 113 added `xmem::map` placeholder and section-commit caching (`fxceless #1103`) and made `VICTORY_COPY_BACK_BESIDE` default-on (`863.4–900.9 writes/frame`, `3.5 fps` / `289.4 ms/frame` at `5,098` draws/frame, `1,855` pipelines loaded, `0` faults, `frame-5964` clean); Loop 114 added SDWA 16-bit `UNUSED_SEXT` sign-extension (`#3124`) and `R9G9B9E5` format family mapping (`#2832`) (`25,694` DXIL cached under key `2b0b236b3a9acaee`, `1,909` pipelines loaded, `0` faults, `frame-5804` clean); Loop 115 converted `xmem::PHYSICAL` (with non-blocking `try_read()` fallback in `write_beside`) and `watch::VIEWS` to `RwLock`, eliminating streaming copy-back lock stalls (`4,378.9 ms -> 128.1 ms` peak copy-back, **`2.34 avg_fps`** / `631.9 ms` average and **`4.0 fps` / `250.4 ms/frame`** steady-state at `5,082` draws/frame with `28.6 ms` copy-back, `2,039` pipelines loaded, `0` faults, `frame-5746` clean)) |
| Stability | In-world runs complete without `0x887A0005` GPUVM faults once Proton null-buffer guards, remap invalidations, submission-held page mappings (`#2303`), >=4 MiB whole-reservation guards (`#153`), fence-retired sparse sweeps (`#194`), heap sentinel clamps (`#144`), dynamic descriptor cursor wrapping (`#201`), private/free memory guard exclusions (`#1147`), aliasing/UAV barrier resource retention (`#2298`), copy-back/remap coherency (`#2299`, `#3031`/`#3036`, `#3037`), list-reset OOM spare fallback (`#3002`), pipeline-worker panic recovery (`#219`), CBV 64 KiB clamping (`#1260`), `LoaderLock` unwind guard (`#3001`), HID NUL termination (`#207`), and `USERPTR` gating landed |
| Memory | ~5.6 GiB commit (~10.9 GiB title private, ~19.9 GiB mapped) |

### Additions over `Belobug/VIctory-Belobug-AMD`
- **Linux / Proton (`vkd3d-proton` + RADV) fixes:**
  - Page-aligned `vkMapMemory` view offsets, batched `UpdateTileMappings`, and null-blob `ID3D12PipelineLibrary` support under Proton.
  - Gated `OpenExistingHeapFromAddress1` (`direct()`) behind `VICTORY_IN_PLACE_WINE=1` on Wine (`MEM_PRIVATE` round-trips, `MEM_MAPPED` reads zeroes, and host-RAM `USERPTR` slows RX 6800 GPU execution ~20× vs VRAM copies).
  - Added `VirtualFree`/`VirtualProtect`/`XMemFreeDefault` remap & full 32-bit `XMem` release invalidations (`#2824`), submission-held page mapping copies (`#2303`), whole-reservation `>= 4 MiB` boundary (`#153`), write-combining tracking on `MEM_COMMIT`, buffer/view span clamping, buffer SRV/UAV range clamping (`#1266`) & 16-byte aligned raw buffer shadow lookups (`#3075`), `HostHeap::stride_table_resource` list lifetime retention (`#2543`), null CBV unmapped-slot skip (`#304`), 64 KiB CBV size cap with 0-byte CBV preservation (`#1260`), null root & null shadow guards, guest list `Close`/`Reset` state validation (`#2979`), `ResourceBarrier` no-op/chain coalescing (`#3129`), `R9G9B9E5` format family mapping (`#2832`), `CopyTextureRegion` subresource/3D-slice/box/BC-alignment validation (`#3194`), `NumRenderTargets <= 8` clamping & unused `RTFormats` clearing (`#3164`), planar depth/stencil subresources (`#3022`/`#3057`) & whole single-plane `D16`/`D32` depth readbacks (`#3006`), linearized 1D indirect mesh group clamping (`#2996`), non-target clear ignore & failed `PLACED_TEXTURES` skip (`#1422`), heap index sentinel clamping (`#144`), dynamic descriptor cursor wrapping (`#201`), aliasing/UAV barrier host-resource retention (`#2298`), `watch::writing` before DirectStorage writes (`#3037`), DirectStorage section-view retry (`#2883`), DirectStorage file reference retention (`#2987`), `CopyMemoryToMemoryX` unmapped/read-only hardening (`#2299`), `XMemMapPhysicalPages` partial-run rollback (`#2742`), `xmem::map` placeholder and section-commit caching (`fxceless #1103`), atomic `map_alias` source retention before destination unmap (`#2199`), freed XMem physical page number recycling (`#1935`), device-removal `u64::MAX` fence sentinel guard (`Belobug #249`), shadow copy-back/remap coherency (`#3031`/`#3036`), deferred post-copy-back CPU copy ordering on shader-written ranges (`#2810`), list-reset OOM spare recovery (`#3002`), volume `depth_pitch` bounds checking (`#21`), fence-retired sparse tile sweeps (`#194`), snapshot pool OOM retry (`#2153`), optional `VICTORY_SECTION_MB` chunked physical sections (`#1274`), opt-in `VICTORY_CONTINUE_INT3` (`#2165`), NUL-terminated DualSense HID path handling (`#207`), lazy Wine `ws2_32` export resolution (`#3091`), present-copy wait before probe/capture readback (`#197`), early-pipeline list save, shared atomic frame pipeline-wait budget (`#2111`) with single-frame wait limit (`#1234`), priority/depth-only pipeline queues + `worker_lost` wake (`#2897`/`#229`), pipeline-worker panic catch (`#219`), `LoaderLock` unwind `Drop` guard (`#3001`), and 64 MB compiler worker stacks with stuck-compile recovery.
- **CPU, watch, bindless & GPU speedups:**
  - Persistent 20-thread `in_parallel` pool, physical-core `dstorage` worker pool sizing (`#1044`) & reusable `DecompressorOxide`/scratch buffers (`#2981`), const-generic `bcpack` lane decode with thread-local `RECORDS` reuse, `SPREAD4` LUT (`#2735`), and completed-lane byte-theft guard (`#2648`), AVX2/SSE2 256B micro-tile texture detiler with `DetilePool` (`#195`), 4-slot LRU detile cache (`32 MiB` budget) for oscillating textures, partial touched-row texture detiling & uploads (`#2985`), `ResourceMinLODClamp` mip skipping (`#2149`), packed linear slice-at-a-time texture copies (`#1201`), and 16 compiler workers with compact lazy `Halves` `Vec<Def>` in `io::raise` (`#2784`), sparse single-pass `propagate_copies` (`#2785`), zero-alloc IR/fields (`#2664`), `USER_DATA` CBV row cache (`#2207`), `O(1)` `def_block`/`def_site`/`done_export` (`#2210`), `DenseMap`/`FxMap` lowering tables + `round16`/`valu16`/buffer fast paths (`#2942`), and shared `Fx` hasher across shader passes (`#2399`); `RwLock` conversion on `GUARDS`, `JOURNALS`, `VIEWS`, `xmem::PHYSICAL` (with non-blocking `try_read()` fallback in `write_beside`), `SHADER_HEAPS` (`#2848`), and descriptor `VIEWS` (split/reduced lock churn `#2848`); 64-MiB-striped sharded `GetWriteWatch` locks (`WRITE_WATCH_LOCKS`) and matching 64-sharded `JOURNALS` with 4 KiB-to-64 KiB page deduplication (`writes()`/`writes_guarded()`/`title_memory_written()`), `64 KiB` per-block region epochs with lock-free per-page `page_epochs` watch-skipping, `KNOWN_HOLES` bulk-read pre-skip, keeping free/committed private memory out of watch guards (`#1147`), cached `MemoryRegionInfo` fallback & `VirtualQuery` multi-block hole-run skipping (`Belobug #247`), pin cleanup on remap, 8 MiB guarded chunks, and disjoint/parallelized `Batch` refresh spans.
  - Default-on ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` (`VICTORY_GPU_UPLOAD=0` opt-out, +45% FPS on RX 6800 16 GiB within 85% VRAM budget), default-on hair strands & simulation shaders (`VICTORY_HAIR=0` opt-out), overlapped next-submission preparation (`#215`) skipping `prewarm` once the predecessor fence has completed (`Belobug #232`), parallel barrier prelude recording (`VICTORY_PARALLEL_PRELUDES`, `#2908`), parallel upload fills (`#2900`), `FxHash` per-list/submission sets (`#2102`), `Deferred::clear()` command-list capacity reuse with loading-outlier shrink (`#1059`/`#2972`), 256 KiB configurable loose shadow windows (`VICTORY_LOOSE_WINDOW_KIB`, `#2088`), loose-window reuse (`#2856`), idle loose-window eviction (`VICTORY_WINDOW_IDLE_SECS`, `#3078`), partial-page `clean_spans` epoch caching in `update_guarded_pages`, `VICTORY_FRAME_INTERVAL_MS`/`VICTORY_TITLE_FPS` byte-checked title frame-rate & `SimRate` unlock (`722c0e1`) with `VICTORY_TIMER_LOG=1`, `VICTORY_OUTPUT` / `VICTORY_RENDER_SCALE` / `VICTORY_UPSCALER` byte-checked output size & render scale overrides (`f63b4b1`), `victory_shader::SOURCE_HASH` build-time shader/pipeline cache keying (`build.rs`, preserving all `2,028` cached DXIL translations and pipeline libraries across non-shader rebuilds), offline 24-thread DXIL warming (`victory translate --cache [--cache-dir] [--no-native16] [--portable-views]`) and Proton headless pipeline warming (`victory warm --used` backed by serialized `GraphicsState` + `used-pipelines.bin` recipes), thread-local buffer lookup cache (`#1237`/`#1245`), render-target bind cache (`VICTORY_BIND_CACHE`) with immediate indirect-clamp invalidation (`Belobug #251`), sharded `HostHeap::kept` locks, lock-free `made_now` & `shadow::has_gone` fast path (`#1146`), 64-bit word `Taken` dynamic descriptor slot bitmap (`fxceless #1297`), lock-ordered `record_bindless` contention skip & single-pass retain (`Belobug #256`), cached alias-copy footprint plans (`#1888`), shared 512B-aligned 64 KiB–4 MiB per-list texture readback chunks (`#2818`), shadow lookup caching (`#46`), wave-coalesced 64 KiB-page GPU shader write-marks bitmap (`VICTORY_WRITE_MARKS`, cutting sparse snapshots 13.7 GB to 0.0–1.3 MB and copy-back to 16–46.5 ms/frame), skip read-only root UAV snapshots & copy-back (`#2948`), default-on persistent unguarded physical-page copy-back (`VICTORY_COPY_BACK_BESIDE=0` opt-out, `Marvisak #3069`, cutting copy-back to `21.1–37.1 ms/frame` without `VirtualProtect` cycles), opt-in shader-written texture readback & console retiling (`VICTORY_TEXTURE_BACK=1`, `texback.rs`, for ocean height/displacement UAV textures), opt-in write-combining copy-back skip (`VICTORY_SKIP_WC_BACK`), opt-in clean private page write-watch reset after copy-back (`VICTORY_BACK_QUIET`, `fxceless #1188`), dirty-generation texture hash bypass (`xmem` persistent views), incremental bindless planning, inline constant and placed view records (`#2099`), static per-stage `TableLayout` & dead `merge_gpu_writes` removal (`#1192`), uncapped-heap table overflow scan skip (`#3061`), shared dynamic descriptor tables across draw stages (`VICTORY_TABLE_STAGES`), lock-free descriptor heap gets, stack root slots, cached `bind_pipeline`, opt-in target-format pipeline variants (`VICTORY_TARGET_VARIANTS`), hoisted & wave-coalesced draw-run indirect gather clamping (`#1171`) with device-local budget, Series X FidelityFX upscaler & render scale support (`#1793`), DirectStorage cross-queue staging bypass and physical-page read deferral on half-full auto-submits, and thin LTO (`codegen-units = 1`) release builds.
- **Shader & rendering correctness:**
  - Ported upstream normalized buffer exact quotient rounding (`da82e444`), render-target XOR `SW_64KB_R_X` addressing (`#2879`), resolve subresource formats (`#2325`), 3D `DispatchMesh` (`#2624`), `bit_exact_format` 32-bit int typed buffer views (`#684`), whole NUL-delimited `PSV0` semantic names (`#3088`), scalar `VCC` `s_cbranch_vccz/vccnz` (`#2406`/`#3023`), `f64` reciprocal/`rsqrt` via 32-bit DXIL ops (`#2799`/`#2811`), flat quad compute rows (`#2999`), `v_readlane`/`v_writelane` lane masking & `EXEC` bypass (`#1312`), quad-local pixel `DPP8` (`#2817`), `TG_SIZE_EN`/`s_get_waveid_in_workgroup` (`#551`), single-lane `v_readfirstlane_b32` (`#2831`), pointer-jumping wave prefix lowering for float sums (`#1408`) & non-sum `Min`/`Max`/`And`/`Or`/`Xor` ops (`#2776`), DXIL `Ubfe`/`Ibfe` intrinsics (`#2954`), RDNA2 legacy `v_mad_legacy_f32` FTZ/`0 * NaN = 0` lowering (`#1026`), POPS pixel-shader root UAV ROVs (`#2490`), `ds_{inc,dec,rsub,mskor}` LDS atomic tests (`#1281`), `v_fmac_f32`/`v_mac_f32`/`v_fmac_legacy_f32` (`#3158`) and `v_fmac_f16`/`v_mac_f16` (`#3192`) VOP3 accumulator `abs`/`neg` modifiers, `v_interp_*` EXEC guards (`#3153`), `v_pk_fmac_f16` inline float constants & literals (`#3148`), BC7 modes 4/5-only channel rotation (`#3111`), `UAVS_EVERY_STAGE` (`a1a2077a`), right-angle rectangle completion (`8ff79135`), sparse `any_mapped()` texture residency (`921d8d3b`), `CalculateLOD` spatial array fixes & packed pixel inputs (`5f4c3df4`), immediate sampler filter reduction bit clearing (`#2983`), mesh `ms_io` unused VS-input skip (`#187`), 32 KiB mesh/pixel shader output linking (`#138`), idle `ScreenProbes_ActivateRays` skip (`6986a571`), `WaveSize(64)` declaration on native wave64 compute shaders for RADV (`#204`), `packed16`/`fmac` modifier guards (`#3032`), integer `v_min3`/`v_max3`/`v_med3` DXIL intrinsics (`#3027`), 32-bit `mul_hi` without `INT64` (`#741`), RDNA2 `f16` output modifier ignore (`#2915`), `v_swaprel_b32` (`#1569`), SM 6.4 `dot4AddPacked.i32` lowering for `v_dot4_*` (`#808`), native `f16` arithmetic in `half` on supported hardware (`#2825`), `abs`/`neg` on `v_cvt_{u16,i16}_f16` (`#668`), SDWA 16-bit `UNUSED_SEXT` sign-extension (`#3124`), post-instruction `code_table` start (`#1006`), post-DCE `TYPED_UAV_LOADS`/`ATOMIC_TYPED64`/`ATOMIC_HEAP64` flag derivation (`#2610`, `#2698`), binary64 `ceil`/`floor`/`fract`/`trig_preop` (`#2713`, `#2728`, `#2988`), independent `s_buffer_load` dword truncation (`#1939`), cube `image_get_resinfo` 6-face depth (`#3010`), FTZ `floor`/`ceil`/`fract` (`#2968`), `d16` float truncation (`#2064`), `V_CVT_F16_F32` directed rounding modes (`#3159`), negative-offset `s_buffer_*` reads via built `CODE` descriptors (`#3155`), cube-map subnormal/signed-zero `v_cubesc/tc/ma_f32` semantics (`#3127`), near-unity 1-ULP `v_log_f32` (`#2160`), waterfall loop `SCC` (`#3016`), exact high dword of Wave64 opaque lane masks (`#2836`), canonical `dx.op.tertiary.f64` (`#202`), DXIL dead-code elimination, one-way branch pruning (`#172`), handle dedup, reachable-only liveness, VOP1/VOP3 CMPX EXEC, `v_cmp_f/tru` (`#173`), whole-wave EXEC (`#191`), scalar quadmask (`#192`), `v_cndmask_b32` (`#193`), `v_interp_p2_f16` (`#189`), `f16` RNE/FTZ `min/max/med3` (`#135`), raw resource-heap buffer loads (`#148`), `v_dot8_i32_i4` & cube `log2` (`#184`), divergent-loop Wave64 fixes, and BC7 dictionary entry 0 fix.

### Building & Running on Linux (with `cargo-xwin` + Proton)
```bash
cargo xwin build --profile diag --target x86_64-pc-windows-msvc -p victory --features diagnostics
cargo xwin build --release --target x86_64-pc-windows-msvc -p victory
```
Launch via Proton (`proton runinprefix`) with `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3dcompiler_47=n"` and `VKD3D_SHADER_CACHE_PATH` pointed outside the repository so `vkd3d-proton.cache.write` stays out of the working tree.

---

## Upstream Forkbase State (`Belobug/VIctory-Belobug-AMD`, 2026-11-25)

**On Belobug's Windows 11 test machine (3 to 5 frames per second):** Windows 11, AMD Radeon RX 9070 XT 16 GB, Ryzen 7 9800X3D, 64 GB RAM, Series X profile. Nothing here has been run on an NVIDIA or Intel card.

| What | State on that machine |
|------|-----------------------|
| Boot, menus | work, 30 fps (as upstream) |
| The world | renders lit, with textures, people, vehicles, vegetation, water and sky |
| Saves | load and store |
| Frame rate in the world | 3 to 5 fps (4.4 to 4.9 standing still in a heavy scene, about 3 while driving) |
| Light in shadow | a constant, less of it in creases and under objects: the game's ray-traced indirect light does not run. One level does not suit every hour (see the known problems) |
| Shadows of the sun and of lamps | the game's own, and they work; too much of the constant light drowns them |
| Distant trees | drawn |
| Vehicles in motion | whole (they lost and stretched parts until the copies of title memory saw writes through every view of a page) |
| In-game phone | **broken**: with the phone out the world is unlit (black) and the phone is an empty outline |
| Memory | about 22 GB of RAM and more video memory than a 16 GB card has |

A driver timeout (Windows resets the display driver, every monitor goes black
for a moment) happened once in the session this was written in. Expect it.

### What this fork changes over upstream

Picture:
- The world's indirect diffuse light is replaced by a constant while the ray
  passes do not run. What the game's resolve pass left instead was bright
  magenta on leaves, cloth and paper, and blocks of colours indoors
  (`VICTORY_AMBIENT=<level>`, 0.6 by default; `off` keeps the game's output).
  Its colour can be set (`VICTORY_AMBIENT_TINT=r,g,b`), and less of it
  reaches creases, corners and the underside of things, worked out from the
  scene's depth alone (`VICTORY_AMBIENT_SHADE=strength,radius`; grainy, it is
  not blurred yet). In diagnostics builds all of it can be tuned in a running
  game from a file (`VICTORY_AMBIENT_FILE`).
- Blending is off on integer render targets in host pipelines: the console
  ignores it there, D3D12 refuses the pipeline. One world pipeline the driver
  refused at every launch is made now (pull request #45, pgm1207 and Arvi23).
- In pixel shaders every read of another pixel of the same 2x2 quad goes
  through quad reads, helper pixels included. Wave reads return 0 from helper
  pixels on AMD, which showed as lines along every triangle on car paint,
  glass and skin. Upstream does this after discards only.
- A helper pixel's `v_readfirstlane` reads its own value (upstream `0601955`).
- Twice the dynamic descriptor slots, so the tables of heavy frames fit.
- Depth and stencil twins for depth textures written by compute, screen
  passes with pipeline workers of their own, textures told apart by the mips
  their memory really holds, and other fixes listed in the history.

Speed:
- The clamps of a run of indirect draws are recorded in a list before the
  list that draws, and run one after another (`VICTORY_HOIST_CLAMPS`, 7 by
  default, 0 for each before its draw): 225 to about 80 ms of GPU time per
  frame in a scene of 9,000 draws.
- The snapshots and the copy-back of the tiles shaders write are taken on
  several threads (pull request #18, PikPakPik): copying back went from
  about 43 to about 33 ms a frame here. The frame rate did not move.
- The title's submissions can return before the host GPU is done, with what
  they owe finished at the next wait (`VICTORY_ASYNC_SUBMIT=1`, off by
  default): stable here, and no faster, because the frame is bound by the
  processor's work, not by the wait.
- Shader cache entries of another build can be reused
  (`VICTORY_SHADER_BUILD_OF=<executable>`, diagnostics).

Memory the title shows at several addresses:
- Every view of a guarded physical page is guarded with it, so a write of
  the title's through another address is seen (the idea of victory-fork
  #115 and victory-amd, written anew here); blocks the kernel is about to
  write stay writable for two seconds. Before this, parts of moving
  vehicles were left behind; the distant trees' quads that crossed the
  screen have not come back since either (cause not proven).

AMD:
- Shaders that crash or never leave the AMD driver's compiler cost only
  their pipeline, not the run.
- Buffer element counts come from the host's own table, not a size query the
  driver's compiler crashes on.

Diagnostics (diagnostics builds): `VICTORY_SHADER_PATCH` changes one word of a
shader's machine code before translation, `VICTORY_SHADER_SWAP` replaces
shader containers by files, `VICTORY_ALWAYS_COPY=1` refreshes every copy of
title memory each frame, `VICTORY_GPU_TIMING=1` logs GPU time per pipeline.
Each is described in a comment where it is read.

### How it is run here

A diagnostics build (`cargo build --profile diag -p victory --features
diagnostics`), with:

```
VICTORY_PRESENT_TARGET=1x1:1
VICTORY_HAIR=1
victory run <game_scarlett_master.exe> --execute --console series-x
```

The first launch after a build translates every shader and builds about
2,500 pipelines; the loading screen can stay frozen, with sound, for minutes.

### Known problems

- With the in-game phone out the world is unlit (see the table).
- The per-draw descriptor slots are few for the heaviest places: when they
  run out, some draws of a frame are dropped.
- Memory: the process commits about 33 GB after a quarter of an hour's
  drive (the title's own 7 GB, 8 GB of textures on the card, copies of its
  buffers, and the system's backing of video memory).
- No ray-traced light; the constant that stands in for it is the same
  everywhere and at every hour. The default of 0.6 is too much at night (it
  fills the shadows the game's lamps cast and the picture goes flat) and too
  little at noon (what the sun does not reach is nearly black). Levels that
  looked right on this machine: 0.07 bluish at night, 0.3 warm at dusk and
  dawn, 1.4 sky blue at noon. Making it follow the game's sky is being
  worked on.
- Some compute programs do not translate (ray-tracing structure builders,
  wind volumes, a few others) and their passes are dropped. With
  `VICTORY_RAYS=1` the builders translate and run, and the GPU stops
  answering about 25 seconds later (device removed): do not expect light
  from it.
- About 380 draws of a frame are dropped in the world for want of a host
  pipeline: programs that do not translate, and six pipelines the AMD driver
  refuses (two compute, four mesh). Which of the two costs most is not
  measured.
- Windows 10 is reported not to get past start-up without a newer D3D12
  runtime (issue #9, pull request #44, not merged or tried here yet); this
  fork has only been run on Windows 11.
- One test fails on this machine, before and after this update:
  `list::tests::lists_are_cut_at_waits_that_follow_work`.
- `crates/victory-shader/src/emit/lower.rs` is larger than the size limit of
  `cargo xtask policy`; it is source code, not an asset.
- Many changes are specific to this title and this GPU, and some are marked
  `shortcut:` in the code where they approximate what the console does.

### Credits

- **[VIctory](https://github.com/Marvisak/VIctory)** by Marvisak and its
  contributors is the base of everything here. Licensed GPL-3.0-or-later; so
  is this fork. Upstream's history and the authors of the pull requests
  carried here are kept in the commit history.
- Pull requests merged into this fork, read line by line and run in the
  world before merging: #45 (pgm1207, sent by Arvi23) and #18 (PikPakPik).
  The others are still to be read; thank you for them and for the test
  reports.
- [fxceless/victory-fork](https://github.com/fxceless/victory-fork) and
  [myokex/victory-amd](https://github.com/myokex/victory-amd) were read for
  comparison; no code of theirs is in this fork.
- [XCL](https://github.com/ivall/xcl) by ivall: its README was read; no code
  of it is here.
- This fork's changes were developed with
  [Claude Code](https://claude.com/claude-code) and Gemini (Antigravity).

---

## From upstream's README

What follows is taken from upstream VIctory's own README: what the project
is, how to build it, its commands and its debugging switches. They apply to
this fork too. Upstream's status section is left out, because it describes
upstream's state and upstream's test machine (an RX 6700 XT), not this fork
or the machine above: read it
[in upstream's repository](https://github.com/Marvisak/VIctory#readme).

# VIctory

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. A DualSense or DualSense Edge (USB or Bluetooth) is read
over HID as well and plays as the first pad when no XInput pad answers. With
no pad connected the keyboard is a pad while the title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, O | A, B |
| Backspace, X, Y | B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--cache] [--cache-dir DIR] [--host-lanes MIN-MAX] [--list] [--jobs N]
victory warm [paths...] --used [--shader-cache-dir DIR] [--pipeline-cache-dir DIR]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not |
| `VICTORY_CRASH_BREAKPOINTS=1` | appends first-chance breakpoints to `crashes.log`, including ones a handler may recover from |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, the allocation hit and the commands in flight |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |

Every build appends unhandled exceptions, panics, and host GPU failures to
`crashes.log` in the title's data folder.

For a startup breakpoint (`0x80000003`), run the diagnostics build with
`VICTORY_CRASH_BREAKPOINTS=1`. Keep the console output along with the new
`crashes.log` entries. A host DLL refusal includes the caller stack in the
console output. The refusal is expected for DLLs outside the host policy,
including `rpcrt4.dll`.

On Series 1.0.2.1, a missing `appdata0.rpf` reproduced the breakpoint at
`+0x2237178` before the first window. Restore the archive from the same
package beside the title executable. `run --execute` checks that it can
read the RPF9 header before starting this build.

To compare hardware and software rendering, repeat with
`VICTORY_D3D12_WARP=1` and check the console for
`host device on the software adapter (WARP)` to confirm the WARP device was
created. Compare the exception's module and offset, stack frames, and
preceding messages across the two runs.

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
