# VIctory — Linux / Proton Fork (AMD Radeon)

Compatibility layer running **Grand Theft Auto VI (Xbox Series X 1.0.2.1)** on **Linux under Proton (`vkd3d-proton` + Mesa RADV)**, built on [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) and [Marvisak/VIctory](https://github.com/Marvisak/VIctory). Combines 160+ commits of Wine/Proton compatibility, lock-free host memory paths, and L0 VRAM placement with custom companion Wine and `vkd3d-proton` forks.

> **No game files, keys, executables, or downloads are included.** Supply your own copy of the game. Not affiliated with or endorsed by Rockstar Games, Take-Two, or Microsoft.

---

## Status at a Glance (`2026-11-27`)

**Test machine:** CachyOS Linux (`7.3.4-1-cachyos`) · Custom Proton (`vkd3d-proton` + Mesa RADV) · AMD Radeon RX 6800 16 GB (RDNA2) · Ryzen 9 5900X · 62 GiB RAM · Series X profile

| Metric | Current State (Linux + Proton, RX 6800) |
|--------|-----------------------------------------|
| **Boot & menus** | Works (**22–30 fps**) |
| **In-world rendering** | Lit open world, characters, textures, water, sky, and geometry (constant ambient fallback; no RT) |
| **Playable** | **Yes** — controller input, walking, and interactions work |
| **Story progress** | Opening scene only (not progressed further yet) |
| **World frame rate** | **7.3–7.5 fps** outdoors (`133.3–137.5 ms/frame` steady state across 5 intervals, `4.98 avg_fps`, ~4,920 draws/frame with Loop 145 `Belobug #503` write-watched graphics allocations + `VICTORY_INDIRECT_REST=0` + `VICTORY_WC_NO_BACK=1` + `EARLY_DEPTH` + `VICTORY_PORT_PLACED=1` + `VICTORY_COLD_SKIP=1` + `VICTORY_CCD=1`) · up to **22.9 fps** indoors (`43.6 ms/frame`) · `14.6 fps` cutscene peak |
| **Stability** | **0** `0x887A0005` GPUVM/page faults across test runs (long-duration session not yet formally timed) |
| **Memory** | `~5.6 GiB` commit (`~10.9 GiB` title private, `~19.9 GiB` mapped) |

---

## Full Linux / Proton Runtime Stack

| Repository | Role | Highlights |
|------------|------|------------|
| [`CaullenOmdahl/Victory`](https://github.com/CaullenOmdahl/Victory) | Core compatibility layer & RDNA2→DXIL translator | 160+ commits over `Belobug/VIctory-Belobug-AMD`: Proton/Wine fixes, lock-free host paths, and `VK_LAYER_VICTORY_placed` L0 VRAM placement |
| [`CaullenOmdahl/victory-cachyos-radv-runtime`](https://github.com/CaullenOmdahl/victory-cachyos-radv-runtime) | Build scripts, patch exports & A/B harness | Automated `ntdll.so` / `vkd3d-proton` builds, minimal patch exports, and serialized `flock` A/B benchmarking |
| [`CaullenOmdahl/wine-cachyos-victory-radv`](https://github.com/CaullenOmdahl/wine-cachyos-victory-radv) (`victory-radv-rx6800`) | Custom Wine (`cachyos-11.0-20261015-native` base) | 128-way 2 MiB `vm_prot_stripes`, lockless `PAGEMAP_SCAN`, `vm_tree_state` rwlock + AVX2 range queries, native `winevulkan`/`win32u` `VK_EXT_map_memory_placed` |
| [`CaullenOmdahl/vkd3d-proton-victory-radv`](https://github.com/CaullenOmdahl/vkd3d-proton-victory-radv) (`victory-radv-rx6800`) | Custom `vkd3d-proton` (`3.2.0` base) | `VK_KHR_map_memory2` + `VK_EXT_map_memory_placed` wiring for zero-copy placed heap placement |

---

## Measured Performance & Engineering Highlights

| Area | Change | Measured Result | Ref |
|------|--------|-----------------|-----|
| **Watched Graphics Heap** | Loop 145 `XMemAllocDefault` (`D3D_ALLOCATOR = 128`, `>= 64 KiB` write-watched, `Belobug #503`) | **7.3–7.5 fps** (`133.3–137.5 ms/f` outdoors across 5 intervals, `4.98 avg_fps`, `0` faults) | — |
| **WC Copy-Back Bypass** | `VICTORY_WC_NO_BACK=1` (`Belobug #477`, including sparse shadows) | Copy-back **23.5 → 1.7 ms/f** (`39.2 → 0.5 MB/f`); **7.0–7.2 fps** (`139.6–142.9 ms/f`) outdoors with `VICTORY_INDIRECT_REST=0`, `0` faults | [`271eea4`](https://github.com/CaullenOmdahl/Victory/commit/271eea4) |
| **Placed L0 VRAM** | `VK_LAYER_VICTORY_placed` + `VICTORY_PORT_PLACED=1` | Host refreshes `82.9 → 39.5 GB/f`; **5.1–5.5 fps** (`181–194 ms/f`) outdoors, `0` faults | [`8755d8c`](https://github.com/CaullenOmdahl/Victory/commit/8755d8c) |
| **Custom Proton Runtime** | 128-way `vm_prot_stripes`, lockless `PAGEMAP_SCAN`, native `VK_EXT_map_memory_placed` | `prot lock` `59–69 → 8.8–10.9 ms` (>6×); **4.6–5.0 → 4.9–5.3 fps** A/B uplift (`prep -12%`, `copies -33%`, `watches -38%`) | [`victory-cachyos-radv-runtime`](https://github.com/CaullenOmdahl/victory-cachyos-radv-runtime) |
| **Copy-Back Lock Stalls** | `RwLock` on `xmem::PHYSICAL` + `watch::VIEWS` | Peak streaming copy-back stall **4,378.9 → 128.1 ms** | [`15c1b16`](https://github.com/CaullenOmdahl/Victory/commit/15c1b16) |
| **Write-Watch Sharding** | 64-way `watch::JOURNALS` + 4 KiB→64 KiB dedup | Up to **22.9 fps** (`43.6 ms/f`) indoors, `0` faults | [`07410ac`](https://github.com/CaullenOmdahl/Victory/commit/07410ac) |
| **Shader & Copy-Back Pipeline** | Default-on `VICTORY_COPY_BACK_BESIDE` + offline 24T DXIL / `victory warm --used` | `21–37 ms/f` copy-back without `VirtualProtect`; `25,694` DXIL cached (`0` misses) | [`518977f`](https://github.com/CaullenOmdahl/Victory/commit/518977f), [`ce0e6da`](https://github.com/CaullenOmdahl/Victory/commit/ce0e6da) |

<details>
<summary>Technical details & tuning flags</summary>

- **Linux / Proton (`vkd3d-proton` + RADV) compatibility:**
  - Page-aligned `vkMapMemory` view offsets, batched `UpdateTileMappings`, and null-blob `ID3D12PipelineLibrary` support under Proton.
  - Gated `OpenExistingHeapFromAddress1` (`direct()`) behind `VICTORY_IN_PLACE_WINE=1` (Wine `MEM_PRIVATE` round-trips, zeroed `MEM_MAPPED` reads, and host-RAM `USERPTR` slow RX 6800 GPU execution ~20× vs VRAM copies).
  - Wine stack unwind guards (`bcd337e`), lazy `ws2_32` export resolution (`#3091`), optional Wine temp-file physical-section backing (`VICTORY_PHYSICAL_BACKING_DIR` / `VICTORY_PHYSICAL_PAGEFILE_FIRST_MB`, `14fdfb8`), `LoaderLock` unwind `Drop` guard (`#3001`), NUL-terminated DualSense HID paths (`#207`), and remap/guard/descriptor fixes eliminating `0x887A0005` GPUVM faults.
- **Tuning & performance flags:**
  - `VICTORY_PORT_PLACED=1` + `ENABLE_VICTORY_PLACED=1` (`tools/vk-layer-placed/` + `port_placed.rs`): backs 32-bit `XMem` reservations directly in RADV L0 VRAM (`8755d8c`).
  - `VICTORY_WC_NO_BACK=1` (`wc.rs`, `VICTORY_WC_NO_BACK_MIN_KB=64` default): leaves `>= 64 KiB` write-combined GPU writes on card/sparse shadows without snapshots or copy-back (`copy-back` `23.5 -> 1.7 ms/f`, `0.5 MB/f`, **`7.0–7.2 fps` / `139.6–142.9 ms/f`**).
  - `VICTORY_INDIRECT_REST=0` (`indirect.rs`, `fork-cgbj 449830f`): prevents an oversized indirect record from exhausting the rest of a command list's indirect budget so trailing draws keep executing.
  - `VICTORY_COPY_BACK_BESIDE` (default-on, `0` to disable) & `VICTORY_TEXTURE_BACK=1` (`texback.rs`, shader-written ocean/displacement UAV texture readback).
  - `VICTORY_COLD_SKIP=1` (`ada5c7a`), `VICTORY_CCD=1` (`622e793`), `VICTORY_CARD_FETCH_INFLIGHT=1` (`39c7ad6`), `VICTORY_HEAP_WORKER=1` (`129f8c4`), `VICTORY_PORT_CBV=1` (`c65cda7`), `VICTORY_PORT_GUARD=1` (`3961805`).
  - Default-on ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` (`VICTORY_GPU_UPLOAD=0` to disable, +45% FPS on RX 6800 16 GiB) and hair simulation (`VICTORY_HAIR=0` to disable).
  - AVX2 copy-back merges & idle `HostList`/chunk trimming (`0e04b91`), `RwLock` on `shadow::BUFFERS` and `shadow::LOOSE` with off-thread `victory-bury` retirement (`7922b08`, `f905d00`), and `victory_shader::SOURCE_HASH` cache keying.
- **Shader & rendering correctness:**
  - `ForceEarlyDepthStencil` (`EARLY_DEPTH`, `avatarmanz #435`, [`2e31861`](https://github.com/CaullenOmdahl/Victory/commit/2e31861)) on `DEPTH_BEFORE_SHADER` pixel shaders (prevents glass prepass depth/motion writes over foreground characters; **`5.5–5.6 fps` / `178.9–183.4 ms/frame`** steady state, `frame-4270.png` clean).
- Console DXGI formats 189, 190, and 191 (`35c0619`), `R9G9B9E5` format family mapping & SDWA 16-bit `UNUSED_SEXT` sign-extension (`dac3e58`).
  - Directed `f16` rounding (`#3159`) & negative-offset `CODE` descriptor reads (`#3155`, `4b8c9d9`).
  - Cube-map subnormal/signed-zero semantics (`#3127`) & near-unity 1-ULP `v_log_f32` (`#2160`, `12efdd0`).
  - 16-bit typed buffer view preservation (`79b950f`, `half_views` unsigned `R16*_UINT` twins + `bufferLoad.i32`/`bufferStore.i32`).
  - Selected RDNA2-to-DXIL wave, `f16`/`f64`, LDS atomic, BC7, and mesh/pixel linking fixes ported from upstream PRs, `DexNom/VIctory`, and `fxceless/victory-fork`.

</details>

---

## Known Limitations

- Ray-traced indirect lighting does not run (replaced by a constant ambient fallback).
- The in-game phone is broken (opening it leaves the world unlit and the phone as an outline).
- Story progress has not yet been tested beyond the opening scene.
- Long-duration crash-free stability has not yet been formally timed.
- Tested only on AMD Radeon RX 6800 on Linux under Proton (`vkd3d-proton` + Mesa RADV); untested on NVIDIA or Intel.

---

## Quick Start (Linux + Proton)

**Requirements:** Linux x86_64, Rust toolchain (`rust-toolchain.toml`), `cargo-xwin`, a Vulkan 1.3+ GPU with Shader Model 6.6 + mesh shaders (tested on Mesa RADV), and Proton (`vkd3d-proton`).

```bash
# 1. Install asset-policy commit hook and run host unit tests (no game files needed)
cargo xtask install-hooks
cargo test --workspace

# 2. Cross-compile the Windows MSVC binary from Linux
cargo xwin build --release --target x86_64-pc-windows-msvc -p victory
# Or diagnostics build:
cargo xwin build --profile diag --target x86_64-pc-windows-msvc -p victory --features diagnostics
```

Copy `victory.example.toml` to `victory.toml` (git-ignored) to configure paths outside the repository. For best performance, build the custom Wine + `vkd3d-proton` runtime from [`victory-cachyos-radv-runtime`](https://github.com/CaullenOmdahl/victory-cachyos-radv-runtime) and launch via Proton (`proton runinprefix`) with:
- `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3dcompiler_47=n"`
- `VKD3D_SHADER_CACHE_PATH` pointing outside the repository so `vkd3d-proton.cache.write` stays out of the working tree

```bash
victory.exe run <path/to/game_scarlett_master.exe> --execute --console series-x
```

**Input:** Supports XInput controllers, DualSense / DualSense Edge (USB or Bluetooth HID), and keyboard fallback (`WASD`/`IJKL` sticks, arrows D-pad, `Enter`/`O` A/B, `Backspace`/`X`/`Y` face buttons, `Q`/`E`/`Z`/`C` shoulders/triggers, `Esc`/`Tab` Menu/View).

For full CLI commands (`inspect`, `census`, `translate`, `warm`, `config`) and diagnostics switches, see [docs/architecture.md](docs/architecture.md) and upstream [Marvisak/VIctory](https://github.com/Marvisak/VIctory#readme).

---

## Credits & License

- Built on [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) and [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (full upstream commit history preserved).
- Includes selected fixes referenced from upstream PRs, [DexNom/VIctory](https://github.com/DexNom/VIctory), and [fxceless/victory-fork](https://github.com/fxceless/victory-fork).
- Licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE).
