# VIctory — Linux / Proton Fork (AMD Radeon)

`CaullenOmdahl/Victory` is a source-only compatibility layer fork built on [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) and [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (with selected fixes from upstream PRs, [DexNom/VIctory](https://github.com/DexNom/VIctory), and [fxceless/victory-fork](https://github.com/fxceless/victory-fork)) to run **Grand Theft Auto VI (Xbox Series X 1.0.2.1)** on **Linux under Proton (`vkd3d-proton` + Mesa RADV)**.

**This repository contains no game files, keys, executables, or downloads.** You must supply your own copy of the game. This project is not affiliated with or endorsed by Rockstar Games, Take-Two, or Microsoft.

---

## Linux / Proton Status (`CaullenOmdahl/Victory`, 2026-11-27)

- **Test machine:** CachyOS Linux (`7.3.4-1-cachyos`), Proton (`proton-patched`, `vkd3d-proton` + Mesa RADV), AMD Radeon RX 6800 16 GB (RDNA2), Ryzen 9 5900X (12C/24T), 62 GiB RAM, Series X profile.

| Metric | Status on Linux + Proton (RX 6800) |
|--------|------------------------------------|
| **Executable & boot** | Loads and boots (`game_scarlett_master.exe`, Series X 1.0.2.1) |
| **Main menu** | Works at **22–30 fps** |
| **In-world rendering** | Renders lit open world, characters, textures, water, sky, and geometry (ray-traced indirect light replaced by constant ambient fallback; in-game phone broken) |
| **Playable** | **Yes** — controller input, walking around in the world, and interactions work |
| **Story progress** | **Opening scene only** (have not progressed further yet) |
| **World frame rate** | **2.1–4.8 fps** outdoors (`206.3–538.1 ms/frame` at `~5,000` draws/frame), up to **22.9 fps** (`43.6 ms/frame`) indoors (`14.6 fps` cutscene peak) |
| **Stability** | **0** `0x887A0005` GPUVM/page faults across repeated in-world test runs; long-duration crash-free stability not yet formally measured |
| **Memory** | `~5.6 GiB` commit (`~10.9 GiB` title private, `~19.9 GiB` mapped) |

---

## What's Unique to This Fork

Over 160 commits on top of `Belobug/VIctory-Belobug-AMD` focused on running cleanly and faster under Linux / Proton (`vkd3d-proton` + RADV):

### Linux / Proton (`vkd3d-proton` + RADV) Compatibility
- **Vulkan/D3D12 interop:** Page-aligned `vkMapMemory` view offsets, batched `UpdateTileMappings`, and null-blob `ID3D12PipelineLibrary` support under Proton.
- **Avoided Wine `USERPTR` slowdown:** Gated `OpenExistingHeapFromAddress1` (`direct()`) behind `VICTORY_IN_PLACE_WINE=1` because Wine `MEM_PRIVATE` round-trips, zeroed `MEM_MAPPED` reads, and host-RAM `USERPTR` slow RX 6800 GPU execution ~20× compared to VRAM copies.
- **Wine runtime & memory-section stability:** Wine stack unwind guards (`bcd337e`), lazy `ws2_32` export resolution (`#3091`), optional Wine temp-file physical-section backing (`VICTORY_PHYSICAL_BACKING_DIR` / `VICTORY_PHYSICAL_PAGEFILE_FIRST_MB`, `14fdfb8`), `LoaderLock` unwind `Drop` guard (`#3001`), NUL-terminated DualSense HID paths (`#207`), and remap/guard/descriptor fixes that eliminated `0x887A0005` GPUVM faults.

### Measured Performance & Concurrency Wins
- **Sharded write-watch & 64 KiB page dedup (`07410ac`):** Sharded `watch::JOURNALS` into 64 aligned `RwLock` stripes and deduplicated 4 KiB `GetWriteWatch` pages into 64 KiB pages — reaching **4.52 avg_fps** overall and **22.9 fps / 43.6 ms/frame** indoors with `0` GPUVM/page faults.
- **`RwLock` physical & view tables (`15c1b16`):** Converted `xmem::PHYSICAL` (with non-blocking `try_read()` fallback in `write_beside`) and `watch::VIEWS` to `RwLock`, cutting peak streaming copy-back lock stalls from **4,378.9 ms to 128.1 ms** (**4.0 fps** steady-state, `28.6 ms` copy-back).
- **Zero-protect copy-back & GPU texture readback (`518977f`, `ce0e6da`):** Enabled `VICTORY_COPY_BACK_BESIDE` by default (`21.1–37.1 ms/frame` copy-back without `VirtualProtect` cycles) and added opt-in `VICTORY_TEXTURE_BACK=1` (`texback.rs`) for tiling shader-written ocean/displacement UAV textures back to title memory cleanly.
- **Lock-free / low-contention host paths (`0e04b91`, `7922b08`, `f905d00`, `ada5c7a`, `622e793`, `39c7ad6`, `129f8c4`, `c65cda7`, `3961805`):**
  - AVX2 32-byte copy-back merges in `shadow.rs` and idle readback-chunk / `HostList` trimming (`0e04b91`).
  - `RwLock` on `shadow::BUFFERS` and `shadow::LOOSE` with off-thread `victory-bury` retirement (`7922b08`, `f905d00`).
  - `VICTORY_COLD_SKIP=1` cold 2 MiB chunk refresh skip (`ada5c7a`), `VICTORY_CCD=1` Ryzen CCD worker pinning & same-page remap skip (`622e793`), AVX2 texture hashing & `VICTORY_CARD_FETCH_INFLIGHT=1` (`39c7ad6`), plus opt-in `VICTORY_HEAP_WORKER=1` (`129f8c4`), `VICTORY_PORT_CBV=1` (`c65cda7`), and `VICTORY_PORT_GUARD=1` (`3961805`).
- **Offline shader & pipeline warming:** `victory_shader::SOURCE_HASH` build-time cache keying plus offline 24-thread DXIL translation (`victory translate --cache`) and Proton headless pipeline warming (`victory warm --used`), keeping `25,694` DXIL shaders cached (`0` shader misses in steady-state runs).
- **Default-on ReBAR & hair simulation:** `D3D12_HEAP_TYPE_GPU_UPLOAD` (`VICTORY_GPU_UPLOAD=0` to disable, +45% FPS on RX 6800 16 GiB) and hair strands (`VICTORY_HAIR=0` to disable).

### Shader & Rendering Correctness
- Console DXGI formats 189, 190, and 191 (`35c0619`), `R9G9B9E5` format family mapping & SDWA 16-bit `UNUSED_SEXT` sign-extension (`dac3e58`).
- Directed `f16` rounding (`#3159`) & negative-offset `CODE` descriptor reads (`#3155`, `4b8c9d9`).
- Cube-map subnormal/signed-zero semantics (`#3127`) & near-unity 1-ULP `v_log_f32` (`#2160`, `12efdd0`).
- 16-bit typed buffer view preservation (`79b950f`, `half_views` unsigned `R16*_UINT` twins + `bufferLoad.i32`/`bufferStore.i32`).
- Ported upstream RDNA2-to-DXIL wave, `f16`/`f64`, LDS atomic, BC7, and mesh/pixel linking fixes.

---

## Known Limitations

- Ray-traced indirect lighting does not run (replaced by a constant ambient fallback).
- The in-game phone is broken (opening it leaves the world unlit and the phone as an outline).
- Story progress has not yet been tested beyond the opening scene.
- Long-duration crash-free stability has not yet been formally timed.
- Tested only on AMD Radeon RX 6800 on Linux under Proton (`vkd3d-proton` + Mesa RADV); untested on NVIDIA or Intel.

---

## Building & Running on Linux (`cargo-xwin` + Proton)

Requirements: Linux x86_64, Rust toolchain (`rust-toolchain.toml`), `cargo-xwin`, a Vulkan 1.3+ GPU with shader model 6.6 + mesh shaders (tested on Mesa RADV), and Proton (`vkd3d-proton`).

```bash
# 1. Install asset-policy commit hook and run host unit tests (no game files needed)
cargo xtask install-hooks
cargo test --workspace

# 2. Cross-compile the Windows MSVC binary from Linux
cargo xwin build --release --target x86_64-pc-windows-msvc -p victory
# Or diagnostics build:
cargo xwin build --profile diag --target x86_64-pc-windows-msvc -p victory --features diagnostics
```

Copy `victory.example.toml` to `victory.toml` (git-ignored) to configure local paths outside the repo, then launch through Proton (`proton runinprefix`) with:
- `WINEDLLOVERRIDES="d3d12,d3d12core,dxgi,d3dcompiler_47=n"`
- `VKD3D_SHADER_CACHE_PATH` pointing outside the repository so `vkd3d-proton.cache.write` stays out of the working tree.

Example execution:
```bash
victory.exe run <path/to/game_scarlett_master.exe> --execute --console series-x
```

Input supports XInput controllers, DualSense / DualSense Edge over USB or Bluetooth HID, and keyboard fallback (`WASD`/`IJKL` sticks, arrows D-pad, `Enter`/`O` A/B, `Backspace`/`X`/`Y` face buttons, `Q`/`E`/`Z`/`C` shoulders/triggers, `Esc`/`Tab` Menu/View).

For full CLI commands (`inspect`, `census`, `translate`, `warm`, `config`) and diagnostics environment variables, see [docs/architecture.md](docs/architecture.md) and upstream [Marvisak/VIctory](https://github.com/Marvisak/VIctory#readme).

---

## Credits & License

- Built on [Belobug/VIctory-Belobug-AMD](https://github.com/Belobug/VIctory-Belobug-AMD) and [Marvisak/VIctory](https://github.com/Marvisak/VIctory) (full upstream commit history preserved).
- Includes selected fixes referenced from upstream PRs, [DexNom/VIctory](https://github.com/DexNom/VIctory), and [fxceless/victory-fork](https://github.com/fxceless/victory-fork).
- Licensed under **GPL-3.0-or-later**. See [LICENSE](LICENSE).
