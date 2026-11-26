//! Opt-in raw-byte constant buffer reads (`VICTORY_PORT_CBV=1`, off by default).
//!
//! D3D12 constant buffer views require 256-byte-aligned addresses and force
//! misaligned console constant buffers (16-byte aligned) to be copied into
//! upload scratch buffers at submission. When `VICTORY_PORT_CBV=1` is enabled,
//! `victory-shader` lowers CBV reads to raw `R32_TYPELESS` buffer SRVs (for
//! tables and bindless heap slots, 16-byte aligned) and root SRVs (for root
//! CBVs, 4-byte aligned), letting the host bind title buffers directly without
//! per-draw scratch copies.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether `VICTORY_PORT_CBV=1` is enabled (shared with `victory-shader`).
pub(crate) fn on() -> bool {
    victory_shader::emit::constants_raw()
}

/// Whether a buffer GPU address is eligible for a raw `R32_TYPELESS` SRV view
/// (`FirstElement = offset / 4`, restricted to 16-byte console CBV alignment).
pub(crate) fn raw_ok(gpu: u64) -> bool {
    gpu.is_multiple_of(16)
}

/// `[0]`: table CBVs bound directly as raw SRVs (count), `[1]`: their bytes,
/// `[2]`: table CBVs falling back to scratch copies, `[3]`: heap/bindless CBVs
/// made as raw SRVs.
static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub(crate) fn in_copy(size: u32) {
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(u64::from(size), Relaxed);
}

pub(crate) fn copied() {
    STATS[2].fetch_add(1, Relaxed);
}

pub(crate) fn twinned() {
    STATS[3].fetch_add(1, Relaxed);
}

pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS
        .iter()
        .map(|s| s.swap(0, Relaxed) as f64 / frames.max(1e-9))
        .collect();
    tracing::info!(
        "port_cbv (VICTORY_PORT_CBV=1) per frame: {:.0} table CBVs direct ({:.1} KB, no fill_scratch), {:.0} scratch fallbacks, {:.0} heap/bindless raw SRVs",
        s[0],
        s[1] / 1024.0,
        s[2],
        s[3],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_views_need_sixteen_bytes() {
        assert!(raw_ok(0x3500_0010));
        assert!(raw_ok(0x3500_0100));
        assert!(!raw_ok(0x3500_0008));
        assert!(!raw_ok(0x3500_0004));
    }
}
