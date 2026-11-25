//! Opt-in tracking of write-combining title reservations (`VICTORY_SKIP_WC_BACK=1`).
//!
//! When enabled, GPU writes to buffers whose declared ranges lie wholly inside
//! a single write-combining `XMemVirtualAlloc` reservation (`PAGE_WRITECOMBINE`,
//! `0x400`) are left in host GPU memory instead of being copied back to title
//! memory every frame.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

struct Reservation {
    len: usize,
    protect: u32,
}

static RESERVATIONS: Mutex<BTreeMap<usize, Reservation>> = Mutex::new(BTreeMap::new());
static SKIPPED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether copying back shader writes to write-combining title memory is
/// skipped (`VICTORY_SKIP_WC_BACK=1`; off by default).
pub(crate) fn skip_wc_back() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_SKIP_WC_BACK").is_some_and(|v| v == "1");
        if on {
            tracing::warn!(
                "VICTORY_SKIP_WC_BACK=1: shader writes to write-combining title memory are left on the card"
            );
        }
        on
    })
}

/// Records a constrained `XMemVirtualAlloc` reservation `[at, at + len)` with
/// the console `protect` flags before host sanitization strips WC bits.
pub fn title_memory_reserved(at: usize, len: usize, protect: u32) {
    if !(skip_wc_back() || cfg!(test)) || at == 0 || len == 0 {
        return;
    }
    RESERVATIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(at, Reservation { len, protect });
}

fn all_wc_in(reservations: &BTreeMap<usize, Reservation>, ranges: &[(usize, usize)]) -> bool {
    !ranges.is_empty()
        && ranges.iter().all(|&(from, to)| {
            // Seeking from the exclusive end detects any reservation boundary inside `[from, to)`.
            from < to
                && reservations.range(..to).next_back().is_some_and(|(&base, r)| {
                    base <= from
                        && r.protect & 0x400 != 0
                        && base.checked_add(r.len).is_some_and(|end| to <= end)
                })
        })
}

/// Returns true and accounts `bytes` if `VICTORY_SKIP_WC_BACK=1` and every
/// declared `[from, to)` range lies inside one write-combining reservation.
pub(crate) fn skip_back(ranges: &[(usize, usize)], bytes: u64) -> bool {
    if !skip_wc_back() {
        return false;
    }
    let wc = {
        let reservations = RESERVATIONS.lock().unwrap_or_else(|e| e.into_inner());
        all_wc_in(&reservations, ranges)
    };
    if wc {
        let n = SKIPPED[0].fetch_add(1, Ordering::Relaxed) + 1;
        SKIPPED[1].fetch_add(bytes, Ordering::Relaxed);
        if n.is_multiple_of(8192) {
            tracing::info!(
                "write-combining copy-backs skipped (VICTORY_SKIP_WC_BACK=1): {n} pieces, {} MB",
                SKIPPED[1].load(Ordering::Relaxed) >> 20
            );
        }
    }
    wc
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{Reservation, all_wc_in};

    fn reservation(len: usize, protect: u32) -> Reservation {
        Reservation { len, protect }
    }

    #[test]
    fn mixed_and_boundary_spanning_ranges_are_not_skipped() {
        const MIB: usize = 1 << 20;
        let window = 8 * MIB;
        let middle = window + 2 * MIB;
        let map = BTreeMap::from([
            (window, reservation(2 * MIB, 0x404)),
            (middle, reservation(2 * MIB, 0x04)),
        ]);
        let wc = (middle - 65536, middle);
        let ordinary = (middle, middle + 65536);
        assert!(all_wc_in(&map, &[wc]));
        assert!(!all_wc_in(&map, &[ordinary]));
        assert!(!all_wc_in(&map, &[wc, ordinary]));
        assert!(!all_wc_in(&map, &[(wc.0, ordinary.1)]));
    }

    #[test]
    fn declaration_must_fit_within_one_wc_reservation() {
        let map = BTreeMap::from([
            (0x10000, reservation(0x10000, 0x404)),
            (0x20000, reservation(0x10000, 0x404)),
            (usize::MAX - 8, reservation(16, 0x404)),
        ]);
        assert!(all_wc_in(&map, &[(0x10000, 0x20000), (0x20000, 0x30000)]));
        assert!(!all_wc_in(&map, &[(0x1ffff, 0x20001)]));
        assert!(!all_wc_in(&map, &[(0x30000, 0x30001)]));
        assert!(!all_wc_in(&map, &[(0x10000, 0x10000)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, 1)]));
        assert!(!all_wc_in(&map, &[(usize::MAX - 8, usize::MAX)]));
    }
}
