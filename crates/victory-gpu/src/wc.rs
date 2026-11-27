//! Opt-in tracking of write-combining title reservations (`VICTORY_SKIP_WC_BACK=1`).
//!
//! When enabled, GPU writes to buffers whose declared ranges lie wholly inside
//! a single write-combining `XMemVirtualAlloc` reservation (`PAGE_WRITECOMBINE`,
//! `0x400`) are left in host GPU memory instead of being copied back to title
//! memory every frame.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

/// Kinds of title memory, by the protection requested on allocation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub(crate) enum Kind {
    Unknown = 0,
    Cached = 1,
    WriteCombined = 2,
    Uncached = 3,
}

const UNCACHED: u32 = 0x200;
const WRITE_COMBINED: u32 = 0x400;

/// Whether title memory protections are tracked (`VICTORY_WC_NO_BACK=1`).
pub(crate) fn tracking_protection() -> bool {
    no_back() || skip_wc_back() || cfg!(test)
}

/// `VICTORY_WC_NO_BACK=1` (off by default): what shaders write into >= 64 KiB
/// (`VICTORY_WC_NO_BACK_MIN_KB`) write-combined ranges in card-backed,
/// write-watched title memory stays on the card (skipping snapshot, compare,
/// and copy-back).
pub(crate) fn no_back() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_WC_NO_BACK").is_some_and(|v| v == "1");
        if on {
            tracing::warn!(
                min_kib = no_back_min() >> 10,
                "VICTORY_WC_NO_BACK=1: shader writes >= min_kib into watched card WC memory stay on the card"
            );
        }
        on
    })
}

fn no_back_min() -> usize {
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VICTORY_WC_NO_BACK_MIN_KB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(64)
            << 10
    })
}

/// `[declarations_skipped, bytes_skipped, small_kept, small_bytes,
/// unwatched_refreshes, unwatched_bytes, uncarded_bytes,
/// kept_unwatched_bytes, kept_uncarded_bytes]`
static NO_BACK_STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

pub(crate) fn unwatched(bytes: u64) {
    NO_BACK_STATS[4].fetch_add(1, Ordering::Relaxed);
    NO_BACK_STATS[5].fetch_add(bytes, Ordering::Relaxed);
}

pub(crate) fn uncarded(bytes: u64) {
    NO_BACK_STATS[6].fetch_add(bytes, Ordering::Relaxed);
}

/// Returns true if `VICTORY_WC_NO_BACK=1` keeps `[address, address + len)` on
/// the card without snapshot or copy-back.
pub(crate) fn skip_no_back(address: usize, len: usize, carded: bool, watched: bool) -> bool {
    if !no_back() || len == 0 {
        return false;
    }
    let Some(end_byte) = address.checked_add(len - 1) else {
        return false;
    };
    if kind(address) != Kind::WriteCombined || kind(end_byte) != Kind::WriteCombined {
        return false;
    }
    if len < no_back_min() {
        NO_BACK_STATS[2].fetch_add(1, Ordering::Relaxed);
        NO_BACK_STATS[3].fetch_add(len as u64, Ordering::Relaxed);
        return false;
    }
    if !carded || !watched {
        NO_BACK_STATS[if carded { 7 } else { 8 }].fetch_add(len as u64, Ordering::Relaxed);
        return false;
    }
    NO_BACK_STATS[0].fetch_add(1, Ordering::Relaxed);
    NO_BACK_STATS[1].fetch_add(len as u64, Ordering::Relaxed);
    true
}

/// A set of disjoint `[start, end)` ranges, joined where they touch.
#[derive(Default)]
pub(crate) struct Ranges(BTreeMap<usize, usize>);

impl Ranges {
    pub(crate) fn add(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let (mut from, mut to) = (from, to);
        if let Some((&s, &e)) = self.0.range(..=from).next_back()
            && e >= from
        {
            if e >= to {
                return;
            }
            from = s;
        }
        let inside: Vec<usize> = self.0.range(from..=to).map(|(&s, _)| s).collect();
        for s in inside {
            if let Some(e) = self.0.remove(&s) {
                to = to.max(e);
            }
        }
        self.0.insert(from, to);
    }

    pub(crate) fn overlapping(
        &self,
        from: usize,
        to: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let first = self
            .0
            .range(..from)
            .next_back()
            .filter(|(_, e)| **e > from)
            .map(|(&s, &e)| (s, e));
        first
            .into_iter()
            .chain(self.0.range(from..to).map(|(&s, &e)| (s, e)))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn remove(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let cut: Vec<(usize, usize)> = self.overlapping(from, to).collect();
        for (s, e) in cut {
            self.0.remove(&s);
            if s < from {
                self.0.insert(s, from);
            }
            if e > to {
                self.0.insert(to, e);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn covers(&self, from: usize, to: usize) -> bool {
        from >= to || self.0.range(..=from).next_back().is_some_and(|(_, &e)| e >= to)
    }

    #[cfg(test)]
    pub(crate) fn contains(&self, at: usize) -> bool {
        self.0.range(..=at).next_back().is_some_and(|(_, &e)| e > at)
    }
}

fn ranges() -> &'static RwLock<BTreeMap<usize, (usize, Kind)>> {
    static RANGES: OnceLock<RwLock<BTreeMap<usize, (usize, Kind)>>> = OnceLock::new();
    RANGES.get_or_init(|| RwLock::new(BTreeMap::new()))
}

fn kind_of_protection(protect: u32) -> Kind {
    if protect & WRITE_COMBINED != 0 {
        Kind::WriteCombined
    } else if protect & UNCACHED != 0 {
        Kind::Uncached
    } else {
        Kind::Cached
    }
}

/// Records the console `protect` flags for `[at, at + len)`.
pub fn title_memory_protection(at: usize, len: usize, protect: u32) {
    if !tracking_protection() || at == 0 || len == 0 {
        return;
    }
    insert(
        &mut ranges().write().unwrap_or_else(|e| e.into_inner()),
        at,
        at.saturating_add(len),
        kind_of_protection(protect),
    );
}

fn insert(map: &mut BTreeMap<usize, (usize, Kind)>, from: usize, to: usize, kind: Kind) {
    if let Some((&start, &(end, k))) = map.range(..from).next_back()
        && end > from
    {
        map.insert(start, (from, k));
        if end > to {
            map.insert(to, (end, k));
        }
    }
    let inside: Vec<usize> = map.range(from..to).map(|(&s, _)| s).collect();
    for start in inside {
        if let Some((end, k)) = map.remove(&start)
            && end > to
        {
            map.insert(to, (end, k));
        }
    }
    map.insert(from, (to, kind));
}

pub(crate) fn kind(address: usize) -> Kind {
    let map = ranges().read().unwrap_or_else(|e| e.into_inner());
    match map.range(..=address).next_back() {
        Some((_, &(end, k))) if address < end => k,
        _ => Kind::Unknown,
    }
}

pub(crate) fn take(frames: f64) -> Option<[f64; 9]> {
    if !no_back() {
        return None;
    }
    let per_frame = |v: u64| v as f64 / frames.max(1.0);
    Some(std::array::from_fn(|i| {
        per_frame(NO_BACK_STATS[i].swap(0, Ordering::Relaxed))
    }))
}

pub(crate) fn log(sample: Option<[f64; 9]>) {
    let Some([n, bytes, small, small_bytes, unwatched, unwatched_bytes, uncarded, kept_unwatched, kept_uncarded]) =
        sample
    else {
        return;
    };
    tracing::info!(
        "wc_no_back (VICTORY_WC_NO_BACK=1, min {} KiB): {n:.0} skipped ({:.1} MB declared), \
         {small:.0} small kept ({:.1} MB), {unwatched:.1} unwatched skips ({:.2} MB), \
         uncarded={:.2} MB, kept(unwatched={:.1} MB, uncarded={:.1} MB)",
        no_back_min() >> 10,
        bytes / 1e6,
        small_bytes / 1e6,
        unwatched_bytes / 1e6,
        uncarded / 1e6,
        kept_unwatched / 1e6,
        kept_uncarded / 1e6,
    );
}

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
    title_memory_protection(at, len, protect);
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

    #[test]
    fn newest_protection_wins_and_cuts_older_ranges() {
        use super::{Kind, insert, kind_of_protection};
        let mut map = BTreeMap::new();
        insert(&mut map, 0x1000, 0x9000, Kind::Cached);
        insert(&mut map, 0x3000, 0x5000, Kind::WriteCombined);
        let got: Vec<_> = map.iter().map(|(&s, &(e, k))| (s, e, k)).collect();
        assert_eq!(
            got,
            [
                (0x1000, 0x3000, Kind::Cached),
                (0x3000, 0x5000, Kind::WriteCombined),
                (0x5000, 0x9000, Kind::Cached),
            ]
        );
        insert(&mut map, 0x2000, 0x6000, Kind::Uncached);
        let got: Vec<_> = map.iter().map(|(&s, &(e, k))| (s, e, k)).collect();
        assert_eq!(
            got,
            [
                (0x1000, 0x2000, Kind::Cached),
                (0x2000, 0x6000, Kind::Uncached),
                (0x6000, 0x9000, Kind::Cached),
            ]
        );
        assert_eq!(kind_of_protection(0x2404), Kind::WriteCombined);
        assert_eq!(kind_of_protection(0x2204), Kind::Uncached);
        assert_eq!(kind_of_protection(0x22004), Kind::Cached);
    }

    #[test]
    fn ranges_join_remove_and_overlap() {
        use super::Ranges;
        let mut r = Ranges::default();
        r.add(10, 20);
        r.add(30, 40);
        r.add(20, 25);
        r.add(12, 15);
        assert_eq!(r.0.iter().map(|(&s, &e)| (s, e)).collect::<Vec<_>>(), [(10, 25), (30, 40)]);
        assert_eq!(r.overlapping(0, 11).collect::<Vec<_>>(), [(10, 25)]);
        assert!(r.overlapping(25, 30).next().is_none());
        assert_eq!(r.overlapping(24, 31).collect::<Vec<_>>(), [(10, 25), (30, 40)]);
        r.remove(15, 35);
        assert_eq!(r.0.iter().map(|(&s, &e)| (s, e)).collect::<Vec<_>>(), [(10, 15), (35, 40)]);
        assert!(r.covers(10, 15) && !r.covers(10, 16) && r.contains(35) && !r.contains(40));
    }
}
