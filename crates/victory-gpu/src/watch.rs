//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! The title shows one physical page at several addresses. A write of its
//! own through any of them is seen: the runtime tells us the views it maps
//! ([`title_memory_view`]), guarding a block makes every other view of its
//! page read-only too, and a fault at one of them opens them all. Without
//! this a copy taken at one address stayed as it was for the rest of the
//! run when the title wrote through another (parts of moving vehicles left
//! behind and stretched across the screen, 2026-11-25). The idea is
//! victory-fork #115's (Spexivus, joroyt) as carried by victory-amd; the
//! code here is written anew. `VICTORY_VIEW_GUARD=0` turns it off.
//!
//! shortcut: the runtime's own writes into physical pages through a view of
//! its own (xmem's `write_physical`, DirectStorage's page destinations) are
//! told by address ([`title_memory_written`]), not through this. A page
//! shown at more than [`MANY_VIEWS`] addresses is guarded one view at a
//! time. A system call writing into a guarded block fails (none is known
//! to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-block (64 KiB) epoch table (or `VICTORY_EPOCH_SHIFT=26` for 64 MiB).
/// Incremented for only the block(s) touched whenever a guarded block is
/// opened (VEH guard fault or `writing`), remapped (`title_memory_remapped`),
/// or rewritten (`title_memory_written`). While a span's region epoch stays
/// unchanged, any range already guarded and copied at that epoch cannot have
/// been modified by the title — and `copy_back` on one buffer no longer
/// invalidates clean spans in other blocks.
const REGION_BUCKETS: usize = 1 << 20; // 64 GiB of 64 KiB blocks before two share a bucket
static REGION_EPOCHS: [AtomicU64; REGION_BUCKETS] =
    [const { AtomicU64::new(1) }; REGION_BUCKETS];

#[inline]
fn region_shift() -> usize {
    static SHIFT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *SHIFT.get_or_init(|| {
        std::env::var("VICTORY_EPOCH_SHIFT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or(16, |s| s.clamp(16, 30))
    })
}

#[inline]
fn bump_region_epochs(start: usize, end: usize) {
    if start >= end {
        return;
    }
    let shift = region_shift();
    let first = start >> shift;
    let last = (end - 1) >> shift;
    let count = last - first + 1;
    if count >= REGION_BUCKETS {
        for slot in &REGION_EPOCHS {
            slot.fetch_add(1, Ordering::Release);
        }
    } else {
        for r in first..=last {
            REGION_EPOCHS[r & (REGION_BUCKETS - 1)].fetch_add(1, Ordering::Release);
        }
    }
}

#[inline]
pub(crate) fn region_epoch(start: usize, end: usize) -> u64 {
    if start >= end {
        return 0;
    }
    let shift = region_shift();
    let first = start >> shift;
    let last = (end - 1) >> shift;
    let count = last - first + 1;
    let mut hash = 0u64;
    if count >= REGION_BUCKETS {
        for slot in &REGION_EPOCHS {
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(slot.load(Ordering::Acquire));
        }
    } else {
        for r in first..=last {
            hash = hash
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .wrapping_add(REGION_EPOCHS[r & (REGION_BUCKETS - 1)].load(Ordering::Acquire));
        }
    }
    hash
}

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
}

static ASKED: AtomicU64 = AtomicU64::new(0);

static JOURNAL: RwLock<Journal> = RwLock::new(Journal {
    written: BTreeMap::new(),
});

/// Serializes `GetWriteWatch(WRITE_WATCH_FLAG_RESET)` with recording its
/// cleared pages into `JOURNAL` so concurrent askers cannot miss a reset.
static WRITE_WATCH_LOCK: Mutex<()> = Mutex::new(());

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
/// Whether `start` lies inside a block the guards already hold (read-only
/// or a known unmapped hole): asking `GetWriteWatch` on mapped view memory
/// always fails, so callers skip the lock and kernel syscall.
fn known_view(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    });
    if ask {
        return false;
    }
    let g = guards_read();
    let first = start / BLOCK;
    g.protected.contains(&first) || g.holes.contains(&first)
}

pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    if guarded && known_view(start) {
        return writes_guarded(start, end);
    }
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let needed = (last - first) * PAGE / KERNEL_PAGE;
    thread_local! {
        static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let ww = WRITE_WATCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let (failed, pages): (bool, Vec<usize>) = FOUND.with(|cell| {
        let mut found = cell.borrow_mut();
        if found.len() < needed {
            found.resize(needed, 0);
        }
        let mut count = needed;
        let mut granularity = 0;
        // SAFETY: the answer buffer holds at least `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        if failed {
            (true, Vec::new())
        } else {
            (false, found[..count].iter().map(|&at| at / PAGE).collect())
        }
    });
    if failed {
        drop(ww);
        if !guarded {
            return None;
        }
        return writes_guarded(start, end);
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    let now = ASKED.fetch_add(1, Ordering::AcqRel) + 1;
    if pages.is_empty() {
        drop(ww);
        let j = JOURNAL.read().unwrap_or_else(|e| e.into_inner());
        return Some((
            now,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    for page in pages {
        j.written.insert(page, now);
    }
    let out = j
        .written
        .range(first..last)
        .map(|(&p, &w)| (p, w))
        .collect();
    drop(j);
    drop(ww);
    Some((now, out))
}

/// Same as [`writes`] for memory already known to be guarded `MEM_MAPPED`
/// view memory: skips the `GetWriteWatch` syscall (which always fails on
/// mapped views).
pub(crate) fn writes_guarded(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let guarded = guard(first * PAGE, last * PAGE)?;
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
    }
    let now = ASKED.fetch_add(1, Ordering::AcqRel) + 1;
    if guarded.is_empty() {
        let j = JOURNAL.read().unwrap_or_else(|e| e.into_inner());
        return Some((
            now,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
    for page in guarded {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
    /// Unmapped blocks (`VirtualProtect` failed) within guarded ranges;
    /// cleared by [`title_memory_remapped`] when the runtime maps them.
    holes: BTreeSet<usize>,
    /// Disjoint `[lo, hi)` block intervals already probed and covered by
    /// guards; cleared on remap by [`title_memory_remapped`].
    covered: BTreeMap<usize, usize>,
    /// Blocks in `covered` (or opened/pinned/aliased) that are not clean
    /// and not holes, so the fast path touches only dirty blocks.
    dirty: BTreeSet<usize>,
    /// Blocks whose page was guarded together with its other views (those
    /// views' blocks are in it too): a write to one opens them all.
    aliased: BTreeSet<usize>,
    /// Blocks the kernel is about to write for the runtime ([`writing`]),
    /// from, to, and until when they are left writable: a read may be
    /// asynchronous and land after the call that asked for it returned, and
    /// the kernel does not fault into our handler, it fails. Until then
    /// they are not made read-only, and every check reports them written.
    pins: Vec<(usize, usize, std::time::Instant)>,
}

static GUARDS: RwLock<Guards> = RwLock::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
    holes: BTreeSet::new(),
    covered: BTreeMap::new(),
    dirty: BTreeSet::new(),
    aliased: BTreeSet::new(),
    pins: Vec::new(),
});

/// How long blocks stay writable after [`writing`].
const PIN: std::time::Duration = std::time::Duration::from_secs(2);

impl Guards {
    /// Whether `block` is left writable for a kernel write.
    fn pinned(&self, block: usize) -> bool {
        self.pins.iter().any(|&(from, to, _)| from <= block && block < to)
    }

    /// Removes block range `[from, to)` from active pins, splitting any pin
    /// that extends past it so neighboring blocks stay pinned. Returns
    /// whether any pin overlapped `[from, to)`.
    fn drop_pins(&mut self, from: usize, to: usize) -> bool {
        if self.pins.is_empty() || from >= to {
            return false;
        }
        let mut changed = false;
        let mut kept = Vec::with_capacity(self.pins.len());
        for (p_from, p_to, until) in self.pins.drain(..) {
            if p_to <= from || p_from >= to {
                kept.push((p_from, p_to, until));
            } else {
                changed = true;
                if p_from < from {
                    kept.push((p_from, from, until));
                }
                if p_to > to {
                    kept.push((to, p_to, until));
                }
            }
        }
        self.pins = kept;
        changed
    }

    /// Whether every block in `[first, last)` is already inside a covered interval.
    fn is_covered(&self, first: usize, last: usize) -> bool {
        if first >= last {
            return true;
        }
        self.covered
            .range(..=first)
            .next_back()
            .is_some_and(|(_, &hi)| hi >= last)
    }

    /// Records `[first, last)` in `covered`, coalescing overlapping or adjacent intervals.
    fn mark_covered(&mut self, first: usize, last: usize) {
        use std::ops::Bound::{Excluded, Included};
        if first >= last {
            return;
        }
        let mut lo = first;
        let mut hi = last;
        let mut to_remove = Vec::new();
        if let Some((&k, &v)) = self.covered.range(..=first).next_back()
            && v >= first
        {
            lo = k;
            hi = hi.max(v);
            to_remove.push(k);
        }
        for (&k, &v) in self.covered.range((Excluded(first), Included(last))) {
            hi = hi.max(v);
            to_remove.push(k);
        }
        for k in to_remove {
            self.covered.remove(&k);
        }
        self.covered.insert(lo, hi);
    }

    /// Subtracts `[first, last)` from `covered` and clears `dirty` entries in that span.
    fn uncover(&mut self, first: usize, last: usize) -> bool {
        if first >= last {
            return false;
        }
        let mut removed = false;
        let dirty: Vec<usize> = self.dirty.range(first..last).copied().collect();
        if !dirty.is_empty() {
            removed = true;
            for b in dirty {
                self.dirty.remove(&b);
            }
        }
        let mut to_remove = Vec::new();
        let mut to_insert = Vec::new();
        if let Some((&k, &v)) = self.covered.range(..first).next_back()
            && v > first
        {
            removed = true;
            to_remove.push(k);
            to_insert.push((k, first));
            if v > last {
                to_insert.push((last, v));
            }
        }
        for (&k, &v) in self.covered.range(first..last) {
            removed = true;
            to_remove.push(k);
            if v > last {
                to_insert.push((last, v));
            }
        }
        for k in to_remove {
            self.covered.remove(&k);
        }
        for (k, v) in to_insert {
            self.covered.insert(k, v);
        }
        removed
    }
}

/// Views of one page past which its views are not guarded together (the
/// page the title shows wherever it has nothing has thousands).
pub(crate) const MANY_VIEWS: usize = 64;

/// The views of physical pages the runtime mapped: by address (length,
/// section, offset in it), and the addresses each page of a section is
/// shown at.
#[derive(Default)]
struct Views {
    by_addr: BTreeMap<usize, (usize, usize, u64)>,
    pages: std::collections::HashMap<(usize, u64), Vec<usize>>,
}

static VIEWS: Mutex<Option<Views>> = Mutex::new(None);

/// Whether all the views of a guarded page are guarded together.
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

impl Views {
    /// Forgets the views overlapping `[at, at + len)`, whole.
    fn forget(&mut self, at: usize, len: usize) {
        let end = at.saturating_add(len);
        let first = self
            .by_addr
            .range(..=at)
            .next_back()
            .filter(|(a, (n, _, _))| *a + n > at)
            .map(|(a, _)| *a);
        let hit: Vec<usize> = first
            .into_iter()
            .chain(self.by_addr.range(at.saturating_add(1)..end).map(|(a, _)| *a))
            .collect();
        for a in hit {
            let Some((n, section, offset)) = self.by_addr.remove(&a) else {
                continue;
            };
            for k in 0..n / BLOCK {
                let key = (section, offset / BLOCK as u64 + k as u64);
                if let Some(shown) = self.pages.get_mut(&key) {
                    shown.retain(|&x| x != a + k * BLOCK);
                    if shown.is_empty() {
                        self.pages.remove(&key);
                    }
                }
            }
        }
    }

    /// Every address the page shown at `at` (a block's start) is shown at,
    /// `at` included; empty if `at` is in no view we were told of.
    fn shown(&self, at: usize) -> &[usize] {
        let Some((&base, &(n, section, offset))) = self.by_addr.range(..=at).next_back() else {
            return &[];
        };
        if at >= base + n {
            return &[];
        }
        let key = (section, offset / BLOCK as u64 + ((at - base) / BLOCK) as u64);
        self.pages.get(&key).map_or(&[], Vec::as_slice)
    }
}

/// Every address the page of the block at `at` is shown at (see
/// [`Views::shown`]).
fn aliases(at: usize) -> Vec<usize> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    views.as_ref().map_or(Vec::new(), |v| v.shown(at - at % BLOCK).to_vec())
}

/// The runtime mapped a view at `[at, at + len)` of the bytes from `offset`
/// on of the physical-page section `section`. The view is writable, so the
/// same pages guarded at another view are opened: their copies check them
/// again, which also guards the new view with them.
pub fn title_memory_view(at: usize, len: usize, section: usize, offset: u64) {
    if !view_guard() || len == 0 || !at.is_multiple_of(BLOCK) || !offset.is_multiple_of(BLOCK as u64) {
        return;
    }
    let mut others = Vec::new();
    {
        let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
        let views = views.get_or_insert_with(Views::default);
        views.forget(at, len);
        views.by_addr.insert(at, (len, section, offset));
        for k in 0..len / BLOCK {
            let shown = views.pages.entry((section, offset / BLOCK as u64 + k as u64)).or_default();
            if shown.len() < MANY_VIEWS {
                others.extend_from_slice(shown);
            }
            shown.push(at + k * BLOCK);
        }
    }
    if !others.is_empty() {
        let mut g = guards_mut();
        for a in others {
            open_here(&mut g, a, a + BLOCK);
        }
    }
}

/// The runtime unmapped the views in `[at, at + len)`.
pub fn title_memory_view_gone(at: usize, len: usize) {
    if let Some(views) = VIEWS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        views.forget(at, len);
    }
}

fn guards_read() -> std::sync::RwLockReadGuard<'static, Guards> {
    GUARDS.read().unwrap_or_else(|e| e.into_inner())
}

fn guards_mut() -> std::sync::RwLockWriteGuard<'static, Guards> {
    GUARDS.write().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Shared read-lock fast path: 97%+ of checks on covered spans have no
    // dirty/pinned blocks and complete with zero lock contention across workers.
    {
        let g = guards_read();
        if g.is_covered(first, last) && g.dirty.range(first..last).next().is_none() {
            return Some(Vec::new());
        }
    }
    // Write-lock fast path: if [first, last) is already covered, walk only the
    // dirty/pinned/re-opened blocks in `g.dirty` (O(log N + k)).
    {
        let mut g = guards_mut();
        if g.is_covered(first, last) {
            if !g.pins.is_empty() {
                let now = std::time::Instant::now();
                g.pins.retain(|&(_, _, until)| until > now);
            }
            let dirty: Vec<usize> = g.dirty.range(first..last).copied().collect();
            if dirty.is_empty() {
                return Some(Vec::new());
            }
            let mut written = Vec::new();
            let mut run: Option<(usize, usize)> = None;
            for block in dirty {
                if !g.pins.is_empty() && g.pinned(block) {
                    if let Some((r_from, r_to)) = run.take() {
                        protect(&mut g, r_from, r_to, &mut written);
                    }
                    written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                    bump_region_epochs(block * BLOCK, (block + 1) * BLOCK);
                } else if g.holes.contains(&block) || g.clean.contains(&block) {
                    g.dirty.remove(&block);
                } else if g.protected.contains(&block) {
                    if let Some((r_from, r_to)) = run.take() {
                        protect(&mut g, r_from, r_to, &mut written);
                    }
                    g.clean.insert(block);
                    g.dirty.remove(&block);
                    written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                } else {
                    match run {
                        Some((r_from, r_to)) if r_to == block => run = Some((r_from, block + 1)),
                        Some((r_from, r_to)) => {
                            protect(&mut g, r_from, r_to, &mut written);
                            run = Some((block, block + 1));
                        }
                        None => run = Some((block, block + 1)),
                    }
                }
            }
            if let Some((r_from, r_to)) = run {
                protect(&mut g, r_from, r_to, &mut written);
            }
            return Some(written);
        }
    }
    // Slow path (first encounter): `start` is itself a mapped view, or if
    // `start` is an unmapped hole (common in 2 GiB sparse shadows), jump past
    // leading holes via `VirtualQuery` to find the first committed page.
    let probe = if mapped(start) {
        start
    } else {
        first_mapped(start, end)?
    };
    let check_each = probe != start;
    install();
    let mut g = guards_mut();
    if !g.pins.is_empty() {
        let now = std::time::Instant::now();
        g.pins.retain(|&(_, _, until)| until > now);
    }
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    for block in first..=last {
        if block < last && check_each && !mapped(block * BLOCK) {
            if let Some(from) = run.take() {
                protect(&mut g, from, block, &mut written);
            }
            g.holes.insert(block);
            g.dirty.remove(&block);
            continue;
        }
        let open = block < last && !g.clean.contains(&block) && !g.holes.contains(&block);
        if open {
            if !g.pins.is_empty() && g.pinned(block) {
                if let Some(from) = run.take() {
                    protect(&mut g, from, block, &mut written);
                }
                // The kernel may be writing it: left writable, reported
                // written at every check until the pin ends.
                g.dirty.insert(block);
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                // No range holding it is clean meanwhile (`region_epoch`).
                bump_region_epochs(block * BLOCK, (block + 1) * BLOCK);
                continue;
            } else if g.protected.contains(&block) {
                if let Some(from) = run.take() {
                    protect(&mut g, from, block, &mut written);
                }
                // Written through the runtime, still read-only.
                g.clean.insert(block);
                g.dirty.remove(&block);
                written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
                continue;
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block, &mut written);
        }
    }
    g.mark_covered(first, last);
    Some(written)
}

/// Scans `[start, end)` jumping over unmapped holes; returns the first
/// committed address if it is a mapped view, or `None` if `[start, end)` has
/// no committed mapped view (or starts with committed non-view memory).
fn first_mapped(start: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut at = start;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries our own address space without dereferencing `at`.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || info.RegionSize == 0 || info.State == MEM_FREE {
            return None;
        }
        if info.State == MEM_COMMIT {
            return (info.Type == MEM_MAPPED).then_some(at);
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at {
            return None;
        }
        at = next;
    }
    None
}

/// Whether the current allocation at `at` is a mapped view. Asking only for
/// the allocation type avoids enumerating pages with identical protection,
/// which can span an entire large view after its blocks were guarded.
fn mapped(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_MAPPED, MEMORY_BASIC_INFORMATION, MemoryRegionInfo, QueryVirtualMemoryInformation,
        VirtualQuery, WIN32_MEMORY_REGION_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut region = WIN32_MEMORY_REGION_INFORMATION::default();
    // SAFETY: both queries describe an address without dereferencing it. The
    // output buffers match their information classes.
    let queried = unsafe {
        QueryVirtualMemoryInformation(
            GetCurrentProcess(),
            at as *const _,
            MemoryRegionInfo,
            (&raw mut region).cast(),
            size_of_val(&region),
            std::ptr::null_mut(),
        )
    };
    if queried != 0 {
        // These are the Windows SDK's named bitfields in declaration order.
        // windows-sys exposes only the union's raw bits.
        const PRIVATE: u32 = 1 << 0;
        const MAPPED_DATA_FILE: u32 = 1 << 1;
        const MAPPED_IMAGE: u32 = 1 << 2;
        const MAPPED_PAGE_FILE: u32 = 1 << 3;
        const MAPPED_PHYSICAL: u32 = 1 << 4;
        // SAFETY: this union was filled by the successful query above.
        match unsafe { region.Anonymous.Flags } {
            MAPPED_DATA_FILE | MAPPED_PAGE_FILE | MAPPED_PHYSICAL => return true,
            PRIVATE | MAPPED_IMAGE => return false,
            _ => {}
        }
    }
    // Unknown flag combinations and failed allocation queries keep the
    // VirtualQuery type check, including for a free address.
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: the output buffer has the required layout and size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.Type == MEM_MAPPED
}

/// From `block` on (to `to` at most): the end of the run of blocks whose
/// memory is of one kind, and whether that kind may be guarded (anything
/// but committed private memory, images, and free memory).
fn view_run(block: usize, to: usize) -> (usize, bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: queries our own address space without dereferencing `block`.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.RegionSize == 0 {
        return (block + 1, false);
    }
    let end = (info.BaseAddress as usize)
        .saturating_add(info.RegionSize)
        .div_ceil(BLOCK)
        .clamp(block + 1, to);
    let non_view = info.State == MEM_FREE
        || (info.State == MEM_COMMIT && (info.Type == MEM_PRIVATE || info.Type == MEM_IMAGE));
    (end, !non_view)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// walks region by region so views beside committed private memory or free
/// blocks never make non-view memory (like the process heap) read-only.
fn protect(g: &mut Guards, from: usize, to: usize, written: &mut Vec<usize>) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, written: &mut Vec<usize>, from: usize, to: usize| {
        for block in from..to {
            g.protected.insert(block);
            g.clean.insert(block);
            g.dirty.remove(&block);
        }
        written.extend(from * BLOCK / PAGE..to * BLOCK / PAGE);
    };
    let mut at = from;
    while at < to {
        let (end, can_guard) = view_run(at, to);
        if !can_guard {
            for block in at..end {
                g.holes.insert(block);
                g.dirty.remove(&block);
            }
        } else if set(at, end) {
            note(g, written, at, end);
        } else {
            for block in at..end {
                if set(block, block + 1) {
                    note(g, written, block, block + 1);
                } else {
                    g.holes.insert(block);
                    g.dirty.remove(&block);
                }
            }
        }
        at = end;
    }
    if !view_guard() {
        return;
    }
    // The same pages at their other views: read-only too, so that a write
    // through one of those faults. They are not noted clean: the copies
    // taken at those views have not taken the pages, and their next check
    // must report them (a block read-only and not clean is reported).
    // One look at the views for the whole run, and no list made for a page
    // with one view (most of them): this runs under the guards' lock, which
    // every copy's check waits for (6 % of the submitting thread's time when
    // it asked block by block, run p1).
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(views) = views.as_ref() else { return };
    for block in from..to {
        let shown = views.shown(block * BLOCK);
        if shown.len() < 2 || shown.len() > MANY_VIEWS || !g.protected.contains(&block) {
            continue;
        }
        for &at in shown {
            let alias = at / BLOCK;
            g.aliased.insert(alias);
            if alias != block && !g.protected.contains(&alias) && !g.pinned(alias) {
                let mut old = 0;
                // SAFETY: title view memory; only its protection changes.
                if unsafe { VirtualProtect(at as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
                    g.protected.insert(alias);
                    g.dirty.insert(alias);
                }
            }
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same pages at their
/// other views are opened too: every guard over them must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    if start >= end || g.aliased.is_empty() {
        return any;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for at in aliases(block * BLOCK) {
            if at / BLOCK != block {
                open_here(g, at, at + BLOCK);
            }
            g.aliased.remove(&(at / BLOCK));
        }
    }
    any
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    if blocks.is_empty() {
        return false;
    }
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        g.dirty.insert(block);
    }
    // Runs of blocks next to each other in one call (fewer kernel
    // transitions and TLB shootdowns). One call cannot span two views: the
    // title's page mappings are one view a page, and a run over several
    // failed whole, leaving blocks read-only that were no longer noted
    // guarded (a kernel read into them then failed and the title stopped).
    // So a run that fails is given back block by block.
    let give = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: blocks we made read-only; only their protection changes.
        let whole =
            unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READWRITE, &mut old) } != 0;
        if !whole && to - from > 1 {
            for block in from..to {
                // SAFETY: as above, one block.
                unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
            }
        }
    };
    let mut run_start = blocks[0];
    let mut run_end = run_start + 1;
    for &block in &blocks[1..] {
        if block == run_end {
            run_end = block + 1;
        } else {
            give(run_start, run_end);
            run_start = block;
            run_end = block + 1;
        }
    }
    give(run_start, run_end);
    bump_region_epochs(start, end);
    true
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / BLOCK, (at + len).div_ceil(BLOCK));
    let mut g = guards_mut();
    if !g.pins.is_empty() {
        let now = std::time::Instant::now();
        g.pins.retain(|&(_, _, until)| until > now);
    }
    let until = std::time::Instant::now() + PIN;
    g.pins.push((first, last, until));
    open(&mut g, at, at + len);
    for block in first..last {
        g.dirty.insert(block);
    }
    if !view_guard() {
        return;
    }
    // The same pages at their other views: the write lands in them too, so
    // their copies must take them again, now and when it has landed.
    for block in first..last {
        let shown = aliases(block * BLOCK);
        if shown.len() < 2 || shown.len() > MANY_VIEWS {
            continue;
        }
        for alias in shown {
            let ab = alias / BLOCK;
            if ab != block {
                g.pins.push((ab, ab + 1, until));
                open_here(&mut g, alias, alias + BLOCK);
                g.dirty.insert(ab);
            }
        }
    }
}

/// Before our own synchronous write into title memory `[at, at + len)`:
/// opens any guarded blocks (and their guarded aliases) without leaving a
/// multi-second async-I/O pin behind, since the write finishes before the
/// caller returns.
pub(crate) fn open_for_sync_write(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let (first, last) = (at / BLOCK, (at + len).div_ceil(BLOCK));
    {
        let g = guards_read();
        if g.protected.range(first..last).next().is_none() {
            return;
        }
    }
    let mut g = guards_mut();
    if g.protected.range(first..last).next().is_some() {
        open(&mut g, at, at + len);
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded, and any placed
/// textures backed by that range are evicted.
pub fn title_memory_remapped(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    {
        let mut g = guards_mut();
        let (first, last) = (at / BLOCK, end.div_ceil(BLOCK));
        let blocks: Vec<usize> = g
            .protected
            .range(first..last)
            .copied()
            .collect();
        let holes: Vec<usize> = g
            .holes
            .range(first..last)
            .copied()
            .collect();
        let uncovered = g.uncover(first, last);
        let dropped_pins = g.drop_pins(first, last);
        let changed = uncovered || !holes.is_empty() || !blocks.is_empty() || dropped_pins;
        for block in holes {
            g.holes.remove(&block);
        }
        for block in blocks {
            g.protected.remove(&block);
            g.clean.remove(&block);
            // A view mapped a moment ago may already have been made
            // read-only as another view of a guarded page (between the
            // runtime's `title_memory_view` and this call): it is given
            // back, or the title's next write there would fault with no
            // guard to answer for it.
            let mut old = 0;
            // SAFETY: only the protection of title view memory changes; the
            // call fails harmlessly on memory that is no longer a view.
            unsafe {
                windows_sys::Win32::System::Memory::VirtualProtect(
                    (block * BLOCK) as *const _,
                    BLOCK,
                    windows_sys::Win32::System::Memory::PAGE_READWRITE,
                    &mut old,
                )
            };
        }
        let aliased: Vec<usize> = g.aliased.range(first..last).copied().collect();
        for block in aliased {
            g.aliased.remove(&block);
        }
        if changed {
            bump_region_epochs(at, end);
        }
    }
    crate::shadow::remapped(at as u64, end as u64);
    crate::resource::evict_placed_textures(at as u64, end as u64);
    crate::descriptor::host::bindless_memory_changed(at as u64, end as u64);
}

/// The title committed memory `[at, at + len)` it had reserved.
pub fn title_memory_committed(at: usize, len: usize) {
    if len != 0 {
        crate::shadow::filled(at as u64, (at + len) as u64);
    }
}

/// The runtime wrote new contents into title memory `[at, at + len)` (a
/// streamed file arrived): textures placed there are uploaded again at their
/// next use, not when their turn to be looked at comes.
pub fn title_memory_written(at: usize, len: usize) {
    if len != 0 {
        bump_region_epochs(at, at.saturating_add(len));
        // The runtime writes through a view of its own, which neither the
        // kernel's write watch nor our guards see: the pages are noted
        // written here, or host copies of buffers made before the file
        // arrived kept what the memory held then for good (meshes read as
        // zeros: walls, floors and bodies missing, others each launch).
        {
            let now = ASKED.fetch_add(1, Ordering::AcqRel) + 1;
            let mut j = JOURNAL.write().unwrap_or_else(|e| e.into_inner());
            for page in at / PAGE..(at + len).div_ceil(PAGE) {
                j.written.insert(page, now);
            }
        }
        crate::resource::textures_rewritten(at as u64, (at + len) as u64);
    }
}

/// Opens a guarded block at `at` (or returns `true` if another thread already
/// made it read-write). Called both by our own VEH handler and by
/// `host::safe_copy`'s VEH handler on write faults.
pub(crate) fn try_open_guarded(at: usize) -> bool {
    let mut g = guards_mut();
    if open(&mut g, at, at + 1) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    if writable(at) {
        return true;
    }
    // Read-only, in a view we were told of, and not noted guarded: only we
    // make title views read-only, so it is ours to give back.
    if view_guard() && !aliases(at).is_empty() {
        let mut old = 0;
        let block = at - at % BLOCK;
        // SAFETY: only the protection of title view memory changes.
        let opened = unsafe {
            windows_sys::Win32::System::Memory::VirtualProtect(
                block as *const _,
                BLOCK,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
                &mut old,
            )
        } != 0;
        if opened {
            bump_region_epochs(block, block + BLOCK);
            tracing::warn!(at = format_args!("{at:#x}"), "a read-only view block that no guard knew was opened");
        }
        return opened;
    }
    false
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        if try_open_guarded(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Two views of the same two physical pages, told as the runtime tells
    /// them: the two addresses.
    fn two_views() -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write twice.
        unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            assert!(one != 0 && two != 0);
            title_memory_view(one, 2 * BLOCK, section as usize, 0);
            title_memory_view(two, 2 * BLOCK, section as usize, 0);
            (one, two)
        }
    }

    fn drop_views(views: &[usize]) {
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, 2 * BLOCK);
            title_memory_view_gone(view, 2 * BLOCK);
            // SAFETY: our view.
            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        }
    }

    /// The title writes the pages of a guarded copy through another view of
    /// them: the guarded view's next check reports the page written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        let (guarded, other) = two_views();
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view; the handler opens it on the fault.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()));
        drop_views(&[guarded, other]);
    }

    /// Both views have copies: a write through either is reported at both.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        let (one, two) = two_views();
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + 2 * BLOCK), Some(vec![pages(one, 0), pages(one, 1)]));
        // The other view was made read-only with the first, not noted clean.
        assert_eq!(guard(two, two + 2 * BLOCK), Some(vec![pages(two, 0), pages(two, 1)]));
        assert_eq!(guard(one, one + 2 * BLOCK), Some(Vec::new()));
        assert_eq!(guard(two, two + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: the first view; the handler opens it on the fault.
        unsafe { ((one + 16) as *mut u8).write(7) };
        assert_eq!(guard(two, two + 2 * BLOCK), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + 2 * BLOCK), Some(vec![pages(one, 0)]));
        // SAFETY: the same byte through the other view.
        assert_eq!(unsafe { ((two + 16) as *const u8).read() }, 7);
        drop_views(&[one, two]);
    }

    /// A view mapped beside a guarded one is writable: the guarded view
    /// checks its pages again.
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let (section, guarded) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (section, MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize)
        };
        title_memory_view(guarded, 2 * BLOCK, section as usize, 0);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()));
        // SAFETY: a second view of the second page only.
        let other = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, BLOCK as u32, BLOCK).Value as usize };
        assert_ne!(other, 0);
        title_memory_view(other, BLOCK, section as usize, BLOCK as u64);
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(1)]));
        title_memory_remapped(other, BLOCK);
        title_memory_view_gone(other, BLOCK);
        // SAFETY: our view.
        unsafe {
            windows_sys::Win32::System::Memory::UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: other as *mut _ },
            )
        };
        drop_views(&[guarded]);
    }

    /// A write changes the epoch of the ranges that hold its block and of no
    /// other: the block beside it stays known clean.
    #[test]
    fn a_write_changes_the_epoch_of_its_block_alone() {
        let (one, two) = two_views();
        assert!(guard(one, one + 2 * BLOCK).is_some());
        let first = region_epoch(one, one + BLOCK);
        let second = region_epoch(one + BLOCK, one + 2 * BLOCK);
        let both = region_epoch(one, one + 2 * BLOCK);
        let part = region_epoch(one + BLOCK + 4096, one + BLOCK + 8192);
        // SAFETY: the view's second block; the handler opens it on the fault.
        unsafe { ((one + BLOCK + 8) as *mut u8).write(3) };
        assert_eq!(region_epoch(one, one + BLOCK), first, "the block beside it");
        assert_ne!(region_epoch(one + BLOCK, one + 2 * BLOCK), second);
        assert_ne!(region_epoch(one, one + 2 * BLOCK), both);
        assert_ne!(region_epoch(one + BLOCK + 4096, one + BLOCK + 8192), part);
        drop_views(&[one, two]);
    }

    /// `Guards::drop_pins` removes only the remapped blocks from active pins,
    /// splitting partially overlapping pins and leaving disjoint ones intact.
    #[test]
    fn remapping_drops_overlapping_pins_and_keeps_adjacent_ones() {
        let until = std::time::Instant::now() + PIN;
        let mut g = Guards {
            protected: BTreeSet::new(),
            clean: BTreeSet::new(),
            holes: BTreeSet::new(),
            covered: BTreeMap::new(),
            dirty: BTreeSet::new(),
            aliased: BTreeSet::new(),
            pins: vec![
                (10, 12, until),
                (14, 18, until),
                (17, 19, until),
                (19, 23, until),
                (15, 22, until),
                (25, 27, until),
            ],
        };
        assert!(!g.drop_pins(30, 32));
        assert!(g.drop_pins(16, 20));
        assert_eq!(
            g.pins,
            vec![
                (10, 12, until),
                (14, 16, until),
                (20, 23, until),
                (15, 16, until),
                (20, 22, until),
                (25, 27, until),
            ]
        );
        for b in 16..20 {
            assert!(!g.pinned(b), "remapped block {b} must not stay pinned");
        }
        assert!(g.pinned(15));
        assert!(g.pinned(20));
    }

    /// Unmapping and remapping a view clears any `writing()` pin on its blocks,
    /// so a later view mapped at the same address is guarded cleanly instead
    /// of inheriting the stale pin.
    #[test]
    fn remapped_view_does_not_inherit_pins_from_earlier_writing() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize;
            (section, view)
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some());
        writing(view + 100, 13);
        assert!(guards_read().pinned(view / BLOCK));

        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
        }
        title_memory_remapped(view, 2 * BLOCK);
        assert!(
            !guards_read().pinned(view / BLOCK),
            "pin must not outlive title_memory_remapped"
        );

        // SAFETY: same live pagefile section, mapped read-write.
        let view2 = unsafe { MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize };
        assert_ne!(view2, 0);
        assert_eq!(
            guard(view2, view2 + 2 * BLOCK),
            Some((view2 / PAGE..(view2 + 2 * BLOCK) / PAGE).collect())
        );
        assert_eq!(guard(view2, view2 + 2 * BLOCK), Some(Vec::new()));

        // SAFETY: our view and section handle.
        unsafe {
            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view2 as *mut _,
            });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(view2, 2 * BLOCK);
    }

    #[test]
    fn allocation_query_recognizes_file_views_and_rejects_nonviews() {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
        };
        let file = tempfile::tempfile().unwrap();
        file.set_len(PAGE as u64).unwrap();
        // SAFETY: a valid, open file and a read-only mapping of its contents.
        let section = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                PAGE_READONLY,
                0,
                0,
                std::ptr::null(),
            )
        };
        assert!(!section.is_null());
        // SAFETY: an open section containing at least one page.
        let view = unsafe { MapViewOfFile(section, FILE_MAP_READ, 0, 0, PAGE) };
        assert!(!view.Value.is_null());
        assert!(mapped(view.Value as usize + 13));
        // SAFETY: our successful mapping and its section handle.
        unsafe {
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
        assert!(!mapped((&raw const file) as usize)); // private stack allocation
        assert!(!mapped(mapped as *const () as usize)); // executable image
        assert!(!mapped(0)); // no allocation to query
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// `open_for_sync_write` opens guarded blocks without pinning them: the
    /// next `guard` check reports only the opened block and immediately
    /// re-protects it so the check after that sees nothing changed.
    #[test]
    fn sync_write_opens_without_two_second_pin() {
        let (guarded, other) = two_views();
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(guarded, 0), pages(guarded, 1)]));
        assert_eq!(guard(other, other + 2 * BLOCK), Some(vec![pages(other, 0), pages(other, 1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()));
        assert_eq!(guard(other, other + 2 * BLOCK), Some(Vec::new()));

        open_for_sync_write(guarded + BLOCK, 64);
        // SAFETY: synchronous write into block 1 now succeeds without taking a VEH fault.
        unsafe { ((guarded + BLOCK + 16) as *mut u8).write(42) };
        assert_eq!(guard(other, other + 2 * BLOCK), Some(vec![pages(other, 1)]));
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(vec![pages(guarded, 1)]));
        // Without a 2-second pin, block 1 is guarded and clean immediately again.
        assert_eq!(guard(guarded, guarded + 2 * BLOCK), Some(Vec::new()));
        assert_eq!(guard(other, other + 2 * BLOCK), Some(Vec::new()));
        drop_views(&[guarded, other]);
    }

    #[test]
    fn free_memory_is_not_tracked() {
        use windows_sys::Win32::System::Memory::{
            MEM_FREE, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualAlloc,
            VirtualFree, VirtualQuery,
        };
        // SAFETY: a reservation of ours, released at once: free memory.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * BLOCK, MEM_RESERVE, PAGE_READWRITE) }
            as usize;
        assert_ne!(r, 0);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
        let tracked = guard(r, r + 2 * BLOCK).is_some();
        // SAFETY: plain data filled in by the query, of the right size.
        let still_free = unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(r as *const _, &mut info, size_of_val(&info)) != 0
                && info.State == MEM_FREE
                && info.BaseAddress as usize + info.RegionSize >= r + 2 * BLOCK
        };
        assert!(!(tracked && still_free), "free memory");
    }

    #[test]
    fn committed_private_memory_beside_a_view_is_not_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
            MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER,
            MEMORY_BASIC_INFORMATION, MapViewOfFile3, PAGE_NOACCESS, PAGE_READONLY,
            UnmapViewOfFile2, VirtualAlloc2, VirtualFree, VirtualQuery,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of two blocks, split; a pagefile
        // section's block mapped over the first, private memory committed
        // over the second.
        let (p, section, view) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                2 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            assert_ne!(
                VirtualFree(p as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                0
            );
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, p);
            let private = VirtualAlloc2(
                GetCurrentProcess(),
                (p + BLOCK) as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_eq!(private, p + BLOCK);
            (p, section, view)
        };
        title_memory_remapped(p, BLOCK);
        assert!(guard(p, p + 2 * BLOCK).is_some());
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query, of the right size.
            unsafe {
                let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(
                    VirtualQuery(at as *const _, &mut info, size_of_val(&info)),
                    0
                );
                info.Protect
            }
        };
        assert_eq!(protection(p), PAGE_READONLY, "the view is guarded");
        assert_eq!(protection(p + BLOCK), PAGE_READWRITE, "the private block is not");
        // SAFETY: committed private memory of ours, writable.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        title_memory_remapped(p, 2 * BLOCK);
        // SAFETY: our view, section and private block.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), view, 0);
            VirtualFree((p + BLOCK) as *mut _, 0, MEM_RELEASE);
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }
}
