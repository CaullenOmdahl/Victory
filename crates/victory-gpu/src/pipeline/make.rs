//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! Passes over the screen (no vertex input, no depth) have workers and a
//! longer budget of their own: the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth
//! of pipelines they waited minutes; a run that lost them had no diffuse
//! light at all.
//!
//! shortcut: the budgets (half a second of waiting per title frame, 20 s
//! for screen passes) are guesses; another pass the title runs only once
//! and whose pipeline takes longer is lost.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline, TargetFormats};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// Its worker was lost to it: it will not be made.
    lost: AtomicBool,
    /// When a worker began it ([`stuck`]).
    begun: OnceLock<Instant>,
    /// Whether a replacement worker was started for it ([`stuck`]).
    replaced: AtomicBool,
    /// A pass over the screen (see [`GraphicsState::screen_pass`]).
    screen: bool,
    /// A single-stage compute pipeline, prioritized and budgeted separately under Wine.
    compute: bool,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
    /// First title frame that waited for it (`u64::MAX`: none yet).
    waited_frame: AtomicU64,
    /// The formats its state declares for what a draw draws into (zeros
    /// without a graphics state).
    pub(crate) formats: TargetFormats,
    /// The same shaders and state for other formats, bound at a draw
    /// ([`for_formats`]): a chain from the title's pipeline, one link per
    /// set of formats met, never shortened.
    variant: OnceLock<Arc<Pending>>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            screen: graphics.as_ref().is_some_and(|g| g.screen_pass()),
            compute: matches!(shaders.as_slice(), [cs] if cs.stage == "compute"),
            formats: graphics.as_ref().map_or([0; 9], |g| g.formats()),
            variant: OnceLock::new(),
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            lost: AtomicBool::new(false),
            begun: OnceLock::new(),
            replaced: AtomicBool::new(false),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            waited_frame: AtomicU64::new(u64::MAX),
        }
    }

    #[cfg(test)]
    pub(crate) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.make_with(|| HostPipeline::new(&self.shaders, self.graphics.as_deref()));
    }

    fn make_with(&self, create: impl FnOnce() -> Option<HostPipeline>) {
        let _ = self.begun.set(Instant::now());
        // Not `get_or_init`: a worker lost in it would leave it locked.
        let made = match catch_unwind(AssertUnwindSafe(create)) {
            Ok(made) => made,
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("unknown panic payload");
                tracing::error!(
                    shaders = ?self.shaders,
                    panic = message,
                    "a panic while making a pipeline: it is refused"
                );
                None
            }
        };
        let _ = self.made.set(made);
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// How long a pipeline may be in the making before it counts as stuck
/// (`VICTORY_PIPELINE_STUCK_S`, 15 s by default).
fn stuck_after() -> Duration {
    #[cfg(test)]
    return Duration::from_millis(50);
    #[cfg(not(test))]
    {
        static AFTER: OnceLock<Duration> = OnceLock::new();
        *AFTER.get_or_init(|| {
            Duration::from_secs(
                std::env::var("VICTORY_PIPELINE_STUCK_S")
                    .ok()
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .unwrap_or(15),
            )
        })
    }
}

/// Stack of each pipeline worker: room for the driver's shader compiler
/// (reserved address space; committed only as used).
const PIPELINE_STACK: usize = 64 << 20;

static NEXT_REPLACEMENT: AtomicUsize = AtomicUsize::new(200);

/// Whether `p`'s worker began making it longer ago than [`stuck_after`]
/// and has not returned: stop waiting for it and spawn a replacement worker.
fn stuck(p: &Pending) -> bool {
    let Some(begun) = p.begun.get() else {
        return false;
    };
    if p.made.get().is_some() || p.lost.load(Ordering::Relaxed) || begun.elapsed() < stuck_after() {
        return false;
    }
    if !p.replaced.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            shaders = ?p.shaders,
            seconds = stuck_after().as_secs(),
            "a host pipeline's compile has not come back: not waiting for it and spawning a replacement worker"
        );
        if WORKERS_RUNNING.load(Ordering::Relaxed) > 0 {
            worker(NEXT_REPLACEMENT.fetch_add(1, Ordering::Relaxed), p.screen);
        }
    }
    true
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None if stuck(p) => Some(format!("stuck {:?}", p.shaders)),
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

impl Pending {
    fn waits_alone(&self) -> bool {
        self.compute && !compute_budget().is_zero()
    }

    fn priority_lane(&self) -> bool {
        self.screen || self.waits_alone()
    }
}

/// Dedicated per-frame wait budget for first-use compute pipelines under Wine (`VICTORY_COMPUTE_WAIT_MS`).
fn compute_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let asked = std::env::var("VICTORY_COMPUTE_WAIT_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        let millis = asked.unwrap_or(if crate::shadow::on_wine() { 10_000 } else { 0 });
        if millis != 0 {
            tracing::info!(
                millis,
                "compute passes wait for their host pipeline by a budget of their own (VICTORY_COMPUTE_WAIT_MS)"
            );
        }
        Duration::from_millis(millis)
    })
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// The same for passes over the screen (below the title's 30 s watchdog).
const SCREEN_BUDGET: Duration = Duration::from_secs(20);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
        let stages: Vec<(&str, &str)> = p
            .shaders
            .iter()
            .map(|s| (s.stage, s.hash.as_str()))
            .collect();
        super::library::note_used_recipe(&p.key, &stages, p.graphics.as_deref());
    }
    if p.made.get().is_none() {
        start(p, true);
        if p.lost.load(Ordering::Relaxed) || stuck(p) {
            return None;
        }
        // Also with a world's worth queued: passes the title runs only once
        // set up what later ones loop over.
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Pipelines made again for the formats bound at a draw, since the start.
static VARIANTS: AtomicU64 = AtomicU64::new(0);

/// Whether a draw into views of other formats than its pipeline declares
/// gets a host pipeline for those formats ([`for_formats`]). On by
/// default; `VICTORY_TARGET_VARIANTS=0` sends such a draw to the host with
/// the title's pipeline, as before.
fn variants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TARGET_VARIANTS").is_none_or(|v| v != "0"))
}

/// The host pipeline for a draw with `p` into views of the formats
/// `bound`, which `p` does not declare (`host::formats_differ`): that of
/// the same shaders and state for those formats. It is a pipeline like the
/// title's own: queued at its first draw, made by a worker, waited for
/// within the frame's budget, and the draw is dropped while it is not
/// there ([`ready`]).
#[cold]
pub(crate) fn for_formats<'a>(p: &'a Arc<Pending>, bound: &TargetFormats) -> Option<&'a HostPipeline> {
    match p.graphics.as_deref() {
        Some(g) if variants_on() => ready(variant(p, g, &super::host::formats_for(&p.formats, bound))),
        _ => ready(p),
    }
}

/// The pipeline of `p`'s shaders and its state `g` for the formats
/// `bound`: found in the chain that starts at `p`, or added to it (logged
/// then: once per pipeline and set of formats).
fn variant<'a>(p: &'a Arc<Pending>, g: &GraphicsState, bound: &TargetFormats) -> &'a Arc<Pending> {
    let mut at = p;
    for _ in 0..64 {
        let next = at.variant.get_or_init(|| {
            let so_far = VARIANTS.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::info!(
                pipeline = ?p.shaders,
                declared = ?p.formats,
                bound = ?bound,
                so_far,
                "a pipeline is made again for the render target formats bound at its draw (eight render targets, then depth; VICTORY_TARGET_VARIANTS=0: the draw goes out with the title's pipeline)"
            );
            Arc::new(Pending::new(p.shaders.clone(), Some(Box::new(g.with_formats(bound)))))
        });
        if next.formats == *bound {
            return next;
        }
        at = next;
    }
    p
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>, now: bool) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        queue(p.clone(), now);
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, false);
    }
}

/// The waiting of a title frame, over all threads. The title records its
/// lists on several threads, and a thread may begin to wait while another
/// waits: the frame's waits end together. Time counts while any thread
/// waits, once however many do, and not between waits.
struct Waits(Mutex<Spent>);

struct Spent {
    /// The newest frame a wait began in (`u64::MAX`: none yet).
    frame: u64,
    /// Time of that frame in which a thread waited, without the waits
    /// under way.
    used: Duration,
    /// Waits under way, and when the first of them began (within `frame`).
    under_way: u32,
    since: Option<Instant>,
    /// As `used` over all frames, since [`Waits::take`].
    total: Duration,
}

impl Waits {
    const fn new() -> Self {
        Self(Mutex::new(Spent {
            frame: u64::MAX,
            used: Duration::ZERO,
            under_way: 0,
            since: None,
            total: Duration::ZERO,
        }))
    }

    fn spent(&self) -> std::sync::MutexGuard<'_, Spent> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A wait begins at `now` in title frame `frame`: when it ends at the
    /// latest. A later frame starts with the whole `budget` (waits of the
    /// frame before that are still under way count from `now`); a thread
    /// that names an earlier frame (it read the frame's number, then lost
    /// the processor) waits on the newest frame's budget.
    fn begin(&self, frame: u64, now: Instant, budget: Duration) -> Instant {
        let mut s = self.spent();
        if s.frame == u64::MAX || frame > s.frame {
            if let Some(since) = s.since.replace(now) {
                s.total += now.saturating_duration_since(since);
            }
            (s.frame, s.used) = (frame, Duration::ZERO);
        }
        if s.under_way == 0 {
            s.since = Some(now);
        }
        s.under_way += 1;
        s.since.unwrap_or(now) + budget.saturating_sub(s.used)
    }

    /// A wait that began with [`Waits::begin`] ends at `now`.
    fn end(&self, now: Instant) {
        let mut s = self.spent();
        s.under_way = s.under_way.saturating_sub(1);
        if s.under_way == 0
            && let Some(since) = s.since.take()
        {
            let waited = now.saturating_duration_since(since);
            s.used += waited;
            s.total += waited;
        }
    }

    /// The time in which a thread waited since the last call, the waits
    /// under way up to `now`.
    fn take(&self, now: Instant) -> Duration {
        let mut s = self.spent();
        if let Some(since) = s.since {
            let waited = now.saturating_duration_since(since);
            // Counted here; the frame's budget keeps it too, and the
            // waits under way go on from `now`.
            s.used += waited;
            s.total += waited;
            s.since = Some(now);
        }
        std::mem::take(&mut s.total)
    }
}

/// The waits of draws, screen passes, and compute passes for their pipelines.
static WAITS: [Waits; 3] = [Waits::new(), Waits::new(), Waits::new()];

/// Milliseconds in which a recording thread waited for a pipeline since the
/// last call (for the frame line): time the title's frame stood still that
/// no submission's time holds, as the waits are outside the submissions.
pub(crate) fn take_waited_ms() -> f64 {
    let now = Instant::now();
    (WAITS[0].take(now) + WAITS[1].take(now) + WAITS[2].take(now)).as_secs_f64() * 1000.0
}

/// Whether a pipeline is waited for in every frame that draws with it
/// (`VICTORY_PIPELINE_WAIT_EVERY_FRAME=1`), not just the first frame with budget.
fn wait_every_frame() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_EVERY_FRAME").is_some_and(|v| v == "1"))
}

/// Claims `frame` as the single frame in which `p` may be waited for.
fn claim_wait(p: &Pending, frame: u64) -> bool {
    if wait_every_frame() {
        return true;
    }
    match p.waited_frame.compare_exchange(u64::MAX, frame, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => true,
        Err(first) => first == frame,
    }
}

/// Waits for `p` while frame `frame` has budget of its kind left.
fn wait(p: &Pending, frame: u64) {
    if stuck(p) {
        return;
    }
    if p.waits_alone() {
        wait_within(&WAITS[2], compute_budget(), p, frame);
        return;
    }
    let kind = usize::from(p.screen);
    let budget = [wait_budget(), SCREEN_BUDGET][kind];
    if budget.is_zero() {
        return;
    }
    wait_within(&WAITS[kind], budget, p, frame);
}

/// Waits for `p` until it is made, stuck, or the waiting `waits` allows frame
/// `frame` of `budget` is over.
fn wait_within(waits: &Waits, budget: Duration, p: &Pending, frame: u64) {
    /// Ends the wait however the function is left.
    struct Under<'a>(&'a Waits);
    impl Drop for Under<'_> {
        fn drop(&mut self) {
            self.0.end(Instant::now());
        }
    }
    let now = Instant::now();
    let end = waits.begin(frame, now, budget);
    let _under = Under(waits);
    if end <= now || !claim_wait(p, frame) {
        return;
    }
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() && !p.lost.load(Ordering::Relaxed) && !stuck(p) {
        let rest = end.saturating_duration_since(Instant::now());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// What the workers take from (normal, then screen passes), each in priority
/// order: waited-for draws, early depth-only pipelines, and other early starts.
static QUEUES: [(Mutex<[VecDeque<Arc<Pending>>; 3]>, Condvar); 2] = [
    (Mutex::new([const { VecDeque::new() }; 3]), Condvar::new()),
    (Mutex::new([const { VecDeque::new() }; 3]), Condvar::new()),
];

fn queue_index(p: &Pending, now: bool) -> usize {
    if now {
        0
    } else if p
        .graphics
        .as_deref()
        .is_some_and(GraphicsState::writes_depth_only)
    {
        1
    } else {
        2
    }
}

static WORKERS_RUNNING: AtomicUsize = AtomicUsize::new(0);

/// What the workers (by thread) are making.
static AT_WORK: Mutex<Vec<(u32, Arc<Pending>)>> = Mutex::new(Vec::new());

/// Starts worker `i`, for screen passes or for the others.
fn worker(i: usize, screen: bool) {
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{i}"))
        .stack_size(PIPELINE_STACK)
        .spawn(move || {
            WORKERS_RUNNING.fetch_add(1, Ordering::Relaxed);
            // Below the title's own threads (loading, recording).
            // SAFETY: a plain call on our own thread.
            unsafe {
                use windows::Win32::System::Threading::*;
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            }
            let (lock, work) = &QUEUES[usize::from(screen)];
            loop {
                let p = {
                    let mut q = lock.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        if let Some(p) = q[0]
                            .pop_front()
                            .or_else(|| q[1].pop_front())
                            .or_else(|| q[2].pop_front())
                        {
                            break p;
                        }
                        q = work.wait(q).unwrap_or_else(|e| e.into_inner());
                    }
                };
                // SAFETY: a plain call.
                let id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
                AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).push((id, p.clone()));
                p.make();
                AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != id);
                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                    drained();
                } else {
                    super::library::save(false);
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no pipeline worker");
    }
}

/// A worker is lost to a compile that crashed or never ends: another takes
/// its place (its pipeline stays unmade). With all of them lost, the rest of
/// a world's pipelines waited forever.
pub(super) fn worker_lost(id: u32) {
    static LOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
    let mut lost = None;
    AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| {
        if w.0 == id {
            w.1.lost.store(true, Ordering::Relaxed);
            lost = Some(w.1.priority_lane());
        }
        w.0 != id
    });
    if let Some(priority) = lost {
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
        drop(_held);
        WAITING.fetch_sub(1, Ordering::Relaxed);
        worker(LOST.fetch_add(1, Ordering::Relaxed), priority);
    }
}

/// Hands `p` to the workers (half the logical processors, 2 to 8, and two
/// for priority passes), started on first use.
fn queue(p: Arc<Pending>, now: bool) {
    static STARTED: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
    let priority = p.priority_lane();
    let slot = usize::from(priority);
    STARTED[slot].call_once(|| {
        let workers = if priority {
            2
        } else {
            std::thread::available_parallelism().map_or(2, |n| (n.get() * 3 / 4).clamp(2, 16))
        };
        for i in 0..workers {
            worker(i + if priority { 50 } else { 0 }, priority);
        }
    });
    let index = queue_index(&p, now);
    let (lock, work) = &QUEUES[slot];
    lock.lock().unwrap_or_else(|e| e.into_inner())[index].push_back(p);
    work.notify_one();
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panic_refuses_the_pipeline() {
        let p = Pending::new(Vec::new(), None);
        p.make_with(|| panic!("pipeline creation panicked"));
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn a_stuck_pipeline_is_not_waited_for_and_its_worker_is_replaced() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        p.queued.store(true, Ordering::Relaxed);
        let _ = p.begun.set(Instant::now());
        WORKERS_RUNNING.store(1, Ordering::Relaxed);
        assert!(!stuck(&p));
        std::thread::sleep(stuck_after() + Duration::from_millis(20));
        let before = NEXT_REPLACEMENT.load(Ordering::Relaxed);
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(started.elapsed() < Duration::from_millis(100), "waited for a stuck pipeline");
        assert_eq!(NEXT_REPLACEMENT.load(Ordering::Relaxed), before + 1);
        assert!(stuck(&p));
        assert_eq!(NEXT_REPLACEMENT.load(Ordering::Relaxed), before + 1);
        assert!(p.made.get().is_none());
    }

    /// Waits of one frame end together: a thread that begins to wait while
    /// another waits gets no budget of its own, waits one after the other
    /// share it, and the next frame has it whole again.
    #[test]
    fn waits_of_a_frame_share_its_budget() {
        let waits = Waits::new();
        let budget = Duration::from_millis(500);
        let ms = Duration::from_millis;
        let start = Instant::now();
        // Three threads begin within the first's wait: all end with it.
        let first = waits.begin(7, start, budget);
        assert_eq!(first, start + budget);
        assert_eq!(waits.begin(7, start + ms(200), budget), first);
        assert_eq!(waits.begin(7, start + ms(499), budget), first);
        for _ in 0..3 {
            waits.end(first);
        }
        // The budget is spent: a later wait of the frame ends at once.
        let later = start + ms(900);
        assert_eq!(waits.begin(7, later, budget), later);
        waits.end(later);
        // A thread that still names the frame before does not bring a
        // budget back.
        assert_eq!(waits.begin(6, later, budget), later);
        waits.end(later);
        // The next frame: waits one after the other add up.
        let next = start + ms(2000);
        assert_eq!(waits.begin(8, next, budget), next + budget);
        waits.end(next + ms(300));
        let again = next + ms(1000);
        assert_eq!(waits.begin(8, again, budget), again + ms(200));
        // A frame that begins while a wait is under way has its budget
        // from then on; the wait under way keeps its own end.
        let third = again + ms(100);
        assert_eq!(waits.begin(9, third, budget), third + budget);
        waits.end(again + ms(200));
        waits.end(third + budget);
        // Counted once, however many threads waited: 500 ms of frame 7,
        // 300 + 100 of frame 8 and 500 of frame 9.
        assert_eq!(waits.take(third + budget), ms(1400));
        assert_eq!(waits.take(third + budget), Duration::ZERO);
        // Taken while a wait is under way: counted up to then, the rest
        // at its end, and the wait's end stays where it was.
        let fourth = third + ms(5000);
        let end = waits.begin(10, fourth, budget);
        assert_eq!(waits.take(fourth + ms(120)), ms(120));
        assert_eq!(waits.begin(10, fourth + ms(200), budget), end);
        waits.end(end);
        waits.end(end);
        assert_eq!(waits.take(end), ms(380));
    }

    /// Recording threads that wait for pipelines never made, each beginning
    /// while the one before still waits: the last, which begins near the
    /// end of the first's wait, returns with it (it took a whole budget
    /// from its own start: the frame waited for nearly two).
    #[test]
    fn a_thread_that_waits_while_another_does_gets_no_budget_of_its_own() {
        static WAITS: Waits = Waits::new();
        let budget = Duration::from_millis(800);
        let started = Instant::now();
        let threads: Vec<_> = (0..4u32)
            .map(|n| {
                std::thread::spawn(move || {
                    std::thread::sleep(budget * n / 4);
                    let p = Pending::new(Vec::new(), None);
                    let began = Instant::now();
                    wait_within(&WAITS, budget, &p, 3);
                    (began.duration_since(started), began.elapsed())
                })
            })
            .collect();
        let waits: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        eprintln!("staggered waits on an 800 ms budget, (began, waited): {waits:?}");
        // Every wait is over a budget after the first of them began (or
        // at once, if it began later than that on a busy machine): the
        // last, 600 ms in, has 200 ms left to it and not 800.
        let first = waits.iter().map(|w| w.0).min().unwrap();
        let slack = Duration::from_millis(250);
        for (began, waited) in &waits {
            let over = (*began).max(first + budget) + slack;
            assert!(*began + *waited <= over, "{waits:?}");
        }
    }

    #[test]
    fn a_pipeline_is_waited_for_in_one_frame_only() {
        let p = Pending::new(Vec::new(), None);
        p.queued.store(true, Ordering::Relaxed);
        let frame = crate::frame::current();
        p.waited_frame.store(frame.wrapping_add(1), Ordering::Relaxed);
        let p = Arc::new(p);
        let started = Instant::now();
        assert!(ready(&p).is_none());
        assert!(started.elapsed() < Duration::from_millis(50), "waited {:?}", started.elapsed());
    }

    #[test]
    fn the_first_frame_claims_the_wait() {
        let p = Pending::new(Vec::new(), None);
        assert!(claim_wait(&p, 7));
        assert!(claim_wait(&p, 7));
        assert!(!claim_wait(&p, 8));
        assert_eq!(p.waited_frame.load(Ordering::Relaxed), 7);
    }

    #[test]
    fn depth_only_pipelines_go_ahead_of_other_early_starts() {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D32_FLOAT;
        let pending = |depth: bool| {
            let mut g = Box::new(GraphicsState::default());
            if depth {
                let subobject = [16u32, DXGI_FORMAT_D32_FLOAT.0 as u32];
                // SAFETY: a whole depth-format subobject of a stream.
                unsafe { g.read(16, subobject.as_ptr().cast()) };
            }
            Pending::new(Vec::new(), Some(g))
        };
        let (shadow, other) = (pending(true), pending(false));
        assert_eq!(queue_index(&shadow, false), 1);
        assert_eq!(queue_index(&other, false), 2);
        assert_eq!(queue_index(&shadow, true), 0);
        assert_eq!(queue_index(&other, true), 0);
    }

    #[test]
    fn lost_worker_wakes_waiter_immediately() {
        static WAITS: Waits = Waits::new();
        let p = Arc::new(Pending::new(Vec::new(), None));
        let waiter_p = p.clone();
        let waiter = std::thread::spawn(move || {
            let started = Instant::now();
            wait_within(&WAITS, Duration::from_secs(5), &waiter_p, 1);
            started.elapsed()
        });

        std::thread::sleep(Duration::from_millis(10));
        p.lost.store(true, Ordering::Relaxed);
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
        drop(_held);

        let elapsed = waiter.join().expect("waiter thread must not panic");
        assert!(elapsed < Duration::from_secs(2));
    }

    #[test]
    fn a_pipeline_for_bound_formats_is_made_once() {
        let p = Arc::new(Pending::new(
            Vec::new(),
            Some(Box::new(GraphicsState::declaring(24))),
        ));
        assert_eq!(p.formats, [24, 0, 0, 0, 0, 0, 0, 0, 0]);
        let g = p.graphics.as_deref().unwrap();
        let (a, b) = ([26, 0, 0, 0, 0, 0, 0, 0, 0], [28, 0, 0, 0, 0, 0, 0, 0, 0]);
        let first = variant(&p, g, &a);
        assert_eq!(first.formats, a);
        assert_ne!(first.key, p.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        let second = variant(&p, g, &b);
        assert_eq!(second.formats, b);
        assert!(!Arc::ptr_eq(first, second));
        assert_ne!(second.key, first.key);
        assert!(Arc::ptr_eq(first, variant(&p, g, &a)));
        assert!(Arc::ptr_eq(second, variant(&p, g, &b)));
        assert!(!first.queued.load(Ordering::Relaxed));
        assert!(!p.queued.load(Ordering::Relaxed));
    }
}
