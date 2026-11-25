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

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

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
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
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
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            lost: AtomicBool::new(false),
            begun: OnceLock::new(),
            replaced: AtomicBool::new(false),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    fn make(&self) {
        let _ = self.begun.set(Instant::now());
        // Not `get_or_init`: a worker lost in it would leave it locked.
        let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
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
    }
    if p.made.get().is_none() {
        start(p);
        if p.lost.load(Ordering::Relaxed) || stuck(p) {
            return None;
        }
        // Also with a world's worth queued: passes the title runs only once
        // set up what later ones loop over.
        wait(p, crate::frame::current());
    }
    p.made.get()?.as_ref()
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>) {
    if !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        queue(p.clone());
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p);
    }
}

/// Waits for `p` while frame `frame` has budget of its kind left.
fn wait(p: &Pending, frame: u64) {
    static FRAME: [AtomicU64; 2] = [const { AtomicU64::new(u64::MAX) }; 2];
    static WAITED_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let kind = usize::from(p.screen);
    let budget = [wait_budget(), SCREEN_BUDGET][kind];
    if budget.is_zero() || stuck(p) {
        return;
    }
    let waited = &WAITED_US[kind];
    if FRAME[kind].swap(frame, Ordering::Relaxed) != frame {
        waited.store(0, Ordering::Relaxed);
    }
    let left = budget.saturating_sub(Duration::from_micros(waited.load(Ordering::Relaxed)));
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() && !stuck(p) {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    waited.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What the workers take from: the ordinary pipelines, the screen passes.
static RECEIVER: [OnceLock<Mutex<std::sync::mpsc::Receiver<Arc<Pending>>>>; 2] =
    [const { OnceLock::new() }; 2];

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
            let Some(receiver) = RECEIVER[usize::from(screen)].get() else { return };
            loop {
                let next = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                let Ok(p) = next else { return };
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
            lost = Some(w.1.screen);
        }
        w.0 != id
    });
    if let Some(screen) = lost {
        WAITING.fetch_sub(1, Ordering::Relaxed);
        worker(LOST.fetch_add(1, Ordering::Relaxed), screen);
    }
}

/// Hands `p` to the workers (half the logical processors, 2 to 8, and two
/// for screen passes), started on first use.
fn queue(p: Arc<Pending>) {
    static SENDER: [OnceLock<std::sync::mpsc::Sender<Arc<Pending>>>; 2] =
        [const { OnceLock::new() }; 2];
    let screen = p.screen;
    let sender = SENDER[usize::from(screen)].get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Arc<Pending>>();
        let _ = RECEIVER[usize::from(screen)].set(Mutex::new(receiver));
        let workers = if screen {
            2
        } else {
            std::thread::available_parallelism().map_or(2, |n| (n.get() * 3 / 4).clamp(2, 16))
        };
        for i in 0..workers {
            worker(i + if screen { 50 } else { 0 }, screen);
        }
        sender
    });
    if let Err(e) = sender.send(p) {
        // No worker: made here instead.
        e.0.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }
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
}
