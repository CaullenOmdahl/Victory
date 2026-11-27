//! Asynchronous texture detiling and `COPY`-queue upload worker
//! (`VICTORY_HEAP_WORKER=1`, off by default).
//!
//! Bindless descriptor placement (`crate::texture::upload_first`) calls
//! [`enqueue`] as soon as a placed texture enters a slot or view, instead of
//! waiting until submission time (`Early::take`). A dedicated background thread:
//! 1. runs `Staging::refresh` (memory commit checks, write-watch / BLAKE3,
//!    and CPU detiling into UPLOAD staging buffers) ahead of submission, and
//! 2. for textures still in `D3D12_RESOURCE_STATE_COMMON` (`0`), executes
//!    `CopyTextureRegion` on a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue,
//!    returning a `(ID3D12Fence, value)` for the `DIRECT` queue to GPU-wait on.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::resource::HostTexture;
use crate::texture::Staging;

const BATCH_LIMIT: usize = 64;
const ALLOCATORS: usize = 4;

static COPIED: AtomicU64 = AtomicU64::new(0);
static READY_DIRECT: AtomicU64 = AtomicU64::new(0);
static UNPROCESSED: AtomicU64 = AtomicU64::new(0);
static WORKER_US: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// Per-thread override for [`on`] in unit tests.
    pub(crate) static HEAP_WORKER_HERE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Whether the async heap texture worker is enabled (`VICTORY_HEAP_WORKER=1`,
/// off by default).
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(enabled) = HEAP_WORKER_HERE.get() {
        return enabled;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_WORKER").is_some_and(|v| v == "1"))
}

pub(crate) struct Drain {
    pub(crate) wait: Option<(ID3D12Fence, u64)>,
    pub(crate) copied: Vec<(Arc<Staging>, Arc<HostTexture>)>,
    pub(crate) ready_direct: Vec<(Arc<Staging>, Arc<HostTexture>)>,
    pub(crate) unprocessed: Vec<(Arc<Staging>, Arc<HostTexture>)>,
}

struct CopyGpu {
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    event: usize,
    allocators: [(ID3D12CommandAllocator, u64); ALLOCATORS],
    next_alloc: usize,
    list: ID3D12GraphicsCommandList,
    submitted: u64,
    in_flight: VecDeque<(u64, Vec<(Arc<Staging>, Arc<HostTexture>)>)>,
}

impl Drop for CopyGpu {
    fn drop(&mut self) {
        if self.event != 0 {
            // SAFETY: `self.event` was created by `CreateEventW` and is owned by `self`.
            unsafe {
                let _ = CloseHandle(HANDLE(self.event as *mut _));
            }
        }
    }
}

impl CopyGpu {
    fn new(device: &ID3D12Device) -> Option<Self> {
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            ..Default::default()
        };
        // SAFETY: `device` is a live `ID3D12Device`; descriptions are valid for `COPY` queue/fence/list.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&queue_desc).ok()?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let a0: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
            let a1: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
            let a2: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
            let a3: ID3D12CommandAllocator =
                device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY).ok()?;
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &a0, None)
                .ok()?;
            list.Close().ok()?;
            Some(Self {
                queue,
                fence,
                event: event.0 as usize,
                allocators: [(a0, 0), (a1, 0), (a2, 0), (a3, 0)],
                next_alloc: 0,
                list,
                submitted: 0,
                in_flight: VecDeque::new(),
            })
        }
    }

    fn completed(&self) -> Option<u64> {
        // SAFETY: `self.fence` is a live `ID3D12Fence`.
        let val = unsafe { self.fence.GetCompletedValue() };
        (val != u64::MAX).then_some(val)
    }

    fn reap(&mut self) {
        let Some(done) = self.completed() else {
            return;
        };
        while self
            .in_flight
            .front()
            .is_some_and(|(fence, _)| *fence <= done)
        {
            self.in_flight.pop_front();
        }
    }

    fn submit_batch(
        &mut self,
        items: Vec<(Arc<Staging>, Arc<HostTexture>)>,
    ) -> Option<(ID3D12Fence, u64)> {
        if items.is_empty() {
            return None;
        }
        self.reap();
        let slot = self.next_alloc;
        self.next_alloc = (slot + 1) % ALLOCATORS;
        let reuse_fence = self.allocators[slot].1;
        if reuse_fence != 0 && self.completed().is_some_and(|c| c < reuse_fence) {
            // SAFETY: `self.fence` and `self.event` are owned by `self`.
            unsafe {
                if self
                    .fence
                    .SetEventOnCompletion(reuse_fence, HANDLE(self.event as *mut _))
                    .is_ok()
                {
                    WaitForSingleObject(HANDLE(self.event as *mut _), INFINITE);
                }
            }
            self.reap();
        }
        let alloc = &self.allocators[slot].0;
        // SAFETY: `alloc` is past `reuse_fence` and `self.list` is currently closed.
        unsafe {
            alloc.Reset().ok()?;
            self.list.Reset(alloc, None).ok()?;
            for (staging, copy) in &items {
                staging.record_copies(&self.list, &copy.texture);
            }
            self.list.Close().ok()?;
        }
        let cmd: ID3D12CommandList = self.list.cast().ok()?;
        self.submitted += 1;
        let val = self.submitted;
        // SAFETY: closed `COPY` command list; `self.queue` and `self.fence` are live.
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            if self.queue.Signal(&self.fence, val).is_err() {
                return None;
            }
        }
        self.allocators[slot].1 = val;
        self.in_flight.push_back((val, items));
        Some((self.fence.clone(), val))
    }
}

#[derive(Default)]
struct State {
    pending: VecDeque<(Arc<Staging>, Arc<HostTexture>)>,
    working: bool,
    copied: Vec<(Arc<Staging>, Arc<HostTexture>)>,
    ready_direct: Vec<(Arc<Staging>, Arc<HostTexture>)>,
    last_copy_fence: Option<(ID3D12Fence, u64)>,
    last_waited_fence: u64,
    stop: bool,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    idle: Condvar,
}

// SAFETY: wrapped D3D12 interfaces (`ID3D12CommandQueue`, `ID3D12Fence`, `ID3D12Device`, etc.)
// are free-threaded COM objects and synchronized by `Shared::state`.
unsafe impl Send for Shared {}
// SAFETY: as above.
unsafe impl Sync for Shared {}

pub(crate) struct HeapWorker {
    shared: Arc<Shared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for HeapWorker {
    fn drop(&mut self) {
        {
            let mut s = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            s.stop = true;
            self.shared.wake.notify_all();
        }
        if let Some(handle) = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take() {
            let _ = handle.join();
        }
    }
}

impl HeapWorker {
    pub(crate) fn new(device: Option<ID3D12Device>) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            idle: Condvar::new(),
        });
        let worker_shared = shared.clone();
        let handle = std::thread::Builder::new()
            .name("victory-heap-worker".into())
            .spawn(move || Self::run(worker_shared, device))
            .ok();
        if handle.is_none() {
            tracing::warn!("failed to spawn victory-heap-worker; falling back to inline uploads");
        }
        Self {
            shared,
            thread: Mutex::new(handle),
        }
    }

    pub(crate) fn enqueue(&self, staging: Arc<Staging>, copy: Arc<HostTexture>) -> bool {
        if self.thread.lock().unwrap_or_else(|e| e.into_inner()).is_none() {
            return false;
        }
        let mut s = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.stop {
            return false;
        }
        s.pending.push_back((staging, copy));
        self.shared.wake.notify_one();
        true
    }

    pub(crate) fn flush_for_submit(&self) -> Drain {
        let mut s = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let unprocessed: Vec<_> = s.pending.drain(..).collect();
        while s.working {
            s = self.shared.idle.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        let copied = std::mem::take(&mut s.copied);
        let ready_direct = std::mem::take(&mut s.ready_direct);
        let wait = match &s.last_copy_fence {
            Some((fence, val)) if *val > s.last_waited_fence => Some((fence.clone(), *val)),
            _ => None,
        };
        if let Some((_, val)) = &wait {
            s.last_waited_fence = *val;
        }
        UNPROCESSED.fetch_add(unprocessed.len() as u64, Ordering::Relaxed);
        Drain {
            wait,
            copied,
            ready_direct,
            unprocessed,
        }
    }

    fn run(shared: Arc<Shared>, device: Option<ID3D12Device>) {
        victory_core::affinity::pin_worker_ccd();
        let mut copy_gpu = device.as_ref().and_then(CopyGpu::new);
        loop {
            let batch: Vec<(Arc<Staging>, Arc<HostTexture>, std::sync::atomic::AtomicBool)> = {
                let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                while s.pending.is_empty() && !s.stop {
                    s = shared.wake.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                if s.pending.is_empty() && s.stop {
                    return;
                }
                let mut seen = crate::fx::Set::default();
                let mut batch = Vec::with_capacity(s.pending.len().min(BATCH_LIMIT));
                while batch.len() < BATCH_LIMIT
                    && let Some((staging, copy)) = s.pending.pop_front()
                {
                    let key = (
                        Arc::as_ptr(&staging) as usize,
                        Arc::as_ptr(&copy) as usize,
                    );
                    if seen.insert(key) {
                        batch.push((staging, copy, std::sync::atomic::AtomicBool::new(false)));
                    }
                }
                s.working = true;
                batch
            };

            let started = std::time::Instant::now();
            if batch.len() > 2 {
                crate::host::in_parallel(&batch, 2, |chunk| {
                    for (staging, _, dirty) in chunk {
                        if staging.refresh() {
                            dirty.store(true, Ordering::Relaxed);
                        }
                    }
                });
            } else {
                for (staging, _, dirty) in &batch {
                    if staging.refresh() {
                        dirty.store(true, Ordering::Relaxed);
                    }
                }
            }

            let mut to_copy = Vec::new();
            let mut to_direct = Vec::new();
            for (staging, copy, dirty) in batch {
                if !dirty.into_inner() {
                    continue;
                }
                if copy_gpu.is_some() && copy.state.load(Ordering::Acquire) == 0 {
                    to_copy.push((staging, copy));
                } else {
                    to_direct.push((staging, copy));
                }
            }

            let mut copied_now = Vec::new();
            let mut fence_now = None;
            if !to_copy.is_empty() {
                if let Some(gpu) = copy_gpu.as_mut() {
                    let clone_items = to_copy.clone();
                    if let Some(f) = gpu.submit_batch(clone_items) {
                        COPIED.fetch_add(to_copy.len() as u64, Ordering::Relaxed);
                        copied_now = to_copy;
                        fence_now = Some(f);
                    } else {
                        to_direct.extend(to_copy);
                    }
                } else {
                    to_direct.extend(to_copy);
                }
            }
            if !to_direct.is_empty() {
                READY_DIRECT.fetch_add(to_direct.len() as u64, Ordering::Relaxed);
            }
            WORKER_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);

            let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(f) = fence_now {
                s.last_copy_fence = Some(f);
            }
            if !copied_now.is_empty() {
                s.copied.extend(copied_now);
            }
            if !to_direct.is_empty() {
                s.ready_direct.extend(to_direct);
            }
            if let Some(gpu) = copy_gpu.as_mut() {
                gpu.reap();
            }
            s.working = false;
            shared.idle.notify_all();
        }
    }
}

static WORKER: OnceLock<HeapWorker> = OnceLock::new();

fn worker() -> &'static HeapWorker {
    WORKER.get_or_init(|| HeapWorker::new(crate::host::host().map(|h| h.device.clone())))
}

/// Queues a newly visible bindless texture to detile and upload asynchronously.
pub(crate) fn enqueue(staging: Arc<Staging>, copy: Arc<HostTexture>) -> bool {
    if !on() {
        return false;
    }
    worker().enqueue(staging, copy)
}

/// Drains finished async uploads before a `DIRECT` submission runs.
pub(crate) fn flush_before_submit() -> Option<Drain> {
    if !on() {
        return None;
    }
    WORKER.get().map(HeapWorker::flush_for_submit)
}

#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let copied = COPIED.swap(0, Ordering::Relaxed);
    let direct = READY_DIRECT.swap(0, Ordering::Relaxed);
    let unproc = UNPROCESSED.swap(0, Ordering::Relaxed);
    let ms = WORKER_US.swap(0, Ordering::Relaxed) as f64 / 1000.0;
    if copied + direct + unproc > 0 || ms > 0.0 {
        tracing::info!(
            "heap_worker (VICTORY_HEAP_WORKER=1) per frame: {:.1} copy_queue uploads, \
             {:.1} pre-detiled direct, {:.1} inline fallback ({:.1} worker ms/frame)",
            copied as f64 / frames,
            direct as f64 / frames,
            unproc as f64 / frames,
            ms / frames,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::Host;
    use crate::texture::{create_staging, tests::test_helpers};

    #[test]
    fn heap_worker_off_by_default_and_toggles_per_thread() {
        struct Reset(Option<bool>);
        impl Drop for Reset {
            fn drop(&mut self) {
                HEAP_WORKER_HERE.set(self.0);
            }
        }
        assert_eq!(
            on(),
            std::env::var_os("VICTORY_HEAP_WORKER").is_some_and(|v| v == "1")
        );
        let _r = Reset(HEAP_WORKER_HERE.replace(Some(true)));
        assert!(on());
        HEAP_WORKER_HERE.set(Some(false));
        assert!(!on());
    }

    #[test]
    fn copy_queue_uploads_common_texture_before_direct_readback() {
        if let Some(host) = Host::new_warp() {
            run_copy_queue_roundtrip(&host);
        } else if let Some(host) = crate::host::host() {
            run_copy_queue_roundtrip(host);
        }
    }

    fn run_copy_queue_roundtrip(host: &Host) {
        let (width, height) = (4, 4);
        let (copy, tex) = test_helpers::make_rgba_texture(host, width, height);
        let layout = tex.planes[0].layout.clone();
        let mut title_mem = vec![0u8; layout.size as usize];
        let expected = [0x33, 0x66, 0x99, 0xff];
        test_helpers::fill_linear_rgba(&layout, &mut title_mem, expected);

        let staging = create_staging(
            host,
            None,
            0,
            std::ptr::null(),
            &copy.texture,
            layout,
            title_mem.as_ptr(),
        )
        .expect("create staging");

        let hw = HeapWorker::new(Some(host.device.clone()));
        assert!(hw.enqueue(staging, copy.clone()));

        // Poll briefly for background worker to execute the COPY batch.
        let mut drain = None;
        for _ in 0..200 {
            let d = hw.flush_for_submit();
            if d.wait.is_some() || !d.ready_direct.is_empty() || !d.copied.is_empty() {
                drain = Some(d);
                break;
            }
            if !d.unprocessed.is_empty() {
                // Re-enqueue if flush raced before worker dequeued it.
                for (s, c) in d.unprocessed {
                    assert!(hw.enqueue(s, c));
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let drain = drain.expect("heap_worker should process COMMON upload");
        assert!(!drain.copied.is_empty(), "expected copy_queue upload for COMMON texture");
        let (copy_fence, val) = drain.wait.expect("copy_queue fence returned");

        // GPU wait on the DIRECT queue, transition COMMON -> SHADER_RESOURCE, then read back.
        host.wait_on_queue(&copy_fence, val);
        let list = host.new_list().expect("transition list");
        // SAFETY: `copy.texture` starts in COMMON (`0`) after COPY queue implicit decay.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                &copy.texture,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATES(0x40 | 0x80),
            )]);
        }
        copy.state.store(0x40 | 0x80, Ordering::Relaxed);
        list.close();
        host.execute(&[&list]);

        let got = test_helpers::read_back_rgba(host, &copy, width, height);
        assert!(test_helpers::all_pixels_eq(&got, expected), "pixel mismatch after COPY upload");
    }
}
