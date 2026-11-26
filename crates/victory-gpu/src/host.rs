//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

/// An asynchronous submission still running on the GPU: its staging textures
/// and bindless plans stay alive until `fence` completes.
struct InFlight {
    fence: u64,
    submission: u64,
    _early_copies: Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )>,
    _plans: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// What is done once it has run ([`Host::finish_submission`]): its
    /// lists, kept alive for that, and the ranges its shaders wrote.
    finish: Option<Finish>,
}

/// See [`InFlight::finish`].
struct Finish {
    lists: Vec<std::sync::Arc<HostList>>,
    back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    late: Vec<(usize, usize, usize, bool)>,
    late_refresh: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    card_plan: Option<crate::card::CardPlan>,
}
// SAFETY: host lists are free-threaded objects, run by one thread once
// closed; the rest is ours and locked.
unsafe impl Send for Finish {}

/// Whether the title's submissions may return before the host GPU is done
/// (`VICTORY_ASYNC_SUBMIT=1`; off by default).
fn async_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1"))
}

/// Microseconds spent bringing buffer copies up to date while the
/// submission before ran ([`Host::prewarm`]), since the last frame line.
static PREWARM_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EARLY_FILLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EARLY_PLAN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static EARLY_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The newest unwaited submission (its fence value) that is finished:
/// waited for and its results put back.
static FINISHED_FENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds_supported: bool,
    /// Resizable BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) supported by device/OS.
    pub(crate) gpu_upload: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// Async submissions not yet retired, oldest first.
    in_flight: Mutex<std::collections::VecDeque<InFlight>>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// Device-local VRAM card buffer engine (Wave64 GPU page hashing), lazily initialized.
    card: OnceLock<Option<Mutex<crate::card::CardEngine>>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission for most of a second in the world: freeing there was
/// a GPU page fault on the freed texture (DXGI_ERROR_DEVICE_HUNG,
/// 2026-11-24, RX 7600).
static RETIRED: Mutex<std::collections::VecDeque<Retired>> =
    Mutex::new(std::collections::VecDeque::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Presented host frames a retired object is also kept for. Worker threads
/// record lists ahead of submission and a title frame spans many
/// submissions, so finishing the submissions running when it was retired
/// is not enough on its own. Ages are in frames presented to the host, not
/// the title's 60 Hz clock (which ticks many times per host frame while
/// the host runs at 1-2 fps).
const KEEP_FRAMES: u64 = 32;

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired {
    /// The submission count it waits for.
    submission: u64,
    /// The presented frame it was retired in.
    frame: u64,
    #[allow(dead_code)]
    object: Box<dyn std::any::Any>,
}
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Drops `object` once every submission running now has finished and
/// [`KEEP_FRAMES`] more frames were presented.
pub(crate) fn retire<T: 'static>(object: T) {
    let submission = STARTED.load(Ordering::Acquire);
    let frame = crate::queue::presented();
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push_back(Retired {
        submission,
        frame,
        object: Box::new(object),
    });
}

fn bury_aside_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BURY_ASIDE").is_none_or(|v| v != "0"))
}

/// Drops what waited for submissions up to `finished` and is old enough
/// in presented frames.
fn bury(finished: u64) {
    // Diagnostics (`VICTORY_KEEP_RETIRED`): nothing is freed, to tell a use
    // after free from a wild address.
    if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_KEEP_RETIRED").is_some() {
        return;
    }
    let now = crate::queue::presented();
    let done: Vec<Retired> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        // In the order they came: the old ones are at the front.
        let n = retired
            .iter()
            .take_while(|r| {
                r.submission <= finished && now.saturating_sub(r.frame) >= KEEP_FRAMES
            })
            .count();
        retired.drain(..n).collect()
    };
    if done.is_empty() {
        return;
    }
    if !bury_aside_enabled() {
        drop(done);
        return;
    }
    static TO: std::sync::OnceLock<Option<Mutex<std::sync::mpsc::Sender<Vec<Retired>>>>> =
        std::sync::OnceLock::new();
    let to = TO.get_or_init(|| {
        let (to, from) = std::sync::mpsc::channel::<Vec<Retired>>();
        std::thread::Builder::new()
            .name("victory-bury".into())
            .spawn(move || {
                for done in from {
                    drop(done);
                }
            })
            .ok()
            .map(|_| Mutex::new(to))
    });
    match to {
        Some(to) => {
            if let Err(e) = to.lock().unwrap_or_else(|e| e.into_inner()).send(done) {
                drop(e.0);
            }
        }
        None => drop(done),
    }
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Time spent retiring completed asynchronous submissions in `reap_completed`
/// since the last frame-rate line, in microseconds: total reap, then
/// `finish_submission` (copy-back + deferred copies + list resets).
#[cfg(feature = "diagnostics")]
static ASYNC_REAP_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Bindless heaps' slots made ready (microseconds), since the last
/// frame-rate line: plans at submission (part of the preparing), and
/// placed textures as draws are recorded.
pub(crate) static BINDLESS_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU's memory use as its driver reports it, `[(used, budget)]`
/// in video memory and in system memory; `None` before the device is
/// made (asking does not make it).
pub(crate) fn video_memory() -> Option<[(u64, u64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let host = HOST.get()?.as_ref()?;
    let adapter = ADAPTER
        .get_or_init(|| {
            // SAFETY: plain DXGI queries on a live device.
            unsafe {
                let luid = host.device.GetAdapterLuid();
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid::<IDXGIAdapter3>(luid).ok()
            }
        })
        .as_ref()?;
    let query = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: node 0 of the device's own adapter, a sized out structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage, info.Budget))
    };
    Some([
        query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?,
        query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?,
    ])
}

pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

/// Waits for any in-flight async host GPU submissions without creating the
/// host device if it has not been initialised yet.
/// Whether a submission is still to be waited for and finished.
pub(crate) fn has_in_flight() -> bool {
    matches!(HOST.get(), Some(Some(h)) if h.has_in_flight())
}

pub(crate) fn wait_in_flight_if_ready() {
    if let Some(Some(h)) = HOST.get() {
        h.wait_in_flight();
    }
}

pub(crate) fn in_flight_fence() -> u64 {
    let Some(Some(h)) = HOST.get() else {
        return 0;
    };
    h.in_flight
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .back()
        .map_or(0, |f| f.fence)
}

pub(crate) fn finished_fence() -> u64 {
    FINISHED_FENCE.load(Ordering::Acquire)
}

pub(crate) fn finish_in_flight_apart() {
    static EVENT: OnceLock<Option<usize>> = OnceLock::new();
    let Some(Some(h)) = HOST.get() else { return };
    // SAFETY: plain creation of an unnamed auto-reset event.
    let event = EVENT.get_or_init(|| {
        unsafe { CreateEventW(None, false, false, None) }
            .ok()
            .map(|e| e.0 as usize)
    });
    match event {
        Some(event) => h.finish_in_flight_apart(HANDLE(*event as *mut _)),
        None => h.wait_in_flight(),
    }
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            match fastest_adapter() {
                Ok(adapter) => D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?,
                Err(e) => {
                    tracing::warn!(error = %e, "no adapter preference; the default is used");
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
                }
            }
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let host = Host::from_device(device, messages)?;
        tracing::info!("host D3D12 device ready");
        Ok(host)
    }
}

impl Host {
    fn from_device(
        device: ID3D12Device,
        messages: Option<ID3D12InfoQueue>,
    ) -> windows::core::Result<Host> {
        // SAFETY: plain D3D12 feature query and queue, fence, event and
        // descriptor heap creation.
        unsafe {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            let depth_bounds_supported = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS2,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .is_ok()
                && options.DepthBoundsTestSupported.as_bool();
            let mut options16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            let gpu_upload = !cfg!(test)
                && std::env::var_os("VICTORY_GPU_UPLOAD").is_none_or(|v| v != "0")
                && device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS16,
                        (&raw mut options16).cast(),
                        size_of_val(&options16) as u32,
                    )
                    .is_ok()
                && options16.GPUUploadHeapSupported.as_bool();
            tracing::info!(gpu_upload, "D3D12 GPU upload heap (Resizable BAR)");
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            let event = CreateEventW(None, false, false, None)?;
            let rtvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let dsvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            Ok(Host {
                device,
                depth_bounds_supported,
                gpu_upload,
                queue,
                fence,
                submitted: Mutex::new(0),
                in_flight: Mutex::new(std::collections::VecDeque::new()),
                event,
                rtvs,
                rtv_size,
                dsvs,
                dsv_size,
                next_dsv: AtomicU32::new(0),
                messages,
                next_rtv: AtomicU32::new(0),
                presenter: Mutex::new(None),
                card: OnceLock::new(),
            })
        }
    }

    fn card_engine(&self) -> Option<&Mutex<crate::card::CardEngine>> {
        self.card
            .get_or_init(|| match crate::card::CardEngine::new(self) {
                Ok(engine) => {
                    tracing::info!("VRAM card buffer engine ready (Wave64 GPU hashing)");
                    Some(Mutex::new(engine))
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "VRAM card buffer engine unavailable; falling back to host RAM"
                    );
                    None
                }
            })
            .as_ref()
    }

    pub(crate) fn card_ready(&self) -> bool {
        self.card_engine().is_some()
    }

    /// Executes a single command list on the direct queue and waits for it to
    /// finish. Caller MUST hold `self.submitted`.
    pub(crate) fn execute_and_wait_locked(&self, cmd: ID3D12CommandList, submitted: &mut u64) {
        *submitted += 1;
        let fence_val = *submitted;
        // SAFETY: caller holds `self.submitted`; `cmd` is closed; fence & event are ours.
        unsafe {
            self.queue.ExecuteCommandLists(&[Some(cmd)]);
            let waited = self
                .queue
                .Signal(&self.fence, fence_val)
                .and_then(|()| self.fence.SetEventOnCompletion(fence_val, self.event));
            match waited {
                Ok(()) => {
                    WaitForSingleObject(self.event, INFINITE);
                }
                Err(e) => tracing::warn!(error = %e, "card GPU wait failed"),
            }
        }
    }

    /// Creates an isolated WARP-backed `Host` for unit tests without touching
    /// the process-global host singleton.
    #[cfg(test)]
    pub(crate) fn new_warp() -> Option<Host> {
        let adapter = warp_adapter().ok()?;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `adapter` is a verified software WARP adapter.
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()? };
        Host::from_device(device?, None).ok()
    }
}

/// The adapter Windows ranks fastest, so a machine with an integrated GPU
/// too runs on its discrete one.
fn fastest_adapter() -> windows::core::Result<IDXGIAdapter1> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter1 =
            factory.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)?;
        let desc = adapter.GetDesc1()?;
        let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        tracing::info!(
            name = String::from_utf16_lossy(&desc.Description[..end]),
            vram_mb = desc.DedicatedVideoMemory >> 20,
            "host adapter"
        );
        Ok(adapter)
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn chunks_trim_after_idle_window() {
        let (mut peak, mut window) = (0, 0);
        assert_eq!(super::trim_chunks(12, 1, &mut peak, &mut window), super::CHUNK_KEEP);
        for _ in 1..super::CHUNK_IDLE_RESETS - 1 {
            assert_eq!(super::trim_chunks(8, 1, &mut peak, &mut window), 8);
        }
        assert_eq!(super::trim_chunks(8, 1, &mut peak, &mut window), super::CHUNK_SPARE);
        assert_eq!((peak, window), (0, 0));

        for i in 0..super::CHUNK_IDLE_RESETS - 1 {
            let used = if i == 5 { 4 } else { 1 };
            assert_eq!(super::trim_chunks(8, used, &mut peak, &mut window), 8);
        }
        assert_eq!(super::trim_chunks(8, 1, &mut peak, &mut window), 4);
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    /// A list with no work before its waits, as the title's queues meet one
    /// another halfway: each write goes as soon as the waits before it are
    /// met, and work holds the later ones back.
    #[test]
    fn writes_before_an_unmet_wait_go_early() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut memory = [0u64; 4];
        let at = memory.as_mut_ptr();
        // SAFETY: inside the array.
        let slot = |i: usize| unsafe { at.add(i) };
        let wait = |i: usize| crate::memwait::Wait {
            address: slot(i) as usize,
            comparison: 7,
            reference: 5,
            bytes: 8,
        };
        let write = |i: usize| list.defer_write(slot(i).cast(), crate::queue::Written::Value(5), 8);
        // SAFETY: as above.
        let value = |i: usize| unsafe { slot(i).read_volatile() };
        list.note_wait(wait(0));
        write(1);
        list.note_wait(wait(2));
        list.count_work();
        write(3);
        assert!(!list.write_early() && value(1) == 0);
        // SAFETY: as above.
        unsafe { slot(0).write_volatile(5) };
        assert_eq!(list.unmet(), [wait(2)]);
        assert!(list.write_early() && !list.write_early());
        assert_eq!((value(1), value(3)), (5, 0));
        // SAFETY: as above.
        unsafe { slot(2).write_volatile(5) };
        assert!(list.unmet().is_empty() && !list.write_early());
        // Run: what is left, then all of it anew for another submission.
        list.close();
        host.execute(&[&list]);
        assert_eq!(value(3), 5);
        // SAFETY: as above.
        unsafe { slot(1).write_volatile(0) };
        host.execute(&[&list]);
        assert_eq!(value(1), 5);
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    fn defer_refreshes_matches_one_call_per_range() {
        let Some(host) = host() else { return };
        let Ok(one) = host.new_list() else { return };
        let Ok(batched) = host.new_list() else { return };
        let Some(shadow) = crate::shadow::find(0x1000, 0x100) else {
            return;
        };
        let ranges = [
            (shadow.clone(), 0x1000usize, 0x100usize),
            (shadow.clone(), 0x1100, 0x100),
            (shadow.clone(), 0x1000, 0x100),
        ];
        for (shadow, address, len) in &ranges {
            one.defer_refresh(shadow, *address, *len);
        }
        batched.defer_refreshes(&ranges);
        let one = one.deferred();
        let batched = batched.deferred();
        assert_eq!(one.copies.len(), 2);
        assert_eq!(batched.copies.len(), 2);
        for (a, b) in one.copies.iter().zip(&batched.copies) {
            let (Step::Refresh(a, aa, al), Step::Refresh(b, ba, bl)) = (a, b) else {
                panic!("only refreshes were deferred");
            };
            assert_eq!(std::sync::Arc::as_ptr(a), std::sync::Arc::as_ptr(b));
            assert_eq!((aa, al), (ba, bl));
        }
    }

    #[test]
    fn preludes_planned_in_order_record_on_several_threads() {
        use super::*;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        let Some(host) = super::host() else {
            return;
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 16,
            Height: 16,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let textures: Vec<_> = (0..400)
            .map(|_| {
                let mut texture: Option<ID3D12Resource> = None;
                // SAFETY: plain test resource creation.
                unsafe {
                    host.device.CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut texture,
                    )
                }
                .unwrap();
                std::sync::Arc::new(crate::resource::HostTexture {
                    texture: texture.unwrap(),
                    state: std::sync::atomic::AtomicU32::new(0),
                    bytes: 0,
                    slot: None,
                    skip: 0,
                    written: std::sync::atomic::AtomicU64::new(0),
                    twin: std::sync::OnceLock::new(),
                })
            })
            .collect();
        const STATES: [u32; 4] = [0x4, 0x80, 0x800, 0x40];
        let lists: Vec<_> = (0..8).map(|_| host.new_list().unwrap()).collect();
        let mut later = Vec::new();
        for (k, list) in lists.iter().enumerate() {
            for t in &textures {
                list.expect_texture_state(t, STATES[k % 4]);
            }
            list.close();
            let Prelude::Barriers(barriers) = list.plan_prelude(&[]) else {
                panic!("list {k}: expected barriers prelude");
            };
            later.push((2 * k, list, barriers));
        }
        let made = record_preludes(&later);
        assert_eq!(
            made.iter().map(|(at, _)| *at).collect::<Vec<_>>(),
            [0, 2, 4, 6, 8, 10, 12, 14]
        );
        assert!(made.iter().all(|(_, prelude)| prelude.is_some()));
        for list in &lists {
            list.reset();
        }
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Readback memory for texture-to-buffer copies: 512-byte-aligned
    /// chunks shared by the list's read backs, reused after a reset.
    readbacks: Mutex<Readbacks>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// What the title's list held for this recording, if the title reset or
    /// released it while a submission of it was still to run.
    kept: Mutex<Vec<Box<dyn std::any::Any + Send>>>,
    /// Draws and dispatches recorded.
    work: Mutex<usize>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Queries begun in the list and not ended yet: host heap, type, index,
    /// and whether a draw since was not run on the host (`missed_draw`).
    queries: Mutex<Vec<(ID3D12QueryHeap, D3D12_QUERY_TYPE, u32, bool)>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<Budget>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Host fence value signalled by the last submission that ran this list.
    last_fence: std::sync::atomic::AtomicU64,
    /// Dispatches and buffer copies recorded since the reset: what may have
    /// written indirect arguments (`crate::indirect` clamps hoisted before).
    writes: std::sync::atomic::AtomicU64,
    /// Buffers for clamped indirect arguments when this list holds the
    /// clamps of the list after it (`clamp_room`).
    clamp_chunks: Mutex<ClampChunks>,
    /// Single-record indirect arguments gathered for this list's prelude
    /// (`crate::indirect::gather`).
    gather: Mutex<Gather>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// A list's buffers for hoisted clamped indirect arguments with their
/// states, how many are in use since the reset, the bytes taken of the last
/// of those, whether the list has the barrier that precedes the clamps, and
/// whether it has the clamps' root signature, pipeline and budget bound.
#[derive(Default)]
struct ClampChunks {
    all: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)>,
    used: usize,
    at: u64,
    barrier: bool,
    bound: bool,
}

#[derive(Default)]
struct Gather {
    closed: bool,
    sets: Vec<crate::indirect::GatherSet>,
    budget: Option<ID3D12Resource>,
    spare: Vec<ID3D12Resource>,
    runs: Vec<crate::indirect::GatherSet>,
    run_spare: Vec<(ID3D12Resource, bool)>,
}

const GATHER_SETS: usize = 8;

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 16384;

impl Drop for HostList {
    fn drop(&mut self) {
        let fence = self.last_fence.load(Ordering::Acquire);
        if fence != 0
            && let Some(h) = host()
        {
            h.wait_fence(fence);
        }
        crate::memstat::sub(&crate::memstat::LISTS, 1);
        self.release_kept();
    }
}

// SAFETY: D3D12 lists and allocators may be used from any thread, one at
// a time: the title records a list on one thread at a time, and a
// submission runs it after its recording ended (a reset while one is to
// run moves recording to a new list). The rest is locked.
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

enum Budget {
    Card(ID3D12Resource),
    Shared(ID3D12Resource, usize),
}

/// Whether indirect-work budgets live in device-local VRAM (`VICTORY_INDIRECT_CARD_BUDGET=0`
/// falls back to mapped upload memory).
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Notes how long the host GPU took for a submission: how many took how
/// long, and of those of 3 ms and more which pipelines they ran.
#[cfg(feature = "diagnostics")]
fn gpu_waits(gpu: std::time::Duration, runs: &[&HostList]) {
    let mut waits = GPU_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let us = gpu.as_micros() as u64;
    let bucket = [200, 1000, 3000, 10_000, 30_000].iter().position(|&m| us < m).unwrap_or(5);
    waits.0[bucket].0 += 1;
    waits.0[bucket].1 += us;
    if us >= 3000 {
        let mut pipelines: Vec<String> = runs
            .iter()
            .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .collect();
        pipelines.sort_unstable();
        pipelines.dedup();
        let mut key = format!("{} lists {} pipelines:", runs.len(), pipelines.len());
        for p in pipelines.iter().take(12) {
            key.push(' ');
            key.push_str(p.trim_start_matches("[compute ").trim_end_matches(']'));
        }
        let e = waits.1.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
}

#[cfg(feature = "diagnostics")]
#[allow(clippy::type_complexity)]
static GPU_WAITS: Mutex<([(u32, u64); 6], std::collections::BTreeMap<String, (u32, u64)>)> =
    Mutex::new(([(0, 0); 6], std::collections::BTreeMap::new()));

/// Logs and forgets what [`gpu_waits`] noted.
#[cfg(feature = "diagnostics")]
fn log_gpu_waits() {
    let (buckets, slow) = std::mem::take(&mut *GPU_WAITS.lock().unwrap_or_else(|e| e.into_inner()));
    let mut text = String::new();
    for (name, (n, us)) in ["<0.2", "<1", "<3", "<10", "<30", "more"].iter().zip(buckets) {
        text.push_str(&format!(" {name} ms: {n} x, {:.0} ms;", us as f64 / 1e3));
    }
    tracing::info!("GPU waits since:{text}");
    let mut slow: Vec<_> = slow.into_iter().collect();
    slow.sort_unstable_by_key(|(_, (_, us))| std::cmp::Reverse(*us));
    for (key, (n, us)) in slow.into_iter().take(8) {
        tracing::info!("GPU wait {:.0} ms in {n} x: {key}", us as f64 / 1e3);
    }
}

/// With `VICTORY_GPU_TIMING`: GPU milliseconds and commands by what ran
/// (kind and pipeline), and the frames presented meanwhile.
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<std::collections::HashMap<String, (u32, f64)>>> = Mutex::new(None);
#[cfg(feature = "diagnostics")]
static TIMED_FRAMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Logs and forgets the GPU time per kind of command, per frame.
#[cfg(feature = "diagnostics")]
fn log_gpu_times() {
    let frames = f64::from(TIMED_FRAMES.swap(0, Ordering::Relaxed).max(1));
    let Some(times) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let total: f64 = times.iter().filter(|(k, _)| k.starts_with("all: ")).map(|(_, t)| t.1).sum();
    let mut all: Vec<_> = times.into_iter().collect();
    all.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    tracing::info!("GPU time: {:.1} ms/frame in {} kinds of commands", total / frames, all.len());
    for (key, (n, ms)) in all.into_iter().take(45) {
        tracing::info!("GPU time {:7.2} ms/frame, {:7.1} x: {key}", ms / frames, f64::from(n) / frames);
    }
}

/// Whether each noted command waits for those before it (`VICTORY_GPU_STEP`,
/// diagnostics builds), so that a list stops at the command at fault.
#[cfg(feature = "diagnostics")]
fn step() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_STEP").is_some())
}

/// Whether submitted lists are not run on the host GPU (`VICTORY_NO_GPU_WORK`,
/// diagnostics builds).
#[cfg(feature = "diagnostics")]
fn no_work() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_GPU_WORK").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 512 << 10;
const FIRST_READBACK_CHUNK: u64 = 64 << 10;
const READBACK_CHUNK: u64 = 4 << 20;
const CHUNK_KEEP: usize = 8;
const CHUNK_SPARE: usize = 1;
const CHUNK_IDLE_RESETS: u32 = 64;

fn trim_chunks(len: usize, used_chunks: usize, peak_used: &mut usize, window: &mut u32) -> usize {
    let len = len.min(CHUNK_KEEP);
    *peak_used = (*peak_used).max(used_chunks);
    *window += 1;
    if *window >= CHUNK_IDLE_RESETS {
        let keep = (*peak_used).clamp(CHUNK_SPARE, CHUNK_KEEP);
        *peak_used = 0;
        *window = 0;
        len.min(keep)
    } else {
        len
    }
}

/// Readback chunks for texture-to-buffer copies, front to back: the chunks,
/// the one in use and the bytes used in it. A list's read backs share them,
/// and they are reused after a reset.
#[derive(Default)]
struct Readbacks {
    chunks: Vec<ReadbackChunk>,
    current: usize,
    used: u64,
    peak_used: usize,
    window: u32,
}

// SAFETY: the chunk pointers are persistent mappings of readback buffers.
unsafe impl Send for Readbacks {}

struct ReadbackChunk {
    buffer: ID3D12Resource,
    mapped: usize,
    capacity: u64,
}

impl Readbacks {
    /// `len` bytes (512-byte aligned, at most one chunk) in a readback
    /// chunk, made on demand: the chunk index and the offset in it.
    fn allocate(&mut self, host: &Host, len: u64) -> Option<(usize, u64)> {
        let len = len.max(1).checked_next_multiple_of(512)?;
        if len > READBACK_CHUNK {
            return None;
        }
        while let Some(chunk) = self.chunks.get(self.current) {
            if len <= chunk.capacity - self.used {
                break;
            }
            (self.current, self.used) = (self.current + 1, 0);
        }
        if self.current == self.chunks.len() {
            let next = self.chunks.last().map_or(FIRST_READBACK_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_READBACK_CHUNK).max(next).min(READBACK_CHUNK);
            let (buffer, data) = host.readback_buffer(capacity)?;
            name(&buffer, || format!("list readback chunk {capacity:#x}"));
            self.chunks.push(ReadbackChunk { buffer, mapped: data as usize, capacity });
            self.used = 0;
        }
        let at = self.used;
        self.used += len;
        Some((self.current, at))
    }
}

/// Scratch chunks no list uses: lists give theirs back when reset. Each list
/// keeping the chunks of its largest use added up to gigabytes, as the title
/// has hundreds of lists.
static SCRATCH_FREE: Mutex<Vec<(ID3D12Resource, usize, u64)>> = Mutex::new(Vec::new());
/// Free chunks kept; more are released.
const SCRATCH_KEPT: usize = 256;

impl Scratch {
    fn give_back(&mut self) {
        (self.current, self.used) = (0, 0);
        let mut free = SCRATCH_FREE.lock().unwrap_or_else(|e| e.into_inner());
        free.append(&mut self.chunks);
        let over = free.len().saturating_sub(SCRATCH_KEPT);
        let over: Vec<_> = free.drain(..over).collect();
        drop(free);
        drop(over);
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.chunks.is_empty() {
            return;
        }
        // A submission may still read them.
        let chunks = std::mem::take(&mut self.chunks);
        retire(chunks);
    }
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy; when
    /// `unmarked` is true (e.g. `ResolveQueryData`), no shader write-mark is
    /// emitted on GPU so the host marks the pages itself.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date and, when dirty, copies
    /// it into the host texture on the list's prelude: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    ),
}

/// A texture-to-buffer readback copied from a mapped readback buffer into
/// title memory once the GPU submission completes.
struct Readback {
    dst: usize,
    src: usize,
    dst_pitch: usize,
    src_pitch: usize,
    lines: usize,
    row_bytes: usize,
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload/readback buffers its GPU commands name, kept until reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// How many leading steps had their fills done ahead in `Host::prewarm`.
    prefilled: usize,
    /// Post-submission readbacks from GPU textures into title buffers.
    readbacks: Vec<Readback>,
    /// Title textures of kinds requested via `VICTORY_TEXTURE_BACK` written
    /// by this list's shaders, copied to readback buffers when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Recorded texture readbacks landed back into title memory after execution.
    texture_backs: Vec<crate::texback::Back>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// The host pipelines `bind_pipeline` kept (by address), so a pipeline
    /// the list binds again is not kept again.
    bound: crate::fx::Set<usize>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: crate::fx::Set<(usize, usize, usize)>,
    /// Buffer ranges declared written by shaders since the last title copy
    /// (host copy, address, bytes): declaring them again would change nothing.
    gpu_written: crate::fx::Set<(usize, usize, usize)>,
    /// Upload memory filled from title memory since the last title write
    /// (source, bytes): its host address, which a second fill would find
    /// holding the same bytes.
    filled: crate::fx::Map<(usize, usize), u64>,
    /// View heaps the list's draws index directly (`ResourceDescriptorHeap[]`),
    /// and whether any of their shaders may write memory: their slots are
    /// brought up to date once per submission, after every other step.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    /// Textures whose contents the list brings up to date.
    textures: crate::fx::Set<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Waits for memory values, with how many `writes` precede them.
    waits: Vec<(usize, crate::memwait::Wait)>,
    /// The work before each of `writes`.
    marks: Vec<usize>,
    /// How many of `writes` this submission has done already.
    early: usize,
    wait_marks: Vec<usize>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Deferred {
    /// Clears all per-submission vectors and maps while preserving steady-state
    /// capacities across list resets, shrinking only extreme loading outliers.
    fn clear(&mut self) {
        self.copies.clear();
        self.readbacks.clear();
        self.back_wanted.clear();
        self.texture_backs.clear();
        self.uploads.clear();
        self.kept.clear();
        self.bound.clear();
        self.refreshed.clear();
        self.gpu_written.clear();
        self.filled.clear();
        self.bindless.clear();
        self.textures.clear();
        self.writes.clear();
        self.waits.clear();
        self.marks.clear();
        self.wait_marks.clear();
        self.prefilled = 0;
        self.early = 0;
        self.budget_zeroed = false;
        if self.copies.capacity() > 16_384 {
            self.copies.shrink_to(4096);
        }
        if self.refreshed.capacity() > 8_192 {
            self.refreshed.shrink_to(1024);
        }
        if self.gpu_written.capacity() > 8_192 {
            self.gpu_written.shrink_to(1024);
        }
        if self.filled.capacity() > 8_192 {
            self.filled.shrink_to(1024);
        }
        if self.textures.capacity() > 8_192 {
            self.textures.shrink_to(1024);
        }
        if self.bound.capacity() > 8_192 {
            self.bound.shrink_to(1024);
        }
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            // Probe once whether the driver supports the full tier-3 view heap
            // or caps shader-visible view heaps at `PORTABLE_VIEWS` (1M).
            let probe: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
                self.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: victory_shader::emit::VIEW_SLOTS,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            };
            let view_slots = if probe.is_ok() {
                victory_shader::emit::VIEW_SLOTS
            } else {
                crate::descriptor::host::PORTABLE_VIEWS
            };
            drop(probe);
            let mut options4 = D3D12_FEATURE_DATA_D3D12_OPTIONS4::default();
            // SAFETY: the query writes the structure of the given size.
            let opts4_ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS4,
                        (&raw mut options4).cast(),
                        size_of_val(&options4) as u32,
                    )
                    .is_ok()
            };
            let is_warp = unsafe {
                let luid = self.device.GetAdapterLuid();
                CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .ok()
                    .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(luid).ok())
                    .and_then(|a| a.GetDesc1().ok())
                    .is_some_and(|d| {
                        d.VendorId == 0x1414
                            || (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0
                    })
            };
            let native16 = opts4_ok
                && options4.Native16BitShaderOpsSupported.as_bool()
                && !is_warp
                && !std::env::var_os("VICTORY_F16_IN_F32").is_some_and(|v| v != "0");
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                    view_slots,
                    sampler_slots: victory_shader::emit::SAMPLER_SLOTS,
                    native16,
                }
            } else {
                victory_shader::emit::Target {
                    view_slots,
                    sampler_slots: victory_shader::emit::SAMPLER_SLOTS,
                    native16,
                    ..victory_shader::emit::Target::ANY
                }
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                view_slots = target.view_slots,
                sampler_slots = target.sampler_slots,
                native16 = target.native16,
                "host wave sizes and descriptor heap limits"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that read                      other lanes do not translate, and their dispatches are dropped"
                );
            }
            target
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            crate::memstat::add(&crate::memstat::LISTS, 1);
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                readbacks: Mutex::default(),
                states: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                kept: Mutex::default(),
                work: Mutex::default(),
                writes: Default::default(),
                clamp_chunks: Mutex::default(),
                gather: Mutex::default(),
                pipelines: Mutex::default(),
                queries: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                last_fence: std::sync::atomic::AtomicU64::new(0),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Returns the completed host fence value, or `None` when D3D12 reports
    /// `u64::MAX` (device removed / hung).
    fn fence_completed(fence: &ID3D12Fence) -> Option<u64> {
        // SAFETY: a plain fence query.
        let val = unsafe { fence.GetCompletedValue() };
        if val == u64::MAX {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    "host fence reported u64::MAX (device removed); refusing to retire in-flight work"
                );
            });
            None
        } else {
            Some(val)
        }
    }

    /// Finishes the unwaited submissions that have run, oldest first, and
    /// buries objects retired before them. The caller holds `submitted`:
    /// finishing puts back what shaders wrote, as a submission does.
    fn reap_completed(&self, submitted: &mut u64) {
        #[cfg(feature = "diagnostics")]
        let reap_started = std::time::Instant::now();
        let Some(completed) = Self::fence_completed(&self.fence) else {
            return;
        };
        let mut last_sub = 0u64;
        loop {
            let done = {
                let mut q = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
                match q.front() {
                    Some(front) if front.fence <= completed => q.pop_front(),
                    _ => None,
                }
            };
            let Some(mut done) = done else { break };
            last_sub = last_sub.max(done.submission);
            if let Some(Finish {
                lists,
                back,
                late,
                late_refresh,
                card_plan,
            }) = done.finish.take()
            {
                if let Some(plan) = card_plan {
                    let copying = std::time::Instant::now();
                    if let Some(engine_lock) = self.card_engine() {
                        let mut engine = engine_lock.lock().unwrap_or_else(|e| e.into_inner());
                        let written = engine.finish_back(self, plan, submitted);
                        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                    } else {
                        plan.clear_pending();
                    }
                    COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
                {
                    let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
                    #[cfg(feature = "diagnostics")]
                    let finish_started = std::time::Instant::now();
                    self.finish_submission(&runs, back, late, late_refresh);
                    #[cfg(feature = "diagnostics")]
                    ASYNC_REAP_US[1].fetch_add(
                        finish_started.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                }
                // Before its lists go: a list that goes with it waits for
                // this fence, with `submitted` held here.
                FINISHED_FENCE.fetch_max(done.fence, Ordering::AcqRel);
                drop(lists);
            } else {
                FINISHED_FENCE.fetch_max(done.fence, Ordering::AcqRel);
            }
        }
        if last_sub != 0 {
            FINISHED.fetch_max(last_sub, Ordering::AcqRel);
            bury(last_sub);
            if !self.has_in_flight() {
                crate::marks::clear_pending();
            }
        }
        crate::timeline::deliver(FINISHED_FENCE.load(Ordering::Acquire));
        #[cfg(feature = "diagnostics")]
        ASYNC_REAP_US[0].fetch_add(
            reap_started.elapsed().as_micros() as u64,
            Ordering::Relaxed,
        );
    }

    /// Waits for in-flight GPU work without holding `submitted`, then acquires
    /// `submitted` to finish completed submissions and deliver late signals.
    fn finish_in_flight_apart(&self, event: HANDLE) {
        let target = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .map(|f| f.fence)
            .unwrap_or(0);
        if target == 0 || FINISHED_FENCE.load(Ordering::Acquire) >= target {
            return;
        }
        let Some(completed) = Self::fence_completed(&self.fence) else {
            return;
        };
        if completed < target {
            let waited = std::time::Instant::now();
            // SAFETY: `fence` is ours and free-threaded; `event` belongs to caller thread.
            unsafe {
                if let Err(e) = self.fence.SetEventOnCompletion(target, event) {
                    tracing::warn!(error = %e, "host fence wait failed");
                } else {
                    WaitForSingleObject(event, INFINITE);
                }
            }
            SUBMISSION_US[1].fetch_add(waited.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        self.reap_completed(&mut submitted);
    }

    /// Whether a submission is still to be waited for and finished.
    fn has_in_flight(&self) -> bool {
        !self.in_flight.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Whether the most recently submitted GPU fence is still executing (`Belobug #232`).
    fn in_flight_running(&self) -> bool {
        let target = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .map(|f| f.fence)
            .unwrap_or(0);
        target != 0 && Self::fence_completed(&self.fence).is_some_and(|c| c < target)
    }

    /// Waits until the host GPU fence reaches at least `target` without
    /// holding `self.submitted` across the kernel wait, then reaps completed
    /// submissions.
    pub(crate) fn wait_fence(&self, target: u64) {
        if target == 0 || FINISHED_FENCE.load(Ordering::Acquire) >= target {
            return;
        }
        let Some(completed) = Self::fence_completed(&self.fence) else {
            return;
        };
        if completed < target {
            struct EventDrop(HANDLE);
            impl Drop for EventDrop {
                fn drop(&mut self) {
                    if !self.0.0.is_null() {
                        // SAFETY: `self.0` was created by `CreateEventW` for this thread.
                        unsafe {
                            let _ = windows::Win32::Foundation::CloseHandle(self.0);
                        }
                    }
                }
            }
            thread_local! {
                static WAIT_EVENT: EventDrop = EventDrop(
                    // SAFETY: plain unnamed auto-reset event creation.
                    unsafe { CreateEventW(None, false, false, None) }.unwrap_or_default()
                );
            }
            let raw = WAIT_EVENT.with(|e| e.0.0 as usize);
            if raw != 0 {
                // SAFETY: `self.fence` is free-threaded; `raw` is this thread's event.
                unsafe {
                    if let Err(e) = self.fence.SetEventOnCompletion(target, HANDLE(raw as *mut _)) {
                        tracing::warn!(error = %e, "host fence wait failed");
                    } else {
                        WaitForSingleObject(HANDLE(raw as *mut _), INFINITE);
                    }
                }
            } else {
                let _guard = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
                // SAFETY: fallback under `self.submitted` using `self.event`.
                unsafe {
                    if let Err(e) = self.fence.SetEventOnCompletion(target, self.event) {
                        tracing::warn!(error = %e, "host fence wait failed");
                    } else {
                        WaitForSingleObject(self.event, INFINITE);
                    }
                }
            }
        }
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        self.reap_completed(&mut submitted);
    }

    #[cfg(test)]
    pub(crate) fn wait_on_queue(&self, fence: &ID3D12Fence, value: u64) {
        // SAFETY: `self.queue` and `fence` are live on the same device.
        unsafe {
            self.queue
                .Wait(fence, value)
                .expect("queue Wait on copy fence");
        }
    }

    /// Waits for all in-flight asynchronous submissions to complete.
    pub(crate) fn wait_in_flight(&self) {
        let target = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .map(|f| f.fence)
            .unwrap_or(0);
        if target != 0 {
            self.wait_fence(target);
        }
    }

    /// The last fence value the host GPU has reached.
    pub(crate) fn completed_fence(&self) -> u64 {
        Self::fence_completed(&self.fence).unwrap_or(0)
    }

    /// Takes the submission lock, waits for any in-flight GPU work to finish,
    /// runs `f`, and when `f` reports queue work (`true`) signals the host
    /// fence so callers can tell when that work has completed on the GPU.
    ///
    /// Must not be called from inside [`in_parallel`] while a submission holds
    /// `self.submitted`.
    pub(crate) fn drain_and_signal<R>(&self, f: impl FnOnce() -> (R, bool)) -> (R, Option<u64>) {
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        self.drain_in_flight_locked(&mut submitted);
        let (result, signal) = f();
        let fence = if signal {
            *submitted += 1;
            let fence_val = *submitted;
            // SAFETY: caller holds `self.submitted`; `self.fence` is ours.
            match unsafe { self.queue.Signal(&self.fence, fence_val) } {
                Ok(()) => Some(fence_val),
                Err(e) => {
                    tracing::warn!(error = %e, "host sweep fence not signalled");
                    None
                }
            }
        } else {
            None
        };
        (result, fence)
    }

    /// Waits for any in-flight async submission while `self.submitted` is already held.
    fn drain_in_flight_locked(&self, submitted: &mut u64) {
        let target = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .map(|f| f.fence)
            .unwrap_or(0);
        if target == 0 {
            return;
        }
        let Some(completed) = Self::fence_completed(&self.fence) else {
            return;
        };
        if completed < target {
            let waited = std::time::Instant::now();
            // SAFETY: the fence and event are ours; caller holds `submitted`.
            unsafe {
                if let Err(e) = self.fence.SetEventOnCompletion(target, self.event) {
                    tracing::warn!(error = %e, "host async drain failed");
                } else {
                    WaitForSingleObject(self.event, INFINITE);
                }
            }
            SUBMISSION_US[1].fetch_add(waited.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        self.reap_completed(submitted);
    }

    /// No submission runs while the result lives: its title memory is not
    /// being read or written (a page remap waits here).
    pub(crate) fn hold_submissions(&self) -> std::sync::MutexGuard<'_, u64> {
        self.submitted.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs closed lists; waits synchronously when CPU copy-back, readbacks, or
    /// bottom-of-pipe writes require GPU completion, or returns asynchronously
    /// so CPU recording overlaps GPU execution.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.execute_inner(runs, None);
    }

    /// As [`Host::execute`] for the title's own submissions, the only ones
    /// that may return before the host GPU is done
    /// (`VICTORY_ASYNC_SUBMIT=1`): the title waits for its fences, which are
    /// signalled after the wait (`crate::timeline`). Our own submissions
    /// (present, captures, read-backs, texture uploads) read or free what
    /// they made as soon as the call returns: unwaited, the present's
    /// capture buffer was freed with its copy still pending, which the
    /// debug layer names (#921, object deleted while still in use) and the
    /// AMD driver answered by removing the device at the first frame.
    ///
    /// Such a submission is finished (its shaders' writes put back, its
    /// read-backs and bottom-of-pipe writes done) when it is waited for:
    /// before the next submission's own preparation, a title fence's
    /// signal, a present, or a reset of one of its lists. Meanwhile the
    /// next submission's buffer copies are brought up to date, so the
    /// processor and the card work at the same time for that long.
    pub(crate) fn execute_title(&self, runs: &[std::sync::Arc<HostList>]) {
        let lists: Vec<&HostList> = runs.iter().map(|l| &**l).collect();
        self.execute_inner(&lists, Some(runs));
    }

    /// Brings buffer copies, leading upload fills, and optional staged bindless
    /// plans up to date while the previous GPU submission runs.
    #[allow(clippy::type_complexity)]
    fn prewarm(
        &self,
        runs: &[&HostList],
    ) -> Vec<(
        std::sync::Arc<crate::descriptor::host::HostHeap>,
        Option<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    )> {
        let started = std::time::Instant::now();
        let mut batch = Batch::default();
        static FILLS: OnceLock<bool> = OnceLock::new();
        let mut filling = *FILLS.get_or_init(|| {
            std::env::var_os("VICTORY_EARLY_FILLS").is_none_or(|v| v != "0")
        });
        let owed: Vec<(usize, usize)> = if filling {
            let flights = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            flights
                .iter()
                .filter_map(|f| f.finish.as_ref())
                .flat_map(|f| f.back.iter().map(|&(_, at, len)| (at, at + len)))
                .collect()
        } else {
            Vec::new()
        };
        let (mut filled, mut fill_us) = (0u64, 0u64);
        for list in runs {
            let mut deferred = list.deferred();
            let mut done = 0;
            for (k, step) in deferred.copies.iter().enumerate() {
                match step {
                    Step::Refresh(shadow, address, len) => batch.add(shadow, *address, *len),
                    Step::Copy(..) => filling = false,
                    &Step::Fill(dst, src, len) if filling => {
                        if owed.iter().any(|&(from, to)| src < to && from < src + len) {
                            filling = false;
                        } else {
                            let at = std::time::Instant::now();
                            copy_mapped(dst as *mut u8, src as *const u8, len, false);
                            fill_us += at.elapsed().as_micros() as u64;
                            filled += len as u64;
                        }
                    }
                    _ => {}
                }
                if filling {
                    done = k + 1;
                }
            }
            deferred.prefilled = done;
            for (heap, writes) in &deferred.bindless {
                batch.bindless(heap, *writes);
            }
        }
        if filled > 0 {
            PREPARING[0].fetch_add(fill_us, Ordering::Relaxed);
            PREPARING[1].fetch_add(filled, Ordering::Relaxed);
            EARLY_FILLS.fetch_add(filled, Ordering::Relaxed);
        }
        batch.flush();
        static PLANS: OnceLock<bool> = OnceLock::new();
        let mut ahead = Vec::new();
        if *PLANS.get_or_init(|| std::env::var_os("VICTORY_EARLY_PLAN").is_some_and(|v| v == "1")) {
            for (heap, _) in std::mem::take(&mut batch.bindless) {
                let planning = std::time::Instant::now();
                let plan = heap.plan_ahead();
                let us = planning.elapsed().as_micros() as u64;
                BINDLESS_US[0].fetch_add(us, Ordering::Relaxed);
                EARLY_PLAN_US.fetch_add(us, Ordering::Relaxed);
                ahead.push((heap, plan));
            }
        }
        PREWARM_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        ahead
    }

    fn execute_inner(&self, runs: &[&HostList], keep: Option<&[std::sync::Arc<HostList>]>) {
        #[cfg(feature = "diagnostics")]
        let _inside = crate::profile::here();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // Wait for any previous async submission before mutating shared Shadow
        // buffers in `prepare`. During the gap between submissions, the GPU
        // ran in parallel with the title's CPU command-list recording, and
        // runs on while this submission's buffer copies are refreshed.
        let ahead = if keep.is_some() && self.in_flight_running() {
            self.prewarm(runs)
        } else {
            Vec::new()
        };
        self.drain_in_flight_locked(&mut submitted);
        for (heap, _) in &ahead {
            EARLY_SLOTS.fetch_add(heap.flush_staged() as u64, Ordering::Relaxed);
        }
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch {
            ahead: ahead.into_iter().filter_map(|(_, plan)| plan).collect(),
            ..Batch::default()
        };
        let mut early = Early::default();
        if !runs.is_empty() {
            early.take(&mut batch, runs, false);
        }
        let parallel = parallel_preludes(runs);
        let mut later = Vec::new();
        for list in runs {
            let uploads = list.run_deferred(&mut batch);
            firsts.push(lists.len());
            match list.plan_prelude(&uploads) {
                Prelude::None => {}
                Prelude::Recorded(prelude) => lists.push(Some(prelude)),
                Prelude::Barriers(barriers) if parallel => {
                    later.push((lists.len(), *list, barriers));
                    lists.push(None);
                }
                Prelude::Barriers(barriers) => lists.push(list.record_prelude(&barriers)),
            }
            lists.push(list.list.cast().ok());
        }
        for (at, prelude) in record_preludes(&later) {
            lists[at] = prelude;
        }
        batch.prepare_bindless();
        if !runs.is_empty() {
            early.take(&mut batch, runs, true);
        }
        // The textures live (in `early`) until the submission is done.
        let first = early.record(self);
        // Deduplicate GPU write ranges once before execution so VRAM card
        // shadows get pre-uploaded / Wave64-hashed in the same GPU wait.
        let mut card_writes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> =
            Vec::new();
        let mut cpu_back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        // A copy is either a card's or not, so one set of ranges serves both.
        let mut declared = crate::fx::Set::default();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len, _) = step
                    && declared.insert((std::sync::Arc::as_ptr(shadow) as usize, *address, *len))
                {
                    let target = if shadow.is_card() {
                        &mut card_writes
                    } else {
                        &mut cpu_back
                    };
                    target.push((shadow.clone(), *address, *len));
                }
            }
        }
        for (shadow, address, len) in batch.bindless_writes.drain(..) {
            if declared.insert((std::sync::Arc::as_ptr(&shadow) as usize, address, len)) {
                let target = if shadow.is_card() {
                    &mut card_writes
                } else {
                    &mut cpu_back
                };
                target.push((shadow, address, len));
            }
        }
        let (card_before, card_plan, mut card_after) =
            if self.card.get().and_then(|o| o.as_ref()).is_some() || !card_writes.is_empty() {
                if let Some(engine_lock) = self.card_engine() {
                    let mut engine = engine_lock.lock().unwrap_or_else(|e| e.into_inner());
                    let (before, plan) = engine.record_before(self, &card_writes);
                    let after = if plan.has_writes() {
                        engine.record_after(self, &plan)
                    } else {
                        None
                    };
                    (before, Some(plan), after)
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };
        let prepared = std::time::Instant::now();
        // Each run (prelude and list) is a submission of its own: the host
        // resets the device when one submission runs for about 2 s, and
        // the world's frames (a second or more over dozens of lists) fit
        // that only list by list. One wait at the end; with
        // `VICTORY_GPU_TIMING` (diagnostics) each run is waited for and
        // timed on its own.
        let groups: Vec<&[Option<ID3D12CommandList>]> = firsts
            .iter()
            .zip(firsts.iter().skip(1).chain([&lists.len()]))
            .map(|(&a, &b)| &lists[a..b])
            .collect();
        #[cfg(feature = "diagnostics")]
        let each = timing();
        #[cfg(not(feature = "diagnostics"))]
        let each = false;
        let can_async = !cfg!(test)
            && keep.is_some()
            && !each
            && async_submit()
            && (card_writes.is_empty() || crate::card::fetch_inflight());
        #[cfg(feature = "diagnostics")]
        let skip_copy_wait = no_work();
        #[cfg(not(feature = "diagnostics"))]
        let skip_copy_wait = false;
        if !skip_copy_wait
            && let Some((copy_fence, val)) = early.copy_wait.take()
        {
            // SAFETY: `self.queue` and `copy_fence` are live on the same device.
            unsafe {
                if let Err(e) = self.queue.Wait(&copy_fence, val) {
                    tracing::warn!(error = %e, "host copy-queue wait failed");
                }
            }
        }
        if let Some(first) = first {
            // SAFETY: a closed list.
            unsafe { self.queue.ExecuteCommandLists(&[Some(first)]) };
        }
        if let Some(card_before) = card_before {
            // SAFETY: a closed list.
            unsafe { self.queue.ExecuteCommandLists(&[Some(card_before)]) };
        }
        let mut async_signalled = false;
        let mut card_after_ran = false;
        for (i, group) in groups.iter().enumerate() {
            // Diagnostics: everything but the GPU's work, to look at the
            // order of the title's submissions.
            #[cfg(feature = "diagnostics")]
            if no_work() {
                continue;
            }
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists.
            unsafe { self.queue.ExecuteCommandLists(group) };
            if !each && i + 1 < groups.len() {
                continue;
            }
            if !each
                && let Some(after) = card_after.take()
            {
                // SAFETY: closed list; runs immediately after the final group under the same fence.
                unsafe { self.queue.ExecuteCommandLists(&[Some(after)]) };
                card_after_ran = true;
            }
            *submitted += 1;
            let fence_val = *submitted;
            if can_async {
                // SAFETY: the fence is ours.
                if let Err(e) = unsafe { self.queue.Signal(&self.fence, fence_val) } {
                    tracing::warn!(error = %e, "host async submission not signalled");
                } else {
                    for list in runs {
                        list.last_fence.store(fence_val, Ordering::Release);
                    }
                    async_signalled = true;
                }
                continue;
            }
            // SAFETY: the fence and event are ours.
            unsafe {
                let waited = self
                    .queue
                    .Signal(&self.fence, fence_val)
                    .and_then(|()| self.fence.SetEventOnCompletion(fence_val, self.event));
                match waited {
                    Ok(()) => {
                        WaitForSingleObject(self.event, INFINITE);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            #[cfg(feature = "diagnostics")]
            if each {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                runs[i].add_times(frequency);
            }
            #[cfg(feature = "diagnostics")]
            if each && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        if let Some(after) = card_after.take() {
            #[cfg(feature = "diagnostics")]
            let skip = no_work();
            #[cfg(not(feature = "diagnostics"))]
            let skip = false;
            if !skip {
                self.execute_and_wait_locked(after, &mut submitted);
                card_after_ran = true;
            }
        }
        let _ = &firsts;
        if async_signalled {
            let deferred_card = if let Some(plan) = card_plan {
                if card_after_ran {
                    Some(plan)
                } else {
                    plan.clear_pending();
                    None
                }
            } else {
                None
            };
            SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
            self.in_flight
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push_back(InFlight {
                    fence: *submitted,
                    submission: this_submission,
                    _early_copies: {
                        let mut copies = early.copies;
                        copies.extend(early.copied);
                        copies
                    },
                    _plans: batch.plans,
                    finish: Some(Finish {
                        lists: keep.map(<[_]>::to_vec).unwrap_or_default(),
                        back: cpu_back,
                        late: batch.late,
                        late_refresh: batch.late_refresh,
                        card_plan: deferred_card,
                    }),
                });
            return;
        }
        drop(early);
        FINISHED.fetch_max(this_submission, Ordering::AcqRel);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        gpu_waits(gpu, runs);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        if let Some(plan) = card_plan {
            if card_after_ran {
                if let Some(engine_lock) = self.card_engine() {
                    let mut engine = engine_lock.lock().unwrap_or_else(|e| e.into_inner());
                    let written = engine.finish_back(self, plan, &mut submitted);
                    COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                } else {
                    plan.clear_pending();
                }
            } else {
                plan.clear_pending();
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.finish_submission(runs, cpu_back, batch.late, batch.late_refresh);
        if !self.has_in_flight() {
            crate::marks::clear_pending();
        }
        self.log_messages();
    }

    /// After a submission ran: what its shaders wrote goes back into title
    /// memory, then deferred copies/refreshes touching those writes, then its
    /// read-backs, then the bottom-of-pipe writes that tell the title the GPU is done.
    fn finish_submission(
        &self,
        runs: &[&HostList],
        back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        late: Vec<(usize, usize, usize, bool)>,
        late_refresh: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    ) {
        let copying = std::time::Instant::now();
        if back.len() <= 1 {
            for (shadow, address, len) in &back {
                let written = shadow.copy_back(*address, *len);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        } else {
            // Group by Shadow pointer so distinct shadows run in parallel
            // across CPU cores without lock contention on `before`/`pending`.
            let mut groups: Vec<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>> =
                Vec::new();
            for item in back {
                if let Some(g) = groups
                    .iter_mut()
                    .find(|g| std::sync::Arc::ptr_eq(&g[0].0, &item.0))
                {
                    g.push(item);
                } else {
                    groups.push(vec![item]);
                }
            }
            if groups.len() <= 1 {
                for (shadow, address, len) in &groups[0] {
                    let written = shadow.copy_back(*address, *len);
                    COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                }
            } else {
                in_parallel(&groups, 8, |chunk| {
                    for g in chunk {
                        for (shadow, address, len) in g {
                            let written = shadow.copy_back(*address, *len);
                            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
                        }
                    }
                });
            }
        }
        // Copies kept past the lists' run: what shaders wrote reached title
        // memory in the copy-backs above, so a deferred copy's source is new
        // now (and its destination is not overwritten by them).
        for (dst, src, len, title_dst) in late {
            copy_mapped(dst as *mut u8, src as *const u8, len, title_dst);
        }
        for (shadow, address, len) in late_refresh {
            let copied = shadow.update(address, len);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
        for list in runs {
            for rb in &list.deferred().readbacks {
                let written = if rb.dst_pitch == rb.row_bytes && rb.src_pitch == rb.row_bytes {
                    let bytes = rb.row_bytes * rb.lines;
                    crate::watch::open_for_sync_write(rb.dst, bytes);
                    write_mapped(rb.dst as *mut u8, rb.src as *const u8, bytes);
                    bytes
                } else {
                    let mut n = 0;
                    for line in 0..rb.lines {
                        let to = rb.dst + line * rb.dst_pitch;
                        let from = rb.src + line * rb.src_pitch;
                        crate::watch::open_for_sync_write(to, rb.row_bytes);
                        write_mapped(to as *mut u8, from as *const u8, rb.row_bytes);
                        n += rb.row_bytes;
                    }
                    n
                };
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
            for back in &list.deferred().texture_backs {
                let written = crate::texback::land(back);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Host timestamps the lists resolved, on the title's GPU clock.
        let mut clock = None;
        for list in runs {
            let mut deferred = list.deferred();
            for &(at, what, bytes) in &deferred.writes[deferred.early..] {
                let what = match what {
                    crate::queue::Written::HostTimestamp(raw) => match clock.get_or_insert_with(|| self.clock()) {
                        // SAFETY: the list's mapped read-back memory, which
                        // the submission that ran resolved into.
                        Some(c) => crate::queue::Written::Value(c.convert(unsafe { (raw as *const u64).read_volatile() })),
                        None => crate::queue::Written::GpuTimestamp,
                    },
                    what => what,
                };
                what.write(at as *mut u8, bytes);
            }
            deferred.early = 0;
        }
    }

    /// How host GPU timestamps map onto the title's GPU clock
    /// ([`crate::queue::Written::GpuTimestamp`]): the queue's calibration
    /// of its timestamps against the CPU's performance counter, which
    /// `Instant` reads too.
    fn clock(&self) -> Option<Clock> {
        use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
        let (mut gpu, mut cpu, mut now, mut cpu_frequency) = (0, 0, 0, 0);
        // SAFETY: plain queries into our locals.
        let frequency = unsafe {
            let frequency = self.queue.GetTimestampFrequency().ok()?;
            self.queue.GetClockCalibration(&mut gpu, &mut cpu).ok()?;
            QueryPerformanceCounter(&mut now);
            QueryPerformanceFrequency(&mut cpu_frequency);
            frequency
        };
        let title_now = crate::queue::gpu_clock();
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        // The title's clock when the host GPU's read `gpu`.
        let since = (i128::from(now) - i128::from(cpu)) * ticks / i128::from(cpu_frequency.max(1));
        (frequency > 0).then_some(Clock { gpu, at: i128::from(title_now) - since, frequency, now: title_now })
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

/// A host GPU timestamp `gpu` taken when the title's GPU clock read `at`,
/// the host's ticks per second, and the title's clock when taken.
struct Clock {
    gpu: u64,
    at: i128,
    frequency: u64,
    now: u64,
}

impl Clock {
    /// A host GPU timestamp on the title's clock: never past the time it
    /// was read, so bottom-of-pipe timestamps written after stay later.
    fn convert(&self, raw: u64) -> u64 {
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        let since = (i128::from(raw) - i128::from(self.gpu)) * ticks / i128::from(self.frequency);
        (self.at + since).clamp(0, i128::from(self.now)) as u64
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Adds the GPU time of each noted command to the totals by kind
    /// (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn add_times(&self, frequency: u64) {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return;
        };
        if count == 0 {
            return;
        }
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return;
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
        let times = times.get_or_insert_with(Default::default);
        for i in 0..count {
            let ms = stamps[i + 1].saturating_sub(stamps[i]) as f64 * 1e3 / frequency.max(1) as f64;
            // The kind without its counts: "dispatch [compute ...]".
            let text = &notes[i].0;
            let kind = text.split(' ').next().unwrap_or("");
            let rest = text.find('[').map_or("", |at| &text[at..]);
            let key: String = format!("{kind} {rest}").chars().take(150).collect();
            let k = times.entry(format!("all: {kind}")).or_insert((0, 0.0));
            k.0 += 1;
            k.1 += ms;
            let e = times.entry(key).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms;
        }
    }

    /// Records that a draw or dispatch writes title texture `resource` via UAV;
    /// if its dimensions and format match `VICTORY_TEXTURE_BACK`, queues a
    /// readback copy at list close.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        if !deferred
            .back_wanted
            .iter()
            .any(|target| target.owner() == resource as usize)
            && let Some(target) = crate::texback::Target::new(resource)
        {
            deferred.back_wanted.push(target);
        }
    }

    fn settle_copy(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                drop(states);
                if before != wanted {
                    // SAFETY: an open list and a live host texture whose state changed.
                    unsafe {
                        self.list.ResourceBarrier(&[transition(
                            &copy.texture,
                            D3D12_RESOURCE_STATES(before as i32),
                            D3D12_RESOURCE_STATES(wanted as i32),
                        )]);
                    }
                }
            }
        }
    }

    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            self.settle_copy(&target.copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    pub(crate) fn close(&self) {
        self.textures_back();
        // Gathered indirect scratch buffers leave INDIRECT_ARGUMENT back to
        // COMMON at list end so the next recording starts from COMMON.
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        {
            let Gather { runs, run_spare, .. } = &mut *gather;
            run_spare.extend(runs.drain(..).map(|set| (set.into_scratch(), true)));
            for (scratch, readable) in run_spare.iter_mut().filter(|s| s.1) {
                // SAFETY: an open list and a live buffer left in INDIRECT_ARGUMENT.
                unsafe {
                    self.list.ResourceBarrier(&[transition(
                        scratch,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
                *readable = false;
            }
        }
        for set in gather.sets.iter().filter(|s| s.records() > 0) {
            // SAFETY: an open list and a live scratch buffer left in INDIRECT_ARGUMENT
            // by this list's prelude.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    &set.scratch,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        }
        drop(gather);
        // The host closes no list with queries running; their title ends,
        // in a later list, are dropped (`end_query`).
        for (heap, kind, index, _) in self.queries.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            // SAFETY: an open list and a query begun in it.
            unsafe { self.list.EndQuery(&heap, kind, index) };
        }
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    pub(crate) fn keep_until_run(&self, what: Box<dyn std::any::Any + Send>) {
        self.kept.lock().unwrap_or_else(|e| e.into_inner()).push(what);
    }

    pub(crate) fn release_kept(&self) {
        let kept = std::mem::take(&mut *self.kept.lock().unwrap_or_else(|e| e.into_inner()));
        drop(kept);
    }

    /// Only after the list's last submission finished.
    pub(crate) fn reset(&self) {
        let fence = self.last_fence.load(Ordering::Acquire);
        if fence != 0
            && let Some(h) = host()
        {
            h.wait_fence(fence);
        }
        self.release_kept();
        self.deferred().clear();
        *self.work.lock().unwrap_or_else(|e| e.into_inner()) = 0;
        self.writes.store(0, Ordering::Relaxed);
        {
            let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
            chunks.used = 0;
            chunks.at = 0;
            chunks.barrier = false;
            chunks.bound = false;
        }
        {
            let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
            states.clear();
            if states.capacity() > 256 {
                states.shrink_to(64);
            }
        }
        {
            let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
            let Gather {
                closed,
                sets,
                spare,
                runs,
                run_spare,
                ..
            } = &mut *gather;
            *closed = false;
            for set in sets.drain(..) {
                if spare.len() < GATHER_SETS {
                    spare.push(set.into_scratch());
                }
            }
            run_spare.extend(runs.drain(..).map(|set| (set.into_scratch(), false)));
            run_spare.truncate(GATHER_SETS);
            for spare in run_spare.iter_mut() {
                spare.1 = false;
            }
        }
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        {
            let mut notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
            notes.clear();
            if notes.capacity() > 256 {
                notes.shrink_to(64);
            }
        }
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            scratch.give_back();
        }
        {
            let mut readbacks = self.readbacks.lock().unwrap_or_else(|e| e.into_inner());
            let readbacks = &mut *readbacks;
            let used_chunks = readbacks.current + usize::from(readbacks.used > 0);
            (readbacks.current, readbacks.used) = (0, 0);
            let keep = trim_chunks(
                readbacks.chunks.len(),
                used_chunks,
                &mut readbacks.peak_used,
                &mut readbacks.window,
            );
            readbacks.chunks.truncate(keep);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Records that this list starts and ends with `copy` in `wanted` state
    /// (test helper matching `transition`'s first-use tracking).
    #[cfg(test)]
    pub(crate) fn expect_texture_state(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .entry(std::sync::Arc::as_ptr(copy) as usize)
            .or_insert_with(|| (wanted, wanted, copy.clone()));
    }

    #[cfg(test)]
    pub(crate) fn has_deferred_work(&self) -> bool {
        !self.deferred().copies.is_empty()
    }

    /// Plans this list's submission prelude: none, already recorded (when
    /// texture uploads, gathered indirect clamps, or budget zeroes run), or
    /// barriers-only (which `record_preludes` can record across threads).
    fn plan_prelude(
        &self,
        uploads: &[(
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
        )],
    ) -> Prelude {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let mut pre_barriers = Vec::new();
        for (_, copy) in uploads {
            let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
            if real != COPY_DEST {
                pre_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let post_barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        let gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        let gathered = gather.sets.iter().any(|s| s.records() > 0);
        let card_budget = match self.budget.get().and_then(|o| o.as_ref()) {
            Some(Budget::Card(b)) if self.deferred().budget_zeroed => Some(b.clone()),
            _ => None,
        };
        if uploads.is_empty() && post_barriers.is_empty() && !gathered && card_budget.is_none() {
            return Prelude::None;
        }
        if uploads.is_empty() && !gathered && card_budget.is_none() {
            return Prelude::Barriers(post_barriers);
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !pre_barriers.is_empty() {
                    list.ResourceBarrier(&pre_barriers);
                }
                for (staging, copy) in uploads {
                    staging.record_copies(list, &copy.texture);
                }
                if !post_barriers.is_empty() {
                    list.ResourceBarrier(&post_barriers);
                }
                if let Some(budget_buf) = &card_budget
                    && let Ok(l2) = list.cast::<ID3D12GraphicsCommandList2>()
                {
                    let at = budget_buf.GetGPUVirtualAddress();
                    let zeros = [0u64, 4, 8].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
                        Dest: at + offset,
                        Value: 0,
                    });
                    list.ResourceBarrier(&[transition(
                        budget_buf,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    l2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
                    list.ResourceBarrier(&[transition(
                        budget_buf,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
                if gathered {
                    let filled = match &gather.budget {
                        Some(budget) => {
                            crate::indirect::record_gathers(list, &gather.sets, budget)
                        }
                        None => false,
                    };
                    if !filled {
                        fail(format_args!(
                            "a list's gathered indirect records could not be clamped in its prelude"
                        ));
                    }
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok().map_or(Prelude::None, Prelude::Recorded)
    }

    /// Records the barrier-only prelude planned by `HostList::plan_prelude`.
    fn record_prelude(&self, barriers: &[D3D12_RESOURCE_BARRIER]) -> Option<ID3D12CommandList> {
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing and `self.states` keeps the textures live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                list.ResourceBarrier(barriers);
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            if step() {
                let barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    ..Default::default()
                };
                // SAFETY: an open list; the barrier names no resource.
                unsafe { self.list.ResourceBarrier(&[barrier]) };
            }
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Begins query `index` of `heap` (occlusion or statistics, never a
    /// timestamp) unless it is running in the list already. Whether it
    /// began.
    pub(crate) fn begin_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> bool {
        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        if kind == crate::query::TIMESTAMP || queries.iter().any(|(h, _, i, _)| h == heap && *i == index) {
            return false;
        }
        let kind = D3D12_QUERY_TYPE(kind as i32);
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.BeginQuery(heap, kind, index) };
        queries.push((heap.clone(), kind, index, false));
        drop(queries);
        self.keep(heap);
        true
    }

    /// Ends query `index` of `heap`: a timestamp is taken; another query
    /// ends only if it began in this list, as the type it began as. The
    /// type it ended as, if it did and its result is the host's to give: an
    /// occlusion query with a draw inside that the host did not run counts
    /// too few samples, and a title would cull by it what the draw shows.
    pub(crate) fn end_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> Option<u32> {
        let (kind, missed) = if kind == crate::query::TIMESTAMP {
            (D3D12_QUERY_TYPE(kind as i32), false)
        } else {
            let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
            let at = queries.iter().position(|(h, _, i, _)| h == heap && *i == index)?;
            let (_, kind, _, missed) = queries.swap_remove(at);
            (kind, missed)
        };
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.EndQuery(heap, kind, index) };
        self.keep(heap);
        let occlusion = matches!(kind.0 as u32, crate::query::OCCLUSION | crate::query::BINARY_OCCLUSION);
        (!(missed && occlusion)).then_some(kind.0 as u32)
    }

    /// A draw the title recorded was not run on the host (no pipeline yet,
    /// a shader that does not translate, ...): the queries running around
    /// it did not see it.
    pub(crate) fn missed_draw(&self) {
        for q in self.queries.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            q.3 = true;
        }
    }

    /// Notes a pipeline the list binds and keeps it and its root signature
    /// until the list is reset, the first time only: the list keeps the
    /// first, so its address names the same pipeline from then on.
    pub(crate) fn bind_pipeline(&self, p: &crate::pipeline::host::HostPipeline) {
        let mut deferred = self.deferred();
        if !deferred.bound.insert(p.pso.as_raw() as usize) {
            return;
        }
        deferred.kept.push(p.pso.clone().into());
        deferred.kept.push(p.root.clone().into());
        drop(deferred);
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(p.name.clone());
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
        deferred.filled.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// The host GPU address of `len` bytes of upload memory holding the title
    /// memory at `src` as of the list's submission: one copy for every ask
    /// until a title copy intervenes.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        let known = self.deferred().filled.get(&key).copied();
        if known.is_some() {
            return known;
        }
        let (gpu, to) = self.scratch(len)?;
        self.defer_fill(to, src, len as usize);
        self.deferred().filled.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        self.defer_refreshes(&[(shadow.clone(), address, len)]);
    }

    /// Brings several ranges up to date when the list is submitted, taking
    /// the deferred lock once however many buffers a draw uses.
    pub(crate) fn defer_refreshes(
        &self,
        refreshes: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)],
    ) {
        let mut deferred = self.deferred();
        for (shadow, address, len) in refreshes {
            let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
            if deferred.refreshed.insert(key) {
                deferred
                    .copies
                    .push(Step::Refresh(shadow.clone(), *address, *len));
            }
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len, false));
        }
    }

    /// Like [`HostList::defer_gpu_write`], for non-shader GPU writes (such as
    /// `ResolveQueryData`) that do not emit shader write-marks on the GPU.
    pub(crate) fn defer_gpu_write_unmarked(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len, true));
        } else if let Some(Step::GpuWrite(_, _, _, unmarked)) = deferred.copies.iter_mut().find(|s| {
            matches!(
                s,
                Step::GpuWrite(sh, a, l, _)
                    if std::sync::Arc::as_ptr(sh) as usize == key.0 && (*a, *l) == (address, len)
            )
        }) {
            *unmarked = true;
        }
    }

    /// A wait for a memory value.
    pub(crate) fn note_wait(&self, wait: crate::memwait::Wait) {
        let mut deferred = self.deferred();
        let before = deferred.writes.len();
        deferred.waits.push((before, wait));
        let work = *self.work.lock().unwrap_or_else(|e| e.into_inner());
        deferred.wait_marks.push(work);
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    ///
    /// Work is draws, dispatches, clears and copies: what has to run first.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        let work = *self.work.lock().unwrap_or_else(|e| e.into_inner());
        let mut deferred = self.deferred();
        deferred.writes.push((dst as usize, what, bytes));
        deferred.marks.push(work);
    }

    /// The list's memory waits and writes in order, with how many draws and
    /// dispatches precede each (for the log).
    pub(crate) fn script(&self) -> String {
        let deferred = self.deferred();
        let mut events: Vec<(usize, u8, String)> = Vec::new();
        for (i, (at, what, _)) in deferred.writes.iter().enumerate() {
            events.push((i, 1, format!("W[{at:x}]={what:x?}@{}", deferred.marks.get(i).copied().unwrap_or(0))));
        }
        for (before, w) in &deferred.waits {
            let mark = deferred.wait_marks.get(events.len().saturating_sub(deferred.writes.len())).copied().unwrap_or(0);
            events.push((*before, 0, format!("R[{:x}]>={:x}@{mark}", w.address, w.reference)));
        }
        events.sort_by_key(|e| (e.0, e.1));
        let work = *self.work.lock().unwrap_or_else(|e| e.into_inner());
        let texts: Vec<_> = events.into_iter().map(|e| e.2).collect();
        format!("{} ({work} in all)", texts.join(" "))
    }

    /// Whether the list has work: draws, dispatches, clears or copies.
    pub(crate) fn worked(&self) -> bool {
        *self.work.lock().unwrap_or_else(|e| e.into_inner()) > 0
    }

    /// How many queries begun in the list still run.
    pub(crate) fn open_queries(&self) -> usize {
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Counts a dispatch or a copy: what may write indirect arguments.
    pub(crate) fn count_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.gather_close();
    }

    /// Stops gathering single-record indirect draws for this list's prelude:
    /// a command just recorded may write argument memory.
    pub(crate) fn gather_close(&self) {
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        gather.closed = true;
        let Gather { runs, run_spare, .. } = &mut *gather;
        run_spare.extend(runs.drain(..).map(|set| (set.into_scratch(), true)));
    }

    fn gather_run_slot(
        &self,
        src: &ID3D12Resource,
        src_id: u64,
        offset: u32,
        (kind, stride): (u32, u32),
    ) -> Option<(ID3D12Resource, u64, bool)> {
        let budget = self.budget()?;
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        for set in gather
            .runs
            .iter_mut()
            .filter(|s| s.is_of(src, src_id, (kind, stride)))
        {
            if let Some(slot) = set.slot(offset) {
                return Some((set.scratch.clone(), u64::from(slot) * u64::from(stride), false));
            }
        }
        let (scratch, readable) = match gather.run_spare.pop() {
            Some(spare) => spare,
            None => {
                let scratch = host()?.uav_buffer(
                    u64::from(crate::indirect::RUN_MOST) * u64::from(crate::indirect::GATHER_STRIDE),
                )?;
                name(&scratch, || "indirect arguments, gathered by runs".to_owned());
                (scratch, false)
            }
        };
        let Some(table) = self.scratch(u64::from(crate::indirect::RUN_MOST) * 4) else {
            gather.run_spare.push((scratch, readable));
            return None;
        };
        let mut set =
            crate::indirect::GatherSet::new_run(src.clone(), src_id, (kind, stride), scratch, table);
        let before = if readable {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        if !crate::indirect::record_run(&self.list, &set, before, budget) {
            gather.run_spare.push((set.into_scratch(), readable));
            return None;
        }
        let slot = set.slot(offset)?;
        let scratch = set.scratch.clone();
        gather.runs.push(set);
        Some((scratch, u64::from(slot) * u64::from(stride), true))
    }

    /// Assigns a slot for the scattered record at `offset` in `src` (`src_id`)
    /// to be clamped once per list in `prelude` (or by draw-run after `gather_close`).
    pub(crate) fn gather_slot(
        &self,
        src: &ID3D12Resource,
        src_id: u64,
        offset: u32,
        (kind, stride): (u32, u32),
    ) -> Option<(ID3D12Resource, u64, bool)> {
        let mut gather = self.gather.lock().unwrap_or_else(|e| e.into_inner());
        if gather.closed {
            if crate::indirect::gather_runs() {
                drop(gather);
                return self.gather_run_slot(src, src_id, offset, (kind, stride));
            }
            return None;
        }
        let found = gather
            .sets
            .iter()
            .position(|s| s.is_of(src, src_id, (kind, stride)));
        let at = match found {
            Some(at) => at,
            None => {
                if gather.sets.len() >= GATHER_SETS {
                    return None;
                }
                if self.list.cast::<ID3D12GraphicsCommandList2>().is_err() {
                    return None;
                }
                let host = host()?;
                if gather.budget.is_none() {
                    let budget = host.uav_buffer(256)?;
                    name(&budget, || "indirect budget of gathered records".to_owned());
                    gather.budget = Some(budget);
                }
                let scratch = match gather.spare.pop() {
                    Some(scratch) => scratch,
                    None => {
                        let scratch = host.uav_buffer(
                            u64::from(crate::indirect::GATHER_MOST)
                                * u64::from(crate::indirect::GATHER_STRIDE),
                        )?;
                        name(&scratch, || "indirect arguments, gathered".to_owned());
                        scratch
                    }
                };
                let Some(table) = self.scratch(u64::from(crate::indirect::GATHER_MOST) * 4) else {
                    gather.spare.push(scratch);
                    return None;
                };
                gather.sets.push(crate::indirect::GatherSet::new(
                    src.clone(),
                    src_id,
                    (kind, stride),
                    scratch,
                    table,
                ));
                gather.sets.len() - 1
            }
        };
        let slot = gather.sets[at].slot(offset)?;
        Some((
            gather.sets[at].scratch.clone(),
            u64::from(slot) * u64::from(stride),
            false,
        ))
    }

    /// Dispatches and copies since the reset.
    pub(crate) fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// Room for `bytes` of clamped indirect arguments in this list's
    /// buffers, writable by shaders from here on in the list: the buffer and
    /// the offset. None for more than a buffer holds.
    pub(crate) fn clamp_room(&self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        const CHUNK: u64 = 256 << 10;
        let bytes = bytes.next_multiple_of(16);
        if bytes > CHUNK {
            return None;
        }
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        if chunks.used > 0 && chunks.at + bytes <= CHUNK {
            let at = chunks.at;
            chunks.at += bytes;
            return Some((chunks.all[chunks.used - 1].0.clone(), at));
        }
        // The next buffer, writable.
        if chunks.used == chunks.all.len() {
            let buffer = host()?.uav_buffer(CHUNK)?;
            name(&buffer, || "hoisted indirect arguments".to_string());
            chunks.all.push((buffer, D3D12_RESOURCE_STATE_COMMON));
        }
        let i = chunks.used;
        chunks.used += 1;
        chunks.at = bytes;
        let mut barriers = Vec::new();
        if !chunks.barrier {
            // What earlier lists' shaders wrote first.
            chunks.barrier = true;
            barriers.push(D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            });
        }
        let (buffer, state) = &mut chunks.all[i];
        if *state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            barriers.push(transition(buffer, *state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            *state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        let buffer = buffer.clone();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
        Some((buffer, 0))
    }

    /// Whether the list's clamps still have to bind their root signature,
    /// pipeline and budget: the first after the reset does, and the rest
    /// find them bound (nothing else records compute state in the list).
    pub(crate) fn clamp_unbound(&self) -> bool {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        !std::mem::replace(&mut chunks.bound, true)
    }

    /// The clamped arguments written so far become readable as indirect
    /// arguments by the lists after this one.
    pub(crate) fn clamp_end(&self) {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        let used = chunks.used;
        let barriers: Vec<_> = chunks.all[..used]
            .iter_mut()
            .filter(|(_, state)| *state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .map(|(buffer, state)| {
                *state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
                transition(
                    buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )
            })
            .collect();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// Counts a draw, dispatch, clear or copy.
    pub(crate) fn count_work(&self) {
        *self.work.lock().unwrap_or_else(|e| e.into_inner()) += 1;
    }

    /// Does the list's writes that come before its first wait not met, if no
    /// work of the list precedes them; whether there were any.
    pub(crate) fn write_early(&self) -> bool {
        let mut deferred = self.deferred();
        let first = deferred.waits.iter().find(|(_, wait)| !wait.met(wait.value()));
        let Some(&(before, _)) = first else {
            return false;
        };
        let done = deferred.early;
        // Those before any work (marks only grow).
        let before = before.min(deferred.marks.iter().take_while(|&&m| m == 0).count());
        if before <= done {
            return false;
        }
        for &(at, what, bytes) in &deferred.writes[done..before] {
            what.write(at as *mut u8, bytes);
        }
        deferred.early = before;
        true
    }

    /// The waits of the list that are not met yet, apart from those its own
    /// earlier writes meet.
    pub(crate) fn unmet(&self) -> Vec<crate::memwait::Wait> {
        let deferred = self.deferred();
        deferred
            .waits
            .iter()
            .filter(|(before, wait)| {
                let own = deferred.writes[..*before]
                    .iter()
                    .rev()
                    .find(|w| w.0 == wait.address);
                match own {
                    Some((_, crate::queue::Written::Value(v), _)) => !wait.met(*v),
                    _ => !wait.met(wait.value()),
                }
            })
            .map(|w| w.1)
            .collect()
    }

    /// Copies a GPU texture readback from `src` into title memory at `dst`
    /// once the list has finished running on the host GPU.
    pub(crate) fn defer_readback(
        &self,
        dst: *mut u8,
        src: *const u8,
        dst_pitch: usize,
        src_pitch: usize,
        lines: usize,
        row_bytes: usize,
    ) {
        self.deferred().readbacks.push(Readback {
            dst: dst as usize,
            src: src as usize,
            dst_pitch,
            src_pitch,
            lines,
            row_bytes,
        });
    }

    /// `size` bytes of mapped upload memory, 512-byte aligned for a placed
    /// footprint, alive until the list is reset: the buffer, the offset in
    /// it and the CPU pointer.
    pub(crate) fn upload(&self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        self.scratch_in(size, 512, |buffer, at, _, data| (buffer.clone(), at, data))
    }

    /// `len` bytes of 512-byte-aligned readback memory for a
    /// texture-to-buffer copy: the buffer, its offset in it and the CPU
    /// pointer there. Read backs of one list share the chunks, and the
    /// chunks are reused after a reset.
    pub(crate) fn readback_space(&self, len: u64) -> Option<(ID3D12Resource, u64, *const u8)> {
        let host = host()?;
        let len = len.max(1).checked_next_multiple_of(512)?;
        if len > READBACK_CHUNK {
            let (buffer, data) = host.readback_buffer(len)?;
            name(&buffer, || format!("list readback {len:#x}"));
            self.deferred().uploads.push(buffer.clone());
            return Some((buffer, 0, data));
        }
        let mut s = self.readbacks.lock().unwrap_or_else(|e| e.into_inner());
        let (chunk, at) = s.allocate(host, len)?;
        let chunk = &s.chunks[chunk];
        Some((
            chunk.buffer.clone(),
            at,
            (chunk.mapped + at as usize) as *const u8,
        ))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        self.scratch_in(len, 256, |_, _, gpu, data| (gpu, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the backing buffer, its host GPU address, and its CPU pointer.
    pub(crate) fn scratch_held(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        self.scratch_in(len, 256, |buffer, _, gpu, data| (buffer.clone(), gpu, data))
    }

    fn scratch_one_off<R>(
        &self,
        host: &Host,
        len: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let (buffer, data) = host.upload_buffer(len)?;
        name(&buffer, || format!("list scratch {len:#x}"));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let made = f(&buffer, 0, gpu, data);
        self.deferred().uploads.push(buffer);
        Some(made)
    }

    /// `len` bytes of upload memory at a multiple of `align` until the list
    /// is reset, handed to `f` with the buffer, the offset in it, the host GPU
    /// address and the CPU pointer.
    fn scratch_in<R>(
        &self,
        len: u64,
        align: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let host = host()?;
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            return self.scratch_one_off(host, len, f);
        }
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        if s.used.next_multiple_of(align) + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let free = SCRATCH_FREE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let chunk = match free {
                Some(chunk) => chunk,
                None => match host.upload_buffer(SCRATCH_CHUNK) {
                    Some((buffer, data)) => {
                        name(&buffer, || "list scratch".into());
                        // SAFETY: a live buffer.
                        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                        (buffer, data as usize, gpu)
                    }
                    None => {
                        drop(s);
                        static WARNED: std::sync::Once = std::sync::Once::new();
                        WARNED.call_once(|| {
                            tracing::warn!(
                                len,
                                "list scratch chunk OOM; falling back to exact-size upload buffer"
                            );
                        });
                        return self.scratch_one_off(host, len, f);
                    }
                },
            };
            s.chunks.push(chunk);
            s.used = 0;
        }
        let (buffer, data, gpu) = &s.chunks[s.current];
        let at = s.used.next_multiple_of(align);
        let made = f(buffer, at, gpu + at, (data + at as usize) as *mut u8);
        s.used = at + len;
        Some(made)
    }

    /// The GPU address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let budget = self
            .budget
            .get_or_init(|| {
                let h = host()?;
                if card_budget() && self.list.cast::<ID3D12GraphicsCommandList2>().is_ok() {
                    let b = h.uav_buffer(256)?;
                    name(&b, || "indirect budget".to_owned());
                    Some(Budget::Card(b))
                } else {
                    let (b, d) = h.shared_buffer(256)?;
                    Some(Budget::Shared(b, d as usize))
                }
            })
            .as_ref()?;
        let buffer = match budget {
            Budget::Card(b) => {
                let mut deferred = self.deferred();
                deferred.budget_zeroed = true;
                b
            }
            Budget::Shared(b, data) => {
                let mut deferred = self.deferred();
                if !deferred.budget_zeroed {
                    deferred.budget_zeroed = true;
                    let at = *data;
                    // SAFETY: the list's own mapped words, zeroed before it runs.
                    deferred.copies.push(Step::Host(Box::new(move || unsafe {
                        std::ptr::write_bytes(at as *mut u8, 0, 12);
                    })));
                }
                b
            }
        };
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    #[cfg(test)]
    pub(crate) fn kept_count(&self) -> usize {
        self.deferred().kept.len()
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Queues submission-time materialization and refresh of the bindless
    /// slots in `heap`, once per list; `writes`: the draw's shaders may write
    /// memory, so the slots' unordered access ranges go back to title memory.
    pub(crate) fn defer_bindless(
        &self,
        heap: &std::sync::Arc<crate::descriptor::host::HostHeap>,
        writes: bool,
    ) {
        let mut deferred = self.deferred();
        match deferred
            .bindless
            .iter_mut()
            .find(|(h, _)| std::sync::Arc::ptr_eq(h, heap))
        {
            Some((_, w)) => *w |= writes,
            None => deferred.bindless.push((heap.clone(), writes)),
        }
    }

    /// Whether each heap queued for bindless preparation may be written.
    #[cfg(test)]
    pub(crate) fn bindless_queued(&self) -> Vec<bool> {
        self.deferred().bindless.iter().map(|&(_, w)| w).collect()
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted (and copies it to `texture` on the prelude only if dirty),
    /// keeping both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        texture: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred()
            .copies
            .push(Step::Texture(staging, texture));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Returns the textures
    /// whose staging buffers changed and must be copied onto the prelude.
    fn run_deferred(
        &self,
        batch: &mut Batch,
    ) -> Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )> {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let mut uploads = Vec::new();
        let mut deferred = self.deferred();
        let prefilled = std::mem::take(&mut deferred.prefilled);
        for (heap, writes) in &deferred.bindless {
            batch.bindless(heap, *writes);
        }
        type TexDue = (
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
            std::sync::atomic::AtomicBool,
        );
        let mut tex_due: Vec<TexDue> = Vec::new();
        let flush_textures = |tex_due: &mut Vec<TexDue>,
                              uploads: &mut Vec<(
                                  std::sync::Arc<crate::texture::Staging>,
                                  std::sync::Arc<crate::resource::HostTexture>,
                              )>| {
            if tex_due.is_empty() {
                return;
            }
            in_parallel(tex_due, 2, |chunk| {
                for (staging, _, dirty) in chunk {
                    if staging.refresh() {
                        dirty.store(true, Ordering::Relaxed);
                    }
                }
            });
            for (staging, copy, dirty) in tex_due.drain(..) {
                if dirty.into_inner() {
                    uploads.push((staging, copy));
                }
            }
        };
        let mut copy_bytes = 0u64;
        let mut copy_us = 0u64;
        #[cfg(feature = "diagnostics")]
        let mut copy_count = 0u64;
        #[cfg(feature = "diagnostics")]
        let mut refresh_count = 0u64;
        #[cfg(feature = "diagnostics")]
        let mut refresh_bytes = 0u64;
        let steps = &deferred.copies;
        let workers = fill_workers();
        let mut next = 0;
        while let Some(step) = steps.get(next) {
            let k = next;
            next += 1;
            match step {
                &Step::Copy(dst, src, len) => {
                    // A source the submission's shaders write is still old here
                    // (it lands in title memory in the copy-back), and a
                    // destination they write would lose their bytes to it.
                    // Either way it runs after copy-backs; earlier deferred
                    // destinations chain the same way.
                    if batch.gpu_written(src, len)
                        || batch.gpu_written(dst, len)
                        || batch.late_pending(src, len)
                        || batch.late_pending(dst, len)
                    {
                        batch.late.push((dst, src, len, true));
                    } else {
                        flush_textures(&mut tex_due, &mut uploads);
                        #[cfg(feature = "diagnostics")]
                        {
                            copy_count += 1;
                        }
                        let started = std::time::Instant::now();
                        batch.title_written();
                        copy_mapped(dst as *mut u8, src as *const u8, len, true);
                        copy_us += started.elapsed().as_micros() as u64;
                        copy_bytes += len as u64;
                    }
                }
                Step::Fill(..) if k < prefilled => {}
                &Step::Fill(dst, src, len) if workers < 2 => {
                    #[cfg(feature = "diagnostics")]
                    {
                        copy_count += 1;
                    }
                    let started = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    copy_us += started.elapsed().as_micros() as u64;
                    copy_bytes += len as u64;
                }
                Step::Fill(..) => {
                    // The run of fills this one starts, done at once
                    // (`VICTORY_FILL_WORKERS`).
                    let run = fill_run(&steps[k..]);
                    next += run.len() - 1;
                    let bytes = run.iter().map(|f| f.len).sum::<usize>();
                    let started = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    {
                        copy_count += run.len() as u64;
                    }
                    run_fills(&run, workers);
                    copy_us += started.elapsed().as_micros() as u64;
                    copy_bytes += bytes as u64;
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        refresh_count += 1;
                        refresh_bytes += *len as u64;
                    }
                    batch.add_or_late(shadow, *address, *len);
                }
                Step::Host(work) => {
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                }
                Step::GpuWrite(shadow, address, len, unmarked) => {
                    // The shaders start from the title's data.
                    batch.flush();
                    batch.gpu_write(*address, *len);
                    // A deferred copy's bytes must not land past a write
                    // recorded after it: the shader's stay.
                    batch.clip_late(*address, *len);
                    if *unmarked {
                        crate::marks::mark_cpu_range(*address, *len);
                    }
                    shadow.writing(*address, *len);
                }
                Step::Texture(staging, copy) => {
                    if batch.texture_due(staging) {
                        tex_due.push((
                            staging.clone(),
                            copy.clone(),
                            std::sync::atomic::AtomicBool::new(false),
                        ));
                    }
                }
            }
        }
        flush_textures(&mut tex_due, &mut uploads);
        if copy_bytes > 0 || copy_us > 0 {
            PREPARING[0].fetch_add(copy_us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copy_bytes, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                add(C::Copies, copy_count);
                add(C::CopyBytes, copy_bytes);
            }
        }
        #[cfg(feature = "diagnostics")]
        if refresh_count > 0 {
            add(C::Refreshes, refresh_count);
            add(C::RefreshBytes, refresh_bytes);
        }
        uploads
    }
}

/// Threads a run of a list's upload fills is shared among at submission
/// (`VICTORY_FILL_WORKERS`, 1 to 6; 1, one at a time, by default).
fn fill_workers() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("VICTORY_FILL_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(1)
            .clamp(1, 6)
    })
}

/// A copy of title memory into a list's upload memory ([`Step::Fill`]).
#[derive(Clone, Copy)]
struct Fill {
    dst: usize,
    src: usize,
    len: usize,
}

impl Fill {
    fn run(self) {
        copy_mapped(self.dst as *mut u8, self.src as *const u8, self.len, false);
    }
}

/// The fills `steps` starts with, up to the first other step (which may
/// write title memory or the upload memory, and so ends the run).
fn fill_run(steps: &[Step]) -> Vec<Fill> {
    steps
        .iter()
        .map_while(|s| match *s {
            Step::Fill(dst, src, len) => Some(Fill { dst, src, len }),
            _ => None,
        })
        .collect()
}

/// Threads that take shares of fill runs, started on first use (five, the
/// most [`fill_workers`] asks for besides the submitting thread); `None`
/// if none could be started (runs are done in order then). Threads spawned
/// for each run cost more than the copies (28,279 fills of 448 bytes: 17
/// ms on four threads made per run of 512, 10 ms on one).
fn fill_pool() -> Option<&'static std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>> {
    type Job = Box<dyn FnOnce() + Send>;
    static POOL: OnceLock<Option<std::sync::mpsc::Sender<Job>>> = OnceLock::new();
    POOL.get_or_init(|| {
        let (send, receive) = std::sync::mpsc::channel::<Job>();
        let receive = std::sync::Arc::new(Mutex::new(receive));
        let mut started = 0;
        for i in 0..5 {
            let receive = receive.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("victory-fills-{i}"))
                .spawn(move || {
                    loop {
                        let job = receive.lock().unwrap_or_else(|e| e.into_inner()).recv();
                        match job {
                            // A share that panics is counted done
                            // (`ShareDone`); the thread takes the next.
                            Ok(job) => {
                                drop(std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)))
                            }
                            Err(_) => return,
                        }
                    }
                });
            started += usize::from(spawned.is_ok());
        }
        (started > 0).then_some(send)
    })
    .as_ref()
}

/// Shares of a fill run left, and the wake-up when none is: one fewer when
/// dropped.
struct ShareDone(std::sync::Arc<(Mutex<usize>, std::sync::Condvar)>);

impl Drop for ShareDone {
    fn drop(&mut self) {
        let mut n = self.0.0.lock().unwrap_or_else(|e| e.into_inner());
        *n -= 1;
        if *n == 0 {
            self.0.1.notify_all();
        }
    }
}

/// Fills a thread takes at least: fewer are not worth one.
const FILLS_PER_JOB: usize = 128;

/// Whether `fills` can run in any order: their destinations rise without
/// overlapping (the upload memory a list hands out in turn), and no source
/// reaches into the destinations' span.
fn independent_fills(fills: &[Fill]) -> bool {
    let Some(first) = fills.first() else {
        return false;
    };
    let mut end = first.dst;
    for fill in fills {
        let Some(next) = fill.dst.checked_add(fill.len) else {
            return false;
        };
        if fill.dst < end {
            return false;
        }
        end = next;
    }
    fills.iter().all(|f| {
        f.src
            .checked_add(f.len)
            .is_some_and(|src_end| f.src >= end || src_end <= first.dst)
    })
}

/// Runs a run of fills: shared among up to `workers` threads, at least
/// [`FILLS_PER_JOB`] each, when they are independent, else in order. All
/// are done when it returns (before the next step reads or writes title
/// memory). Returns the threads used.
fn run_fills(fills: &[Fill], workers: usize) -> usize {
    let jobs = workers.min(fills.len() / FILLS_PER_JOB);
    if jobs > 1
        && independent_fills(fills)
        && let Some(pool) = fill_pool()
    {
        let mut chunks = fills.chunks(fills.len().div_ceil(jobs));
        let mine = chunks.next().unwrap_or_default();
        let left = std::sync::Arc::new((Mutex::new(0usize), std::sync::Condvar::new()));
        for chunk in chunks {
            *left.0.lock().unwrap_or_else(|e| e.into_inner()) += 1;
            let (chunk, left) = (chunk.to_vec(), left.clone());
            let job: Box<dyn FnOnce() + Send> = Box::new(move || {
                // Counted done when dropped: after the share, or while a
                // panic in it unwinds (the submitting thread waits on it).
                let _done = ShareDone(left);
                chunk.into_iter().for_each(Fill::run);
            });
            if let Err(std::sync::mpsc::SendError(job)) = pool.send(job) {
                // No pool thread left: the share is done here.
                job();
            }
        }
        mine.iter().copied().for_each(Fill::run);
        let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
        while *n > 0 {
            n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        jobs
    } else {
        fills.iter().copied().for_each(Fill::run);
        1
    }
}

#[cfg(test)]
mod fill_tests {
    use super::*;

    /// Fills of `bytes` each from `source` into consecutive pieces of `dst`.
    fn fills(source: &[u8], dst: &mut [u8], bytes: usize) -> Vec<Fill> {
        dst.chunks_exact_mut(bytes)
            .enumerate()
            .map(|(i, out)| Fill {
                dst: out.as_mut_ptr() as usize,
                src: source[i * bytes..].as_ptr() as usize,
                len: bytes,
            })
            .collect()
    }

    /// 512 fills of 448 bytes shared among four threads copy what one does,
    /// and the source as it is when they run.
    #[test]
    fn parallel_fills_copy_what_serial_ones_do() {
        let mut input: Vec<u8> = (0..512 * 448).map(|i| (i % 251) as u8).collect();
        let mut serial = vec![0; input.len()];
        let mut parallel = vec![0; input.len()];
        let serial_fills = fills(&input, &mut serial, 448);
        let parallel_fills = fills(&input, &mut parallel, 448);
        for turn in [0x5a, 0x37] {
            for b in &mut input {
                *b ^= turn;
            }
            assert_eq!(run_fills(&serial_fills, 1), 1);
            assert_eq!(run_fills(&parallel_fills, 4), 4);
            assert_eq!(parallel, input);
            assert_eq!(parallel, serial);
        }
    }

    /// A share that panics still counts as done: the thread waiting for
    /// the run's shares is woken.
    #[test]
    fn a_panicking_share_counts_as_done() {
        let left = std::sync::Arc::new((Mutex::new(1usize), std::sync::Condvar::new()));
        let done = ShareDone(left.clone());
        let share = std::thread::spawn(move || {
            let _done = done;
            panic!("a share that fails");
        });
        assert!(share.join().is_err());
        let n = left.0.lock().unwrap();
        let (n, waited) = left
            .1
            .wait_timeout_while(n, std::time::Duration::from_secs(5), |n| *n > 0)
            .unwrap();
        assert_eq!((*n, waited.timed_out()), (0, false));
    }

    /// A run of fills ends at the first other step: a title copy between
    /// fills splits them.
    #[test]
    fn a_title_copy_ends_a_run_of_fills() {
        let steps = [
            Step::Fill(0x1000, 0x9000, 16),
            Step::Fill(0x1010, 0x9010, 16),
            Step::Copy(0x9000, 0x5000, 16),
            Step::Fill(0x1020, 0x9000, 16),
        ];
        assert_eq!(fill_run(&steps).len(), 2);
        assert!(fill_run(&steps[2..]).is_empty());
        assert_eq!(fill_run(&steps[3..]).len(), 1);
    }

    /// Time of 28,279 fills of 448 bytes, as DexNom measured them (in runs
    /// of 512), on one thread and on four. Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn fill_runs_bench() {
        const FILLS: usize = 28_279;
        let source: Vec<u8> = (0..FILLS * 448).map(|i| i as u8).collect();
        let mut out = vec![0u8; source.len()];
        let all = fills(&source, &mut out, 448);
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..20 {
            let workers = [1, 4][round % 2];
            let started = std::time::Instant::now();
            for run in all.chunks(512) {
                run_fills(run, workers);
            }
            times[round % 2].push(started.elapsed().as_secs_f64() * 1e3);
        }
        for (workers, mut t) in [1, 4].into_iter().zip(times) {
            t.sort_by(f64::total_cmp);
            eprintln!(
                "{workers} thread(s): median {:.2} ms, best {:.2} ms",
                t[t.len() / 2],
                t[0]
            );
        }
        assert_eq!(out, source);
    }

    /// Fills whose destinations overlap, or one reading another's
    /// destination, run in order: the last write wins, the read sees the
    /// earlier write.
    #[test]
    fn dependent_fills_run_in_order() {
        let source: Vec<u8> = (0..=255).collect();
        let mut output = [0; 256];
        let mut overlapping = fills(&source, &mut output, 1);
        overlapping[255].dst = overlapping[0].dst;
        assert_eq!(run_fills(&overlapping, 2), 1);
        assert_eq!(output[0], 255);
        let mut aliased = fills(&source, &mut output, 1);
        aliased[0].src = source[167..].as_ptr() as usize;
        aliased[1].src = aliased[0].dst;
        assert_eq!(run_fills(&aliased, 2), 1);
        assert_eq!(output[1], 167);
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
/// Per-shadow sorted, disjoint `[from, to)` refresh spans in a [`Batch`].
/// Overlapping and touching intervals coalesce; far-apart intervals stay
/// separate so refreshes do not copy or stall on unrelated gaps.
const MAX_BATCH_SPANS: usize = 64;

enum Spans {
    One((usize, usize)),
    Many(Vec<(usize, usize)>),
}

impl Spans {
    fn add(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        match self {
            Self::One((a, b)) => {
                if from <= *b && to >= *a {
                    *a = (*a).min(from);
                    *b = (*b).max(to);
                } else {
                    let v = if to < *a {
                        vec![(from, to), (*a, *b)]
                    } else {
                        vec![(*a, *b), (from, to)]
                    };
                    *self = Self::Many(v);
                }
            }
            Self::Many(v) => coalesce_span(v, from, to),
        }
    }

    fn as_slice(&self) -> &[(usize, usize)] {
        match self {
            Self::One(range) => std::slice::from_ref(range),
            Self::Many(ranges) => ranges.as_slice(),
        }
    }
}

fn coalesce_span(spans: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    if from >= to {
        return;
    }
    let pos = spans.partition_point(|&(_, end)| end < from);
    let mut lo = from;
    let mut hi = to;
    let mut erase_end = pos;
    while erase_end < spans.len() && spans[erase_end].0 <= hi {
        lo = lo.min(spans[erase_end].0);
        hi = hi.max(spans[erase_end].1);
        erase_end += 1;
    }
    if pos < erase_end {
        spans[pos] = (lo, hi);
        spans.drain(pos + 1..erase_end);
    } else {
        spans.insert(pos, (lo, hi));
    }
    if spans.len() > MAX_BATCH_SPANS
        && let Some(best) = (0..spans.len() - 1).min_by_key(|&i| spans[i + 1].0 - spans[i].1)
    {
        spans[best].1 = spans[best + 1].1;
        spans.remove(best + 1);
    }
}

type LaterPrelude<'a> = (usize, &'a HostList, Vec<D3D12_RESOURCE_BARRIER>);

enum Prelude {
    None,
    Recorded(ID3D12CommandList),
    Barriers(Vec<D3D12_RESOURCE_BARRIER>),
}

// SAFETY: `D3D12_RESOURCE_BARRIER` wraps COM pointers kept alive by `HostList::states` until reset.
unsafe impl Send for Prelude {}
// SAFETY: only read across worker threads while recording distinct lists' preludes.
unsafe impl Sync for Prelude {}

const PARALLEL_PRELUDE_BARRIERS: usize = 2048;

fn parallel_preludes(runs: &[&HostList]) -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on =
        *ON.get_or_init(|| std::env::var_os("VICTORY_PARALLEL_PRELUDES").is_some_and(|v| v == "1"));
    if !on || runs.len() < 4 {
        return false;
    }
    let mut seen = std::collections::HashSet::with_capacity(runs.len());
    runs.iter()
        .all(|list| seen.insert(std::ptr::from_ref(*list)))
}

fn record_preludes(later: &[LaterPrelude<'_>]) -> Vec<(usize, Option<ID3D12CommandList>)> {
    let barriers: usize = later.iter().map(|(_, _, b)| b.len()).sum();
    if later.len() < 4 || barriers < PARALLEL_PRELUDE_BARRIERS {
        return later
            .iter()
            .map(|(at, list, b)| (*at, list.record_prelude(b)))
            .collect();
    }
    let recorded: Vec<Mutex<Option<ID3D12CommandList>>> =
        later.iter().map(|_| Mutex::new(None)).collect();
    let items: Vec<(usize, &LaterPrelude<'_>)> = later.iter().enumerate().collect();
    in_parallel(&items, 1, |chunk| {
        for &(k, (_, list, b)) in chunk {
            *recorded[k].lock().unwrap_or_else(|e| e.into_inner()) = list.record_prelude(b);
        }
    });
    later
        .iter()
        .zip(recorded)
        .map(|((at, ..), r)| (*at, r.into_inner().unwrap_or_else(|e| e.into_inner())))
        .collect()
}

#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, Spans)>,
    /// The same of sparse copies.
    pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: crate::fx::Set<usize>,
    /// View heaps the submission's draws index directly, and whether their
    /// shaders may write memory.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    /// Their plans, which hold the host resources the slots view until the
    /// submission has run.
    plans: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// Bindless unordered access ranges, copied back after the submission.
    bindless_writes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Plans made ahead while the previous submission ran.
    ahead: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// Title ranges the submission's shaders are declared to write so far
    /// `(start, end)`: their results reach title memory in the copy-back
    /// after the lists run, so a deferred copy touching one waits for it.
    gpu_writes: Vec<(usize, usize)>,
    /// Deferred copies kept past the lists' run, in record order (they run
    /// after copy-backs): `(dst, src, len, title_dst)`.
    late: Vec<(usize, usize, usize, bool)>,
    /// Refreshes whose range a deferred copy's destination touches: done
    /// after the late copies, which is where its bytes are.
    late_refresh: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let Some(end) = address.checked_add(len).filter(|&e| e > address) else {
            return;
        };
        if shadow.is_sparse() {
            // It stands for far more memory than a list reads.
            self.pieces.push((shadow.clone(), address, end));
            return;
        }
        match self
            .refreshes
            .iter_mut()
            .find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, spans)) => spans.add(address, end),
            None => self
                .refreshes
                .push((shadow.clone(), Spans::One((address, end)))),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// A list of the submission indexes `heap` directly.
    fn bindless(&mut self, heap: &std::sync::Arc<crate::descriptor::host::HostHeap>, writes: bool) {
        match self
            .bindless
            .iter_mut()
            .find(|(h, _)| std::sync::Arc::ptr_eq(h, heap))
        {
            Some((_, w)) => *w |= writes,
            None => self.bindless.push((heap.clone(), writes)),
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    /// A range the submission's shaders write (`Step::GpuWrite`), in
    /// recorded order: a deferred copy touching it sees its bytes only in
    /// the copy-back.
    fn gpu_write(&mut self, at: usize, len: usize) {
        if let Some(end) = at.checked_add(len).filter(|&e| e > at) {
            self.gpu_writes.push((at, end));
        }
    }

    /// Whether `len` title bytes at `at` touch a range this submission's
    /// shaders write.
    fn gpu_written(&self, at: usize, len: usize) -> bool {
        let Some(end) = at.checked_add(len).filter(|&e| e > at) else {
            return false;
        };
        self.gpu_writes.iter().any(|&(a, b)| a < end && at < b)
    }

    /// Whether `len` bytes at `at` touch a deferred copy's title
    /// destination so later steps chain after it.
    fn late_pending(&self, at: usize, len: usize) -> bool {
        let Some(end) = at.checked_add(len).filter(|&e| e > at) else {
            return false;
        };
        self.late
            .iter()
            .any(|&(d, _, l, t)| t && d < end && at < d + l)
    }

    /// `add`, or deferred past the submission's late copies when overlapping
    /// a deferred copy's title destination.
    fn add_or_late(
        &mut self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        if self.late_pending(address, len) {
            self.late_refresh.push((shadow.clone(), address, len));
        } else {
            self.add(shadow, address, len);
        }
    }

    /// Drops the parts of pending deferred copies a write recorded later covers.
    fn clip_late(&mut self, at: usize, len: usize) {
        let Some(w1) = at.checked_add(len).filter(|&e| e > at) else {
            return;
        };
        let w0 = at;
        let mut kept: Vec<(usize, usize, usize, bool)> = Vec::with_capacity(self.late.len());
        for &(dst, src, l, t) in &self.late {
            if !t || w1 <= dst || dst + l <= w0 {
                kept.push((dst, src, l, t));
                continue;
            }
            if dst < w0 {
                kept.push((dst, src, w0 - dst, t));
            }
            if w1 < dst + l {
                kept.push((w1, src + (w1 - dst), dst + l - w1, t));
            }
        }
        self.late = kept;
    }

    fn flush(&mut self) {
        let refreshes = std::mem::take(&mut self.refreshes);
        in_parallel(&refreshes, 2, |chunk| {
            for (shadow, spans) in chunk {
                update_spans(shadow, spans.as_slice());
            }
        });
        let mut pieces = std::mem::take(&mut self.pieces);
        if pieces.is_empty() {
            return;
        }
        pieces.sort_unstable_by_key(|(s, from, _)| (s.root_key(), *from));
        let mut merged_pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> =
            Vec::new();
        for (shadow, from, to) in pieces {
            if let Some((s, _, end)) = merged_pieces.last_mut()
                && s.root_key() == shadow.root_key()
                && from <= *end + 4096
            {
                *end = (*end).max(to);
                continue;
            }
            merged_pieces.push((shadow, from, to));
        }
        in_parallel(&merged_pieces, 2, |chunk| {
            for (s, a, b) in chunk {
                update(s, *a, b - a);
            }
        });
    }

    /// After every list's steps (no title copy follows): each directly
    /// indexed heap's slots, once per submission however many lists and
    /// draws use it. What they read is refreshed (unchanged pages cost only
    /// the write-watch's answer); their unordered access ranges are kept
    /// for copy-back only when a shader that may write memory uses the heap.
    fn prepare_bindless(&mut self) {
        self.flush();
        let copy = |plan: &crate::descriptor::host::BindlessPlan| {
            let (reads, all) = plan.copy_constants();
            if !reads.is_empty() {
                let started = std::time::Instant::now();
                let copied = std::sync::atomic::AtomicU64::new(0);
                in_parallel(reads, 8, |reads| {
                    for (shadow, address, len) in reads {
                        #[cfg(feature = "diagnostics")]
                        {
                            use crate::copystat::{C, add};
                            add(C::Refreshes, 1);
                            add(C::RefreshBytes, *len as u64);
                        }
                        let n = shadow.update_lazily(*address, *len, all);
                        copied.fetch_add(n as u64, Ordering::Relaxed);
                    }
                });
                PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                PREPARING[1].fetch_add(copied.into_inner(), Ordering::Relaxed);
            }
        };
        for plan in std::mem::take(&mut self.ahead) {
            copy(&plan);
            self.plans.push(plan);
        }
        for (heap, writes) in std::mem::take(&mut self.bindless) {
            let started = std::time::Instant::now();
            let plan = heap.bindless_plan();
            BINDLESS_US[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            let Some(plan) = plan else {
                continue;
            };
            copy(&plan);
            // shortcut: what shaders write through directly indexed slots into
            // copied memory stays in the copy (title memory used in place
            // needs no copy back); copying all of it back cost ~0.3 s a frame.
            if writes && cfg!(test) {
                for (shadow, address, len) in plan.writes.iter() {
                    if !self.bindless_writes.iter().any(|(s, a, l)| {
                        std::sync::Arc::ptr_eq(s, shadow) && (*a, *l) == (*address, *len)
                    }) {
                        shadow.writing(*address, *len);
                        self.bindless_writes.push((shadow.clone(), *address, *len));
                    }
                }
            }
            self.plans.push(plan);
        }
    }
}

/// Uploads before a submission's lists ([`crate::texture::upload_first`]).
#[derive(Default)]
struct Early {
    copies: Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )>,
    copied: Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )>,
    copy_wait: Option<(ID3D12Fence, u64)>,
    before: Vec<D3D12_RESOURCE_BARRIER>,
    after: Vec<D3D12_RESOURCE_BARRIER>,
}

impl Early {
    /// Takes the textures waiting; `late`: after the lists' preludes took
    /// the textures' states, so not those the lists use (they upload them).
    fn take(&mut self, batch: &mut Batch, runs: &[&HostList], late: bool) {
        const COPY_DEST: i32 = D3D12_RESOURCE_STATE_COPY_DEST.0;
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut first = crate::texture::take_first();
        let mut ready_direct = Vec::new();
        if let Some(drain) = crate::heap_worker::flush_before_submit() {
            if let Some((fence, val)) = drain.wait {
                match &mut self.copy_wait {
                    Some((f, cur)) if val > *cur => {
                        *f = fence;
                        *cur = val;
                    }
                    None => self.copy_wait = Some((fence, val)),
                    _ => {}
                }
            }
            for (staging, copy) in drain.copied {
                let _ = batch.texture_due(&staging);
                self.copied.push((staging, copy));
            }
            ready_direct = drain.ready_direct;
            first.extend(drain.unprocessed);
        }
        if !first.is_empty() || !ready_direct.is_empty() {
            let mut used = crate::fx::Set::default();
            if late {
                for list in runs {
                    let states = list.states.lock().unwrap_or_else(|e| e.into_inner());
                    used.extend(states.values().map(|s| std::sync::Arc::as_ptr(&s.2) as usize));
                }
            }
            let due: Vec<(
                std::sync::Arc<crate::texture::Staging>,
                std::sync::Arc<crate::resource::HostTexture>,
                std::sync::atomic::AtomicBool,
            )> = first
                .into_iter()
                .filter(|(staging, copy)| {
                    !used.contains(&(std::sync::Arc::as_ptr(copy) as usize))
                        && batch.texture_due(staging)
                })
                .map(|(staging, copy)| (staging, copy, std::sync::atomic::AtomicBool::new(false)))
                .collect();
            if !due.is_empty() {
                in_parallel(&due, 2, |chunk| {
                    for (staging, _, dirty) in chunk {
                        if staging.refresh() {
                            dirty.store(true, Ordering::Relaxed);
                        }
                    }
                });
            }
            let mut push_direct = |staging, copy: std::sync::Arc<crate::resource::HostTexture>| {
                let real = copy.state.swap(SHADER_RESOURCE, Ordering::Relaxed) as i32;
                let (to, from) = (D3D12_RESOURCE_STATES(COPY_DEST), D3D12_RESOURCE_STATES(real));
                if real != COPY_DEST {
                    self.before.push(transition(&copy.texture, from, to));
                }
                self.after
                    .push(transition(&copy.texture, to, D3D12_RESOURCE_STATES(SHADER_RESOURCE as i32)));
                self.copies.push((staging, copy));
            };
            for (staging, copy, dirty) in due {
                if dirty.into_inner() {
                    push_direct(staging, copy);
                }
            }
            for (staging, copy) in ready_direct {
                if !used.contains(&(std::sync::Arc::as_ptr(&copy) as usize))
                    && batch.texture_due(&staging)
                {
                    push_direct(staging, copy);
                }
            }
        }
        if late && !self.copied.is_empty() {
            for (_, copy) in &self.copied {
                if copy
                    .state
                    .compare_exchange(0, SHADER_RESOURCE, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.after.push(transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATES(SHADER_RESOURCE as i32),
                    ));
                }
            }
        }
    }

    /// The list with the uploads, if there are any. One submission runs at a
    /// time and is waited for, so there is one list.
    fn record(&self, host: &Host) -> Option<ID3D12CommandList> {
        if self.copies.is_empty() && self.before.is_empty() && self.after.is_empty() {
            return None;
        }
        struct List(ID3D12CommandAllocator, ID3D12GraphicsCommandList);
        // SAFETY: free-threaded objects, used under the submission's lock.
        unsafe impl Send for List {}
        // SAFETY: as above.
        unsafe impl Sync for List {}
        static LIST: OnceLock<Option<List>> = OnceLock::new();
        // SAFETY: the list is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        unsafe {
            let List(allocator, list) = LIST
                .get_or_init(|| {
                    let allocator: ID3D12CommandAllocator = host
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .ok()?;
                    let list: ID3D12GraphicsCommandList = host
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                        .ok()?;
                    list.Close().ok()?;
                    Some(List(allocator, list))
                })
                .as_ref()?;
            let recorded = allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !self.before.is_empty() {
                    list.ResourceBarrier(&self.before);
                }
                for (staging, copy) in &self.copies {
                    staging.record_copies(list, &copy.texture);
                }
                if !self.after.is_empty() {
                    list.ResourceBarrier(&self.after);
                }
                list.Close()
            });
            if let Err(e) = recorded {
                fail(format_args!("texture uploads did not record ({e})"));
            }
            list.cast().ok()
        }
    }
}

/// Persistent worker pool backing [`in_parallel`], avoiding per-submission OS
/// thread creation/teardown overhead under Wine.
struct Pool {
    dispatch: std::sync::Mutex<()>,
    shared: &'static PoolShared,
    threads: usize,
}

struct PoolShared {
    state: std::sync::Mutex<PoolState>,
    wake: std::sync::Condvar,
    done: std::sync::Condvar,
}

struct PoolState {
    epoch: u64,
    job: Option<PoolJob>,
}

#[derive(Clone, Copy)]
struct PoolJob {
    run: unsafe fn(*const (), usize, usize),
    ctx: *const (),
    len: usize,
    piece: usize,
    next: *const std::sync::atomic::AtomicUsize,
    remaining: *const std::sync::atomic::AtomicUsize,
}

// SAFETY: `PoolJob` is only visible while `in_parallel` blocks on the caller
// thread until `remaining` reaches zero, and `T`/`F` inside `ctx` are `Sync`.
unsafe impl Send for PoolJob {}

impl PoolShared {
    fn execute_chunks(&self, job: PoolJob) {
        // SAFETY: `job.next` points to the live stack `AtomicUsize` on the
        // calling thread until `remaining` hits zero.
        let next = unsafe { &*job.next };
        loop {
            let from = next.fetch_add(job.piece, Ordering::Relaxed);
            if from >= job.len {
                break;
            }
            let to = (from + job.piece).min(job.len);
            // SAFETY: `job.ctx` is the live `Ctx` on the caller's stack and
            // `[from..to]` is within `list.len()`.
            unsafe { (job.run)(job.ctx, from, to) };
        }
    }
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let total = std::thread::available_parallelism()
            .map_or(6, |n| n.get().saturating_sub(2).clamp(4, 20));
        let shared: &'static PoolShared = Box::leak(Box::new(PoolShared {
            state: std::sync::Mutex::new(PoolState { epoch: 0, job: None }),
            wake: std::sync::Condvar::new(),
            done: std::sync::Condvar::new(),
        }));
        let mut spawned = 0usize;
        for i in 0..total.saturating_sub(1) {
            if std::thread::Builder::new()
                .name(format!("victory-pool-{i}"))
                .spawn(move || {
                    struct FinishGuard {
                        shared: &'static PoolShared,
                        remaining: *const std::sync::atomic::AtomicUsize,
                    }
                    impl Drop for FinishGuard {
                        fn drop(&mut self) {
                            // SAFETY: caller keeps `remaining` live until it drops to 0.
                            let rem = unsafe { &*self.remaining };
                            if rem.fetch_sub(1, Ordering::AcqRel) == 1 {
                                let _held =
                                    self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                                self.shared.done.notify_one();
                            }
                        }
                    }
                    let mut seen_epoch = 0u64;
                    let mut guard = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    loop {
                        while guard.epoch == seen_epoch || guard.job.is_none() {
                            guard = shared.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
                        }
                        seen_epoch = guard.epoch;
                        let Some(job) = guard.job else { continue };
                        drop(guard);
                        let _done = FinishGuard {
                            shared,
                            remaining: job.remaining,
                        };
                        shared.execute_chunks(job);
                        drop(_done);
                        guard = shared.state.lock().unwrap_or_else(|e| e.into_inner());
                    }
                })
                .is_ok()
            {
                spawned += 1;
            }
        }
        Pool {
            dispatch: std::sync::Mutex::new(()),
            shared,
            threads: spawned + 1,
        }
    })
}

/// Runs `f` on pieces of `list`, of `least` items or more, on a persistent worker pool.
pub(crate) fn in_parallel<T: Sync, F: Fn(&[T]) + Sync>(list: &[T], least: usize, f: F) {
    if list.len() <= least {
        return f(list);
    }
    let pool = pool();
    if pool.threads <= 1 {
        return f(list);
    }
    let Ok(_dispatch_guard) = pool.dispatch.try_lock() else {
        // Nested or concurrent `in_parallel` call: run inline instead of spawning or deadlocking.
        return f(list);
    };
    let bg_workers = pool.threads - 1;
    let piece = (list.len() / (pool.threads * 4)).max(least);
    if list.len() <= piece {
        return f(list);
    }
    struct Ctx<'a, T, F> {
        list: &'a [T],
        f: &'a F,
    }
    unsafe fn trampoline<T, F: Fn(&[T]) + Sync>(ctx: *const (), from: usize, to: usize) {
        // SAFETY: `ctx` is a valid `&Ctx<T, F>` live on the caller's stack until all workers finish.
        let ctx = unsafe { &*ctx.cast::<Ctx<'_, T, F>>() };
        (ctx.f)(&ctx.list[from..to]);
    }
    let ctx = Ctx { list, f: &f };
    let next = std::sync::atomic::AtomicUsize::new(0);
    let remaining = std::sync::atomic::AtomicUsize::new(bg_workers);
    let job = PoolJob {
        run: trampoline::<T, F>,
        ctx: (&raw const ctx).cast(),
        len: list.len(),
        piece,
        next: &raw const next,
        remaining: &raw const remaining,
    };
    {
        let mut state = pool.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.job = Some(job);
        state.epoch = state.epoch.wrapping_add(1);
        pool.shared.wake.notify_all();
    }
    pool.shared.execute_chunks(job);
    let mut state = pool.shared.state.lock().unwrap_or_else(|e| e.into_inner());
    while remaining.load(Ordering::Acquire) != 0 {
        state = pool.shared.done.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    state.job = None;
}

/// Brings `[from, from + len)` of a host copy up to date now.
fn update(shadow: &crate::shadow::Shadow, from: usize, len: usize) {
    let started = std::time::Instant::now();
    let copied = shadow.update(from, len);
    let us = started.elapsed().as_micros() as u64;
    #[cfg(feature = "diagnostics")]
    crate::copystat::range(from, len, us, 0);
    PREPARING[0].fetch_add(us, Ordering::Relaxed);
    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
}

/// Refreshes multiple sorted non-overlapping `[from, to)` spans of `shadow` in
/// one batch so write-watch syscalls coalesce across adjacent spans.
fn update_spans(shadow: &crate::shadow::Shadow, spans: &[(usize, usize)]) {
    if spans.is_empty() {
        return;
    }
    let started = std::time::Instant::now();
    let copied = shadow.update_spans(spans, false);
    let us = started.elapsed().as_micros() as u64;
    #[cfg(feature = "diagnostics")]
    {
        let first = spans[0].0;
        let last = spans[spans.len() - 1].1;
        crate::copystat::range(first, last.saturating_sub(first), us, 0);
    }
    PREPARING[0].fetch_add(us, Ordering::Relaxed);
    PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 || title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // Our synchronous copy writes the destination: open any guarded blocks without pinning.
        crate::watch::open_for_sync_write(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    if !title_dst
        && len >= 64 << 10
        && (crate::watch::has_known_hole(src as usize, (src as usize).saturating_add(len))
            || crate::watch::has_cached_unmapped_hole(
                src as usize,
                (src as usize).saturating_add(len),
            ))
    {
        read_sparse(src, dst, len);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
        return;
    }
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies unless `VICTORY_HOLES_ONCE=1` is active (in
/// which case holes clear on map/unmap/commit).
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        if let Some(hole_end) = crate::watch::cached_unmapped_end(at as usize) {
            let n = hole_end.saturating_sub(at as usize).min(remaining);
            if n > 0 {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                // SAFETY: `to` is a writable host staging slice covering `remaining` bytes.
                unsafe { to.write_bytes(0, n) };
                offset += n;
                continue;
            }
        }
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            if info.Type == MEM_MAPPED {
                let lo = info.BaseAddress as usize;
                let hi = lo.saturating_add(info.RegionSize);
                crate::watch::record_unmapped_hole(lo, hi);
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_safe_copy",
    ".globl victory_safe_copy_start",
    ".globl victory_safe_copy_end",
    ".globl victory_safe_copy_fault",
    "victory_safe_copy:",
    "victory_safe_copy_start:",
    "cmp r8, 64",
    "jb 2f",
    "1:",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqu xmm2, xmmword ptr [rdx + 32]",
    "movdqu xmm3, xmmword ptr [rdx + 48]",
    "movdqu xmmword ptr [rcx], xmm0",
    "movdqu xmmword ptr [rcx + 16], xmm1",
    "movdqu xmmword ptr [rcx + 32], xmm2",
    "movdqu xmmword ptr [rcx + 48], xmm3",
    "add rdx, 64",
    "add rcx, 64",
    "sub r8, 64",
    "cmp r8, 64",
    "jae 1b",
    "2:",
    "cmp r8, 16",
    "jb 3f",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmmword ptr [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "jmp 2b",
    "3:",
    "cmp r8, 8",
    "jb 4f",
    "mov rax, qword ptr [rdx]",
    "mov qword ptr [rcx], rax",
    "add rdx, 8",
    "add rcx, 8",
    "sub r8, 8",
    "4:",
    "test r8, r8",
    "jz 5f",
    "mov al, byte ptr [rdx]",
    "mov byte ptr [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jnz 4b",
    "5:",
    "victory_safe_copy_end:",
    "mov eax, 1",
    "ret",
    "victory_safe_copy_fault:",
    "xor eax, eax",
    "ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_safe_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_safe_copy_start: u8;
    static victory_safe_copy_end: u8;
    static victory_safe_copy_fault: u8;
}

/// Installs our Vectored Exception Handler once, ahead of the crash reporter,
/// so `safe_copy` can copy mapped memory in pure user mode without kernel
/// syscalls and recover cleanly if a page is unmapped.
#[cfg(target_arch = "x86_64")]
fn ensure_safe_copy_veh() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_IN_PAGE_ERROR, NTSTATUS,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: Windows passes valid exception and context pointers.
        let r = unsafe { &*(*info).ExceptionRecord };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.ExceptionCode != EXCEPTION_IN_PAGE_ERROR as NTSTATUS
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `ContextRecord` is valid for x86_64 exceptions.
        let ctx = unsafe { &mut *(*info).ContextRecord };
        let rip = ctx.Rip as usize;
        let start = (&raw const victory_safe_copy_start) as usize;
        let end = (&raw const victory_safe_copy_end) as usize;
        if rip >= start && rip < end {
            const WRITE: usize = 1;
            if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                && r.NumberParameters >= 2
                && r.ExceptionInformation[0] == WRITE
                && crate::watch::try_open_guarded(r.ExceptionInformation[1])
            {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            ctx.Rip = (&raw const victory_safe_copy_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: registers a read-only IP-range check handler ahead of crash reporting.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Whether to force kernel `ReadProcessMemory` / `WriteProcessMemory` copies
/// (`VICTORY_KERNEL_COPY=1`, diagnostic escape hatch).
fn kernel_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KERNEL_COPY").is_some())
}

/// Zero-syscall user-mode copy for non-overlapping ranges; returns `false` if
/// either side hits an unmapped or inaccessible page.
#[cfg(target_arch = "x86_64")]
fn try_user_copy(dst: *mut u8, src: *const u8, len: usize) -> Option<bool> {
    if kernel_copy() {
        return None;
    }
    let (d, s) = (dst as usize, src as usize);
    if len == 0 {
        return Some(true);
    }
    // Reject obviously invalid low addresses without even taking a fault.
    if d < 0x10000 || s < 0x10000 {
        return Some(false);
    }
    let (Some(d_end), Some(s_end)) = (d.checked_add(len), s.checked_add(len)) else {
        return Some(false);
    };
    // Overlapping ranges fall back to kernel copy (memmove semantics).
    if d < s_end && s < d_end {
        return None;
    }
    ensure_safe_copy_veh();
    // SAFETY: leaf assembly uses only volatile registers and recovers via VEH
    // if `src` or `dst` hits an unmapped page.
    Some(unsafe { victory_safe_copy(dst, src, len) != 0 })
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

type NtReadVirtualMemoryFn = unsafe extern "system" fn(
    *mut std::ffi::c_void,
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
    usize,
    *mut usize,
) -> i32;

type NtWriteVirtualMemoryFn = unsafe extern "system" fn(
    *mut std::ffi::c_void,
    *mut std::ffi::c_void,
    *const std::ffi::c_void,
    usize,
    *mut usize,
) -> i32;

fn ntdll_export(name: &std::ffi::CStr) -> Option<usize> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
    // SAFETY: plain lookup in `ntdll.dll`, which is always loaded.
    unsafe {
        let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
        if ntdll.is_null() {
            return None;
        }
        GetProcAddress(ntdll, name.as_ptr().cast()).map(|f| f as usize)
    }
}

fn nt_read_virtual_memory() -> Option<NtReadVirtualMemoryFn> {
    static FN: OnceLock<Option<usize>> = OnceLock::new();
    let addr = (*FN.get_or_init(|| ntdll_export(c"NtReadVirtualMemory")))?;
    // SAFETY: `addr` came from `GetProcAddress(ntdll, "NtReadVirtualMemory")`.
    Some(unsafe { std::mem::transmute::<usize, NtReadVirtualMemoryFn>(addr) })
}

fn nt_write_virtual_memory() -> Option<NtWriteVirtualMemoryFn> {
    static FN: OnceLock<Option<usize>> = OnceLock::new();
    let addr = (*FN.get_or_init(|| ntdll_export(c"NtWriteVirtualMemory")))?;
    // SAFETY: `addr` came from `GetProcAddress(ntdll, "NtWriteVirtualMemory")`.
    Some(unsafe { std::mem::transmute::<usize, NtWriteVirtualMemoryFn>(addr) })
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if let Some(ok) = try_user_copy(dst, src, len) {
        return ok;
    }
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    if let Some(nt_read) = nt_read_virtual_memory() {
        // SAFETY: destination is writable mapped memory of `len` bytes; the
        // kernel validates `src` and returns a negative status on unmapped pages.
        let status = unsafe { nt_read(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) };
        return status >= 0 && done == len;
    }
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        #[cfg(target_arch = "x86_64")]
        if let Some(ok) = try_user_copy(at as *mut u8, from, n) {
            return ok;
        }
        let mut done = 0;
        if let Some(nt_write) = nt_write_virtual_memory() {
            // SAFETY: `from` points into our mapped host copy; the kernel
            // validates destination pages and returns an NTSTATUS.
            let status = unsafe { nt_write(GetCurrentProcess(), at as _, from.cast(), n, &mut done) };
            return status >= 0 && done == n;
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in GPU-upload (Resizable BAR VRAM, when supported and within
    /// 85% of the live VRAM budget) or standard upload heap, mapped for CPU
    /// writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let use_gpu_upload = self.gpu_upload
            && video_memory().is_none_or(|[(used, budget), _]| {
                budget == 0 || used.saturating_add(size) < budget * 85 / 100
            });
        let try_make = |heap_type: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain creation and mapping of an upload buffer.
            let made = unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut buffer,
                    )
                    .and_then(|()| {
                        let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                        b.Map(0, None, Some(&mut data))
                    })
            };
            made.map(|()| (buffer, data))
        };
        let made = if use_gpu_upload {
            try_make(D3D12_HEAP_TYPE_GPU_UPLOAD, D3D12_RESOURCE_STATE_COMMON)
                .or_else(|_| try_make(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
        } else {
            try_make(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)
        };
        match made {
            Ok((buffer, data)) => {
                let buffer = buffer?;
                crate::memstat::track(&buffer, 0, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                wait_if_removed(&self.device, &e);
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer in the readback heap, mapped for reading.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                let buffer = buffer?;
                crate::memstat::track(&buffer, 1, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    log_frames: u32,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            log_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        self.log_frames += 1;
        #[cfg(feature = "diagnostics")]
        TIMED_FRAMES.fetch_add(1, Ordering::Relaxed);
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let [plans_ms, records_ms] =
            [0, 1].map(|i| BINDLESS_US[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
        let dropped = crate::draw::take_dropped() as f64 / frames;
        // Outside the submissions: in none of the times above.
        let waited = crate::pipeline::take_waited_ms() / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        #[cfg(feature = "diagnostics")]
        let [reap_ms, finish_ms] = [0, 1].map(|i| {
            ASYNC_REAP_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            tracing::info!("sparse writes since: {}", crate::shadow::sparse_writes_text());
            #[cfg(feature = "diagnostics")]
            log_gpu_waits();
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "async retire since: {reap_ms:.1} ms/frame reap ({finish_ms:.1} ms/frame finish_submission)"
            );
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "prewarm since: {:.1} ms, {:.2} MB of fills done ahead, {:.1} ms of plans made ahead, {} slots staged and copied",
                PREWARM_US.swap(0, Ordering::Relaxed) as f64 / 1e3,
                EARLY_FILLS.swap(0, Ordering::Relaxed) as f64 / 1e6,
                EARLY_PLAN_US.swap(0, Ordering::Relaxed) as f64 / 1e3,
                EARLY_SLOTS.swap(0, Ordering::Relaxed)
            );
            #[cfg(feature = "diagnostics")]
            log_gpu_times();
            tracing::info!(
                "[{} {}] {fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB, bindless plans \
                 {plans_ms:.1} ms; {gpu:.1} on the GPU; {back_ms:.1} copying back {back_mb:.1} MB; \
                 bindless records {records_ms:.1} ms), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {waited:.1} ms/frame of draws waiting for pipelines",
                crate::descriptor::host::BACKLOG[0].load(Ordering::Relaxed),
                crate::descriptor::host::BACKLOG[1].load(Ordering::Relaxed),
                making = crate::pipeline::making()
            );
            {
                let reuses = crate::shadow::REUSES.load(Ordering::Relaxed);
                let (windows, window_bytes) = crate::shadow::loose_stats();
                tracing::info!(
                    "loose windows: {windows} ({:.0} MB, never freed), {reuses} covered by a wider \
                     one since start",
                    window_bytes as f64 / 1e6,
                );
            }
            #[cfg(feature = "diagnostics")]
            {
                tracing::info!(
                    written = crate::resource::rewrites(),
                    evicted = crate::resource::EVICTED.load(Ordering::Relaxed),
                    late = crate::texture::CHANGED[0].load(Ordering::Relaxed),
                    at_once = crate::texture::CHANGED[1].load(Ordering::Relaxed),
                    "textures changed in place"
                );
                crate::texture::flips::log();
                crate::port_cbv::log(frames);
                crate::heap_worker::log(frames);
                crate::watch::log_port_guard(frames);
                crate::watch::log_holes_once(frames);
                crate::copystat::log(frames);
                let beside_frames = f64::from(self.log_frames.max(1));
                self.log_frames = 0;
                if crate::watch::copy_back_beside() {
                    let (beside_writes, beside_bytes) = crate::watch::take_beside_stats();
                    tracing::info!(
                        "copy-back beside (VICTORY_COPY_BACK_BESIDE=1): {:.1} writes/frame, {:.2} MB/frame",
                        beside_writes as f64 / beside_frames,
                        beside_bytes as f64 / 1e6 / beside_frames
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        self.wait_in_flight();
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ),
            ]);
            p.copy.close();
            self.execute(&[&p.copy]);
            // The capture and the black-frame probe read what the copy wrote
            // and are dropped when this returns: with `VICTORY_ASYNC_SUBMIT`
            // the copy may not have run yet (the device was removed reading
            // a freed probe).
            #[cfg(feature = "diagnostics")]
            if capture.is_some() || probe.is_some() {
                self.wait_in_flight();
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    /// As [`Host::present`], with the texture `over` put over `scene` (each
    /// with the host state it is in and stays in).
    pub(crate) fn present_over(
        &self,
        scene: Option<(&ID3D12Resource, u32)>,
        over: (&ID3D12Resource, u32),
        frame: u64,
    ) -> Option<()> {
        self.wait_in_flight();
        // SAFETY: live resources.
        let (a, b) = unsafe { (scene.map(|s| s.0.GetDesc()), over.0.GetDesc()) };
        // The scene's size; with a smaller one (or none) the screen's.
        let (width, height) = match a {
            Some(a) if a.Height >= 1080 => (a.Width as u32, a.Height),
            _ => (1920, 1080),
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = self.presenter(width, height, format).ok();
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            }
            .ok()?;
            (p.width, p.height, p.format) = (width, height, format);
        }
        let view = |d: &D3D12_RESOURCE_DESC| swap_format(d.Format);
        let planes = [
            scene.and_then(|s| Some((s.0, view(&a?)?))),
            Some((over.0, view(&b)?)),
        ];
        // SAFETY: our swap chain, list and live textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let back = p.swap_chain.GetBuffer::<ID3D12Resource>(index).ok()?;
            p.copy.reset();
            let list = &p.copy.list;
            let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
            let mut into = vec![transition(&back, D3D12_RESOURCE_STATE_PRESENT, target)];
            let mut out = vec![transition(&back, target, D3D12_RESOURCE_STATE_PRESENT)];
            for (texture, state) in scene.iter().chain([&over]) {
                let state = D3D12_RESOURCE_STATES(*state as i32);
                if state != read {
                    into.push(transition(texture, state, read));
                    out.push(transition(texture, read, state));
                }
            }
            list.ResourceBarrier(&into);
            crate::compose::record(self, list, planes, &back, (width, height, format));
            #[cfg(feature = "diagnostics")]
            let capture = {
                list.ResourceBarrier(&[transition(&back, target, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let capture = crate::capture::wanted(frame)
                    .then(|| crate::capture::read_back(&self.device, list, &back))
                    .flatten();
                list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_COPY_SOURCE, target)]);
                capture
            };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&out);
            p.copy.close();
            self.execute(&[&p.copy]);
            // As in `present`: the capture reads what the copy wrote.
            #[cfg(feature = "diagnostics")]
            if capture.is_some() {
                self.wait_in_flight();
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
        }
        Some(())
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                let buffer = buffer?;
                crate::memstat::track(&buffer, 2, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                wait_if_removed(&self.device, &e);
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => {
                if let Some(b) = &buffer {
                    crate::memstat::track(b, 3, size);
                }
                buffer
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let before = markers.iter().filter(|(at, _)| (*at as usize) < from);
    for (at, marker) in before.rev().take(6) {
        let short: String = marker.chars().take(90).collect();
        text.push_str(&format!(" (before, {at}: {short})"));
    }
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 100 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
        // Its constants as they are now: the title may have overwritten them.
        #[cfg(feature = "diagnostics")]
        for c in marker.split("Constant(0x").skip(1).take(4) {
            let mut parts = c.split([',', ')']);
            let at = parts.next().and_then(|a| u64::from_str_radix(a, 16).ok());
            let size = parts.next().and_then(|s| s.trim().parse::<usize>().ok());
            if let (Some(at), Some(size)) = (at, size) {
                let words = crate::shadow::words_text(at, size);
                let same = c.contains(&words);
                text.push_str(&format!(" NOW {at:#x} same={same} {words}"));
                if shown == 1 {
                    let mut slots: Vec<u32> = words
                        .split(['[', ',', ']'])
                        .filter_map(|w| u32::from_str_radix(w.trim(), 16).ok())
                        .filter(|w| (0x1000..0x13_0000).contains(w))
                        .collect();
                    for u in marker.split(" u").skip(1) {
                        let v = u.split_once("=0x").map(|(_, v)| v.split(' ').next().unwrap_or(""));
                        if let Some(v) = v.and_then(|v| u32::from_str_radix(v.trim(), 16).ok())
                            && v % 32 == 0
                            && v >= 0x10000
                        {
                            slots.extend([0, 1, 2, 3, 6, 24, 25].map(|k| v / 32 + k));
                        }
                    }
                    text.push_str(&crate::draw::report_slots(&slots));
                }
            }
        }
        // The records a skinning program takes its views from: those whose
        // slot holds no buffer.
        #[cfg(feature = "diagnostics")]
        if shown <= 3 && marker.contains("CSComputeSkinning") {
            let number = |key: &str, radix: u32, end: char| {
                marker
                    .split_once(key)
                    .and_then(|(_, v)| u64::from_str_radix(v.split(end).next()?, radix).ok())
            };
            let list = marker.split_once("Srv s1+16: placed").and_then(|(_, t)| {
                u64::from_str_radix(t.split_once(" at 0x")?.1.split('"').next()?, 16).ok()
            });
            if let (Some(list), Some(first), Some(count)) =
                (list, number(" u6=0x", 16, ' '), number(" u7=0x", 16, ' '))
            {
                text.push_str(&format!(" RECORDS {first}+{count} of {list:#x}"));
                for k in 0..count.min(64) {
                    let at = list + 192 + (first + k) * 176;
                    let words = crate::shadow::words_text(at, 128);
                    let w: Vec<u32> = words
                        .split(['[', ',', ']'])
                        .filter_map(|w| u32::from_str_radix(w.trim(), 16).ok())
                        .collect();
                    if w.len() < 32 {
                        text.push_str(&format!(" REC {k} unreadable"));
                        continue;
                    }
                    let slots = [w[18], w[15] & 0xff_ffff, w[19] & 0xff_ffff];
                    let what = crate::draw::report_slots(&slots);
                    if k < 1 || what.contains("null") || what.contains("Shader(") || what.contains("DIRTY") {
                        text.push_str(&format!(" REC {k} {w:x?}{what}"));
                    }
                }
            }
        }
        // What its root views start with now.
        #[cfg(feature = "diagnostics")]
        if shown <= 6 {
            for r in marker.split(" root").skip(1).take(10) {
                if let Some((slot, rest)) = r.split_once("=0x")
                    && let Ok(at) = u64::from_str_radix(rest.split(' ').next().unwrap_or(""), 16)
                {
                    text.push_str(&format!(" ROOT {slot} {at:#x} {}", crate::shadow::words_text(at, 80)));
                    if let Some(s) = crate::shadow::find(at, 80)
                        && s.is_sparse()
                    {
                        text.push_str(&format!(" {}", crate::shadow::args_text(at)));
                    }
                }
            }
        }
        // What an indirect command got, as the title's memory has it now.
        if let Some(at) = marker
            .strip_prefix("indirect kind ")
            .and_then(|m| m.split(" at 0x").nth(1))
            .and_then(|m| u64::from_str_radix(m.split(' ').next().unwrap_or(""), 16).ok())
        {
            text.push_str(&format!(" args now {}", crate::shadow::args_text(at)));
        }
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn safe_copy_handles_valid_unaligned_and_unmapped_memory() {
        use super::{copy_mapped, read_into, write_mapped};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        for len in [0usize, 1, 7, 8, 15, 16, 31, 63, 64, 65, 129, 4096, 70_000] {
            let src: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
            let mut dst = vec![0xAAu8; len];
            assert!(read_into(src.as_ptr(), &mut dst), "valid read len={len}");
            assert_eq!(dst, src, "mismatch len={len}");
        }
        // Reserve 2 pages: page 0 committed RW, page 1 uncommitted (faults on access).
        // SAFETY: fresh reservation, freed below.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * 4096, MEM_RESERVE, PAGE_NOACCESS) as usize
        };
        assert_ne!(base, 0);
        // SAFETY: commit only page 0.
        let committed = unsafe {
            VirtualAlloc(base as *mut _, 4096, MEM_COMMIT, PAGE_READWRITE) as usize
        };
        assert_eq!(committed, base);
        // SAFETY: page 0 is committed RW.
        unsafe { std::ptr::write_bytes(base as *mut u8, 0x5A, 4096) };

        let mut buf = vec![0u8; 2 * 4096];
        // Reading across into uncommitted page 1 must fail cleanly without crashing.
        assert!(!read_into(base as *const u8, &mut buf));
        // copy_mapped fills page 0 with 0x5A and zeroes uncommitted page 1.
        copy_mapped(buf.as_mut_ptr(), base as *const u8, 2 * 4096, false);
        assert!(buf[..4096].iter().all(|&b| b == 0x5A));
        assert!(buf[4096..].iter().all(|&b| b == 0));

        // write_mapped writes page 0 and safely skips uncommitted page 1.
        let src = vec![0x3Cu8; 2 * 4096];
        write_mapped(base as *mut u8, src.as_ptr(), 2 * 4096);
        // SAFETY: page 0 is committed RW.
        let page0 = unsafe { std::slice::from_raw_parts(base as *const u8, 4096) };
        assert!(page0.iter().all(|&b| b == 0x3C));

        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn persistent_in_parallel_runs_small_large_and_nested_batches() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let items: Vec<AtomicUsize> = (0..1024).map(|_| AtomicUsize::new(0)).collect();
        super::in_parallel(&items[..4], 8, |chunk| {
            for cell in chunk {
                cell.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert!(items[..4].iter().all(|c| c.load(Ordering::Relaxed) == 1));
        assert!(items[4..].iter().all(|c| c.load(Ordering::Relaxed) == 0));

        for c in &items {
            c.store(0, Ordering::Relaxed);
        }
        super::in_parallel(&items, 4, |chunk| {
            // Nested call must execute inline without deadlocking.
            super::in_parallel(chunk, 1, |inner| {
                for cell in inner {
                    cell.fetch_add(1, Ordering::Relaxed);
                }
            });
        });
        assert!(items.iter().all(|c| c.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn coalesced_spans_keep_disjoint_ranges_separate_and_merge_overlaps() {
        use super::{MAX_BATCH_SPANS, Spans};

        let mut spans = Spans::One((100, 160));
        spans.add(110, 140);
        assert_eq!(spans.as_slice(), &[(100, 160)]);
        spans.add(160, 200);
        assert_eq!(spans.as_slice(), &[(100, 200)]);
        spans.add(80, 120);
        assert_eq!(spans.as_slice(), &[(80, 200)]);
        spans.add(300, 300);
        assert_eq!(spans.as_slice(), &[(80, 200)]);

        spans.add(10, 30);
        spans.add(500, 560);
        spans.add(300, 360);
        assert_eq!(
            spans.as_slice(),
            &[(10, 30), (80, 200), (300, 360), (500, 560)]
        );

        spans.add(200, 300);
        assert_eq!(spans.as_slice(), &[(10, 30), (80, 360), (500, 560)]);

        let mut bounded = Spans::One((0, 10));
        for i in 1..MAX_BATCH_SPANS {
            let start = i * 100;
            bounded.add(start, start + 10);
        }
        assert_eq!(bounded.as_slice().len(), MAX_BATCH_SPANS);
        bounded.add(120, 125);
        assert_eq!(bounded.as_slice().len(), MAX_BATCH_SPANS);
        assert!(bounded.as_slice().contains(&(100, 125)));
    }

    #[test]
    fn readback_spaces_share_chunks_and_are_reused() {
        use windows::core::Interface;
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        let (first, at, ptr) = list.readback_space(4 << 10).unwrap();
        let (second, at2, ptr2) = list.readback_space(4 << 10).unwrap();
        assert_eq!(first.as_raw(), second.as_raw(), "one chunk for the list's read backs");
        assert_eq!(at % 512, 0);
        assert_eq!(at2, at + (4 << 10));
        assert_eq!(ptr2 as usize - ptr as usize, 4 << 10);
        list.close();
        host.execute(&[&list]);
        list.reset();
        let (again, at3, _) = list.readback_space(4 << 10).unwrap();
        assert_eq!((again.as_raw(), at3), (first.as_raw(), at), "reused after reset");
        list.close();
    }

    #[test]
    fn texture_readback_through_shared_chunks_round_trips_texels() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = super::host() else { return };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: plain creation into our out pointer.
        unsafe {
            host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut texture,
            )
        }
        .expect("texture");
        let texture = texture.expect("texture");
        let mut footprint = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 1];
        let mut rows = [0u32; 1];
        let mut total = 0u64;
        // SAFETY: arrays for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(&desc, 0, 1, 0, Some(footprint.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        let f = footprint[0];
        let pitch = f.Footprint.RowPitch as usize;
        let (upload, udata) = host.upload_buffer(total).expect("upload buffer");
        let mut pattern = [0u8; 64];
        for y in 0..4 {
            for x in 0..16 {
                let v = (1 + y * 16 + x) as u8;
                pattern[y * 16 + x] = v;
                // SAFETY: inside the upload buffer.
                unsafe { *udata.add(y * pitch + x) = v };
            }
        }
        let list = host.new_list().unwrap();
        let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
        };
        let placed = |r: &ID3D12Resource, mut p: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, offset: u64| {
            p.Offset = offset;
            D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: p },
            }
        };
        // SAFETY: an open list; the texture is COPY_DEST, the upload buffer is upload memory.
        unsafe {
            list.list.CopyTextureRegion(&sub(&texture, 0), 0, 0, 0, &placed(&upload, f, 0), None);
            list.list.ResourceBarrier(&[super::transition(
                &texture,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        let (readback, rat, rdata) = list.readback_space(total).expect("readback space");
        let rat = rat as usize;
        assert_eq!(rat % 512, 0);
        // SAFETY: the texture is COPY_SOURCE; the readback chunk is COPY_DEST memory.
        unsafe {
            list.list.CopyTextureRegion(&placed(&readback, f, rat as u64), 0, 0, 0, &sub(&texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for y in 0..4 {
            for x in 0..16 {
                // SAFETY: inside the readback chunk, GPU done.
                let got = unsafe { *rdata.add(y * pitch + x) };
                assert_eq!(got, pattern[y * 16 + x], "texel row {y} byte {x}");
            }
        }
        list.reset();
    }
}

#[cfg(test)]
mod copy_order {
    fn pool_page() -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: fresh private allocation of one page for the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0);
        p
    }

    fn free_page(p: usize) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // SAFETY: allocation from `pool_page`.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn a_copy_past_a_gpu_write_sees_what_it_wrote() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return;
        };
        let pools = [pool_page(), pool_page(), pool_page(), pool_page()];
        // SAFETY: our pools, not guarded yet.
        unsafe {
            (pools[0] as *mut u8).write_volatile(0xAA);
            (pools[2] as *mut u8).write_volatile(0xCC);
        }
        crate::shadow::register(pools[0], pools[0] as u64, PAGE as u64);
        let copy = crate::shadow::find(pools[0] as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        copy.refresh(&list, pools[0] as u64, PAGE as u64);
        copy.gpu_writes(&list, pools[0] as u64, PAGE as u64);
        let src_copy = copy.clone();
        list.defer_host(move || src_copy.write_host(pools[0] as u64, &[0xBB]));
        list.defer_copy(pools[1] as *mut u8, pools[0] as *const u8, PAGE);
        list.defer_copy(pools[0] as *mut u8, pools[2] as *const u8, PAGE);
        list.defer_copy(pools[3] as *mut u8, pools[2] as *const u8, PAGE);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let at = |k: usize| unsafe { (pools[k] as *const u8).read_volatile() };
        assert_eq!(at(1), 0xBB, "the copy sees what the shaders wrote");
        assert_eq!(at(0), 0xCC, "a copy over the write lands after it");
        assert_eq!(at(3), 0xCC, "an unrelated copy is copied at once");
        crate::shadow::unregister(pools[0], pools[0] as u64);
        drop(copy);
        for &p in &pools {
            free_page(p);
        }
    }

    #[test]
    fn deferred_copies_chain_and_a_later_gpu_write_wins() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return;
        };
        let pools = [pool_page(), pool_page(), pool_page(), pool_page(), pool_page()];
        // SAFETY: our pools, not guarded yet.
        unsafe {
            (pools[0] as *mut u8).write_volatile(0xAA);
            (pools[4] as *mut u8).write_volatile(0x55);
        }
        crate::shadow::register(pools[0], pools[0] as u64, PAGE as u64);
        crate::shadow::register(pools[4], pools[4] as u64, PAGE as u64);
        let copy0 = crate::shadow::find(pools[0] as u64, PAGE as u64).unwrap();
        let copy4 = crate::shadow::find(pools[4] as u64, PAGE as u64).unwrap();
        let list = host.new_list().unwrap();
        copy0.refresh(&list, pools[0] as u64, PAGE as u64);
        copy0.gpu_writes(&list, pools[0] as u64, PAGE as u64);
        let c0 = copy0.clone();
        list.defer_host(move || c0.write_host(pools[0] as u64, &[0xBB]));
        list.defer_copy(pools[1] as *mut u8, pools[0] as *const u8, PAGE);
        list.defer_copy(pools[3] as *mut u8, pools[1] as *const u8, PAGE);
        list.defer_fill(pools[2] as *mut u8, pools[0] as *const u8, PAGE);
        list.defer_copy(pools[4] as *mut u8, pools[0] as *const u8, PAGE);
        copy4.refresh(&list, pools[4] as u64, PAGE as u64);
        copy4.gpu_writes(&list, pools[4] as u64, PAGE as u64);
        let c4 = copy4.clone();
        list.defer_host(move || c4.write_host(pools[4] as u64, &[0xEE]));
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let at = |k: usize| unsafe { (pools[k] as *const u8).read_volatile() };
        assert_eq!(at(1), 0xBB, "the deferred copy sees the shader's bytes");
        assert_eq!(at(3), 0xBB, "a chained copy reads the deferred bytes");
        assert_eq!(at(2), 0xAA, "a fill copies at once: the old source");
        assert_eq!(at(4), 0xEE, "a later GPU write wins over the deferred copy");
        crate::shadow::unregister(pools[0], pools[0] as u64);
        crate::shadow::unregister(pools[4], pools[4] as u64);
        drop(copy0);
        drop(copy4);
        for &p in &pools {
            free_page(p);
        }
    }

    #[test]
    fn deferred_clear_reuses_steady_state_and_shrinks_outliers() {
        let mut d = super::Deferred {
            prefilled: 5,
            early: 3,
            budget_zeroed: true,
            ..Default::default()
        };
        d.copies.reserve(20_000);
        d.refreshed.reserve(10_000);
        d.gpu_written.reserve(10_000);
        d.filled.reserve(10_000);
        d.textures.reserve(10_000);
        d.bound.reserve(10_000);
        d.marks.reserve(256);
        d.marks.push(42);
        d.bound.insert(7);
        d.clear();
        assert_eq!(d.prefilled, 0);
        assert_eq!(d.early, 0);
        assert!(!d.budget_zeroed);
        assert!(d.marks.is_empty());
        assert!(d.bound.is_empty());
        assert!(d.marks.capacity() >= 256);
        assert!(d.copies.capacity() <= 8192);
        assert!(d.refreshed.capacity() <= 2048);
        assert!(d.gpu_written.capacity() <= 2048);
        assert!(d.filled.capacity() <= 2048);
        assert!(d.textures.capacity() <= 2048);
        assert!(d.bound.capacity() <= 2048);
    }

    #[test]
    fn scratch_one_off_fallback_allocates_exact_aligned_buffer() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().expect("a host list");
        let (off, gpu, data_non_null) = list
            .scratch_one_off(host, 512, |buf, off, gpu, data| {
                // SAFETY: live upload buffer of 512 bytes.
                assert_eq!(unsafe { buf.GetDesc() }.Width, 512);
                (off, gpu, !data.is_null())
            })
            .expect("exact-size scratch buffer");
        assert_eq!(off, 0);
        assert_ne!(gpu, 0);
        assert!(data_non_null);
        assert_eq!(list.deferred().uploads.len(), 1);
        list.close();
    }
}

/// The device is gone: why is logged, and the caller waits, so the thread
/// that runs the submission can report it before the title fails.
fn wait_if_removed(device: &ID3D12Device, error: &windows::core::Error) {
    if error.code().0 as u32 != 0x887A_0005 {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: a plain query.
        let reason = unsafe { device.GetDeviceRemovedReason() };
        tracing::error!(?reason, layer = ?host().and_then(|h| h.log_messages()), "host GPU device gone");
    });
    std::thread::sleep(std::time::Duration::from_secs(20));
}
