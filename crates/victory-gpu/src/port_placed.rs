//! Placed graphics reservations (`VICTORY_PORT_PLACED=1`, off by default):
//! backs the title's 32-bit graphics reservation with a custom L0 write-back
//! `ID3D12Heap` mapped at the title's exact virtual address via
//! `VK_KHR_map_memory2` + `VK_EXT_map_memory_placed` (`VK_LAYER_VICTORY_placed`).
//!
//! Never `Map()` a resource placed inside such a heap (vkd3d-proton's original
//! CPU mapping was released with `VK_MEMORY_UNMAP_RESERVE_BIT_EXT`), and
//! never apply write guards / `GetWriteWatch` on its addresses.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::{GUID, HRESULT, Interface};

const K64: u64 = 64 << 10;
const MOST: u64 = 4 << 30;

/// `VICTORY_PORT_PLACED`: `1` enables placed RADV L0 heap mappings over title
/// reservations (`VK_LAYER_VICTORY_placed`, `ENABLE_VICTORY_PLACED=1`).
pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_PLACED").is_ok_and(|v| v.trim() == "1"))
}

/// `[heaps_placed, bytes_placed, fallbacks, heaps_released, bytes_released]`
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

struct PlacedRegion {
    start: u64,
    end: u64,
    heap: ID3D12Heap,
    device: u64,
    memory: u64,
}

// SAFETY: `ID3D12Heap` is free-threaded; Vulkan handles are opaque integers.
unsafe impl Send for PlacedRegion {}

static PLACED_REGIONS: Mutex<Vec<PlacedRegion>> = Mutex::new(Vec::new());
static PLACED_BOUNDS: [AtomicUsize; 2] = [AtomicUsize::new(usize::MAX), AtomicUsize::new(0)];

fn update_bounds(regions: &[PlacedRegion]) {
    let lo = regions.iter().map(|r| r.start as usize).min().unwrap_or(usize::MAX);
    let hi = regions.iter().map(|r| r.end as usize).max().unwrap_or(0);
    PLACED_BOUNDS[0].store(lo, Ordering::Release);
    PLACED_BOUNDS[1].store(hi, Ordering::Release);
}

// vkd3d-proton's interop interface (`ID3D12DXVKInteropDevice3`).
const IID_INTEROP3: GUID = GUID::from_u128(0x22a70184_a6a4_4c24_bf97_7d6df9f12d8a);

#[repr(C)]
struct InteropVtbl {
    query_interface: usize,
    add_ref: usize,
    release: unsafe extern "system" fn(*mut c_void) -> u32,
    get_dxgi_adapter: usize,
    get_instance_extensions: usize,
    get_device_extensions: usize,
    get_device_features: usize,
    get_vulkan_handles:
        unsafe extern "system" fn(*mut c_void, *mut u64, *mut u64, *mut u64) -> HRESULT,
    get_vulkan_queue_info: usize,
    get_vulkan_image_layout: usize,
    get_vulkan_resource_info: usize,
    lock_command_queue: usize,
    unlock_command_queue: usize,
    get_vulkan_resource_info1: usize,
    create_interop_command_queue: usize,
    create_interop_command_allocator: usize,
    begin_vk_command_buffer_interop: usize,
    end_vk_command_buffer_interop: usize,
    lock_vulkan_queue: usize,
    unlock_vulkan_queue: usize,
    get_vulkan_heap_info:
        unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u64, *mut u64, *mut u32) -> HRESULT,
}

const VK_STRUCTURE_TYPE_MEMORY_MAP_INFO: u32 = 1_000_271_000;
const VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO: u32 = 1_000_271_001;
const VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT: u32 = 1_000_272_002;
const VK_MEMORY_MAP_PLACED_BIT_EXT: u32 = 1;
const VK_MEMORY_UNMAP_RESERVE_BIT_EXT: u32 = 1;

#[repr(C)]
struct VkMemoryMapInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    memory: u64,
    offset: u64,
    size: u64,
}

#[repr(C)]
struct VkMemoryUnmapInfo {
    s_type: u32,
    p_next: *const c_void,
    flags: u32,
    memory: u64,
}

#[repr(C)]
struct VkMemoryMapPlacedInfoExt {
    s_type: u32,
    p_next: *const c_void,
    placed_address: *mut c_void,
}

type PfnMap2 = unsafe extern "system" fn(u64, *const VkMemoryMapInfo, *mut *mut c_void) -> i32;
type PfnUnmap2 = unsafe extern "system" fn(u64, *const VkMemoryUnmapInfo) -> i32;

struct Vk {
    map2: PfnMap2,
    unmap2: PfnUnmap2,
}

fn vk() -> Option<&'static Vk> {
    static VK: OnceLock<Option<Vk>> = OnceLock::new();
    VK.get_or_init(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress, LoadLibraryA};
        // SAFETY: `vulkan-1.dll` exports `vkMapMemory2` and `vkUnmapMemory2` and
        // stays loaded while vkd3d-proton holds the Vulkan device.
        let found = unsafe {
            let mut module = GetModuleHandleA(c"vulkan-1.dll".as_ptr().cast());
            if module.is_null() {
                module = LoadLibraryA(c"vulkan-1.dll".as_ptr().cast());
            }
            if module.is_null() {
                None
            } else {
                let map2 = GetProcAddress(module, c"vkMapMemory2".as_ptr().cast());
                let unmap2 = GetProcAddress(module, c"vkUnmapMemory2".as_ptr().cast());
                match (map2, unmap2) {
                    // SAFETY: function pointers match Vulkan 1.4 `vkMapMemory2` / `vkUnmapMemory2`.
                    (Some(map2), Some(unmap2)) => Some(Vk {
                        map2: std::mem::transmute::<unsafe extern "system" fn() -> isize, PfnMap2>(
                            map2,
                        ),
                        unmap2: std::mem::transmute::<
                            unsafe extern "system" fn() -> isize,
                            PfnUnmap2,
                        >(unmap2),
                    }),
                    _ => None,
                }
            }
        };
        if found.is_none() {
            tracing::warn!(
                "port_placed: vulkan-1.dll has no vkMapMemory2/vkUnmapMemory2; falling back"
            );
        }
        found
    })
    .as_ref()
}

/// Initializes host D3D12 and resolves Vulkan entry points before mapping locks are held.
pub fn warm() {
    let _ = crate::host::host();
    let _ = vk();
}

/// Prepared L0 heap whose vkd3d-proton mapping has been released with `RESERVE`.
pub struct Prepared {
    heap: ID3D12Heap,
    device: u64,
    memory: u64,
    bytes: u64,
}

// SAFETY: `ID3D12Heap` is free-threaded and Vulkan handles are plain values.
unsafe impl Send for Prepared {}

fn unmap_reserving(vk: &Vk, device: u64, memory: u64) -> i32 {
    let info = VkMemoryUnmapInfo {
        s_type: VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO,
        p_next: std::ptr::null(),
        flags: VK_MEMORY_UNMAP_RESERVE_BIT_EXT,
        memory,
    };
    // SAFETY: `device` and `memory` came from `ID3D12DXVKInteropDevice3` for a live heap.
    unsafe { (vk.unmap2)(device, &info) }
}

/// Creates a custom L0 write-back `ID3D12Heap` of `bytes` and releases vkd3d-proton's
/// default CPU mapping with `VK_MEMORY_UNMAP_RESERVE_BIT_EXT`.
pub fn prepare(bytes: usize) -> Result<Prepared, String> {
    let bytes = bytes as u64;
    if bytes == 0 || !bytes.is_multiple_of(K64) || bytes > MOST {
        return Err(format!("{bytes:#x} bytes: not 64 KiB pages, or over 4 GiB"));
    }
    let vk = vk().ok_or("no vkMapMemory2/vkUnmapMemory2 in vulkan-1.dll")?;
    let host = crate::host::host().ok_or("no host device")?;
    let desc = D3D12_HEAP_DESC {
        SizeInBytes: bytes,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        },
        Alignment: 0,
        Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
    };
    let mut heap: Option<ID3D12Heap> = None;
    // SAFETY: valid heap description and output slot.
    unsafe { host.device.CreateHeap(&desc, &mut heap) }
        .map_err(|e| format!("no heap of {bytes:#x} bytes: {e}"))?;
    let heap = heap.ok_or("no heap")?;
    // SAFETY: `heap` is live.
    let made = unsafe { heap.GetDesc() }.SizeInBytes;
    if made != bytes {
        return Err(format!("the heap has {made:#x} bytes, not {bytes:#x}"));
    }
    let mut interop: *mut c_void = std::ptr::null_mut();
    // SAFETY: COM query for `ID3D12DXVKInteropDevice3` on `host.device`.
    let hr = unsafe { host.device.query(&IID_INTEROP3, &mut interop) };
    if hr.is_err() || interop.is_null() {
        return Err(format!("no ID3D12DXVKInteropDevice3 (not vkd3d-proton?): {hr:?}"));
    }
    // SAFETY: `interop` implements `InteropVtbl` (`ID3D12DXVKInteropDevice3`) and is released below.
    let (device, memory, offset, hr) = unsafe {
        let vt = &**(interop as *mut *const InteropVtbl);
        let (mut instance, mut physical, mut device) = (0u64, 0u64, 0u64);
        let handles = (vt.get_vulkan_handles)(interop, &mut instance, &mut physical, &mut device);
        let (mut memory, mut offset, mut kind) = (0u64, u64::MAX, 0u32);
        let hr = if handles.is_ok() {
            (vt.get_vulkan_heap_info)(interop, heap.as_raw(), &mut memory, &mut offset, &mut kind)
        } else {
            handles
        };
        (vt.release)(interop);
        (device, memory, offset, hr)
    };
    if hr.is_err() || device == 0 || memory == 0 {
        return Err(format!("GetVulkanHandles/GetVulkanHeapInfo refused: {hr:?}"));
    }
    if offset != 0 {
        return Err(format!(
            "the heap is at offset {offset:#x} of its Vulkan allocation, not a whole one"
        ));
    }
    let r = unmap_reserving(vk, device, memory);
    if r != 0 {
        return Err(format!("vkUnmapMemory2(RESERVE) of vkd3d's mapping: VkResult {r}"));
    }
    Ok(Prepared {
        heap,
        device,
        memory,
        bytes,
    })
}

/// Maps `prepared` over committed RW process memory at `at` with `VK_MEMORY_MAP_PLACED_BIT_EXT`.
pub fn map(prepared: &Prepared, at: usize) -> Result<(), String> {
    let vk = vk().ok_or("no vulkan entry points")?;
    let placed = VkMemoryMapPlacedInfoExt {
        s_type: VK_STRUCTURE_TYPE_MEMORY_MAP_PLACED_INFO_EXT,
        p_next: std::ptr::null(),
        placed_address: at as *mut c_void,
    };
    let info = VkMemoryMapInfo {
        s_type: VK_STRUCTURE_TYPE_MEMORY_MAP_INFO,
        p_next: (&raw const placed).cast(),
        flags: VK_MEMORY_MAP_PLACED_BIT_EXT,
        memory: prepared.memory,
        offset: 0,
        size: u64::MAX,
    };
    let mut got: *mut c_void = std::ptr::null_mut();
    // SAFETY: `prepared.device` and `prepared.memory` are live and unmapped; `placed` outlives the call.
    let r = unsafe { (vk.map2)(prepared.device, &info, &mut got) };
    if r == 0 && got as usize == at {
        return Ok(());
    }
    if r == 0 && !got.is_null() {
        let plain = VkMemoryUnmapInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_UNMAP_INFO,
            p_next: std::ptr::null(),
            flags: 0,
            memory: prepared.memory,
        };
        // SAFETY: unmaps the fallback address chosen when the extension was inactive.
        unsafe { (vk.unmap2)(prepared.device, &plain) };
    }
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        tracing::warn!(
            result = r,
            got = format_args!("{got:p}"),
            at = format_args!("{at:#x}"),
            "port_placed: vkMapMemory2(PLACED) refused or mapped elsewhere; is ENABLE_VICTORY_PLACED=1 active?"
        );
    });
    Err(format!(
        "vkMapMemory2(PLACED at {at:#x}): VkResult {r}, mapped at {got:p}"
    ))
}

/// Releases the placed mapping (keeping addresses reserved) when wiring fails after `map`.
pub fn unmap(prepared: &Prepared) {
    if let Some(vk) = vk() {
        let r = unmap_reserving(vk, prepared.device, prepared.memory);
        if r != 0 {
            tracing::warn!(result = r, "port_placed: failed to unmap placed mapping");
        }
    }
}

/// Registers a placed heap covering `[at, at + prepared.bytes)`.
pub fn register(prepared: Prepared, at: usize) {
    let Prepared {
        heap,
        device,
        memory,
        bytes,
    } = prepared;
    let start = at as u64;
    let end = start + bytes;
    {
        let mut regions = PLACED_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
        regions.retain(|r| r.end <= start || r.start >= end);
        regions.push(PlacedRegion {
            start,
            end,
            heap,
            device,
            memory,
        });
        update_bounds(&regions);
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(bytes, Ordering::Relaxed);
    tracing::info!(
        at = format_args!("{at:#x}"),
        bytes,
        "port_placed: title reservation backed by placed L0 allocation at title address"
    );
}

/// Releases any placed regions intersecting `[at, at + len)`. Returns bytes released.
pub fn release(at: usize, len: usize) -> u64 {
    let start = at as u64;
    let end = start.saturating_add(len.max(1) as u64);
    let gone = {
        let mut regions = PLACED_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = Vec::new();
        let mut i = 0;
        while i < regions.len() {
            if regions[i].end > start && regions[i].start < end {
                out.push(regions.swap_remove(i));
            } else {
                i += 1;
            }
        }
        update_bounds(&regions);
        out
    };
    let mut bytes = 0;
    for r in gone {
        let span = r.end - r.start;
        crate::shadow::retire_direct(r.start, r.end);
        let res = vk().map_or(-1, |vk| unmap_reserving(vk, r.device, r.memory));
        bytes += span;
        STATS[3].fetch_add(1, Ordering::Relaxed);
        STATS[4].fetch_add(span, Ordering::Relaxed);
        tracing::warn!(
            start = format_args!("{:#x}", r.start),
            bytes = span,
            unmap = res,
            "port_placed: released placed region"
        );
        drop(r.heap);
    }
    bytes
}

/// Fast-path check for whether `[at, at + len)` touches a placed region.
pub fn placed_touches(at: u64, len: u64) -> bool {
    if !on() {
        return false;
    }
    let hi = PLACED_BOUNDS[1].load(Ordering::Acquire);
    let lo = PLACED_BOUNDS[0].load(Ordering::Acquire);
    let addr = at as usize;
    let end_addr = addr.saturating_add(len.max(1) as usize);
    if lo >= hi || addr >= hi || end_addr <= lo {
        return false;
    }
    let end = at.saturating_add(len.max(1));
    PLACED_REGIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|r| r.start < end && at < r.end)
}

/// Places a non-shared buffer resource covering the registered L0 placed heap that
/// contains `[start, start + len)`. Returns `(heap, resource, region_start, region_len)`.
pub(crate) fn place_in(
    device: &ID3D12Device,
    start: u64,
    len: u64,
) -> Option<(ID3D12Heap, ID3D12Resource, u64, u64)> {
    let end = start.checked_add(len.max(1))?;
    let (heap, region_start, size) = {
        let regions = PLACED_REGIONS.lock().unwrap_or_else(|e| e.into_inner());
        let region = regions.iter().find(|r| r.start <= start && end <= r.end)?;
        (region.heap.clone(), region.start, region.end - region.start)
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
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
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: `heap` owns `[0, size)` on `device`.
    let hr = unsafe {
        device.CreatePlacedResource(
            &heap,
            0,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    };
    if let Err(e) = hr {
        tracing::warn!(error = %e, at = format_args!("{region_start:#x}"), size, "port_placed: CreatePlacedResource failed");
        return None;
    }
    Some((heap, resource?, region_start, size))
}

/// Records a fallback to non-placed memory for a reservation.
pub fn fallback(why: &str) {
    let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
    if n <= 8 || n.is_power_of_two() {
        tracing::warn!(why, "port_placed: falling back from placed reservation");
    }
}

/// Logs summary counters alongside periodic frame stats.
pub fn log() {
    if !on() {
        return;
    }
    let s = STATS.each_ref().map(|a| a.load(Ordering::Relaxed));
    tracing::info!(
        "port_placed (VICTORY_PORT_PLACED=1): {} heaps ({} MB), {} fallbacks, {} released ({} MB)",
        s[0],
        s[1] >> 20,
        s[2],
        s[3],
        s[4] >> 20
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placed_heap_maps_at_requested_address_and_places_buffer() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        if !on() {
            return;
        }
        warm();
        let Some(host) = crate::host::host() else {
            return;
        };
        let bytes = 64 << 10;
        // SAFETY: test owns this 64 KiB read-write reservation for the duration of the test.
        let addr = unsafe { VirtualAlloc(std::ptr::null(), bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) }
            as usize;
        assert_ne!(addr, 0);
        let prepared = prepare(bytes).expect("prepare placed L0 heap");
        map(&prepared, addr).expect("vkMapMemory2(PLACED) at exact address");
        register(prepared, addr);
        assert!(placed_touches(addr as u64, bytes as u64));

        // Write via CPU address and verify a placed resource covers the same range on GPU.
        // SAFETY: `[addr, addr + bytes)` is backed by the mapped L0 memory.
        unsafe { std::ptr::write_volatile(addr as *mut u32, 0xCAFE_BABEu32) };
        let (_, resource, aligned, aligned_len) =
            place_in(&host.device, addr as u64, bytes as u64).expect("place_in buffer");
        assert_eq!((aligned, aligned_len), (addr as u64, bytes as u64));
        // SAFETY: live D3D12 resource.
        let gpu = unsafe { resource.GetGPUVirtualAddress() };
        assert_ne!(gpu, 0);
        // SAFETY: `[addr, addr + bytes)` holds our written value.
        assert_eq!(unsafe { std::ptr::read_volatile(addr as *const u32) }, 0xCAFE_BABEu32);
        drop(resource);

        assert_eq!(release(addr, bytes), bytes as u64);
        assert!(!placed_touches(addr as u64, bytes as u64));
        // SAFETY: restores and releases test address range.
        unsafe { assert_ne!(VirtualFree(addr as *mut _, 0, MEM_RELEASE), 0) };
    }
}
