//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_PRIVATE, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQuery,
    VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (Grand Theft
/// Auto VI's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    start_memory_log();
    let raw_protect = protect;
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (Grand Theft Auto VI's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if constrained && !at.is_null() {
        victory_gpu::title_memory_reserved(at as usize, size, raw_protect);
    }
    if kind & MEM_RESERVE_PLACEHOLDER != 0 && !at.is_null() {
        let mut phys = physical();
        drop_overlapping_placeholders(&mut phys.placeholders, at as usize, at as usize + size);
        phys.reservations.push((at as usize, size));
    }
    if !constrained && kind & MEM_COMMIT != 0 && !at.is_null() {
        victory_gpu::title_memory_reserved(at as usize, size, raw_protect);
        victory_gpu::title_memory_committed(at as usize, size);
    }
    if !at.is_null() {
        victory_gpu::title_memory_protection(at as usize, size, raw_protect);
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (Grand Theft Auto VI's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (Grand Theft Auto VI does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        let at = XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
        if !at.is_null() {
            victory_gpu::title_memory_remapped(at as usize, size);
        }
        return at;
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (Grand Theft Auto VI uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            victory_gpu::release_title_allocation(address);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// The top of the console's own title memory: 40 bits (the title's BVH
/// builder refuses a structure above it, decision 0001). A persistent view
/// is placed at or above this address so it cannot collide with title allocations.
const TITLE_MEMORY_CEILING: usize = 1 << 40;

const FREE: u8 = 0;
const COMMITTING: u8 = 1;
const COMMITTED: u8 = 2;
const COMMIT_SPINS: usize = 1 << 14;

/// A section handle, its persistent whole mapping, and per-page commit/write-generation state.
struct Section {
    handle: HANDLE,
    persistent: OnceLock<Option<MappedBase>>,
    size: usize,
    commit_state: Box<[AtomicU8]>,
    generation: Box<[AtomicU32]>,
    fallback_noted: OnceLock<()>,
    holding: Mutex<Vec<(usize, usize)>>,
    /// Base address when a whole reservation is backed by a placed L0 heap (`0` = none).
    placed: AtomicUsize,
}

/// Bytes of the physical-page sections alive (reserved; committed as
/// mapped or written).
static SECTION_BYTES: AtomicUsize = AtomicUsize::new(0);

// Charge each successful section for the lifetime of the process: a mapped
// view can keep its backing alive after the title frees the page numbers.
static PHYSICAL_PAGEFILE_REMAINING: LazyLock<AtomicUsize> = LazyLock::new(|| {
    let bytes = std::env::var("VICTORY_PHYSICAL_PAGEFILE_FIRST_MB")
        .ok()
        .and_then(|mib| mib.trim().parse::<usize>().ok())
        .and_then(|mib| mib.checked_mul(MIB))
        .unwrap_or(0);
    AtomicUsize::new(bytes)
});

fn reserve_pagefile(remaining: &AtomicUsize, size: usize) -> bool {
    remaining
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bytes| {
            bytes.checked_sub(size)
        })
        .is_ok()
}

fn create_section_handle(size: usize, storage: HANDLE, protection: u32) -> std::io::Result<HANDLE> {
    // SAFETY: callers supply either a live writable file handle or
    // `INVALID_HANDLE_VALUE` for pagefile backing.
    let handle = unsafe {
        CreateFileMappingW(
            storage,
            std::ptr::null(),
            protection,
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    Ok(handle)
}

fn physical_section_in(
    size: usize,
    directory: &std::path::Path,
    remaining: &AtomicUsize,
) -> std::io::Result<HANDLE> {
    use std::os::windows::io::AsRawHandle;

    victory_core::ensure_outside_repo(directory).map_err(std::io::Error::other)?;
    if reserve_pagefile(remaining, size) {
        match create_section_handle(size, INVALID_HANDLE_VALUE, PAGE_READWRITE | SEC_COMMIT) {
            Ok(handle) => {
                tracing::info!(bytes = size, "physical pages use budgeted pagefile backing");
                return Ok(handle);
            }
            Err(error) => {
                remaining.fetch_add(size, Ordering::Relaxed);
                tracing::warn!(
                    bytes = size,
                    %error,
                    "pagefile backing failed; using a temporary file"
                );
            }
        }
    }
    let file = tempfile::tempfile_in(directory)?;
    let handle = create_section_handle(size, file.as_raw_handle(), PAGE_READWRITE)?;
    tracing::info!(bytes = size, "physical pages use temporary file backing");
    Ok(handle)
}

/// Creates a section handle whose views share physical pages. On Wine with
/// `VICTORY_PHYSICAL_BACKING_DIR` set outside the repo, sections exceeding
/// `VICTORY_PHYSICAL_PAGEFILE_FIRST_MB` use an anonymous temporary file so Linux
/// can reclaim file-backed pages.
fn physical_section(size: usize) -> std::io::Result<HANDLE> {
    if on_wine()
        && let Some(directory) = std::env::var_os("VICTORY_PHYSICAL_BACKING_DIR")
    {
        return physical_section_in(
            size,
            std::path::Path::new(&directory),
            &PHYSICAL_PAGEFILE_REMAINING,
        );
    }
    // Under Wine a committed section is a sparse file, free until touched, and
    // committing a reserved one page by page costs a wineserver request each
    // that scans every range committed so far: commit it whole there.
    let protection = PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE };
    create_section_handle(size, INVALID_HANDLE_VALUE, protection)
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    fn new(handle: HANDLE, size: usize) -> Section {
        let pages = size.div_ceil(PAGE);
        Section {
            handle,
            persistent: OnceLock::new(),
            size,
            commit_state: (0..pages).map(|_| AtomicU8::new(FREE)).collect(),
            generation: (0..pages).map(|_| AtomicU32::new(0)).collect(),
            fallback_noted: OnceLock::new(),
            holding: Mutex::new(Vec::new()),
            placed: AtomicUsize::new(0),
        }
    }

    fn note_fallback(&self) {
        if self.fallback_noted.set(()).is_ok() {
            tracing::warn!(
                size = self.size,
                "a commit through the persistent view failed; this section's pages keep a temporary view per write"
            );
        }
    }

    fn hold(&self, from: usize, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let mut holding = self.holding.lock().unwrap_or_else(|e| e.into_inner());
        let end = from.saturating_add(len);
        if holding
            .iter()
            .any(|&(at, bytes)| from < at.saturating_add(bytes) && at < end)
        {
            return false;
        }
        holding.push((from, len));
        true
    }

    fn let_go(&self, from: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut holding = self.holding.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(held) = holding.iter().position(|&held| held == (from, len)) {
            holding.swap_remove(held);
        }
    }

    fn persistent_ptr(&self) -> Option<*mut u8> {
        let mapped = self
            .persistent
            .get_or_init(|| map_above_title_memory(self.handle, self.size).map(MappedBase));
        mapped.as_ref().map(|base| base.0)
    }

    fn committed_all(&self, start: usize, count: usize) -> bool {
        let Some(end) = start.checked_add(count) else {
            return false;
        };
        end <= self.commit_state.len()
            && (start..end).all(|page| self.commit_state[page].load(Ordering::Acquire) == COMMITTED)
    }

    fn mark_committed(&self, start: usize, count: usize) {
        for page in start..start.saturating_add(count).min(self.commit_state.len()) {
            self.commit_state[page].store(COMMITTED, Ordering::Release);
        }
    }

    fn mark_free(&self, start: usize, count: usize) {
        for page in start..start.saturating_add(count).min(self.commit_state.len()) {
            let _ = self.commit_state[page].compare_exchange(
                COMMITTING,
                FREE,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    /// Bumps the write generation of pages `[first, first + count)` AFTER bytes are written.
    fn bump_range(&self, first: usize, count: usize) {
        let Some(end) = first.checked_add(count) else {
            return;
        };
        for page in first..end.min(self.generation.len()) {
            self.generation[page].fetch_add(1, Ordering::Release);
        }
    }

    fn generation_of(&self, start: usize, count: usize) -> Option<u64> {
        let end = start.checked_add(count)?;
        if end > self.generation.len() {
            return None;
        }
        let mut digest = 0x9e37_79b9_7f4a_7c15u64;
        for page in start..end {
            digest = mix64(digest ^ (page as u64).wrapping_mul(0x2545_f491_4f6c_dd1d));
            digest = mix64(digest ^ u64::from(self.generation[page].load(Ordering::Acquire)));
        }
        Some(digest)
    }

    fn ensure_committed(&self, base: *mut u8, start_page: usize, count: usize) -> bool {
        let Some(end_page) = start_page.checked_add(count) else {
            return false;
        };
        if !end_page
            .checked_mul(PAGE)
            .is_some_and(|end| end <= self.size)
        {
            return false;
        }
        if self.committed_all(start_page, count) {
            return true;
        }
        for _ in 0..COMMIT_SPINS {
            if self.committed_all(start_page, count) {
                return true;
            }
            let mut claimed = false;
            for page in start_page..end_page {
                if self.commit_state[page]
                    .compare_exchange(FREE, COMMITTING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    claimed = true;
                }
            }
            if !claimed {
                std::hint::spin_loop();
                continue;
            }
            // SAFETY: `[start_page, end_page)` lies inside the section's view.
            let at = unsafe { base.add(start_page * PAGE) };
            if !commit(at as usize, count * PAGE) {
                self.mark_free(start_page, count);
                return false;
            }
            self.mark_committed(start_page, count);
            return true;
        }
        false
    }
}

fn mix64(value: u64) -> u64 {
    let mut x = value;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn above_title_memory(at: usize) -> bool {
    at >= TITLE_MEMORY_CEILING
}

fn highest_ending_address() -> usize {
    let mut info = SYSTEM_INFO::default();
    // SAFETY: `GetSystemInfo` writes only into `info`.
    unsafe { GetSystemInfo(&mut info) };
    (info.lpMaximumApplicationAddress as usize).saturating_sub(PAGE)
}

fn map_above_title_memory(handle: HANDLE, size: usize) -> Option<*mut u8> {
    let rounded = size.max(PAGE).checked_next_multiple_of(PAGE)?;
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: TITLE_MEMORY_CEILING as *mut c_void,
        HighestEndingAddress: highest_ending_address() as *mut c_void,
        Alignment: PAGE,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: `parameter` points at `requirements`, which outlives the call.
    let placeholder = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            rounded,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            &mut parameter,
            1,
        )
    };
    if placeholder.is_null() {
        tracing::warn!(
            size,
            error = %std::io::Error::last_os_error(),
            "reserving a view above the title's own addresses failed; its pages keep a temporary view per write"
        );
        return None;
    }
    // SAFETY: `placeholder` is replaced in place by a view of our section.
    let view = unsafe {
        MapViewOfFile3(
            handle,
            GetCurrentProcess(),
            placeholder,
            0,
            rounded,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    let at = view.Value as usize;
    if at != placeholder as usize {
        // SAFETY: the reservation was not replaced, so it is still ours to free.
        unsafe { VirtualFree(placeholder, 0, MEM_RELEASE) };
        tracing::warn!(
            size,
            error = %std::io::Error::last_os_error(),
            "mapping a physical-page section whole above the title's addresses failed; its pages keep a temporary view per write"
        );
        return None;
    }
    if !above_title_memory(at) {
        // SAFETY: releasing this view also frees its underlying reservation.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, 0) };
        tracing::warn!(
            at = format_args!("{at:#x}"),
            "the host placed a persistent view inside the title's own addresses; its pages keep a temporary view per write"
        );
        return None;
    }
    Some(at as *mut u8)
}

struct MappedBase(*mut u8);

// SAFETY: writers through the persistent mapping hold disjoint byte ranges via `Section::hold`.
unsafe impl Send for MappedBase {}
// SAFETY: as above.
unsafe impl Sync for MappedBase {}

impl Drop for MappedBase {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our persistent view; views keep the section alive independently.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.0.cast(),
            });
        }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        SECTION_BYTES.fetch_sub(self.size, Ordering::Relaxed);
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Freed physical page numbers recycled before advancing `next_page` (`Marvisak #1935`).
    free: Vec<usize>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Reservations that take physical pages: start, length.
    reservations: Vec<(usize, usize)>,
    /// Of them, those shown whole by one view: start, length, the section.
    whole: Vec<(usize, usize, Arc<Section>)>,
    /// Pages with a place in such a view.
    bound: std::collections::HashSet<usize>,
    /// Exact single-placeholder ranges left by [`unmap`] that need no `split_at` before remapping.
    placeholders: std::collections::HashSet<(usize, usize)>,
}

fn drop_overlapping_placeholders(
    placeholders: &mut std::collections::HashSet<(usize, usize)>,
    base: usize,
    end: usize,
) {
    placeholders.retain(|&(p, n)| p.saturating_add(n) <= base || p >= end);
}

static PHYSICAL: LazyLock<RwLock<Physical>> = LazyLock::new(|| {
    RwLock::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> RwLockWriteGuard<'static, Physical> {
    PHYSICAL.write().unwrap_or_else(|e| e.into_inner())
}

fn physical_read() -> RwLockReadGuard<'static, Physical> {
    PHYSICAL.read().unwrap_or_else(|e| e.into_inner())
}

fn section_cap_pages() -> Option<usize> {
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| {
        let mb = std::env::var("VICTORY_SECTION_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)?;
        Some((mb.saturating_mul(MIB) / PAGE).max(1))
    })
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    allocate_physical_pages_with_cap(flags, count, pages, section_cap_pages())
}

fn allocate_physical_pages_with_cap(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
    cap_pages: Option<usize>,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n == 0 || n.checked_mul(PAGE).is_none() {
        return 0;
    }
    let max_chunk = cap_pages.map_or(n, |c| c.max(1));
    let mut sections: Vec<(Arc<Section>, usize)> = Vec::new();
    let mut remaining = n;
    while remaining > 0 {
        let chunk_pages = remaining.min(max_chunk);
        let size = chunk_pages * PAGE;
        // A pagefile-backed section of `size` bytes, its pages committed when
        // first mapped or written (`commit`): Grand Theft Auto VI asks for 6.1
        // GiB at once, which a busy host often cannot commit (error 1450).
        let handle = match physical_section(size) {
            Ok(handle) => handle,
            Err(error) => {
                tracing::warn!(n, chunk_pages, %error, "XMemAllocatePhysicalPages failed");
                // SAFETY: checked non-null.
                unsafe { count.write(0) };
                return 0;
            }
        };
        SECTION_BYTES.fetch_add(size, Ordering::Relaxed);
        sections.push((Arc::new(Section::new(handle, size)), chunk_pages));
        remaining -= chunk_pages;
    }
    tracing::debug!(
        n,
        sections = sections.len(),
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    let mut physical = physical();
    let mut written = 0usize;
    for (section, chunk_pages) in sections {
        for index in 0..chunk_pages {
            let slice = Slice {
                section: section.clone(),
                index,
            };
            let number = take_page_number(&mut physical);
            physical.pages.insert(number, slice);
            // SAFETY: the title passes room for `n` page numbers.
            unsafe { pages.add(written).write(number) };
            written += 1;
        }
    }
    1
}

fn take_page_number(physical: &mut Physical) -> usize {
    if let Some(recycled) = physical.free.pop() {
        recycled
    } else {
        let next = physical.next_page;
        physical.next_page += FRAMES;
        next
    }
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: Grand Theft Auto VI keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        let number = unsafe { *pages.add(i) };
        if physical.pages.remove(&number).is_some() {
            physical.free.push(number);
        }
    }
    1
}

pub(crate) fn parse_persistent_views(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let Some(text) = value.to_str() else {
        return false;
    };
    let trimmed = text.trim();
    trimmed == "1"
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("on")
}

/// Whether pages are written through one persistent view per section
/// (`VICTORY_XMEM_PERSISTENT_VIEW=1`) instead of mapping/unmapping per write.
pub(crate) fn persistent_views() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_persistent_views(std::env::var_os("VICTORY_XMEM_PERSISTENT_VIEW").as_deref())
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mapping {
    Default,
    Persistent,
    Temporary,
}

impl Mapping {
    fn view(self, slice: &Slice, n: usize) -> Option<View> {
        if slice.section.placed.load(Ordering::Acquire) != 0 {
            return View::placed(slice, n);
        }
        let mapping = match self {
            Mapping::Default if persistent_views() => Mapping::Persistent,
            Mapping::Default => Mapping::Temporary,
            forced => forced,
        };
        match mapping {
            Mapping::Persistent => View::persistent(slice, n).or_else(|| View::temporary(slice, n)),
            Mapping::Temporary | Mapping::Default => View::temporary(slice, n),
        }
    }
}

/// A view of our own on consecutive slices of one section, committed and writable.
enum View {
    Temporary {
        at: MEMORY_MAPPED_VIEW_ADDRESS,
        len: usize,
    },
    Persistent {
        section: Arc<Section>,
        ptr: *mut u8,
        from: usize,
        len: usize,
    },
}

impl View {
    fn placed(slice: &Slice, n: usize) -> Option<View> {
        let base = slice.section.placed.load(Ordering::Acquire);
        if base == 0 {
            return None;
        }
        let len = n.checked_mul(PAGE)?;
        let from = slice.index * PAGE;
        if !from.checked_add(len).is_some_and(|end| end <= slice.section.size) {
            return None;
        }
        while !slice.section.hold(from, len) {
            std::thread::yield_now();
        }
        slice.section.mark_committed(slice.index, n);
        // SAFETY: `base..base + size` is committed RW L0 memory and `hold` serializes `[from, from + len)`.
        let ptr = unsafe { (base as *mut u8).add(from) };
        Some(View::Persistent {
            section: slice.section.clone(),
            ptr,
            from,
            len,
        })
    }

    fn temporary(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View::Temporary { at, len };
        commit(at.Value as usize, len).then_some(view)
    }

    fn persistent(slice: &Slice, n: usize) -> Option<View> {
        let base = slice.section.persistent_ptr()?;
        let len = n.checked_mul(PAGE)?;
        let from = slice.index * PAGE;
        if !slice.section.hold(from, len) {
            return None;
        }
        if !slice.section.ensure_committed(base, slice.index, n) {
            slice.section.let_go(from, len);
            slice.section.note_fallback();
            return None;
        }
        // SAFETY: `ensure_committed` verified `[slice.index, slice.index + n)` lies inside the section.
        let ptr = unsafe { base.add(from) };
        Some(View::Persistent {
            section: slice.section.clone(),
            ptr,
            from,
            len,
        })
    }

    fn bytes_at(&mut self, from: usize, len: usize) -> &mut [u8] {
        let (ptr, whole) = match self {
            View::Temporary { at, len } => (at.Value.cast::<u8>(), *len),
            View::Persistent { ptr, len, .. } => (*ptr, *len),
        };
        assert!(
            from.checked_add(len).is_some_and(|end| end <= whole),
            "view bytes {from}+{len} past a view of {whole}"
        );
        // SAFETY: `[from, from + len)` fits within this view; persistent slices hold disjoint byte ranges.
        unsafe { std::slice::from_raw_parts_mut(ptr.add(from), len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        match self {
            View::Temporary { at, .. } => {
                use windows_sys::Win32::System::Memory::UnmapViewOfFile;
                // SAFETY: our temporary view.
                unsafe { UnmapViewOfFile(*at) };
            }
            View::Persistent { section, from, len, .. } => section.let_go(*from, *len),
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical_read();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Writes title memory `[at, at + len)` where one title view of physical
/// pages shows all of it: `write` gets the address of those bytes in our
/// persistent view of the section (`Section::persistent_ptr`), which no guard
/// covers. Takes a non-blocking read lock (`PHYSICAL.try_read`) and releases it
/// before committing/copying so parallel copy-back workers do not serialize on
/// `physical` and immediately fall back when `XMemMapPhysicalPages` holds the
/// write lock; `View::persistent` keeps the `Arc<Section>` alive and guards
/// against concurrent persistent writes.
/// False, with `write` not called, where no physical-page view shows `[at, at + len)`.
pub fn write_beside(at: usize, len: usize, write: &mut dyn FnMut(*mut u8)) -> bool {
    if len == 0 {
        return true;
    }
    let Some(end) = at.checked_add(len) else {
        return false;
    };
    let (view_slice, offset) = {
        let physical = match PHYSICAL.try_read() {
            Ok(guard) => guard,
            Err(TryLockError::Poisoned(e)) => e.into_inner(),
            Err(TryLockError::WouldBlock) => return false,
        };
        let Some((&base, &(size, ref slice))) = physical.views.range(..=at).next_back() else {
            return false;
        };
        if end > base + size {
            return false;
        }
        let offset = slice.index * PAGE + (at - base);
        (
            Slice {
                section: slice.section.clone(),
                index: offset / PAGE,
            },
            offset,
        )
    };
    let page_count = (offset % PAGE + len).div_ceil(PAGE);
    let Some(mut view) = View::persistent(&view_slice, page_count) else {
        return false;
    };
    let ptr = view.bytes_at(offset % PAGE, len).as_mut_ptr();
    write(ptr);
    view_slice.section.bump_range(view_slice.index, page_count);
    true
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    write_physical_with(Mapping::Default, pages, offset, data)
}

fn shown_runs(runs: &[(Slice, usize)]) -> Vec<(usize, usize)> {
    let physical = physical_read();
    let mut shown = Vec::new();
    for (slice, n) in runs {
        let (from, to) = (slice.index, slice.index + n);
        for (&base, (size, view)) in &physical.views {
            let (first, last) = (view.index, view.index + size / PAGE);
            if first < to && from < last && Arc::ptr_eq(&view.section, &slice.section) {
                let (a, b) = (from.max(first), to.min(last));
                shown.push((base + (a - first) * PAGE, (b - a) * PAGE));
            }
        }
    }
    shown
}

fn write_physical_with(mapping: Mapping, pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    // Open guarded blocks before writing through DirectStorage mappings so
    // the watch saves original bytes and holds the blocks open during writes.
    for (at, len) in shown_runs(&runs) {
        victory_gpu::kernel_will_write(at, len);
    }
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = mapping.view(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes_at(within, k).copy_from_slice(&data[done..done + k]);
        slice
            .section
            .bump_range(slice.index + within / PAGE, (within + k).div_ceil(PAGE));
        (within, done) = (0, done + k);
    }
    true
}

/// Registers [`va_generation_digest`] as the GPU crate's page-generation oracle.
pub fn register_page_gen_oracle() {
    victory_gpu::set_page_gen_oracle(va_generation_digest);
}

/// Folded write generation for `[addr, addr + len)` across mapped physical-page
/// views, or `None` when any byte lies outside a known `xmem` view.
pub fn va_generation_digest(addr: usize, len: usize) -> Option<u64> {
    if len == 0 {
        return None;
    }
    let end = addr.checked_add(len)?;
    let phys = physical_read();
    let mut cursor = addr;
    let mut digest = 0x517c_c1b7_2722_0a95u64;
    while cursor < end {
        let (&base, (vlen, slice)) = phys.views.range(..=cursor).next_back()?;
        let view_end = base.checked_add(*vlen)?;
        if view_end <= cursor {
            return None;
        }
        let within = cursor - base;
        let take = (view_end - cursor).min(end - cursor);
        let first_page = slice.index + within / PAGE;
        let page_count = ((within % PAGE) + take).div_ceil(PAGE);
        let folded = slice.section.generation_of(first_page, page_count)?;
        digest = mix64(digest ^ (Arc::as_ptr(&slice.section) as usize as u64));
        digest = mix64(digest ^ (first_page as u64));
        digest = mix64(digest ^ folded);
        cursor += take;
    }
    Some(digest)
}

/// DirectStorage filled bytes `[offset, offset + len)` of `pages`: tells the
/// GPU side the addresses they are shown at (it keeps copies of textures).
pub(crate) fn physical_written(pages: &[usize], offset: usize, len: usize) {
    let Some(runs) = runs(pages, offset, len) else {
        return;
    };
    for (at, len) in shown_runs(&runs) {
        victory_gpu::title_memory_written(at, len);
    }
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
    section: Arc<Section>,
    first_page: usize,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        self.section
            .bump_range(self.first_page, (self.within + self.len).div_ceil(PAGE));
    }
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        self.view.bytes_at(within, len)
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    page_window_with(Mapping::Default, pages, offset, len)
}

fn page_window_with(
    mapping: Mapping,
    pages: &[usize],
    offset: usize,
    len: usize,
) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    let view = mapping.view(slice, *n)?;
    for (at, shown_len) in shown_runs(&runs) {
        victory_gpu::kernel_will_write(at, shown_len);
    }
    Some(PageWindow {
        view,
        within: offset % PAGE,
        len,
        section: slice.section.clone(),
        first_page: slice.index + (offset % PAGE) / PAGE,
    })
}

/// Count of `pages` not known to this allocator.
pub(crate) fn unknown_pages(pages: &[usize]) -> usize {
    let physical = physical_read();
    pages.iter().filter(|p| !physical.pages.contains_key(p)).count()
}

/// Count of 64 KiB pages skipped during `XMemMapPhysicalPages` / `map_alias`
/// because the target address already maps the identical physical slice.
static SAME_PAGE_SKIPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn same_slice_at(physical: &Physical, addr: usize, slice: &Slice) -> bool {
    physical
        .views
        .range(..=addr)
        .next_back()
        .is_some_and(|(&vbase, (vlen, vslice))| {
            addr < vbase + *vlen
                && Arc::ptr_eq(&vslice.section, &slice.section)
                && vslice.index + (addr - vbase) / PAGE == slice.index
        })
}

/// Applies `slices` across `[base, base + slices.len() * PAGE)`, skipping
/// pages that already map the identical section slice while unmapping and
/// remapping only contiguous runs that actually changed.
fn apply_changed_slices(physical: &mut Physical, base: usize, slices: &[Slice]) -> Option<usize> {
    let n = slices.len();
    let mut skipped = 0u64;
    let mut i = 0;
    while i < n {
        if same_slice_at(physical, base + i * PAGE, &slices[i]) {
            skipped += 1;
            i += 1;
            continue;
        }
        let changed_start = i;
        i += 1;
        while i < n && !same_slice_at(physical, base + i * PAGE, &slices[i]) {
            i += 1;
        }
        unmap(physical, base + changed_start * PAGE, (i - changed_start) * PAGE);
        let mut j = changed_start;
        while j < i {
            let first = slices[j].clone();
            let run = 1 + slices[j + 1..i]
                .iter()
                .zip(1usize..)
                .take_while(|(s, k)| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + *k
                })
                .count();
            if !map(physical, base + j * PAGE, run * PAGE, first) {
                if skipped > 0 {
                    SAME_PAGE_SKIPS.fetch_add(skipped, Ordering::Relaxed);
                }
                return Some(j);
            }
            j += run;
        }
    }
    if skipped > 0 {
        SAME_PAGE_SKIPS.fetch_add(skipped, Ordering::Relaxed);
    }
    None
}

/// Maps `n` pages at `address`, replacing any existing mapping in that range;
/// null `pages` unmaps. Returns `address`, or null if a page is unknown or the
/// host fails, leaving the range unmapped after rolling back any runs mapped
/// during this call.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    if n == 0 {
        return address;
    }
    if !pages.is_null() && victory_gpu::port_placed::on() {
        victory_gpu::port_placed::warm();
    }
    let base = address as usize;
    let mut physical = physical();
    if pages.is_null() {
        unmap(&mut physical, base, n * PAGE);
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    if map_whole(&mut physical, base, pages) {
        return address;
    }
    let mut slices = Vec::with_capacity(n);
    for &page in pages {
        let Some(slice) = physical.pages.get(&page).cloned() else {
            tracing::warn!(page, "XMemMapPhysicalPages: unknown page");
            if !slices.is_empty() {
                unmap(&mut physical, base, slices.len() * PAGE);
            }
            return std::ptr::null_mut();
        };
        slices.push(slice);
    }
    if let Some(failed_at) = apply_changed_slices(&mut physical, base, &slices) {
        if failed_at > 0 {
            unmap(&mut physical, base, failed_at * PAGE);
        }
        return std::ptr::null_mut();
    }
    address
}

/// Reservations in this size range are eligible to be mapped whole.
const MIN_WHOLE_RESERVATION: usize = 4 * MIB;
const MAX_WHOLE_RESERVATION: usize = 1 << 30;

/// The host GPU uses a run of title memory in place only where one view
/// shows all of it, and Grand Theft Auto VI maps its 185 MB GPU heap from
/// one allocation of pages as long as the reservation, in three calls. A
/// reservation mapped from such pages is shown whole by one view of them;
/// where the title puts a page is where in the section it goes (pages hold
/// nothing before they are first mapped). True if done.
fn map_whole(physical: &mut Physical, base: usize, pages: &[usize]) -> bool {
    let end = base + pages.len() * PAGE;
    let Some(first) = physical.pages.get(&pages[0]).cloned() else {
        return false;
    };
    if pages.len() > 64 && physical.whole.is_empty() {
        tracing::debug!(base = format_args!("{base:#x}"), n = pages.len(), reservations = ?physical.reservations.iter().map(|r| format!("{:x}+{:x}", r.0, r.1)).collect::<Vec<_>>(), "whole: asked");
    }
    let known = physical
        .whole
        .iter()
        .find(|(at, len, _)| *at <= base && end <= at + len)
        .cloned();
    let (at, _, section) = match known {
        Some(w) => w,
        None => {
            let Some(&(at, len)) = physical
                .reservations
                .iter()
                .find(|(at, len)| *at <= base && end <= at + len)
            else {
                return false;
            };
            if !(MIN_WHOLE_RESERVATION..=MAX_WHOLE_RESERVATION).contains(&len) {
                return false;
            }
            tracing::debug!(at = format_args!("{at:#x}"), len, section = first.section.size, views = ?physical.views.range(..at + len).next_back().map(|(v, (n, _))| (*v, *n)), "whole?");
            if len != first.section.size
                || physical.views.range(..at + len).next_back().is_some_and(|(v, (n, _))| v + n > at)
                || physical.whole.iter().any(|w| Arc::ptr_eq(&w.2, &first.section))
            {
                return false;
            }
            let all = Slice {
                section: first.section.clone(),
                index: 0,
            };
            // The title splits it in pieces before it maps them.
            const COALESCE: u32 = 1;
            drop_overlapping_placeholders(&mut physical.placeholders, at, at + len);
            // SAFETY: placeholders of one reservation, none mapped.
            unsafe { VirtualFree(at as *mut c_void, len, MEM_RELEASE | COALESCE) };
            let placed_ok = victory_gpu::port_placed::on()
                && match map_whole_placed(at, len, &first.section) {
                    Ok(()) => true,
                    Err(why) => {
                        victory_gpu::port_placed::fallback(&why);
                        false
                    }
                };
            if placed_ok {
                physical.whole.push((at, len, first.section.clone()));
            } else {
                if !map_view(at, len, &all) {
                    return false;
                }
                tracing::info!(
                    at = format_args!("{at:#x}"),
                    len,
                    "a reservation of physical pages shown whole"
                );
                victory_gpu::title_memory_remapped(at, len);
                physical.views.insert(at, (len, all));
                physical.whole.push((at, len, first.section.clone()));
            }
            (at, len, first.section.clone())
        }
    };
    let mut owners: HashMap<usize, usize> = physical
        .pages
        .iter()
        .filter(|(_, s)| Arc::ptr_eq(&s.section, &section))
        .map(|(&p, s)| (s.index, p))
        .collect();
    for (k, page) in pages.iter().enumerate() {
        let wanted = (base - at) / PAGE + k;
        let Some(slice) = physical.pages.get(page).cloned() else {
            return false;
        };
        let same = Arc::ptr_eq(&slice.section, &section);
        let owner = owners.get(&wanted).copied();
        let taken = owner.is_some_and(|o| o != *page && physical.bound.contains(&o));
        if !same || slice.index != wanted && (taken || physical.bound.contains(page)) {
            // shortcut: shown as it is; the title sees other bytes than it
            // asked for.
            tracing::warn!(page, wanted, same, "a page mapped again in a reservation shown whole");
            continue;
        }
        if slice.index != wanted {
            if let Some(owner) = owner {
                owners.insert(slice.index, owner);
                if let Some(s) = physical.pages.get_mut(&owner) {
                    s.index = slice.index;
                }
            }
            owners.insert(wanted, *page);
            if let Some(s) = physical.pages.get_mut(page) {
                s.index = wanted;
            }
        }
        physical.bound.insert(*page);
    }
    true
}

fn give_back_placed_placeholders(allocations: &[(usize, usize)]) {
    for &(at, size) in allocations {
        // SAFETY: `at..at + size` was committed in place of a placeholder in `map_whole_placed`.
        unsafe { VirtualFree(at as *mut c_void, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
    }
}

fn map_whole_placed(at: usize, len: usize, section: &Arc<Section>) -> Result<(), String> {
    let end = at.checked_add(len).ok_or("overflow")?;
    let mut pieces = Vec::new();
    let mut here = at;
    while here < end {
        // SAFETY: plain output buffer for `VirtualQuery`.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: queries a live address in our process.
        let asked = unsafe { VirtualQuery(here as *const c_void, &mut info, size_of_val(&info)) };
        if asked == 0
            || info.State != MEM_RESERVE
            || info.Type != MEM_PRIVATE
            || info.AllocationBase as usize != at
            || info.RegionSize == 0
        {
            return Err(format!(
                "{here:#x} is not a private placeholder (state={:#x}, type={:#x})",
                info.State, info.Type
            ));
        }
        let stop = (info.BaseAddress as usize + info.RegionSize).min(end);
        pieces.push((here, stop - here));
        here = stop;
    }
    let prepared = victory_gpu::port_placed::prepare(len)?;
    let mut committed = Vec::with_capacity(pieces.len());
    for &(p_at, p_size) in &pieces {
        // SAFETY: `[p_at, p_at + p_size)` is a reserved placeholder owned by this reservation.
        let got = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                p_at as *const c_void,
                p_size,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        if got as usize != p_at {
            let err = std::io::Error::last_os_error();
            give_back_placed_placeholders(&committed);
            return Err(format!("placeholder {p_at:#x}+{p_size:#x} commit failed: {err}"));
        }
        committed.push((p_at, p_size));
    }
    if let Err(why) = victory_gpu::port_placed::map(&prepared, at) {
        give_back_placed_placeholders(&committed);
        return Err(why);
    }
    let pages = section.size / PAGE;
    let mut copied_pages = 0usize;
    let mut p = 0usize;
    while p < pages {
        if section.commit_state[p].load(Ordering::Acquire) != COMMITTED {
            p += 1;
            continue;
        }
        let run = (p..pages)
            .take_while(|&q| section.commit_state[q].load(Ordering::Acquire) == COMMITTED)
            .count();
        let slice = Slice {
            section: section.clone(),
            index: p,
        };
        let Some(mut src) = View::temporary(&slice, run) else {
            victory_gpu::port_placed::unmap(&prepared);
            give_back_placed_placeholders(&committed);
            return Err(format!("failed copying committed pages {p}..{}", p + run));
        };
        let bytes = run * PAGE;
        // SAFETY: `at + p * PAGE..at + (p + run) * PAGE` is committed RW placed memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.bytes_at(0, bytes).as_ptr(),
                (at + p * PAGE) as *mut u8,
                bytes,
            );
        }
        copied_pages += run;
        p += run;
    }
    section.placed.store(at, Ordering::Release);
    victory_gpu::title_memory_remapped(at, len);
    victory_gpu::port_placed::register(prepared, at);
    tracing::info!(
        at = format_args!("{at:#x}"),
        len,
        placeholders = pieces.len(),
        copied_pages,
        "a whole reservation placed in L0 VRAM (no section view, no userptr)"
    );
    Ok(())
}

/// Whether `[base, base + len)` lies in a reservation shown whole.
fn in_whole(physical: &Physical, base: usize, len: usize) -> bool {
    physical
        .whole
        .iter()
        .any(|(at, n, _)| *at <= base && base + len <= at + n)
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (Grand Theft Auto VI passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): Grand Theft Auto
/// VI maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    if len == 0 {
        return true;
    }
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    if in_whole(&physical, to, len) {
        tracing::warn!(to = format_args!("{to:#x}"), len, "page mappings into a reservation shown whole");
        return false;
    }
    if from == 0 {
        unmap(&mut physical, to, len);
        return true;
    }
    // Resolve and retain every source slice before tearing down overlapping
    // destination mappings or aborting on a source hole (`Marvisak #2199`).
    let mut slices = Vec::with_capacity(len.div_ceil(PAGE));
    for k in (0..len).step_by(PAGE) {
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        slices.push(slice);
    }
    apply_changed_slices(&mut physical, to, &slices).is_none()
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must be placeholders. A view replaces exactly one placeholder, and a
/// reservation is often several by now (each mapping splits it at its
/// ends, unmapping does not join the pieces, titles split their own), so
/// the range is split at its two ends and each placeholder in it gets a
/// view of its part. Placeholders are never joined: that would also join
/// separate reservations, or pieces the title split, for good. All or
/// nothing: if a view fails, those made before it are placeholders again.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    if physical.placeholders.remove(&(base, len)) {
        if !map_view(base, len, &slice) {
            return false;
        }
        victory_gpu::title_memory_remapped(base, len);
        physical.views.insert(base, (len, slice));
        return true;
    }
    let end = base + len;
    drop_overlapping_placeholders(&mut physical.placeholders, base, end);
    if len >= 16 * MIB {
        tracing::debug!(
            base = format_args!("{base:#x}"),
            len,
            index = slice.index,
            section = slice.section.size,
            "physical pages mapped"
        );
    }
    split_at(base);
    split_at(end);
    let mut made: Vec<(usize, usize, Slice)> = Vec::new();
    let mut at = base;
    while at < end {
        // What is not a placeholder fails in `map_view`.
        let next = placeholder(at).map_or(end, |(_, stop)| stop.min(end));
        let offset = at - base;
        let part = Slice {
            index: slice.index + offset / PAGE,
            section: slice.section.clone(),
        };
        // A view must start on a page of the run to show its slice.
        if !offset.is_multiple_of(PAGE) || !map_view(at, next - at, &part) {
            for &(view, len, _) in &made {
                victory_gpu::title_memory_view_gone(view, len);
                let view = MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut c_void,
                };
                // SAFETY: a view made above, back to its placeholder.
                unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
            }
            return false;
        }
        made.push((at, next - at, part));
        at = next;
    }
    for (at, len, part) in made {
        // New contents, read-write: the GPU side's copies no longer guard it.
        victory_gpu::title_memory_remapped(at, len);
        physical.views.insert(at, (len, part));
    }
    true
}

/// Replaces the placeholder `[base, base + len)` with a view of `slice` and
/// the slices after it, committed.
fn map_view(base: usize, len: usize, slice: &Slice) -> bool {
    // SAFETY: replaces the placeholder `[base, base + len)` with a view of
    // the section; the host refuses anything that is not that placeholder.
    let view = unsafe {
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    let pages = len.div_ceil(PAGE);
    if !slice.section.committed_all(slice.index, pages) {
        if !commit(base, len) {
            // SAFETY: the view just mapped, back to a placeholder.
            unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
            return false;
        }
        slice.section.mark_committed(slice.index, pages);
    }
    // The GPU side guards every view of a page together.
    victory_gpu::title_memory_view(
        base,
        len,
        slice.section.handle as usize,
        (slice.index * PAGE) as u64,
    );
    true
}

/// The start and end of the placeholder holding `at`, if it may be one
/// (other reserved memory looks the same, but the host refuses to split it
/// or map over it).
fn placeholder(at: usize) -> Option<(usize, usize)> {
    // SAFETY: plain data; zero is a valid value.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: a query of our own address space into `info`.
    let found = unsafe { VirtualQuery(at as *const c_void, &mut info, size) } == size;
    (found && info.State == MEM_RESERVE && info.Type == MEM_PRIVATE).then(|| {
        (
            info.AllocationBase as usize,
            info.BaseAddress as usize + info.RegionSize,
        )
    })
}

/// Splits the placeholder holding `at` in two there, if it starts below.
/// The same call frees reserved memory that replaced a placeholder back to
/// one, committed pages and all, so everything below `at` in it must be
/// reserved: nothing committed is ever freed here.
fn split_at(at: usize) {
    if let Some((start, _)) = placeholder(at)
        && start < at
        && placeholder(start).is_some_and(|(_, stop)| stop >= at)
    {
        // SAFETY: `[start, at)` is reserved and holds no data; the host
        // fails the call on anything but a placeholder or memory that
        // replaced one, and then changes nothing.
        unsafe {
            VirtualFree(
                start as *mut c_void,
                at - start,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            )
        };
    }
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 || in_whole(physical, base, len) {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        victory_gpu::title_memory_remapped(at, size);
        victory_gpu::title_memory_view_gone(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        let head_ok = at >= base || map(physical, at, base - at, slice.clone());
        let tail_ok = at + size <= end || {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice })
        };
        if head_ok && tail_ok {
            let gap_start = at.max(base);
            let gap_end = (at + size).min(end);
            if gap_end > gap_start {
                physical.placeholders.insert((gap_start, gap_end - gap_start));
            }
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_pagefile_sections_share_a_fixed_budget() {
        let remaining = AtomicUsize::new(4 * PAGE);
        assert!(!reserve_pagefile(&remaining, 5 * PAGE));
        assert!(reserve_pagefile(&remaining, 3 * PAGE));
        assert!(!reserve_pagefile(&remaining, 2 * PAGE));
        assert!(reserve_pagefile(&remaining, PAGE));
        assert!(!reserve_pagefile(&remaining, PAGE));
    }

    #[test]
    fn physical_section_in_backing_dir_uses_temp_file_and_rejects_repo_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let handle = physical_section_in(PAGE, tmp.path(), &AtomicUsize::new(0)).unwrap();
        assert!(!handle.is_null());
        // SAFETY: valid section handle returned by `physical_section_in`.
        unsafe { CloseHandle(handle) };

        assert!(
            physical_section_in(
                PAGE,
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
                &AtomicUsize::new(0),
            )
            .is_err()
        );
    }

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        Mapping::Temporary
            .view(&slice, 1)
            .expect("mapped")
            .bytes_at(0, PAGE)
            .to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// `n` physical pages (consecutive slices of one section) and a
    /// reservation of `n` 64 KiB pages for them.
    fn pages_and_reservation(n: usize) -> (Vec<usize>, *mut c_void) {
        let mut pages = vec![0usize; n];
        let mut count = n;
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut count, pages.as_mut_ptr()),
            1
        );
        let reservation = XMemVirtualAlloc(
            std::ptr::null_mut(),
            n * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!reservation.is_null());
        (pages, reservation)
    }

    /// Unmapping part of a run leaves the range as several host
    /// placeholders; mapping a run over all of them again works, and the
    /// pages keep what was written through the first mapping.
    #[test]
    fn a_run_maps_again_after_a_partial_unmap() {
        let (pages, a) = pages_and_reservation(3);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe {
            for page in 0..3 {
                at(page).cast::<u8>().write(10 + page as u8);
            }
        }
        assert_eq!(XMemMapPhysicalPages(at(1), 1, std::ptr::null()), at(1));
        assert_eq!(query(at(1)).host.State, MEM_RESERVE);

        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        for page in 0..3 {
            assert_eq!(query(at(page)).host.State, MEM_COMMIT, "page {page}");
            // SAFETY: mapped read-write.
            assert_eq!(unsafe { at(page).cast::<u8>().read() }, 10 + page as u8);
        }
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(query(a).host.State, MEM_RESERVE);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// A run that starts inside one host placeholder and ends inside the
    /// next (left by an earlier, shorter run) maps, and the pages around it
    /// stay unmapped.
    #[test]
    fn a_run_maps_across_two_placeholders() {
        let (pages, a) = pages_and_reservation(4);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // Mapping and unmapping pages 0-1 splits the reservation in two.
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);

        assert_eq!(XMemMapPhysicalPages(at(1), 2, pages[1..].as_ptr()), at(1));
        assert_eq!(query(a).host.State, MEM_RESERVE);
        assert_eq!(query(at(1)).host.State, MEM_COMMIT);
        assert_eq!(query(at(2)).host.State, MEM_COMMIT);
        assert_eq!(query(at(3)).host.State, MEM_RESERVE);
        // Page 3, split off by the run, still takes a mapping: the run's
        // last page shows there too.
        // SAFETY: mapped read-write.
        unsafe {
            at(2).cast::<u8>().write(42);
            assert_eq!(XMemMapPhysicalPages(at(3), 1, pages[2..].as_ptr()), at(3));
            assert_eq!(at(3).cast::<u8>().read(), 42);
        }
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// Placeholders a run spans are never joined: a piece the title split
    /// off itself is still its own after the run is unmapped, so the title
    /// can release it.
    #[test]
    fn a_run_keeps_the_placeholders_it_spans() {
        let (pages, a) = pages_and_reservation(4);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // SAFETY: splits the reservation's placeholder, as a title does.
        let split = unsafe { VirtualFree(a, 2 * PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        assert_ne!(split, 0);
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        let info = query(at(2)).host;
        assert_eq!((info.AllocationBase, info.RegionSize), (at(2), 2 * PAGE));
        // SAFETY: the title's own piece, unmapped.
        assert_ne!(unsafe { VirtualFree(at(2), 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// A run whose second part is no placeholder (memory committed in place
    /// of one) fails: its first part is unmapped again, and the committed
    /// memory, which the run ends inside, keeps its data.
    #[test]
    fn a_failed_run_leaves_no_view() {
        let (pages, a) = pages_and_reservation(3);
        let at = |page: usize| a.wrapping_byte_add(page * PAGE);
        // SAFETY: splits the placeholder, makes pages 1-2 an ordinary
        // allocation in place and commits page 1 of it.
        unsafe {
            assert_ne!(
                VirtualFree(a, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                0
            );
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                at(1),
                2 * PAGE,
                MEM_RESERVE | MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(p, at(1));
            assert!(!VirtualAlloc(at(1), PAGE, MEM_COMMIT, PAGE_READWRITE).is_null());
            at(1).cast::<u8>().write(99);
        }
        assert!(XMemMapPhysicalPages(a, 2, pages.as_ptr()).is_null());
        assert_eq!(query(a).host.State, MEM_RESERVE);
        assert_eq!(query(at(1)).host.State, MEM_COMMIT);
        // SAFETY: still committed.
        assert_eq!(unsafe { at(1).cast::<u8>().read() }, 99);
        // Page 0 is a placeholder again: it takes a mapping.
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn failed_second_run_rolls_back_first_run() {
        let pages = [allocate(1)[0], allocate(1)[0]];
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 46) as *mut c_void,
            ..Default::default()
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: a fresh reservation; the address requirements stay live for the call.
        let addr = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null_mut(),
                2 * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                &mut parameter,
                1,
            )
        };
        assert!(!addr.is_null(), "{}", std::io::Error::last_os_error());
        let second = addr.wrapping_byte_add(PAGE);
        // SAFETY: split our reservation so the second page can be released.
        let split = unsafe { VirtualFree(addr, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) };
        assert_ne!(split, 0);
        // SAFETY: releasing the second placeholder makes the second run fail.
        assert_ne!(unsafe { VirtualFree(second, 0, MEM_RELEASE) }, 0);

        assert!(XMemMapPhysicalPages(addr, 2, pages.as_ptr()).is_null());
        assert!(!physical().views.contains_key(&(addr as usize)));
        assert_eq!(query(addr).host.State, MEM_RESERVE);

        // SAFETY: releasing our remaining first placeholder.
        assert_ne!(unsafe { VirtualFree(addr, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    fn slice_of(page: usize) -> Slice {
        physical().pages.get(&page).cloned().expect("known page")
    }

    #[test]
    fn persistent_view_maps_above_the_title_memory_ceiling() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        let base = section
            .persistent_ptr()
            .expect("the host refused a view above 40 bits");
        assert!(
            above_title_memory(base as usize),
            "the persistent view is at {base:p}, inside the title's own addresses"
        );
        assert!(!above_title_memory(TITLE_MEMORY_CEILING - PAGE));
        assert!(above_title_memory(TITLE_MEMORY_CEILING));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn a_view_slice_covers_only_the_bytes_asked_for() {
        let pages = allocate(2);
        let slice = slice_of(pages[0]);
        for (name, mapping) in [
            ("temporary", Mapping::Temporary),
            ("persistent", Mapping::Persistent),
        ] {
            let mut view = mapping.view(&slice, 1).expect("one page mapped");
            assert_eq!(view.bytes_at(5, 3).len(), 3, "{name}: not the whole run");
            view.bytes_at(5, 3).copy_from_slice(&[1, 2, 3]);
            drop(view);
            let page = page_bytes(pages[0]);
            assert_eq!(&page[5..8], &[1, 2, 3], "{name}");
            assert_eq!((page[4], page[8]), (0, 0), "{name}: only those bytes");
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn a_second_overlapping_persistent_view_falls_back_to_a_temporary_one() {
        let pages = allocate(3);
        let first = View::persistent(&slice_of(pages[0]), 1).expect("one page");
        assert!(matches!(first, View::Persistent { .. }));
        assert!(View::persistent(&slice_of(pages[0]), 1).is_none());
        let second = Mapping::Persistent
            .view(&slice_of(pages[0]), 1)
            .expect("a temporary view");
        assert!(matches!(second, View::Temporary { .. }));
        let neighbour = View::persistent(&slice_of(pages[1]), 1).expect("next page");
        assert!(matches!(neighbour, View::Persistent { .. }));
        drop(first);
        let again = View::persistent(&slice_of(pages[0]), 1).expect("free again");
        assert!(matches!(again, View::Persistent { .. }));
        drop((second, neighbour, again));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn va_generation_digest_tracks_mapped_views_and_rejects_gaps() {
        let pages = allocate(3);
        let mapped_pages = &pages[..2];
        let va = XMemVirtualAlloc(std::ptr::null_mut(), 2 * PAGE, MEM_RESERVE | PAGES_64K, 0, 0x04);
        assert!(!va.is_null());
        assert_eq!(XMemMapPhysicalPages(va, 2, mapped_pages.as_ptr()), va);
        let addr = va as usize;

        let d0 = va_generation_digest(addr, 2 * PAGE).expect("mapped view has digest");
        assert_eq!(va_generation_digest(addr, 2 * PAGE), Some(d0));
        assert!(write_physical(mapped_pages, 0, &[0xAB; 64]));
        let d1 = va_generation_digest(addr, 2 * PAGE).expect("still mapped");
        assert_ne!(d1, d0);
        assert!(va_generation_digest(addr, 3 * PAGE).is_none());
        assert!(va_generation_digest(addr, 0).is_none());

        assert_eq!(XMemMapPhysicalPages(va, 2, std::ptr::null()), va);
        assert!(va_generation_digest(addr, 2 * PAGE).is_none());
        assert_ne!(unsafe { VirtualFree(va, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn chunked_physical_sections_map_across_section_boundaries() {
        let mut n = 5usize;
        let mut pages = vec![0usize; n];
        assert_eq!(
            allocate_physical_pages_with_cap(0, &mut n, pages.as_mut_ptr(), Some(2)),
            1
        );
        {
            let phys = physical();
            assert_eq!(phys.pages[&pages[0]].index, 0);
            assert_eq!(phys.pages[&pages[1]].index, 1);
            assert_eq!(phys.pages[&pages[2]].index, 0);
            assert_eq!(phys.pages[&pages[3]].index, 1);
            assert_eq!(phys.pages[&pages[4]].index, 0);
            assert!(!Arc::ptr_eq(&phys.pages[&pages[1]].section, &phys.pages[&pages[2]].section));
        }
        let va = XMemVirtualAlloc(std::ptr::null_mut(), 5 * PAGE, MEM_RESERVE | PAGES_64K, 0, 0x04);
        assert!(!va.is_null());
        assert_eq!(XMemMapPhysicalPages(va, 5, pages.as_ptr()), va);
        // SAFETY: all 5 pages are freshly mapped.
        unsafe {
            *(va as *mut u32) = 0x1111_2222;
            *((va as *mut u8).add(2 * PAGE) as *mut u32) = 0x3333_4444;
            *((va as *mut u8).add(4 * PAGE) as *mut u32) = 0x5555_6666;
            assert_eq!(*(va as *const u32), 0x1111_2222);
            assert_eq!(*((va as *const u8).add(2 * PAGE) as *const u32), 0x3333_4444);
            assert_eq!(*((va as *const u8).add(4 * PAGE) as *const u32), 0x5555_6666);
        }
        assert_eq!(XMemMapPhysicalPages(va, 5, std::ptr::null()), va);
        assert_ne!(unsafe { VirtualFree(va, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(5, pages.as_ptr()), 1);
    }

    #[test]
    fn freed_physical_page_numbers_are_recycled() {
        // Test deterministic recycling directly on a local `Physical` state so parallel
        // `xmem::tests` cannot race on the global `PHYSICAL` allocator.
        let mut local = Physical {
            next_page: 0x10_0000,
            ..Default::default()
        };
        let p0 = take_page_number(&mut local);
        let p1 = take_page_number(&mut local);
        assert_eq!((p0, p1), (0x10_0000, 0x10_0000 + FRAMES));
        local.free.push(p0);
        let recycled = take_page_number(&mut local);
        assert_eq!(recycled, p0);
        assert_eq!(local.next_page, 0x10_0000 + 2 * FRAMES);

        // Also exercise the live FFI alloc -> free -> alloc path.
        let first = allocate(4);
        assert_eq!(XMemFreePhysicalPages(4, first.as_ptr()), 1);
        let second = allocate(4);
        assert_eq!(second.len(), 4);
        assert_eq!(XMemFreePhysicalPages(4, second.as_ptr()), 1);
    }

    #[test]
    fn map_alias_preserves_dst_on_source_hole_and_supports_overlap() {
        let pages = allocate(3);
        let va = XMemVirtualAlloc(std::ptr::null_mut(), 3 * PAGE, MEM_RESERVE | PAGES_64K, 0, 0x04);
        assert!(!va.is_null());
        assert_eq!(XMemMapPhysicalPages(va, 2, pages.as_ptr()), va);
        // SAFETY: pages 0 and 1 of `va` are mapped.
        unsafe {
            *(va as *mut u32) = 0xDEAD_BEEF;
            *((va as *mut u8).add(PAGE) as *mut u32) = 0x1234_5678;
        }
        // Source [va + PAGE, va + 3 * PAGE) hits an unmapped hole at page 2: abort without
        // unmapping existing destination page 0.
        assert!(!map_alias(va as u64, (va as u64) + PAGE as u64, (2 * PAGE) as u64));
        // SAFETY: destination page 0 remained intact.
        assert_eq!(unsafe { *(va as *const u32) }, 0xDEAD_BEEF);

        // Map page 2, then run overlapping alias shift by 1 page.
        let p2 = unsafe { va.cast::<u8>().add(2 * PAGE).cast::<c_void>() };
        assert_eq!(
            unsafe { XMemMapPhysicalPages(p2, 1, pages.as_ptr().add(2)) },
            p2
        );
        // SAFETY: page 2 is now mapped.
        unsafe {
            *((va as *mut u8).add(2 * PAGE) as *mut u32) = 0xCAFE_BABE;
        }
        assert!(map_alias(va as u64, (va as u64) + PAGE as u64, (2 * PAGE) as u64));
        // SAFETY: aliased pages reflect original pages 1 and 2.
        unsafe {
            assert_eq!(*(va as *const u32), 0x1234_5678);
            assert_eq!(*((va as *const u8).add(PAGE) as *const u32), 0xCAFE_BABE);
        }

        assert!(map_alias(va as u64, 0, (3 * PAGE) as u64));
        assert_ne!(unsafe { VirtualFree(va, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn writes_beside_land_through_guarded_views() {
        let _gate = write_beside_test_gate().lock().unwrap_or_else(|e| e.into_inner());
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
        let pages = allocate(2);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                2 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve() as usize, reserve() as usize);
        assert!(a != 0 && b != 0);
        assert_eq!(XMemMapPhysicalPages(a as _, 2, pages.as_ptr()), a as _);
        assert_eq!(XMemMapPhysicalPages(b as _, 1, pages[1..].as_ptr()), b as _);
        let mut old = 0;
        // SAFETY: our view, made read-only as a guard makes it.
        assert_ne!(
            unsafe { VirtualProtect(a as _, 2 * PAGE, PAGE_READONLY, &mut old) },
            0
        );
        let put = |at: usize, len: usize| {
            write_beside(at, len, &mut |to| {
                // SAFETY: `len` writable bytes the writer gives.
                unsafe { std::ptr::write_bytes(to, 9, len) }
            })
        };
        assert!(put(a + PAGE + 8, 4));
        // SAFETY: both mappings, readable.
        unsafe {
            assert_eq!(
                std::slice::from_raw_parts((a + PAGE + 8) as *const u8, 4),
                [9; 4]
            );
            assert_eq!(
                std::slice::from_raw_parts((b + 8) as *const u8, 4),
                [9; 4]
            );
            assert_eq!(((a + PAGE + 12) as *const u8).read(), 0);
        }
        assert!(!put(b + PAGE - 2, 4), "past the end of the view at b");
        let private = [0u8; 4];
        assert!(!put(private.as_ptr() as usize, 4), "not a view of pages");
        assert_eq!(private, [0; 4]);
        // SAFETY: as above.
        assert_eq!(unsafe { ((b + PAGE - 2) as *const u8).read() }, 0);
        assert_eq!(XMemMapPhysicalPages(b as _, 1, std::ptr::null()), b as _);
        assert_eq!(XMemMapPhysicalPages(a as _, 2, std::ptr::null()), a as _);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    #[test]
    fn remapping_unmapped_page_reuses_cached_placeholder() {
        let pages = allocate(2);
        let va = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!va.is_null());
        let addr = va as usize;

        assert_eq!(XMemMapPhysicalPages(va, 2, pages.as_ptr()), va);
        // SAFETY: both mapped 64 KiB pages are committed and writable.
        unsafe {
            *(va as *mut u32) = 0x1122_3344;
            *((va as *mut u8).add(PAGE) as *mut u32) = 0x5566_7788;
        }

        // Unmapping splits nothing further and records the exact unmapped range as a cached placeholder.
        assert_eq!(XMemMapPhysicalPages(va, 1, std::ptr::null()), va);
        assert!(physical().placeholders.contains(&(addr, PAGE)));

        // Remapping page 1 onto the same single-page range consumes the cached placeholder and keeps its bytes.
        assert_eq!(XMemMapPhysicalPages(va, 1, pages[1..].as_ptr()), va);
        assert!(!physical().placeholders.contains(&(addr, PAGE)));
        // SAFETY: page 1 is now mapped at `va`.
        assert_eq!(unsafe { *(va as *const u32) }, 0x5566_7788);

        // Unmapping/remapping a multi-page range clears any overlapping cached placeholders.
        assert_eq!(XMemMapPhysicalPages(va, 1, std::ptr::null()), va);
        assert!(physical().placeholders.contains(&(addr, PAGE)));
        assert_eq!(XMemMapPhysicalPages(va, 2, std::ptr::null()), va);
        assert_eq!(XMemMapPhysicalPages(va, 2, pages.as_ptr()), va);
        assert!(!physical().placeholders.contains(&(addr, PAGE)));
        // SAFETY: page 0 is back at `va`.
        assert_eq!(unsafe { *(va as *const u32) }, 0x1122_3344);

        assert_eq!(XMemMapPhysicalPages(va, 2, std::ptr::null()), va);
        // SAFETY: `va` is our whole reservation.
        assert_ne!(unsafe { VirtualFree(va, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    fn write_beside_test_gate() -> &'static Mutex<()> {
        static GATE: OnceLock<Mutex<()>> = OnceLock::new();
        GATE.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn write_beside_falls_back_immediately_when_physical_is_write_locked() {
        let _gate = write_beside_test_gate().lock().unwrap_or_else(|e| e.into_inner());
        let pages = allocate(1);
        let va = XMemVirtualAlloc(
            std::ptr::null_mut(),
            PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!va.is_null());
        assert_eq!(XMemMapPhysicalPages(va, 1, pages.as_ptr()), va);

        {
            let _write_lock = physical();
            let mut called = false;
            assert!(!write_beside(va as usize, 4, &mut |_| called = true));
            assert!(!called);
        }

        let mut called = false;
        assert!(write_beside(va as usize, 4, &mut |to| {
            called = true;
            // SAFETY: 4 bytes inside the mapped page.
            unsafe { std::ptr::write_bytes(to, 0xAB, 4) };
        }));
        assert!(called);
        // SAFETY: mapped page, readable.
        assert_eq!(unsafe { std::slice::from_raw_parts(va as *const u8, 4) }, [0xAB; 4]);

        assert_eq!(XMemMapPhysicalPages(va, 1, std::ptr::null()), va);
        // SAFETY: `va` is our whole reservation.
        assert_ne!(unsafe { VirtualFree(va, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn remapping_same_physical_pages_skips_remap_and_preserves_data() {
        let pages = allocate(3);
        let va = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!va.is_null());
        assert_eq!(XMemMapPhysicalPages(va, 2, pages[0..2].as_ptr()), va);
        // SAFETY: both mapped 64 KiB pages are committed and writable.
        unsafe {
            *(va as *mut u32) = 0xCAFE_BABE;
            *((va as *mut u8).add(PAGE) as *mut u32) = 0xDEAD_BEEF;
        }

        let skips_before = super::SAME_PAGE_SKIPS.load(Ordering::Relaxed);
        // Re-mapping identical pages `[0, 1]` skips both pages cleanly.
        assert_eq!(XMemMapPhysicalPages(va, 2, pages[0..2].as_ptr()), va);
        assert_eq!(
            super::SAME_PAGE_SKIPS.load(Ordering::Relaxed) - skips_before,
            2
        );
        // SAFETY: both pages still hold their written values.
        unsafe {
            assert_eq!(*(va as *const u32), 0xCAFE_BABE);
            assert_eq!(*((va as *const u8).add(PAGE) as *const u32), 0xDEAD_BEEF);
        }

        // Replacing only page 1 (`[0, 2]`) skips page 0 and remaps only page 1.
        let mixed = [pages[0], pages[2]];
        let skips_mid = super::SAME_PAGE_SKIPS.load(Ordering::Relaxed);
        assert_eq!(XMemMapPhysicalPages(va, 2, mixed.as_ptr()), va);
        assert_eq!(super::SAME_PAGE_SKIPS.load(Ordering::Relaxed) - skips_mid, 1);
        // SAFETY: page 0 retains its contents while page 1 reflects the new page 2.
        unsafe {
            assert_eq!(*(va as *const u32), 0xCAFE_BABE);
            assert_eq!(*((va as *const u8).add(PAGE) as *const u32), 0);
        }

        assert_eq!(XMemMapPhysicalPages(va, 2, std::ptr::null()), va);
        // SAFETY: `va` is our whole reservation.
        assert_ne!(unsafe { VirtualFree(va, 0, MEM_RELEASE) }, 0);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn placed_whole_reservation_places_and_routes_writes_when_enabled() {
        if !victory_gpu::port_placed::on() {
            return;
        }
        let len = MIN_WHOLE_RESERVATION;
        let count = len / PAGE;
        let pages = allocate(count);
        let va = XMemVirtualAlloc(
            std::ptr::null_mut(),
            len,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!va.is_null());
        assert_eq!(XMemMapPhysicalPages(va, count, pages.as_ptr()), va);
        assert!(victory_gpu::port_placed::placed_touches(va as u64, len as u64));

        let payload = [0xA5u8; 4096];
        assert!(write_physical(&pages, PAGE, &payload));
        // SAFETY: `va + PAGE` is backed by the placed L0 reservation.
        let landed = unsafe { std::slice::from_raw_parts((va as *const u8).add(PAGE), 4096) };
        assert_eq!(landed, &payload);

        let _ = victory_gpu::port_placed::release(va as usize, len);
        {
            let mut phys = physical();
            phys.whole.retain(|&(at, _, _)| at != va as usize);
        }
        assert_ne!(XMemFreePhysicalPages(count, pages.as_ptr()), 0);
    }
}

/// Logs every `VICTORY_MEMORY_LOG_SECS` seconds (5 by default, 0: never)
/// where the process's commit charge goes, for runs that near the host's
/// commit limit (loading Grand Theft Auto VI's world): committed memory by
/// kind and its largest private allocations, the physical pages, and what
/// the GPU side holds.
fn start_memory_log() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let secs = std::env::var("VICTORY_MEMORY_LOG_SECS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(5);
        if secs == 0 {
            return;
        }
        let _ = std::thread::Builder::new()
            .name("memory log".into())
            .spawn(move || {
                let mut peak = 0;
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(secs));
                    log_memory(&mut peak);
                }
            });
    });
}

fn log_memory(peak: &mut usize) {
    use windows_sys::Win32::System::Memory::{MEM_IMAGE, MEM_MAPPED, MEM_PRIVATE, VirtualQuery};
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let commit = counters.PagefileUsage;
    *peak = (*peak).max(commit);
    // Committed bytes by kind, and private ones by allocation.
    let (mut private, mut mapped, mut image) = (0usize, 0usize, 0usize);
    // Private memory below 2^40 is the title's (`XMemVirtualAlloc` keeps
    // it there); above, the host heap's, the driver's and the title's own
    // kernel32 allocations. Allocations by size: under 1, 16, 64 MiB, more.
    let mut low = 0usize;
    let mut sizes = [(0usize, 0usize); 4];
    let mut by_base: HashMap<usize, usize> = HashMap::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
        if n == 0 || info.RegionSize == 0 {
            break;
        }
        if info.State == MEM_COMMIT {
            match info.Type {
                MEM_PRIVATE => {
                    private += info.RegionSize;
                    *by_base.entry(info.AllocationBase as usize).or_default() += info.RegionSize;
                }
                MEM_MAPPED => mapped += info.RegionSize,
                MEM_IMAGE => image += info.RegionSize,
                _ => {}
            }
        }
        let Some(next) = (info.BaseAddress as usize).checked_add(info.RegionSize) else {
            break;
        };
        at = next;
    }
    for (&base, &n) in &by_base {
        if base < 1 << 40 {
            low += n;
        }
        let bucket = match n {
            n if n < MIB => 0,
            n if n < 16 * MIB => 1,
            n if n < 64 * MIB => 2,
            _ => 3,
        };
        sizes[bucket].0 += 1;
        sizes[bucket].1 += n;
    }
    let sizes: Vec<String> = ["<1M", "<16M", "<64M", ">=64M"]
        .iter()
        .zip(sizes)
        .map(|(name, (count, n))| format!("{name} {count}x {} MiB", n / MIB))
        .collect();
    let mut largest: Vec<(usize, usize)> = by_base.into_iter().collect();
    largest.sort_unstable_by_key(|&(_, n)| std::cmp::Reverse(n));
    let largest: Vec<String> = largest
        .iter()
        .take(8)
        .map(|(base, n)| format!("{base:#x} {} MiB", n / MIB))
        .collect();
    let pages = physical_read().pages.len() * PAGE;
    tracing::info!(
        target: "victory::memory",
        "commit {} MiB (peak {} MiB): private {} MiB (title range {} MiB; by allocation: {}), \
         mapped {} MiB, image {} MiB; \
         physical pages {} MiB of sections {} MiB (same-page skips {}); {}; largest private: {}",
        commit / MIB,
        *peak / MIB,
        private / MIB,
        low / MIB,
        sizes.join(", "),
        mapped / MIB,
        image / MIB,
        pages / MIB,
        SECTION_BYTES.load(Ordering::Relaxed) / MIB,
        SAME_PAGE_SKIPS.load(Ordering::Relaxed),
        victory_gpu::memory_report(),
        largest.join(", "),
    );
}
