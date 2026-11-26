//! GPU shader write-mark bitmap (`u0`, `victory_shader::emit::WRITE_MARKS_SPACE`):
//! a persistent 2 MiB `shared_buffer` in cached system RAM with 1 bit per 64 KiB
//! title-memory page across the 40-bit console address space (`2^24` bits =
//! `524,288` `u32` words). Shaders OR bits using wave-coalescing; the CPU checks
//! only declared/dirty pages during `copy_back` / `sparse_back` and clears
//! observed/recorded dirty words once all in-flight GPU work has drained.

use std::sync::atomic::Ordering;
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::host;

const PAGE_SHIFT: usize = 16;
const WORDS: usize = 1 << (40 - PAGE_SHIFT - 5); // 524,288 u32 words = 2 MiB
const BITMAP_BYTES: u64 = (WORDS * 4) as u64;

struct Marks {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

static MARKS: OnceLock<Option<Marks>> = OnceLock::new();
static DIRTY_WORDS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Whether GPU shader write-marks are enabled (`VICTORY_WRITE_MARKS=0` disables).
pub(crate) fn enabled() -> bool {
    victory_shader::emit::write_marks()
}

/// Whether `copy_back` / `sparse_back` should filter pages via the GPU bitmap.
/// Disabled under `#[cfg(test)]` where existing unit tests write directly into
/// host copies or mock lists without translated shader write-marks.
pub(crate) fn filter_copy_back() -> bool {
    !cfg!(test) && enabled()
}

fn init() -> Option<&'static Marks> {
    MARKS
        .get_or_init(|| {
            let host = host()?;
            let (buffer, ptr) = host.shared_buffer(BITMAP_BYTES)?;
            crate::host::name(&buffer, || "shader write marks".into());
            // SAFETY: freshly allocated `BITMAP_BYTES` shared buffer.
            unsafe { std::ptr::write_bytes(ptr, 0, BITMAP_BYTES as usize) };
            // SAFETY: live D3D12 buffer resource.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Marks {
                _buffer: buffer,
                data: ptr as usize,
                gpu,
            })
        })
        .as_ref()
}

/// GPU virtual address of the global 2 MiB 64 KiB-page write-mark bitmap.
pub(crate) fn gpu_address() -> Option<u64> {
    init().map(|m| m.gpu)
}

#[inline]
fn word_and_bit(page_addr: usize) -> (usize, u32) {
    let page = (page_addr >> PAGE_SHIFT) & ((1 << (40 - PAGE_SHIFT)) - 1);
    (page >> 5, 1u32 << (page & 31))
}

/// Bitmap word index covering the 64 KiB page containing `page_addr`.
#[inline]
pub(crate) fn word_index(page_addr: usize) -> u32 {
    word_and_bit(page_addr).0 as u32
}

#[inline]
fn is_marked_in(m: &Marks, page_addr: usize) -> bool {
    let (word_idx, bit) = word_and_bit(page_addr);
    // SAFETY: `word_idx < WORDS` and `m.data` points to `BITMAP_BYTES` mapped RAM.
    let word = unsafe { (m.data as *const u32).add(word_idx).read_volatile() };
    (word & bit) != 0
}

fn mark_range_in(m: &Marks, address: usize, len: usize) {
    if len == 0 {
        return;
    }
    let start_page = address & !((1 << PAGE_SHIFT) - 1);
    let end = address.saturating_add(len);
    let mut touched = Vec::new();
    let mut at = start_page;
    while at < end {
        let (word_idx, bit) = word_and_bit(at);
        // SAFETY: `word_idx < WORDS` inside the persistent bitmap.
        unsafe {
            let p = (m.data as *mut u32).add(word_idx);
            let old = p.read_volatile();
            if old & bit == 0 {
                p.write_volatile(old | bit);
            }
        }
        if touched.last().copied() != Some(word_idx as u32) {
            touched.push(word_idx as u32);
        }
        match at.checked_add(1 << PAGE_SHIFT) {
            Some(next) => at = next,
            None => break,
        }
    }
    note_dirty_words(&touched);
}

fn clear_pending_in(m: &Marks) {
    let words = {
        let mut dirty = DIRTY_WORDS.lock().unwrap_or_else(|e| e.into_inner());
        if dirty.is_empty() {
            return;
        }
        std::mem::take(&mut *dirty)
    };
    for word_idx in words {
        let idx = word_idx as usize;
        if idx < WORDS {
            // SAFETY: `idx < WORDS` and no GPU work is currently in flight.
            unsafe { (m.data as *mut u32).add(idx).write_volatile(0) };
        }
    }
    std::sync::atomic::fence(Ordering::Release);
}

/// Returns `true` if the 64 KiB page containing `page_addr` has its write-mark
/// bit set in the GPU bitmap. If the bitmap is uninitialized or disabled,
/// conservatively returns `true`.
#[inline]
pub(crate) fn is_page_marked(page_addr: usize) -> bool {
    if !enabled() {
        return true;
    }
    let Some(m) = init() else {
        return true;
    };
    is_marked_in(m, page_addr)
}

/// Records `word_idx` values observed set during `copy_back` / `sparse_back`
/// so [`clear_pending`] zeroes them once all GPU work has drained.
pub(crate) fn note_dirty_words(words: &[u32]) {
    if words.is_empty() {
        return;
    }
    DIRTY_WORDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend_from_slice(words);
}

/// Marks the 64 KiB page bits covering `[address, address + len)` from the CPU
/// for non-shader GPU writes (such as `ResolveQueryData`).
pub(crate) fn mark_cpu_range(address: usize, len: usize) {
    if !enabled() || len == 0 {
        return;
    }
    let Some(m) = init() else {
        return;
    };
    mark_range_in(m, address, len);
}

/// Clears all dirty write-mark bitmap words recorded since the last clear.
/// Caller must ensure no GPU submission writing to the bitmap is currently in flight.
pub(crate) fn clear_pending() {
    if !enabled() {
        return;
    }
    let Some(m) = init() else {
        return;
    };
    clear_pending_in(m);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_range_marks_and_clears_across_page_and_word_boundaries() {
        let Some(m) = init() else {
            return;
        };
        clear_pending_in(m);

        // Span two consecutive 64 KiB pages crossing a 32-page (2 MiB) word boundary.
        let addr = (32usize * (1 << PAGE_SHIFT)) - 16;
        mark_range_in(m, addr, 32);

        assert!(is_marked_in(m, addr));
        assert!(is_marked_in(m, addr + 31));
        assert!(!is_marked_in(m, addr - (1 << PAGE_SHIFT)));
        assert!(!is_marked_in(m, addr + 32 + (1 << PAGE_SHIFT)));

        clear_pending_in(m);
        assert!(!is_marked_in(m, addr));
        assert!(!is_marked_in(m, addr + 31));
    }
}
