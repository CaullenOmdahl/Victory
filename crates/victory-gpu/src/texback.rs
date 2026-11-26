//! Textures shaders write that the title's own code then reads on the CPU
//! (`VICTORY_TEXTURE_BACK=1` or comma-separated `<w>x<h>f<fmt>`).
//! On the console the CPU and the GPU share memory, so what a compute pass
//! stores into a texture is in title memory for the CPU to read. Here it stays
//! in the texture's host copy. The title's ocean is one: its wave passes write
//! a 256x256 height map (`R32_FLOAT`) and a displacement map (`R32G32_FLOAT`),
//! and its CPU code samples both so boats float instead of sinking.
//!
//! For textures matching `VICTORY_TEXTURE_BACK`, a command list in which shaders
//! write one via UAV copies it into a readback buffer when closing, and once
//! submitted & finished (`Host::finish_submission`) the texels are tiled back
//! into the texture's title memory in Xbox console layout.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::*;

use crate::object::This;

const DEFAULT_KINDS: &str = "256x256f41,256x256f16,64x64f16,128x128f36";

fn parse_kinds(spec: &str) -> Vec<(u64, u32, u32)> {
    spec.split(',')
        .filter_map(|k| {
            let (size, format) = k.trim().split_once('f')?;
            let (w, h) = size.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?, format.parse().ok()?))
        })
        .collect()
}

/// The kinds copied back: `(width, height, dxgi_format)`.
fn kinds() -> &'static [(u64, u32, u32)] {
    static KINDS: OnceLock<Vec<(u64, u32, u32)>> = OnceLock::new();
    KINDS.get_or_init(|| {
        let list = match std::env::var("VICTORY_TEXTURE_BACK").as_deref() {
            Ok("1") => DEFAULT_KINDS.to_owned(),
            Ok(spec) => spec.to_owned(),
            Err(_) => String::new(),
        };
        let parsed = parse_kinds(&list);
        if !parsed.is_empty() {
            tracing::info!(
                ?parsed,
                "textures of these kinds that shaders write are copied back into title memory (VICTORY_TEXTURE_BACK)"
            );
        }
        parsed
    })
}

/// Whether title texture `resource` is of a kind copied back.
pub(crate) fn wanted(resource: This) -> bool {
    let list = kinds();
    !list.is_empty() && crate::resource::texture_kind(resource).is_some_and(|k| list.contains(&k))
}

/// An outstanding write to a specific allocation. Mapping and resource
/// changes invalidate only the writes that could touch the replaced memory.
struct Ticket {
    owner: usize,
    at: usize,
    end: usize,
    valid: AtomicBool,
}

static TICKETS: Mutex<Vec<Weak<Ticket>>> = Mutex::new(Vec::new());
static TRACKING: AtomicBool = AtomicBool::new(false);

fn track(owner: usize, at: usize, end: usize) -> Arc<Ticket> {
    let ticket = Arc::new(Ticket {
        owner,
        at,
        end,
        valid: AtomicBool::new(true),
    });
    let mut tickets = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    tickets.retain(|weak| weak.strong_count() != 0);
    tickets.push(Arc::downgrade(&ticket));
    TRACKING.store(true, Ordering::Relaxed);
    ticket
}

fn invalidate(matches: impl Fn(&Ticket) -> bool) {
    if !TRACKING.load(Ordering::Relaxed) {
        return;
    }
    TICKETS.lock().unwrap_or_else(|e| e.into_inner()).retain(|weak| {
        let Some(ticket) = weak.upgrade() else {
            return false;
        };
        if matches(&ticket) {
            ticket.valid.store(false, Ordering::Release);
        }
        true
    });
}

pub(crate) fn released(owner: usize) {
    invalidate(|ticket| ticket.owner == owner);
}

pub(crate) fn replaced(at: usize) {
    invalidate(|ticket| ticket.at == at);
}

pub(crate) fn remapped(at: usize, len: usize) {
    let end = at.saturating_add(len);
    invalidate(|ticket| at < ticket.end && ticket.at < end);
}

pub(crate) struct Target {
    pub(crate) copy: Arc<crate::resource::HostTexture>,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Target {
    pub(crate) fn new(resource: This) -> Option<Self> {
        let copy = crate::resource::host_copy(resource)?;
        let (layout, at, size) = crate::resource::title_texture(resource)?;
        let end = at.checked_add(size).filter(|&end| at != 0 && end > at)?;
        let ticket = track(resource as usize, at, end);
        Some(Self {
            copy,
            layout,
            ticket,
        })
    }

    pub(crate) fn owner(&self) -> usize {
        self.ticket.owner
    }
}

/// The mapped readback buffer and target layout/range the texels belong to.
pub(crate) struct Back {
    _buffer: ID3D12Resource,
    data: usize,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    mips: u32,
    layout: victory_formats::texture::Layout,
    ticket: Arc<Ticket>,
}

impl Back {
    pub(crate) fn range(&self) -> (usize, usize) {
        (self.ticket.at, self.ticket.end)
    }
}

/// Records a host copy from a texture already in `COPY_SOURCE`.
pub(crate) fn record(list: &ID3D12GraphicsCommandList, target: Target) -> Option<Back> {
    if !target.ticket.valid.load(Ordering::Acquire) {
        return None;
    }
    let host = crate::host::host()?;
    let texture = &target.copy.texture;
    // SAFETY: live host texture retained by `Target.copy`.
    let desc = unsafe { texture.GetDesc() };
    let mips = u32::from(desc.MipLevels.max(1));
    let subresources = mips * u32::from(desc.DepthOrArraySize.max(1));
    let n = subresources as usize;
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n];
    let mut rows = vec![0u32; n];
    let mut total = 0u64;
    // SAFETY: out slices hold `n` elements matching `subresources`.
    unsafe {
        host.device.GetCopyableFootprints(
            &desc,
            0,
            subresources,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // The last footprint's advertised extent can omit row-end padding; the
    // tiler reads a full pitched slice, so ensure the readback buffer covers it.
    for (footprint, &row_count) in footprints.iter().zip(&rows) {
        let format = footprint.Footprint;
        let pitched = u64::from(format.RowPitch)
            .checked_mul(u64::from(row_count))?
            .checked_mul(u64::from(format.Depth))?;
        total = total.max(footprint.Offset.checked_add(pitched)?);
    }
    let (buffer, mapped) = host.readback_buffer(total)?;
    if mapped.is_null() {
        return None;
    }
    for (i, footprint) in footprints.iter().enumerate() {
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed COM pointer that D3D12_TEXTURE_COPY_LOCATION never releases.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above; `texture` stays alive for the list's lifetime.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        // SAFETY: open command list with `texture` transitioned to COPY_SOURCE.
        unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    Some(Back {
        _buffer: buffer,
        data: mapped as usize,
        footprints,
        rows,
        mips,
        layout: target.layout,
        ticket: target.ticket,
    })
}

/// Once the list's submission has finished on the GPU, tile its linear texels
/// back into the corresponding title texture memory; returns bytes written.
pub(crate) fn land(back: &Back) -> usize {
    static COPIED: AtomicU64 = AtomicU64::new(0);
    static STALE: AtomicU64 = AtomicU64::new(0);
    let _ticket_guard = TICKETS.lock().unwrap_or_else(|e| e.into_inner());
    if !back.ticket.valid.load(Ordering::Acquire) {
        let skipped = STALE.fetch_add(1, Ordering::Relaxed) + 1;
        if skipped.is_power_of_two() {
            tracing::info!(
                skipped,
                "skipping texture readbacks whose title allocation changed"
            );
        }
        return 0;
    }
    let (at, end) = back.range();
    let size = end - at;
    let mut tiled = vec![0u8; size];
    // Preserve title bytes that the subresources do not overwrite.
    if !crate::host::read_into(at as *const u8, &mut tiled) {
        return 0;
    }
    let data = back.data as *const u8;
    for (i, (footprint, &row_count)) in back.footprints.iter().zip(&back.rows).enumerate() {
        let (mip, slice) = (i as u32 % back.mips, i as u32 / back.mips);
        let f = footprint.Footprint;
        let pitch = f.RowPitch as usize;
        let len = pitch * row_count as usize * f.Depth as usize;
        // SAFETY: mapped readback buffer covers `[Offset, Offset + len)` and GPU work completed.
        let linear = unsafe { std::slice::from_raw_parts(data.add(footprint.Offset as usize), len) };
        if let Err(e) = back
            .layout
            .tile(&mut tiled, mip, slice, linear, pitch, pitch * row_count as usize)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(?e, mip, slice, "a texture copied back could not be tiled");
            });
            return 0;
        }
    }
    if !crate::watch::write_beside(at, &tiled) {
        crate::watch::open_for_sync_write(at, size);
        crate::host::write_mapped(at as *mut u8, tiled.as_ptr(), size);
    }
    let copied = COPIED.fetch_add(1, Ordering::Relaxed) + 1;
    if copied == 1 || copied.is_multiple_of(128) {
        tracing::info!(copied, bytes = size, "texture readback landed");
    }
    size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_kinds_parses_default_and_custom_specs() {
        assert_eq!(
            parse_kinds(DEFAULT_KINDS),
            vec![(256, 256, 41), (256, 256, 16), (64, 64, 16), (128, 128, 36)]
        );
        assert_eq!(
            parse_kinds(" 512x512f10, invalid, 128x64f28 "),
            vec![(512, 512, 10), (128, 64, 28)]
        );
        assert!(parse_kinds("").is_empty());
    }

    #[test]
    fn only_readbacks_for_changed_allocations_are_invalidated() {
        let first = track(91, 0x1000, 0x2000);
        let next = track(92, 0x2000, 0x3000);
        let far = track(93, 0x4000, 0x5000);

        remapped(0x0f00, 0x200);
        assert!(!first.valid.load(Ordering::Acquire));
        assert!(next.valid.load(Ordering::Acquire));
        replaced(0x2000);
        assert!(!next.valid.load(Ordering::Acquire));
        assert!(far.valid.load(Ordering::Acquire));

        released(93);
        assert!(!far.valid.load(Ordering::Acquire));
        assert!(track(94, 0x2000, 0x3000).valid.load(Ordering::Acquire));
    }
}
