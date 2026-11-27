//! Worker-thread CCD / L3-cache affinity pinning (`VICTORY_CCD`).
//!
//! On multi-CCD Ryzen processors (such as a 2-CCD Ryzen 9 5900X), pinning
//! Victory's background copy/upload/dstorage worker threads onto the second CCD
//! (`VICTORY_CCD=1`) or an explicit hex processor mask keeps host copy traffic
//! from thrashing L3 cache and scheduler slots used by the title's main CCD.

#[cfg(windows)]
use std::sync::OnceLock;

#[cfg(any(windows, test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CcdTarget {
    Index(usize),
    Mask(usize),
}

#[cfg(any(windows, test))]
fn parse_ccd_spec(raw: Option<&str>) -> Option<CcdTarget> {
    let s = raw?.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        let mask = usize::from_str_radix(hex, 16).ok()?;
        return (mask != 0).then_some(CcdTarget::Mask(mask));
    }
    s.parse::<usize>().ok().map(CcdTarget::Index)
}

#[cfg(windows)]
fn resolve_ccd_mask(target: CcdTarget) -> Option<usize> {
    match target {
        CcdTarget::Mask(mask) => Some(mask),
        CcdTarget::Index(idx) => {
            use windows_sys::Win32::System::SystemInformation::{
                GetLogicalProcessorInformationEx, RelationCache,
                SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            };
            let mut bytes = 0u32;
            // SAFETY: null buffer query retrieves required byte length in `bytes`.
            unsafe {
                GetLogicalProcessorInformationEx(
                    RelationCache,
                    std::ptr::null_mut(),
                    &mut bytes,
                );
            }
            let mut l3_masks: Vec<usize> = Vec::new();
            if bytes > 0 {
                let mut buf = vec![0u8; bytes as usize];
                // SAFETY: `buf` is sized to `bytes` bytes as requested above.
                let ok = unsafe {
                    GetLogicalProcessorInformationEx(
                        RelationCache,
                        buf.as_mut_ptr().cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(),
                        &mut bytes,
                    )
                };
                if ok != 0 {
                    let mut off = 0usize;
                    while off + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
                        <= bytes as usize
                    {
                        // SAFETY: `off` aligns to valid structs within `[0..bytes)` as written by Windows.
                        let info = unsafe {
                            &*buf
                                .as_ptr()
                                .add(off)
                                .cast::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>()
                        };
                        if info.Relationship == RelationCache && info.Size as usize >= 32 {
                            // SAFETY: union variant is `Cache` when `Relationship == RelationCache`
                            // and `GroupMask` is populated for `RelationCache` entries.
                            let mask = unsafe { info.Anonymous.Cache.Anonymous.GroupMask.Mask };
                            if unsafe { info.Anonymous.Cache.Level } == 3 && mask != 0
                                && !l3_masks.contains(&mask)
                            {
                                l3_masks.push(mask);
                            }
                        }
                        if info.Size == 0 {
                            break;
                        }
                        off += info.Size as usize;
                    }
                }
            }
            if l3_masks.len() > 1 {
                Some(l3_masks[idx % l3_masks.len()])
            } else {
                None
            }
        }
    }
}

#[cfg(windows)]
fn configured_mask() -> Option<usize> {
    static MASK: OnceLock<Option<usize>> = OnceLock::new();
    *MASK.get_or_init(|| {
        let val = std::env::var("VICTORY_CCD").ok();
        let target = parse_ccd_spec(val.as_deref())?;
        let resolved = resolve_ccd_mask(target);
        if let Some(mask) = resolved {
            tracing::info!(mask = format_args!("{mask:#x}"), "VICTORY_CCD: worker affinity mask active");
        }
        resolved
    })
}

/// Pins the current worker thread to the configured CCD or processor mask when
/// `VICTORY_CCD` is set (`0`, `1`, or `0x...` hex mask). No-op when unset.
pub fn pin_worker_ccd() {
    #[cfg(windows)]
    if let Some(mask) = configured_mask() {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
        // SAFETY: `GetCurrentThread()` returns a valid pseudo-handle for this worker thread.
        unsafe {
            SetThreadAffinityMask(GetCurrentThread(), mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{CcdTarget, parse_ccd_spec};

    #[test]
    fn parses_ccd_index_and_hex_masks() {
        assert_eq!(parse_ccd_spec(None), None);
        assert_eq!(parse_ccd_spec(Some("")), None);
        assert_eq!(parse_ccd_spec(Some("0")), Some(CcdTarget::Index(0)));
        assert_eq!(parse_ccd_spec(Some("1")), Some(CcdTarget::Index(1)));
        assert_eq!(
            parse_ccd_spec(Some("0x00FC0FC0")),
            Some(CcdTarget::Mask(0x00FC_0FC0))
        );
        assert_eq!(parse_ccd_spec(Some("0x0")), None);
        assert_eq!(parse_ccd_spec(Some("abc")), None);
    }
}
