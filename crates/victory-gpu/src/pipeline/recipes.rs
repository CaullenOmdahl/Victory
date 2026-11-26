//! Cross-platform persistence for pipeline recipes (`used-pipelines.bin`),
//! separating OS-independent recipe encoding/reading from D3D12 library calls.

use std::collections::HashSet;
use std::path::Path;

pub(crate) const RECIPES_FILE: &str = "used-pipelines.bin";
const MAGIC: &[u8; 4] = b"VPR1";

/// One host pipeline recipe recorded when the title draws or dispatches with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct UsedRecipe {
    pub(crate) key: String,
    pub(crate) stages: Vec<(String, String)>,
    pub(crate) graphics: Option<Vec<u8>>,
}

pub(crate) fn encode_recipes(recipes: &[UsedRecipe]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + recipes.len() * 96);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(recipes.len() as u32).to_le_bytes());
    for r in recipes {
        put_str(&mut out, &r.key);
        out.extend_from_slice(&(r.stages.len() as u32).to_le_bytes());
        for (stage, hash) in &r.stages {
            put_str(&mut out, stage);
            put_str(&mut out, hash);
        }
        match &r.graphics {
            Some(g) => {
                out.push(1);
                out.extend_from_slice(&(g.len() as u32).to_le_bytes());
                out.extend_from_slice(g);
            }
            None => out.push(0),
        }
    }
    out
}

pub(crate) fn decode_recipes(mut bytes: &[u8]) -> Option<Vec<UsedRecipe>> {
    if bytes.get(..4)? != MAGIC {
        return None;
    }
    bytes = &bytes[4..];
    let count = take_u32(&mut bytes)? as usize;
    let mut out = Vec::with_capacity(count.min(16_384));
    for _ in 0..count {
        let key = take_str(&mut bytes)?;
        let stage_count = take_u32(&mut bytes)? as usize;
        let mut stages = Vec::with_capacity(stage_count.min(8));
        for _ in 0..stage_count {
            let stage = take_str(&mut bytes)?;
            let hash = take_str(&mut bytes)?;
            stages.push((stage, hash));
        }
        let has_graphics = *bytes.first()?;
        bytes = &bytes[1..];
        let graphics = match has_graphics {
            0 => None,
            1 => {
                let len = take_u32(&mut bytes)? as usize;
                if bytes.len() < len {
                    return None;
                }
                let (g, rest) = bytes.split_at(len);
                bytes = rest;
                Some(g.to_vec())
            }
            _ => return None,
        };
        out.push(UsedRecipe {
            key,
            stages,
            graphics,
        });
    }
    bytes.is_empty().then_some(out)
}

pub(crate) fn load_used_recipes(pipeline_dir: &Path) -> Vec<UsedRecipe> {
    let path = pipeline_dir
        .join(crate::shader_cache::build_key())
        .join(RECIPES_FILE);
    std::fs::read(path)
        .ok()
        .and_then(|b| decode_recipes(&b))
        .unwrap_or_default()
}

/// Returns the set of 16-char BLAKE3 container hashes referenced by
/// `used-pipelines.bin` for the current shader-cache build key.
pub fn needed_shader_hashes(pipeline_cache_dir: &Path) -> HashSet<String> {
    load_used_recipes(pipeline_cache_dir)
        .into_iter()
        .flat_map(|r| r.stages.into_iter().map(|(_, h)| h))
        .collect()
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn take_u32(bytes: &mut &[u8]) -> Option<u32> {
    if bytes.len() < 4 {
        return None;
    }
    let (head, rest) = bytes.split_at(4);
    *bytes = rest;
    Some(u32::from_le_bytes(head.try_into().ok()?))
}

fn take_str(bytes: &mut &[u8]) -> Option<String> {
    let len = take_u32(bytes)? as usize;
    if bytes.len() < len {
        return None;
    }
    let (head, rest) = bytes.split_at(len);
    *bytes = rest;
    std::str::from_utf8(head).ok().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipes_round_trip_and_reject_damage() {
        let sample = vec![
            UsedRecipe {
                key: "0123456789abcdef0123456789abcdef".into(),
                stages: vec![
                    ("vertex".into(), "aaaabbbbccccdddd".into()),
                    ("pixel".into(), "1111222233334444".into()),
                ],
                graphics: Some(vec![1, 2, 3, 4]),
            },
            UsedRecipe {
                key: "fedcba9876543210fedcba9876543210".into(),
                stages: vec![("compute".into(), "5555666677778888".into())],
                graphics: None,
            },
        ];
        let encoded = encode_recipes(&sample);
        assert_eq!(decode_recipes(&encoded), Some(sample));
        assert_eq!(decode_recipes(b"BAD0"), None);
        assert_eq!(decode_recipes(&encoded[..encoded.len() - 1]), None);
        let mut extra = encoded.clone();
        extra.push(0);
        assert_eq!(decode_recipes(&extra), None);
    }
}
