use std::ops::Index;

use crate::hash::{HashMap, HashSet};
use crate::ir::{Program, Value};

pub(super) type FxMap<K, V> = HashMap<K, V>;
pub(super) type FxSet<T> = HashSet<T>;

/// Dense table keyed by [`Value`], avoiding per-operand hashing in [`super::Lower`].
pub(super) struct DenseMap<T> {
    slots: Vec<Option<T>>,
}

impl<T: Copy> DenseMap<T> {
    pub(super) fn with_capacity(len: usize) -> Self {
        Self {
            slots: vec![None; len],
        }
    }

    #[inline]
    pub(super) fn get(&self, v: &Value) -> Option<&T> {
        self.slots.get(v.0 as usize).and_then(Option::as_ref)
    }

    #[inline]
    pub(super) fn contains_key(&self, v: &Value) -> bool {
        self.slots.get(v.0 as usize).is_some_and(Option::is_some)
    }

    #[inline]
    pub(super) fn insert(&mut self, v: Value, val: T) {
        let i = v.0 as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        self.slots[i] = Some(val);
    }

    #[inline]
    pub(super) fn entry(&mut self, v: Value) -> DenseEntry<'_, T> {
        let i = v.0 as usize;
        if i >= self.slots.len() {
            self.slots.resize(i + 1, None);
        }
        DenseEntry(&mut self.slots[i])
    }
}

pub(super) struct DenseEntry<'a, T>(&'a mut Option<T>);

impl<'a, T> DenseEntry<'a, T> {
    #[inline]
    pub(super) fn or_insert(self, default: T) -> &'a mut T {
        self.0.get_or_insert(default)
    }
}

impl<T: Copy> Index<&Value> for DenseMap<T> {
    type Output = T;

    #[inline]
    fn index(&self, v: &Value) -> &T {
        self.get(v).expect("no entry found for Value")
    }
}

/// One past the highest [`Value`] id in `p`.
pub(super) fn value_count(p: &Program) -> usize {
    let mut max = 0usize;
    for &(_, v) in &p.inputs {
        max = max.max(v.0 as usize + 1);
    }
    for block in &p.blocks {
        for phi in &block.phis {
            max = max.max(phi.value.0 as usize + 1);
        }
        for inst in &block.insts {
            for &(_, v) in &inst.writes {
                max = max.max(v.0 as usize + 1);
            }
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::lower::{round16, round16_after};
    use crate::ir;

    const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn dense_map_and_round16_fast_path_behave_identically() {
        let mut m = DenseMap::<u32>::with_capacity(4);
        assert_eq!(m.get(&Value(0)), None);
        assert!(!m.contains_key(&Value(99)));
        m.insert(Value(2), 42);
        m.insert(Value(10), 7);
        assert_eq!(m.get(&Value(2)), Some(&42));
        assert_eq!(m[&Value(2)], 42);
        assert_eq!(m.get(&Value(10)), Some(&7));
        assert!(m.contains_key(&Value(10)));
        assert!(!m.contains_key(&Value(3)));

        let plain: Vec<u8> = [0x0600_0100u32, S_ENDPGM]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let p = ir::build(&plain, false, 8).unwrap();
        assert_eq!(round16(&p), vec![Some(0)]);

        let rtz: Vec<u8> = [
            0xBFA4_000Cu32, // s_round_mode 0xC (RTZ for f16/f64)
            0xBF82_0000,    // s_branch +0 -> next block
            0x0600_0100,
            S_ENDPGM,
        ]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
        let p2 = ir::build(&rtz, false, 8).unwrap();
        let modes = round16(&p2);
        assert_eq!(modes[0], Some(0));
        assert_eq!(round16_after(&p2.blocks[0].insts[0], Some(0)), Some(3));
    }
}
