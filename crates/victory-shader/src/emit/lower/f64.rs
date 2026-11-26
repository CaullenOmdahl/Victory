//! Binary64 `ceil`, `floor`, `fract`, and `trig_preop` lowering without native
//! DXIL `f64` or `i64` feature bits (`SFI0` `DOUBLES` / `INT64`).

use super::{Attr, Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, Shader, V, Why, pred};
use crate::emit::bitcode::Module;

/// The first 1184 fractional bits of 2/pi, most significant word first.
/// The final zero is the tail past the measured RDNA2 table (AnyPS5 #447).
const TWO_OVER_PI: [u32; 38] = [
    0xa2f9_836e,
    0x4e44_1529,
    0xfc27_57d1,
    0xf534_ddc0,
    0xdb62_9599,
    0x3c43_9041,
    0xfe51_63ab,
    0xdebb_c561,
    0xb724_6e3a,
    0x424d_d2e0,
    0x0649_2eea,
    0x09d1_921c,
    0xfe1d_eb1c,
    0xb129_a73e,
    0xe882_35f5,
    0x2ebb_4484,
    0xe99c_7026,
    0xb45f_7e41,
    0x3991_d639,
    0x8353_39f4,
    0x9c84_5f8b,
    0xbdf9_283b,
    0x1ff8_97ff,
    0xde05_980f,
    0xef2f_118b,
    0x5a0a_6d1f,
    0x6d36_7ecf,
    0x27cb_09b7,
    0x4f46_3f66,
    0x9e5f_ea2d,
    0x7527_bac7,
    0xebe5_f17b,
    0x3d07_39f7,
    0x8a52_92ea,
    0x6bfb_5fb1,
    0x1f8d_5d08,
    0x5603_3046,
    0,
];

impl<'a> Lower<'a> {
    /// Reads the raw `(low, high)` 32-bit words of a binary64 source operand
    /// without marking the shader as using native f64 or int64 operations.
    fn f64_raw_words(&mut self, inst: &Inst, op: Operand) -> Result<(V, V), Why> {
        let pair = |l: &mut Self, lo: Reg, hi: Reg| -> Result<(V, V), Why> {
            let lo = l.i32_of(inst.read(lo).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(hi).ok_or("an unread operand")?)?;
            Ok((lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => {
                let next = reg.checked_add(1).ok_or("a missing operand high word")?;
                pair(self, Reg::V(reg), Reg::V(next))
            }
            Operand::Scalar { code, .. } if code != NULL => {
                let next = code.checked_add(1).ok_or("a missing operand high word")?;
                pair(self, Reg::S(code), Reg::S(next))
            }
            Operand::Literal(high) => Ok((self.s.m.i32(0), self.s.m.i32(high))),
            Operand::Const(code @ 240..=248) => {
                let bits = if code == 248 {
                    0x3fc4_5f30_6dc9_c882 // Binary64 1/(2π).
                } else {
                    let value = crate::resource::inline(code).ok_or("an unknown inline float")?;
                    f64::from(f32::from_bits(value)).to_bits()
                };
                Ok((self.s.m.i32(bits as u32), self.s.m.i32((bits >> 32) as u32)))
            }
            _ => {
                let high = if matches!(op, Operand::Const(193..=208)) {
                    u32::MAX
                } else {
                    0
                };
                Ok((self.operand(inst, &op)?, self.s.m.i32(high)))
            }
        }
    }

    /// Ceil or floor a binary64 value in its two 32-bit register words.
    pub(super) fn round_directed_f64(
        &mut self,
        inst: &Inst,
        guard: Option<V>,
        ceil: bool,
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("binary64 rounding modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 destination pair");
        };
        let dst_hi = reg.checked_add(1).ok_or("a missing destination high word")?;
        let source = *v.src.first().ok_or("a missing binary64 source")?;
        let (lo, mut hi) = self.f64_raw_words(inst, source)?;

        let c = self.cur;
        let m = &mut self.s.m;
        let i32t = self.s.t.i32;
        let zero = m.i32(0);
        let ones = m.i32(u32::MAX);
        let sign_mask = m.i32(0x8000_0000);
        let mag_mask = m.i32(0x7fff_ffff);
        if v.abs & 1 != 0 {
            hi = m.bin(c, Bin::And, hi, mag_mask);
        }
        if v.neg & 1 != 0 {
            hi = m.bin(c, Bin::Xor, hi, sign_mask);
        }
        let sign = m.bin(c, Bin::And, hi, sign_mask);
        let away = m.cmp(c, if ceil { pred::EQ } else { pred::NE }, sign, zero);

        let mag_hi = m.bin(c, Bin::And, hi, mag_mask);
        let k20 = m.i32(20);
        let exponent = m.bin(c, Bin::LShr, mag_hi, k20);
        let integral_exponent = m.i32(1023 + 52);
        let integral = m.cmp(c, pred::IUGE, exponent, integral_exponent);
        let n = m.bin(c, Bin::Sub, integral_exponent, exponent);
        let (k31, k32) = (m.i32(31), m.i32(32));
        let clears_low = m.cmp(c, pred::IUGE, n, k32);
        let clears_high_bits = m.cmp(c, pred::IUGT, n, k32);
        let shift = m.bin(c, Bin::And, n, k31);
        let keep = m.bin(c, Bin::Shl, ones, shift);
        let discard = m.bin(c, Bin::Xor, keep, ones);

        // Add ones in the discarded bits when rounding away from zero.
        // Then clearing those bits advances to the next integer only if
        // the fraction was nonzero.
        let low_bias = m.select(c, clears_low, ones, discard);
        let low_bias = m.select(c, away, low_bias, zero);
        let sum_lo = m.bin(c, Bin::Add, lo, low_bias);
        let carry = m.cmp(c, pred::IULT, sum_lo, lo);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);
        let high_increments = m.bin(c, Bin::And, away, clears_high_bits);
        let high_bias = m.select(c, high_increments, discard, zero);
        let sum_hi = m.bin(c, Bin::Add, hi, high_bias);
        let sum_hi = m.bin(c, Bin::Add, sum_hi, carry);
        let round_lo = m.bin(c, Bin::And, sum_lo, keep);
        let round_lo = m.select(c, clears_low, zero, round_lo);
        let round_hi = m.bin(c, Bin::And, sum_hi, keep);
        let round_hi = m.select(c, clears_high_bits, round_hi, sum_hi);

        // Below one, the result is signed zero or signed one.
        let magnitude = m.bin(c, Bin::Or, mag_hi, lo);
        let nonzero = m.cmp(c, pred::NE, magnitude, zero);
        let to_one = m.bin(c, Bin::And, away, nonzero);
        let one_hi = m.i32(0x3ff0_0000);
        let small_hi = m.select(c, to_one, one_hi, zero);
        let small_hi = m.bin(c, Bin::Or, small_hi, sign);
        let bias = m.i32(1023);
        let below_one = m.cmp(c, pred::IULT, exponent, bias);
        let round_hi = m.select(c, below_one, small_hi, round_hi);

        let result_lo = m.select(c, integral, lo, round_lo);
        let result_hi = m.select(c, integral, hi, round_hi);
        self.write(inst, Reg::V(reg), result_lo, guard);
        self.write(inst, Reg::V(dst_hi), result_hi, guard);
        Ok(())
    }

    /// Extract the binary64 fraction without requiring native doubles or int64.
    pub(super) fn fract_f64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("binary64 fraction modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 destination pair");
        };
        let dst_high = reg.checked_add(1).ok_or("a missing destination high word")?;
        let source = *v.src.first().ok_or("a missing binary64 source")?;
        let (low, mut high) = self.f64_raw_words(inst, source)?;

        let c = self.cur;
        let i32t = self.s.t.i32;
        let first_bit = |s: &mut Shader, op, word| {
            s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, op, &[word])
        };
        let m = &mut self.s.m;
        let zero = m.i32(0);
        let one = m.i32(1);
        let ones = m.i32(u32::MAX);
        let sign_mask = m.i32(0x8000_0000);
        let magnitude_mask = m.i32(0x7fff_ffff);
        let mantissa_mask = m.i32(0x000f_ffff);
        let hidden_bit = m.i32(0x0010_0000);
        let magnitude_high = m.bin(c, Bin::And, high, magnitude_mask);
        if v.abs & 1 != 0 {
            high = magnitude_high;
        }
        if v.neg & 1 != 0 {
            high = m.bin(c, Bin::Xor, high, sign_mask);
        }
        let sign = m.bin(c, Bin::And, high, sign_mask);
        let is_negative = m.cmp(c, pred::NE, sign, zero);
        let shift20 = m.i32(20);
        let exponent = m.bin(c, Bin::LShr, magnitude_high, shift20);
        let exponent_one = m.i32(1023);
        let exponent_half = m.i32(1022);
        let exponent_integral = m.i32(1075);
        let small = m.cmp(c, pred::IULT, exponent, exponent_half);
        let integral = m.cmp(c, pred::IUGE, exponent, exponent_integral);
        let mantissa_high = m.bin(c, Bin::And, high, mantissa_mask);
        let significand_high = m.bin(c, Bin::Or, mantissa_high, hidden_bit);

        // For |x| >= 1/2, the fractional field uses at most 53 bits.
        // For smaller negative values, round |x| to units of 2^-53 before
        // subtracting it from 1.0. The guard, sticky and parity select ties to even.
        let bits53 = m.i32(53);
        let tail_bits = m.bin(c, Bin::Sub, exponent_integral, exponent);
        let tail_bits = m.select(c, small, bits53, tail_bits);
        let tail_bits = m.select(c, integral, zero, tail_bits);
        let guard_shift = m.i32(1021);
        let guard_shift = m.bin(c, Bin::Sub, guard_shift, exponent);
        let guard_in_range = m.cmp(c, pred::IULT, guard_shift, bits53);
        let guard_shift = m.select(c, guard_in_range, guard_shift, bits53);
        let (guard_low, guard_high) =
            shift_fraction_words(m, c, low, significand_high, guard_shift, false);
        let (scaled_low, scaled_high) =
            shift_fraction_words(m, c, guard_low, guard_high, one, false);
        let guard_bit = m.bin(c, Bin::And, guard_low, one);
        let odd = m.bin(c, Bin::And, scaled_low, one);
        let low_nonzero = m.cmp(c, pred::NE, low, zero);
        let first_low = first_bit(&mut self.s, 32, low);
        let first_high = first_bit(&mut self.s, 32, significand_high);
        let m = &mut self.s.m;
        let bits32 = m.i32(32);
        let first_high = m.bin(c, Bin::Add, first_high, bits32);
        let first = m.select(c, low_nonzero, first_low, first_high);
        let sticky = m.cmp(c, pred::IULT, first, guard_shift);
        let sticky = m.cast(c, Cast::ZExt, sticky, i32t);
        let round_up = m.bin(c, Bin::Or, sticky, odd);
        let round_up = m.bin(c, Bin::And, round_up, guard_bit);
        let rounded_low = m.bin(c, Bin::Add, scaled_low, round_up);
        let carry = m.cmp(c, pred::IULT, rounded_low, scaled_low);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);
        let rounded_high = m.bin(c, Bin::Add, scaled_high, carry);
        let significand_low = m.select(c, small, rounded_low, low);
        let significand_high = m.select(c, small, rounded_high, significand_high);

        // For negative x, complementing modulo the width of its fractional
        // field computes 1 - |fraction| and also keeps integer results at zero.
        let borrow = m.cmp(c, pred::NE, significand_low, zero);
        let borrow = m.cast(c, Cast::ZExt, borrow, i32t);
        let negated_low = m.bin(c, Bin::Sub, zero, significand_low);
        let negated_high = m.bin(c, Bin::Sub, zero, significand_high);
        let negated_high = m.bin(c, Bin::Sub, negated_high, borrow);
        let fraction_low = m.select(c, is_negative, negated_low, significand_low);
        let fraction_high = m.select(c, is_negative, negated_high, significand_high);
        let low_full = m.cmp(c, pred::IUGE, tail_bits, bits32);
        let bits31 = m.i32(31);
        let tail_shift = m.bin(c, Bin::And, tail_bits, bits31);
        let tail_mask = m.bin(c, Bin::Shl, ones, tail_shift);
        let tail_mask = m.bin(c, Bin::Xor, tail_mask, ones);
        let low_mask = m.select(c, low_full, ones, tail_mask);
        let high_mask = m.select(c, low_full, tail_mask, zero);
        let fraction_low = m.bin(c, Bin::And, fraction_low, low_mask);
        let fraction_high = m.bin(c, Bin::And, fraction_high, high_mask);

        // Normalize the fixed-point fraction and apply the binary64 exponent bias.
        let leading_low = first_bit(&mut self.s, 33, fraction_low);
        let leading_high = first_bit(&mut self.s, 33, fraction_high);
        let m = &mut self.s.m;
        let bits63 = m.i32(63);
        let msb_low = m.bin(c, Bin::Sub, bits31, leading_low);
        let msb_high = m.bin(c, Bin::Sub, bits63, leading_high);
        let high_nonzero = m.cmp(c, pred::NE, fraction_high, zero);
        let msb = m.select(c, high_nonzero, msb_high, msb_low);
        let bits52 = m.i32(52);
        let shift = m.bin(c, Bin::Sub, bits52, msb);
        let (normal_low, normal_high) =
            shift_fraction_words(m, c, fraction_low, fraction_high, shift, true);
        let normal_high = m.bin(c, Bin::And, normal_high, mantissa_mask);
        let adjust = m.bin(c, Bin::Sub, msb, tail_bits);
        let normal_exp = m.bin(c, Bin::Add, exponent_one, adjust);
        let normal_exp = m.bin(c, Bin::Shl, normal_exp, shift20);
        let normal_high = m.bin(c, Bin::Or, normal_high, normal_exp);

        // Negative values too small to change the rounded 1.0 still return
        // the largest value below one. Exact integers and -0 return +0.
        let input_magnitude = m.bin(c, Bin::Or, magnitude_high, low);
        let input_nonzero = m.cmp(c, pred::NE, input_magnitude, zero);
        let small_negative = m.bin(c, Bin::And, small, is_negative);
        let round_to_limit = m.bin(c, Bin::And, small_negative, input_nonzero);
        let limit_high = m.i32(0x3fef_ffff);
        let empty_low = m.select(c, round_to_limit, ones, zero);
        let empty_high = m.select(c, round_to_limit, limit_high, zero);
        let fraction = m.bin(c, Bin::Or, fraction_low, fraction_high);
        let has_fraction = m.cmp(c, pred::NE, fraction, zero);
        let result_low = m.select(c, has_fraction, normal_low, empty_low);
        let result_high = m.select(c, has_fraction, normal_high, empty_high);

        // Positive inputs already below one include subnormals and need no
        // arithmetic. The IEEE nonfinite cases produce a quiet NaN.
        let below_one = m.cmp(c, pred::IULT, exponent, exponent_one);
        let is_positive = m.cmp(c, pred::EQ, sign, zero);
        let keep = m.bin(c, Bin::And, is_positive, below_one);
        let result_low = m.select(c, keep, low, result_low);
        let result_high = m.select(c, keep, magnitude_high, result_high);
        let inf_exp = m.i32(0x7ff);
        let nonfinite = m.cmp(c, pred::EQ, exponent, inf_exp);
        let quiet = m.i32(0x0008_0000);
        let nan_high = m.bin(c, Bin::Or, high, quiet);
        let result_low = m.select(c, nonfinite, low, result_low);
        let result_high = m.select(c, nonfinite, nan_high, result_high);

        self.write(inst, Reg::V(reg), result_low, guard);
        self.write(inst, Reg::V(dst_high), result_high, guard);
        Ok(())
    }

    /// Lower binary64 trigonometric argument-reduction segment lookup (`v_trig_preop_f64`).
    pub(super) fn trig_preop_f64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a double result outside a VGPR pair");
        };
        let high = reg
            .checked_add(1)
            .ok_or("a missing destination high word")?;
        let op = *v.src.first().ok_or("a missing operand")?;
        // Only the exponent matters. All inline numbers are below the first
        // shift threshold, and input sign modifiers do not change it.
        let input_high = match op {
            Operand::Literal(high) => self.s.m.i32(high),
            Operand::Const(128..=208 | 240..=248) => self.s.m.i32(0),
            _ => self.f64_raw_words(inst, op)?.1,
        };
        let selector = self.operand(inst, v.src.get(1).ok_or("a missing selector")?)?;
        let table = self.trig_preop_table();
        let c = self.cur;
        let m = &mut self.s.m;

        let shift20 = m.i32(20);
        let exp_mask = m.i32(0x7ff);
        let exp = m.bin(c, Bin::LShr, input_high, shift20);
        let exp = m.bin(c, Bin::And, exp, exp_mask);
        let threshold = m.i32(1077);
        let past_threshold = m.cmp(c, pred::IUGT, exp, threshold);
        let extra = m.bin(c, Bin::Sub, exp, threshold);
        let zero = m.i32(0);
        let extra = m.select(c, past_threshold, extra, zero);
        let mask = m.i32(31);
        let width = m.i32(53);
        let segment = m.bin(c, Bin::And, selector, mask);
        let start = m.bin(c, Bin::Mul, segment, width);
        let start = m.bin(c, Bin::Add, start, extra);

        let mantissa = read_window(m, c, table, start);
        let large_exp = m.i32(1968);
        let large = m.cmp(c, pred::IUGE, exp, large_exp);
        let boost = m.i32(128);
        let boost = m.select(c, large, boost, zero);
        // Largest possible exponent of a window scaled by 2^(-53 - start).
        let biased = m.i32(1022);
        let biased = m.bin(c, Bin::Sub, biased, start);
        let biased = m.bin(c, Bin::Add, biased, boost);
        let [lo, hi] = self.trig_preop_pack(mantissa, biased);
        // The result is positive and below one, so clamp does not affect it.
        // f64 OMOD is ignored with denormals kept, the model used here.
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(high), hi, guard);
        Ok(())
    }

    fn trig_preop_table(&mut self) -> V {
        if let Some(table) = self.trig_preop_table {
            return table;
        }
        let m = &mut self.s.m;
        let words: Vec<_> = TWO_OVER_PI.iter().map(|&word| m.i32(word)).collect();
        let ty = m.array(words.len() as u32, self.s.t.i32);
        let init = m.aggregate(ty, &words);
        let table = m.constant_global("trig_preop_two_over_pi", init);
        self.trig_preop_table = Some(table);
        table
    }

    fn trig_preop_pack(&mut self, [lo, hi]: [V; 2], biased: V) -> [V; 2] {
        let c = self.cur;
        let ty = self.s.t.i32;
        let low_leading = self.s.op(c, "unaryBits.i32", Attr::ReadNone, ty, 33, &[lo]);
        let high_leading = self.s.op(c, "unaryBits.i32", Attr::ReadNone, ty, 33, &[hi]);
        let m = &mut self.s.m;
        let zero = m.i32(0);
        let hi_base = m.i32(-11_i32 as u32);
        let lo_base = m.i32(21);
        let shift_hi = m.bin(c, Bin::Add, high_leading, hi_base);
        let shift_lo = m.bin(c, Bin::Add, low_leading, lo_base);
        let has_hi = m.cmp(c, pred::NE, hi, zero);
        let leading = m.select(c, has_hi, shift_hi, shift_lo);
        let bits = m.bin(c, Bin::Or, hi, lo);
        let nonzero = m.cmp(c, pred::NE, bits, zero);

        let biased = m.bin(c, Bin::Sub, biased, leading);
        let is_normal = m.cmp(c, pred::SGT, biased, zero);
        let [low, high] = normalize(m, c, [lo, hi], leading);
        let fraction = m.i32(0x000f_ffff);
        let fraction = m.bin(c, Bin::And, high, fraction);
        let shift20 = m.i32(20);
        let exp = m.bin(c, Bin::Shl, biased, shift20);
        let normal_high = m.bin(c, Bin::Or, exp, fraction);

        // Below the normal range, keep the integral subnormal bits and
        // discard the rest. This opcode always rounds toward zero.
        let one = m.i32(1);
        let subnormal_shift = m.bin(c, Bin::Sub, one, biased);
        let [sub_low, sub_high] = truncate(m, c, [low, high], subnormal_shift);
        let result_lo = m.select(c, is_normal, low, sub_low);
        let result_hi = m.select(c, is_normal, normal_high, sub_high);
        [
            m.select(c, nonzero, result_lo, zero),
            m.select(c, nonzero, result_hi, zero),
        ]
    }
}

/// Shift a pair of words without any IR shift count reaching 32.
fn shift_fraction_words(
    m: &mut Module,
    c: usize,
    low: V,
    high: V,
    count: V,
    left: bool,
) -> (V, V) {
    let (near, far, shift, cross) = if left {
        (low, high, Bin::Shl, Bin::LShr)
    } else {
        (high, low, Bin::LShr, Bin::Shl)
    };
    let zero = m.i32(0);
    let bits31 = m.i32(31);
    let bits32 = m.i32(32);
    let large = m.cmp(c, pred::IUGE, count, bits32);
    let offset = m.bin(c, Bin::And, count, bits31);
    let shifted = m.bin(c, shift, near, offset);
    let from_far = m.bin(c, shift, far, offset);
    let carry_shift = m.bin(c, Bin::Sub, zero, offset);
    let carry_shift = m.bin(c, Bin::And, carry_shift, bits31);
    let carry = m.bin(c, cross, near, carry_shift);
    let needs_carry = m.cmp(c, pred::NE, offset, zero);
    let carry = m.select(c, needs_carry, carry, zero);
    let into_far = m.bin(c, Bin::Or, from_far, carry);
    let near = m.select(c, large, zero, shifted);
    let far = m.select(c, large, shifted, into_far);
    if left { (near, far) } else { (far, near) }
}

fn read_window(m: &mut Module, c: usize, table: V, start: V) -> [V; 2] {
    let five = m.i32(5);
    let index = m.bin(c, Bin::LShr, start, five);
    let end = m.i32((TWO_OVER_PI.len() - 1) as u32);
    let [first, second, third] = [0, 1, 2].map(|word| {
        let offset = m.i32(word);
        let index = m.bin(c, Bin::Add, index, offset);
        let past_end = m.cmp(c, pred::IUGT, index, end);
        let index = m.select(c, past_end, end, index);
        let address = m.element(c, table, index);
        m.load(c, address)
    });
    let mask = m.i32(31);
    let inner = m.bin(c, Bin::And, start, mask);
    let upper = funnel_left(m, c, first, second, inner);
    let lower = funnel_left(m, c, second, third, inner);
    let remove = m.i32(11);
    let carry = m.i32(21);
    let hi = m.bin(c, Bin::LShr, upper, remove);
    let upper_lo = m.bin(c, Bin::Shl, upper, carry);
    let lower_lo = m.bin(c, Bin::LShr, lower, remove);
    [m.bin(c, Bin::Or, upper_lo, lower_lo), hi]
}

/// Align two adjacent words by a count in 0..32, keeping the leading word.
fn funnel_left(m: &mut Module, c: usize, lead: V, following: V, inner: V) -> V {
    let zero = m.i32(0);
    let mask = m.i32(31);
    let complement = m.bin(c, Bin::Sub, zero, inner);
    let complement = m.bin(c, Bin::And, complement, mask);
    let cross = m.bin(c, Bin::LShr, following, complement);
    let aligned = m.cmp(c, pred::EQ, inner, zero);
    let cross = m.select(c, aligned, zero, cross);
    let shifted = m.bin(c, Bin::Shl, lead, inner);
    m.bin(c, Bin::Or, shifted, cross)
}

/// Left shift a nonzero 53-bit window to put its leading bit at bit 52.
fn normalize(m: &mut Module, c: usize, [lo, hi]: [V; 2], leading: V) -> [V; 2] {
    let zero = m.i32(0);
    let mask = m.i32(31);
    let word = m.i32(32);
    let inner = m.bin(c, Bin::And, leading, mask);
    let crosses_word = m.cmp(c, pred::IUGE, leading, word);
    let shifted_lo = m.bin(c, Bin::Shl, lo, inner);
    let shifted_hi = funnel_left(m, c, hi, lo, inner);
    [
        m.select(c, crosses_word, zero, shifted_lo),
        m.select(c, crosses_word, shifted_lo, shifted_hi),
    ]
}

/// Shift the normalized window right, returning zero past its last bit.
fn truncate(m: &mut Module, c: usize, [lo, hi]: [V; 2], count: V) -> [V; 2] {
    let zero = m.i32(0);
    let mask = m.i32(31);
    let word = m.i32(32);
    let two_words = m.i32(64);
    let inner = m.bin(c, Bin::And, count, mask);
    let complement = m.bin(c, Bin::Sub, zero, inner);
    let complement = m.bin(c, Bin::And, complement, mask);
    let cross = m.bin(c, Bin::Shl, hi, complement);
    let aligned = m.cmp(c, pred::EQ, inner, zero);
    let cross = m.select(c, aligned, zero, cross);
    let shifted_lo = m.bin(c, Bin::LShr, lo, inner);
    let shifted_lo = m.bin(c, Bin::Or, shifted_lo, cross);
    let shifted_hi = m.bin(c, Bin::LShr, hi, inner);
    let crosses_word = m.cmp(c, pred::IUGE, count, word);
    let past_end = m.cmp(c, pred::IUGE, count, two_words);
    let low = m.select(c, crosses_word, shifted_hi, shifted_lo);
    [
        m.select(c, past_end, zero, low),
        m.select(c, crosses_word, zero, shifted_hi),
    ]
}

#[cfg(test)]
mod tests {
    use super::super::{Parts, lower};
    use crate::emit::Target;
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{ShaderError, dxbc, io, ir, resource, structure};

    fn compile(inst_words: &[u32]) -> Result<super::super::Lowered, ShaderError> {
        let mut words = vec![
            0xBE8C_0302u32, // s_mov_b32 s12, s2
            0xBE8D_0380,    // s_mov_b32 s13, 0
            0xF408_0406,
            0xFA00_0000,    // s_load_dwordx4 s[16:19], s[12:13], 0x0
            0x7E02_0280,    // v_mov_b32 v1, 0
            0x7E04_0280,    // v_mov_b32 v2, 0
            0xBE86_0380,    // s_mov_b32 s6, 0
            0xBE87_0380,    // s_mov_b32 s7, 0
            0xBF8C_C07F,    // s_waitcnt lgkmcnt(0)
        ];
        words.extend_from_slice(inst_words);
        words.extend([
            0x340C_0082, // v_lshlrev_b32 v6, 2, v0
            0xE070_1000,
            0x8004_0406, // buffer_store_dword v4, v6, s[16:19], 0 offen
            0xE070_1000,
            0x8004_0506, // buffer_store_dword v5, v6, s[16:19], 0 offen
            0xBF81_0000, // s_endpgm
        ]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 2,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 2,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 3 << 1,
                threads: [32, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage)?;
        let io = io::raise(&p, &native)?;
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            target: Target::ANY,
            psv: None,
            console_cf: false,
            per_lane_only: false,
        })
    }

    fn sfi0_bits(dxil: &[u8]) -> u64 {
        let container = dxbc::Container::parse(dxil).unwrap();
        let sfi0 = container.part(dxbc::FourCC::SFI0).unwrap();
        u64::from_le_bytes(sfi0.data[..8].try_into().unwrap())
    }

    const fn vop1(op: u16, dst: u8, src: u16) -> u32 {
        0x7E00_0000 | (dst as u32) << 17 | (op as u32) << 9 | (src as u32 & 0x1FF)
    }

    const fn vop3(op: u16, dst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
        [
            0x35 << 26 | (op as u32 & 0x3FF) << 16 | dst as u32,
            (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
        ]
    }

    #[test]
    fn f64_round_fract_and_trig_preop_lower_without_doubles_or_int64() {
        for op in [0x018u16, 0x01a, 0x03e] {
            for src in [257u16, 6u16] {
                let lowered = compile(&[vop1(op, 4, src)]).unwrap();
                assert_eq!(sfi0_bits(&lowered.dxil) & 0x8001, 0, "op {op:#x} src {src}");
            }
        }
        // VOP3 literal with neg / abs / clamp / omod.
        for opcode in [0x198u16, 0x19a, 0x1be] {
            let [w0, w1] = vop3(opcode, 4, 255, 0, 0);
            let lowered = compile(&[w0 | 1 << 8, w1 | 1 << 29, 0x402b_0000]).unwrap();
            assert_eq!(sfi0_bits(&lowered.dxil) & 0x8001, 0, "vop3 {opcode:#x}");
            // Clamp or omod fails closed for ceil/floor/fract.
            assert!(compile(&[w0 | 1 << 15, w1, 0x402b_0000]).is_err());
            assert!(compile(&[w0, w1 | 1 << 27, 0x402b_0000]).is_err());
        }
        // v_trig_preop_f64 across VGPR, SGPR, literal, inline const, and modifiers.
        for (w0, w1, lit) in [
            (vop3(0x174, 4, 257, 128 + 20, 128)[0], vop3(0x174, 4, 257, 128 + 20, 128)[1], None),
            (vop3(0x174, 4, 6, 258, 128)[0], vop3(0x174, 4, 6, 258, 128)[1], None),
            (vop3(0x174, 4, 193, 128, 128)[0], vop3(0x174, 4, 193, 128, 128)[1], None),
            (
                vop3(0x174, 4, 255, 128, 128)[0] | 1 << 15 | 1 << 8,
                vop3(0x174, 4, 255, 128, 128)[1] | 1 << 29 | 2 << 27,
                Some(0x7b00_0000),
            ),
        ] {
            let mut inst = vec![w0, w1];
            if let Some(l) = lit {
                inst.push(l);
            }
            let lowered = compile(&inst).unwrap();
            assert_eq!(sfi0_bits(&lowered.dxil) & 0x8001, 0);
        }
    }
}
