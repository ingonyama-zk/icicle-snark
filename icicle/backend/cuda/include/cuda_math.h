#pragma once

#include <cstdio>
#include <cstdint>
#include <cassert>
#include "icicle/utils/modifiers.h"
#include "icicle/math/storage.h"
#include "ptx.h"

namespace cuda_math {

  template <unsigned OPS_COUNT = UINT32_MAX, bool CARRY_IN = false, bool CARRY_OUT = false>
  struct carry_chain {
    unsigned index;

    constexpr __device__ __forceinline__ carry_chain() : index(0) {}

    __device__ __forceinline__ uint32_t add(const uint32_t x, const uint32_t y)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return ptx::add(x, y);
      else if (index == 1 && !CARRY_IN)
        return ptx::add_cc(x, y);
      else if (index < OPS_COUNT || CARRY_OUT)
        return ptx::addc_cc(x, y);
      else
        return ptx::addc(x, y);
    }

    __device__ __forceinline__ uint32_t sub(const uint32_t x, const uint32_t y)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return ptx::sub(x, y);
      else if (index == 1 && !CARRY_IN)
        return ptx::sub_cc(x, y);
      else if (index < OPS_COUNT || CARRY_OUT)
        return ptx::subc_cc(x, y);
      else
        return ptx::subc(x, y);
    }

    __device__ __forceinline__ uint32_t mad_lo(const uint32_t x, const uint32_t y, const uint32_t z)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return ptx::mad_lo(x, y, z);
      else if (index == 1 && !CARRY_IN)
        return ptx::mad_lo_cc(x, y, z);
      else if (index < OPS_COUNT || CARRY_OUT)
        return ptx::madc_lo_cc(x, y, z);
      else
        return ptx::madc_lo(x, y, z);
    }

    __device__ __forceinline__ uint32_t mad_hi(const uint32_t x, const uint32_t y, const uint32_t z)
    {
      index++;
      if (index == 1 && OPS_COUNT == 1 && !CARRY_IN && !CARRY_OUT)
        return ptx::mad_hi(x, y, z);
      else if (index == 1 && !CARRY_IN)
        return ptx::mad_hi_cc(x, y, z);
      else if (index < OPS_COUNT || CARRY_OUT)
        return ptx::madc_hi_cc(x, y, z);
      else
        return ptx::madc_hi(x, y, z);
    }
  };

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT>
  static constexpr __device__ __forceinline__ uint32_t add_sub_u32(const uint32_t* x, const uint32_t* y, uint32_t* r)
  {
    r[0] = SUBTRACT ? ptx::sub_cc(x[0], y[0]) : ptx::add_cc(x[0], y[0]);
    for (unsigned i = 1; i < NLIMBS; i++)
      r[i] = SUBTRACT ? ptx::subc_cc(x[i], y[i]) : ptx::addc_cc(x[i], y[i]);
    if (!CARRY_OUT) {
      ptx::addc(0, 0);
      return 0;
    }
    return SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0);
  }

  template <unsigned NLIMBS, bool SUBTRACT, bool CARRY_OUT, bool IS_U32 = true>
  static constexpr __device__ __forceinline__ uint32_t
  add_sub_limbs(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t* r = rs.limbs;
    return add_sub_u32<NLIMBS, SUBTRACT, CARRY_OUT>(x, y, r);
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    UNROLL
    for (size_t i = 0; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  mul_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS, size_t start_i = 0)
  {
    UNROLL
    for (size_t i = start_i; i < n; i += 2) {
      acc[i] = ptx::mul_lo(a[i], bi);
      acc[i + 1] = ptx::mul_hi(a[i], bi);
    }
  }

  template <unsigned NLIMBS, bool CARRY_IN = false>
  static __device__ __forceinline__ void
  cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS, uint32_t optional_carry = 0)
  {
    if (CARRY_IN) ptx::add_cc(UINT32_MAX, optional_carry);
    acc[0] = CARRY_IN ? ptx::madc_lo_cc(a[0], bi, acc[0]) : ptx::mad_lo_cc(a[0], bi, acc[0]);
    acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  template <unsigned NLIMBS, bool EVEN_PHASE>
  static __device__ __forceinline__ void cmad_n_msb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    if (EVEN_PHASE) {
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
      acc[1] = ptx::madc_hi_cc(a[0], bi, acc[1]);
    } else {
      acc[1] = ptx::mad_hi_cc(a[0], bi, acc[1]);
    }

    UNROLL
    for (size_t i = 2; i < n; i += 2) {
      acc[i] = ptx::madc_lo_cc(a[i], bi, acc[i]);
      acc[i + 1] = ptx::madc_hi_cc(a[i], bi, acc[i + 1]);
    }
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void cmad_n_lsb(uint32_t* acc, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    if (n > 1)
      acc[0] = ptx::mad_lo_cc(a[0], bi, acc[0]);
    else
      acc[0] = ptx::mad_lo(a[0], bi, acc[0]);

    size_t i;
    UNROLL
    for (i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, acc[i]);
      if (i == n - 2)
        acc[i + 1] = ptx::madc_lo(a[i + 1], bi, acc[i + 1]);
      else
        acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, acc[i + 1]);
    }
    if (i == n - 1) acc[i] = ptx::madc_hi(a[i - 1], bi, acc[i]);
  }

  template <unsigned NLIMBS, bool CARRY_OUT = false, bool CARRY_IN = false>
  static __device__ __forceinline__ uint32_t mad_row(
    uint32_t* odd,
    uint32_t* even,
    const uint32_t* a,
    uint32_t bi,
    size_t n = NLIMBS,
    uint32_t ci = 0,
    uint32_t di = 0,
    uint32_t carry_for_high = 0,
    uint32_t carry_for_low = 0)
  {
    cmad_n<NLIMBS, CARRY_IN>(odd, a + 1, bi, n - 2, carry_for_low);
    odd[n - 2] = ptx::madc_lo_cc(a[n - 1], bi, ci);
    odd[n - 1] = CARRY_OUT ? ptx::madc_hi_cc(a[n - 1], bi, di) : ptx::madc_hi(a[n - 1], bi, di);
    uint32_t cr = CARRY_OUT ? ptx::addc(0, 0) : 0;
    cmad_n<NLIMBS>(even, a, bi, n);
    if (CARRY_OUT) {
      odd[n - 1] = ptx::addc_cc(odd[n - 1], carry_for_high);
      cr = ptx::addc(cr, 0);
    } else
      odd[n - 1] = ptx::addc(odd[n - 1], carry_for_high);
    return cr;
  }

  template <unsigned NLIMBS, bool EVEN_PHASE>
  static __device__ __forceinline__ void
  mad_row_msb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    cmad_n_msb<NLIMBS, !EVEN_PHASE>(odd, EVEN_PHASE ? a : (a + 1), bi, n - 2);
    odd[EVEN_PHASE ? (n - 1) : (n - 2)] = ptx::madc_lo_cc(a[n - 1], bi, 0);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::madc_hi(a[n - 1], bi, 0);
    cmad_n_msb<NLIMBS, EVEN_PHASE>(even, EVEN_PHASE ? (a + 1) : a, bi, n - 1);
    odd[EVEN_PHASE ? n : (n - 1)] = ptx::addc(odd[EVEN_PHASE ? n : (n - 1)], 0);
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  mad_row_lsb(uint32_t* odd, uint32_t* even, const uint32_t* a, uint32_t bi, size_t n = NLIMBS)
  {
    // bi here is constant so we can do a compile-time check for zero (which does happen once for bls12-381 scalar field
    // modulus)
    if (bi != 0) {
      if (n > 1) cmad_n_lsb<NLIMBS>(odd, a + 1, bi, n - 1);
      cmad_n_lsb<NLIMBS>(even, a, bi, n);
    }
    return;
  }

  template <unsigned NLIMBS>
  static __device__ __forceinline__ uint32_t
  mul_n_and_add(uint32_t* acc, const uint32_t* a, const uint32_t bi, const uint32_t* extra, size_t n = (NLIMBS >> 1))
  {
    acc[0] = ptx::mad_lo_cc(a[0], bi, extra[0]);

    UNROLL
    for (size_t i = 1; i < n - 1; i += 2) {
      acc[i] = ptx::madc_hi_cc(a[i - 1], bi, extra[i]);
      acc[i + 1] = ptx::madc_lo_cc(a[i + 1], bi, extra[i + 1]);
    }

    acc[n - 1] = ptx::madc_hi_cc(a[n - 2], bi, extra[n - 1]);
    return ptx::addc(0, 0);
  }

  /**
   * This method multiplies `a` and `b` (both assumed to have NLIMBS / 2 limbs) and adds `in1` and `in2` (NLIMBS limbs
   * each) to the result which is written to `even`.
   *
   * It is used to compute the "middle" part of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} =
   * (a_{hi} - a_{lo})(b_{lo} - b_{hi}) + a_{lo} \cdot b_{lo} + a_{hi} \cdot b_{hi} \f$. Currently this method assumes
   * that the top bit of \f$ a_{hi} \f$ and \f$ b_{hi} \f$ are unset. This ensures correctness by allowing to keep the
   * result inside NLIMBS limbs and ignore the carries from the highest limb.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  multiply_and_add_short_raw(const uint32_t* a, const uint32_t* b, uint32_t* even, uint32_t* in1, uint32_t* in2)
  {
    __align__(16) uint32_t odd[NLIMBS - 2];
    uint32_t first_row_carry = mul_n_and_add<NLIMBS>(even, a, b[0], in1);
    uint32_t carry = mul_n_and_add<NLIMBS>(odd, a + 1, b[0], &in2[1]);

    size_t i;
    UNROLL
    for (i = 2; i < ((NLIMBS >> 1) - 1); i += 2) {
      carry = mad_row<NLIMBS, true, false>(
        &even[i], &odd[i - 2], a, b[i - 1], NLIMBS >> 1, in1[(NLIMBS >> 1) + i - 2], in1[(NLIMBS >> 1) + i - 1], carry);
      carry = mad_row<NLIMBS, true, false>(
        &odd[i], &even[i], a, b[i], NLIMBS >> 1, in2[(NLIMBS >> 1) + i - 1], in2[(NLIMBS >> 1) + i], carry);
    }
    mad_row<NLIMBS, false, true>(
      &even[NLIMBS >> 1], &odd[(NLIMBS >> 1) - 2], a, b[(NLIMBS >> 1) - 1], NLIMBS >> 1, in1[NLIMBS - 2],
      in1[NLIMBS - 1], carry, first_row_carry);
    // merge |even| and |odd| plus the parts of `in2` we haven't added yet (first and last limbs)
    even[0] = ptx::add_cc(even[0], in2[0]);
    for (i = 0; i < (NLIMBS - 2); i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], in2[i + 1]);
  }

  /**
   * This method multiplies `a` and `b` and writes the result into `even`. It assumes that `a` and `b` are NLIMBS/2
   * limbs long. The usual schoolbook algorithm is used.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void multiply_short_raw(const uint32_t* a, const uint32_t* b, uint32_t* even)
  {
    __align__(16) uint32_t odd[NLIMBS - 2];
    mul_n<NLIMBS>(even, a, b[0], NLIMBS >> 1);
    mul_n<NLIMBS>(odd, a + 1, b[0], NLIMBS >> 1);
    mad_row<NLIMBS>(&even[2], &odd[0], a, b[1], NLIMBS >> 1);

    size_t i;
    UNROLL
    for (i = 2; i < ((NLIMBS >> 1) - 1); i += 2) {
      mad_row<NLIMBS>(&odd[i], &even[i], a, b[i], NLIMBS >> 1);
      mad_row<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1], NLIMBS >> 1);
    }
    // merge |even| and |odd|
    even[1] = ptx::add_cc(even[1], odd[0]);
    for (i = 1; i < NLIMBS - 2; i++)
      even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
    even[i + 1] = ptx::addc(even[i + 1], 0);
  }

  /**
   * This method multiplies `as` and `bs` and writes the (wide) result into `rs`.
   *
   * It is assumed that the highest bits of `as` and `bs` are unset which is true for all the numbers icicle had to deal
   * with so far. This method implements [subtractive
   * Karatsuba](https://en.wikipedia.org/wiki/Karatsuba_algorithm#Implementation).
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  multiply_raw(const storage<NLIMBS>& as, const storage<NLIMBS>& bs, storage<2 * NLIMBS>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = bs.limbs;
    uint32_t* r = rs.limbs;
    if constexpr (NLIMBS > 2) {
      // Next two lines multiply high and low halves of operands (\f$ a_{lo} \cdot b_{lo}; a_{hi} \cdot b_{hi} \$f) and
      // write the results into `r`.
      multiply_short_raw<NLIMBS>(a, b, r);
      multiply_short_raw<NLIMBS>(&a[NLIMBS >> 1], &b[NLIMBS >> 1], &r[NLIMBS]);
      __align__(16) uint32_t middle_part[NLIMBS];
      __align__(16) uint32_t diffs[NLIMBS];
      // Differences of halves \f$ a_{hi} - a_{lo}; b_{lo} - b_{hi} \$f are written into `diffs`, signs written to
      // `carry1` and `carry2`.
      uint32_t carry1 = add_sub_u32<(NLIMBS >> 1), true, true>(&a[NLIMBS >> 1], a, diffs);
      uint32_t carry2 = add_sub_u32<(NLIMBS >> 1), true, true>(b, &b[NLIMBS >> 1], &diffs[NLIMBS >> 1]);
      // Compute the "middle part" of Karatsuba: \f$ a_{lo} \cdot b_{hi} + b_{lo} \cdot a_{hi} \f$.
      // This is where the assumption about unset high bit of `a` and `b` is relevant.
      multiply_and_add_short_raw<NLIMBS>(diffs, &diffs[NLIMBS >> 1], middle_part, r, &r[NLIMBS]);
      // Corrections that need to be performed when differences are negative.
      // Again, carry doesn't need to be propagated due to unset high bits of `a` and `b`.
      if (carry1)
        add_sub_u32<(NLIMBS >> 1), true, false>(
          &middle_part[NLIMBS >> 1], &diffs[NLIMBS >> 1], &middle_part[NLIMBS >> 1]);
      if (carry2) add_sub_u32<(NLIMBS >> 1), true, false>(&middle_part[NLIMBS >> 1], diffs, &middle_part[NLIMBS >> 1]);
      // Now that middle part is fully correct, it can be added to the result.
      add_sub_u32<NLIMBS, false, true>(&r[NLIMBS >> 1], middle_part, &r[NLIMBS >> 1]);

      // Carry from adding middle part has to be propagated to the highest limb.
      for (size_t i = NLIMBS + (NLIMBS >> 1); i < 2 * NLIMBS; i++)
        r[i] = ptx::addc_cc(r[i], 0);
    } else if (NLIMBS == 2) {
      auto a_128b = static_cast<__uint128_t>(*(uint64_t*)(as.limbs));
      auto b_64b = *(uint64_t*)bs.limbs;
      __uint128_t r_128b = a_128b * b_64b; // 64b * 64b --> 128b
      r[0] = r_128b;
      r[1] = r_128b >> 32;
      r[2] = r_128b >> 64;
      r[3] = r_128b >> 96;

    } else if (NLIMBS == 1) {
      r[0] = ptx::mul_lo(a[0], b[0]);
      r[1] = ptx::mul_hi(a[0], b[0]);
    }
  }

  /**
   * A function that computes wide product \f$ rs = as \cdot bs \f$ that's correct for the higher NLIMBS + 1 limbs with
   * a small maximum error.
   *
   * The way this function saves computations (as compared to regular school-book multiplication) is by not including
   * terms that are too small. Namely, limb product \f$ a_i \cdot b_j \f$ is excluded if \f$ i + j < NLIMBS - 2 \f$ and
   * only the higher half is included if \f$ i + j = NLIMBS - 2 \f$. All other limb products are included. So, the error
   * i.e. difference between true product and the result of this function written to `rs` is exactly the sum of all
   * dropped limbs products, which we can bound: \f$ a_0 \cdot b_0 + 2^{32}(a_0 \cdot b_1 + a_1 \cdot b_0) + \dots +
   * 2^{32(NLIMBS - 3)}(a_{NLIMBS - 3} \cdot b_0 + \dots + a_0 \cdot b_{NLIMBS - 3}) + 2^{32(NLIMBS -
   * 2)}(\floor{\frac{a_{NLIMBS - 2} \cdot b_0}{2^{32}}} + \dots + \floor{\frac{a_0 \cdot b_{NLIMBS - 2}}{2^{32}}}) \leq
   * 2^{64} + 2\cdot 2^{96} + \dots + (NLIMBS - 2) \cdot 2^{32(NLIMBS - 1)} + (NLIMBS - 1) \cdot 2^{32(NLIMBS - 1)} \leq
   * 2(NLIMBS - 1) \cdot 2^{32(NLIMBS - 1)}\f$.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void
  multiply_msb_raw(const storage<NLIMBS>& as, const storage<NLIMBS>& bs, storage<2 * NLIMBS>& rs)
  {
    if constexpr (NLIMBS > 1) {
      const uint32_t* a = as.limbs;
      const uint32_t* b = bs.limbs;
      uint32_t* even = rs.limbs;
      __align__(16) uint32_t odd[2 * NLIMBS - 2];

      even[NLIMBS - 1] = ptx::mul_hi(a[NLIMBS - 2], b[0]);
      odd[NLIMBS - 2] = ptx::mul_lo(a[NLIMBS - 1], b[0]);
      odd[NLIMBS - 1] = ptx::mul_hi(a[NLIMBS - 1], b[0]);
      size_t i;
      UNROLL
      for (i = 2; i < NLIMBS - 1; i += 2) {
        mad_row_msb<NLIMBS, true>(&even[NLIMBS - 2], &odd[NLIMBS - 2], &a[NLIMBS - i - 1], b[i - 1], i + 1);
        mad_row_msb<NLIMBS, false>(&odd[NLIMBS - 2], &even[NLIMBS - 2], &a[NLIMBS - i - 2], b[i], i + 2);
      }
      mad_row<NLIMBS>(&even[NLIMBS], &odd[NLIMBS - 2], a, b[NLIMBS - 1]);

      // merge |even| and |odd|
      ptx::add_cc(even[NLIMBS - 1], odd[NLIMBS - 2]);
      for (i = NLIMBS - 1; i < 2 * NLIMBS - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], 0);
    } else {
      multiply_raw<NLIMBS>(as, bs, rs);
    }
  }

  /**
   * A function that computes the low half of the fused multiply-and-add \f$ rs = as \cdot bs + cs \f$ where
   * \f$ bs = 2^{32*nof_limbs} \f$.
   *
   * For efficiency, this method does not include terms that are too large. Namely, limb product \f$ a_i \cdot b_j \f$
   * is excluded if \f$ i + j > NLIMBS - 1 \f$ and only the lower half is included if \f$ i + j = NLIMBS - 1 \f$. All
   * other limb products are included.
   */
  template <unsigned NLIMBS>
  static __device__ __forceinline__ void multiply_and_add_lsb_neg_modulus_raw(
    const storage<NLIMBS>& as, const storage<NLIMBS>& neg_mod, const storage<NLIMBS>& cs, storage<NLIMBS>& rs)
  {
    const uint32_t* a = as.limbs;
    const uint32_t* b = neg_mod.limbs;
    const uint32_t* c = cs.limbs;
    uint32_t* even = rs.limbs;

    if constexpr (NLIMBS > 2) {
      __align__(16) uint32_t odd[NLIMBS - 1];
      size_t i;
      // `b[0]` is \f$ 2^{32} \f$ minus the last limb of prime modulus. Because most scalar (and some base) primes
      // are necessarily NTT-friendly, `b[0]` often turns out to be \f$ 2^{32} - 1 \f$. This actually leads to
      // less efficient SASS generated by nvcc, so this case needed separate handling.
      if (b[0] == UINT32_MAX) {
        add_sub_u32<NLIMBS, true, false>(c, a, even);
        for (i = 0; i < NLIMBS - 1; i++)
          odd[i] = a[i];
      } else {
        mul_n_and_add<NLIMBS>(even, a, b[0], c, NLIMBS);
        mul_n<NLIMBS>(odd, a + 1, b[0], NLIMBS - 1);
      }
      mad_row_lsb<NLIMBS>(&even[2], &odd[0], a, b[1], NLIMBS - 1);
      UNROLL
      for (i = 2; i < NLIMBS - 1; i += 2) {
        mad_row_lsb<NLIMBS>(&odd[i], &even[i], a, b[i], NLIMBS - i);
        mad_row_lsb<NLIMBS>(&even[i + 2], &odd[i], a, b[i + 1], NLIMBS - i - 1);
      }

      // merge |even| and |odd|
      even[1] = ptx::add_cc(even[1], odd[0]);
      for (i = 1; i < NLIMBS - 2; i++)
        even[i + 1] = ptx::addc_cc(even[i + 1], odd[i]);
      even[i + 1] = ptx::addc(even[i + 1], odd[i]);
    } else if (NLIMBS == 2) {
      uint64_t res = *(uint64_t*)a * *(uint64_t*)b + *(uint64_t*)c;
      even[0] = res;
      even[1] = res >> 32;

    } else if (NLIMBS == 1) {
      even[0] = ptx::mad_lo(a[0], b[0], c[0]);
    }
  }

  /**
   * @brief Return upper/lower half of x (into r).
   * @tparam NLIMBS Number of 32bit limbs in r.
   * @tparam HIGHER what half to be returned. default is false=lower.
   * @param x 2*NLIMBS sized multiprecision input.
   * @param r NLIMBS sized multiprecision upper/lower half output.
   */
  template <unsigned NLIMBS, bool HIGHER = false>
  static __device__ __forceinline__ void get_half_32(const uint32_t* x, uint32_t* r)
  {
    for (unsigned i = 0; i < NLIMBS; i++) {
      r[i] = x[HIGHER ? i + NLIMBS : i];
    }
  }

  template <unsigned NLIMBS, unsigned SLACK_BITS>
  static constexpr __device__ __forceinline__ void
  get_higher_with_slack(const storage<2 * NLIMBS>& xs, storage<NLIMBS>& out)
  {
    UNROLL
    for (unsigned i = 0; i < NLIMBS; i++) {
      out.limbs[i] = __funnelshift_lc(xs.limbs[i + NLIMBS - 1], xs.limbs[i + NLIMBS], 2 * SLACK_BITS);
    }
  }

  /**
   * This method reduces a Wide number `xs` modulo `p` and returns the result as a Field element.
   *
   * It is assumed that the high `2 * slack_bits` bits of `xs` are unset which is always the case for the product of 2
   * numbers with their high `slack_bits` unset. Larger Wide numbers should be reduced by subtracting an appropriate
   * factor of `modulus_squared` first.
   *
   * This function implements ["multi-precision Barrett"](https://github.com/ingonyama-zk/modular_multiplication). As
   * opposed to Montgomery reduction, it doesn't require numbers to have a special representation but lets us work with
   * them as-is. The general idea of Barrett reduction is to estimate the quotient \f$ l \approx \floor{\frac{xs}{p}}
   * \f$ and return \f$ xs - l \cdot p \f$. But since \f$ l \f$ is inevitably computed with an error (it's always less
   * or equal than the real quotient). So the modulus `p` might need to be subtracted several times before the result is
   * in the desired range \f$ [0;p-1] \f$. The estimate of the error is as follows: \f[ \frac{xs}{p} - l = \frac{xs}{p}
   * - \frac{xs \cdot m}{2^{2n}} + \frac{xs \cdot m}{2^{2n}} - \floor{\frac{xs}{2^k}}\frac{m}{2^{2n-k}}
   *  + \floor{\frac{xs}{2^k}}\frac{m}{2^{2n-k}} - l \leq p^2(\frac{1}{p}-\frac{m}{2^{2n}}) + \frac{m}{2^{2n-k}} + 2(TLC
   * - 1) \cdot 2^{-32} \f] Here \f$ l \f$ is the result of [multiply_msb_raw](@ref multiply_msb_raw) function and the
   * last term in the error is due to its approximation. \f$ n \f$ is the number of bits in \f$ p \f$ and \f$ k = 2n -
   * 32\cdot TLC \f$. Overall, the error is always less than 2 so at most 2 reductions are needed. However, in most
   * cases it's less than 1, so setting the [num_of_reductions](@ref num_of_reductions) variable for a field equal to 1
   * will cause only 1 reduction to be performed.
   */
  template <unsigned NLIMBS, unsigned SLACK_BITS, unsigned NOF_REDUCTIONS>
  static constexpr __device__ __forceinline__ storage<NLIMBS> barrett_reduce(
    const storage<2 * NLIMBS>& xs,
    const storage<NLIMBS>& ms,
    const storage<NLIMBS>& mod1,
    const storage<NLIMBS>& mod2,
    const storage<NLIMBS>& neg_mod)
  {
    storage<2 * NLIMBS> l = {}; // the approximation of l for a*b = l*p + r mod p
    storage<NLIMBS> r = {};

    // `xs` is left-shifted by `2 * slack_bits` and higher half is written to `xs_hi`
    storage<NLIMBS> xs_hi = {};
    get_higher_with_slack<NLIMBS, SLACK_BITS>(xs, xs_hi);
    multiply_msb_raw<NLIMBS>(xs_hi, ms, l); // MSB mult by `m`.
    // Note: taking views is zero copy but unsafe
    storage<NLIMBS> l_hi = {};
    storage<NLIMBS> xs_lo = {};
    get_half_32<NLIMBS, true>(l.limbs, l_hi.limbs);
    get_half_32<NLIMBS, false>(xs.limbs, xs_lo.limbs);
    // Here we need to compute the lsb of `xs - l \cdot p` and to make use of fused multiply-and-add, we rewrite it as
    // `xs + l \cdot (2^{32 \cdot TLC}-p)` which is the same as original (up to higher limbs which we don't care about).
    multiply_and_add_lsb_neg_modulus_raw(l_hi, neg_mod, xs_lo, r);
    // As mentioned, either 2 or 1 reduction can be performed depending on the field in question.
    if constexpr (NOF_REDUCTIONS == 2) {
      storage<NLIMBS> r_reduced = {};
      const auto borrow = add_sub_limbs<NLIMBS, true, true>(r, mod2, r_reduced);
      // If r-2p has no borrow then we are done
      if (!borrow) return r_reduced;
    }
    // if r-2p has borrow then we need to either subtract p or we are already in [0,p).
    // so we subtract p and based on the borrow bit we know which case it is
    storage<NLIMBS> r_reduced = {};
    const auto borrow = add_sub_limbs<NLIMBS, true, true>(r, mod1, r_reduced);
    return borrow ? r : r_reduced;
  }

  template <unsigned NLIMBS>
  static constexpr __device__ __forceinline__ bool is_equal(const storage<NLIMBS>& xs, const storage<NLIMBS>& ys)
  {
    const uint32_t* x = xs.limbs;
    const uint32_t* y = ys.limbs;
    uint32_t limbs_or = x[0] ^ y[0];
    UNROLL
    for (unsigned i = 1; i < NLIMBS; i++)
      limbs_or |= x[i] ^ y[i];
    return limbs_or == 0;
  }

  template <unsigned NLIMBS, unsigned BITS>
  static constexpr __device__ __forceinline__ storage<NLIMBS> right_shift(const storage<NLIMBS>& xs)
  {
    if constexpr (BITS == 0)
      return xs;
    else {
      constexpr unsigned BITS32 = BITS % 32;
      constexpr unsigned LIMBS_GAP = BITS / 32;
      storage<NLIMBS> out{};
      if constexpr (LIMBS_GAP < NLIMBS - 1) {
        for (unsigned i = 0; i < NLIMBS - LIMBS_GAP - 1; i++)
          out.limbs[i] = (xs.limbs[i + LIMBS_GAP] >> BITS32) + (xs.limbs[i + LIMBS_GAP + 1] << (32 - BITS32));
      }
      if constexpr (LIMBS_GAP < NLIMBS) out.limbs[NLIMBS - LIMBS_GAP - 1] = (xs.limbs[NLIMBS - 1] >> BITS32);
      return out;
    }
  }
  // this function checks if the given index is within the array range
  static constexpr __device__ __forceinline__ void index_err(uint32_t index, uint32_t max_index)
  {
    if (index > max_index) {
      printf("CUDA ERROR: field.h: index out of range: given index - %u > max index - %u", index, max_index);
      assert(false);
    }
  }
  // Assumes the number is even!
  template <unsigned NLIMBS>
  static constexpr __device__ __forceinline__ void div2(const storage<NLIMBS>& xs, storage<NLIMBS>& rs)
  {
    const uint32_t* x = xs.limbs;
    uint32_t* r = rs.limbs;
    if constexpr (NLIMBS > 1) {
      UNROLL
      for (unsigned i = 0; i < NLIMBS - 1; i++) {
        r[i] = __funnelshift_rc(x[i], x[i + 1], 1);
      }
    }
    r[NLIMBS - 1] = x[NLIMBS - 1] >> 1;
  }
} // namespace cuda_math
