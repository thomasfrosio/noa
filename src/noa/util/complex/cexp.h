// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include <cstdint>

#include "noa/util/complex/math_private.h"

// Implementation for Math::exp(Complex<double>)
namespace Noa::Math::Details::Complex {
    /*
     * Compute exp(x), scaled to avoid spurious overflow.  An exponent is
     * returned separately in 'expt'.
     *
     * Input:  ln(DBL_MAX) <= x < ln(2 * DBL_MAX / DBL_MIN_DENORM) ~= 1454.91
     * Output: 2**1023 <= y < 2**1024
     */
    NOA_IHD double frexp_exp(double x, int* expt) {
        const uint32_t k = 1799; // constant for reduction
        const double kln2 = 1246.97177782734161156; // k * ln2

        double exp_x;
        uint32_t hx;

        // We use exp(x) = exp(x - kln2) * 2**k, carefully chosen to
        // minimize |exp(kln2) - 2**k|. We also scale the exponent of
        // exp_x to MAX_EXP so that the result can be multiplied by
        // a tiny number without losing accuracy due to denormalization.
        exp_x = Math::exp(x - kln2);
        get_high_word(hx, exp_x);
        *expt = static_cast<int>((hx >> 20) - (0x3ff + 1023) + k);
        set_high_word(exp_x, (hx & 0xfffff) | ((0x3ff + 1023) << 20));
        return exp_x;
    }

    NOA_IHD Noa::Complex<double> ldexp_cexp(Noa::Complex<double> z, int expt) {
        double x, y, exp_x, scale1, scale2;
        int ex_expt, half_expt;

        x = z.real();
        y = z.imag();
        exp_x = frexp_exp(x, &ex_expt);
        expt += ex_expt;

        // Arrange so that scale1 * scale2 == 2**expt.  We use this to
        // compensate for scalbn being horrendously slow.
        half_expt = expt / 2;
        insert_words(scale1, static_cast<uint32_t>((0x3ff + half_expt) << 20), 0);
        half_expt = expt - half_expt;
        insert_words(scale2, static_cast<uint32_t>((0x3ff + half_expt) << 20), 0);

        return Noa::Complex<double>(cos(y) * exp_x * scale1 * scale2,
                                    sin(y) * exp_x * scale1 * scale2);
    }

    NOA_IHD Noa::Complex<double> cexp(const Noa::Complex<double>& z) {
        double x, y, exp_x;
        uint32_t hx, hy, lx, ly;

        const uint32_t exp_ovfl = 0x40862e42; // high bits of MAX_EXP * ln2 ~= 710
        const uint32_t cexp_ovfl = 0x4096b8e4; // (MAX_EXP - MIN_DENORM_EXP) * ln2

        x = z.real();
        y = z.imag();

        extract_words(hy, ly, y);
        hy &= 0x7fffffff;

        // cexp(x + I 0) = exp(x) + I 0
        if ((hy | ly) == 0)
            return Noa::Complex<double>(exp(x), y);
        extract_words(hx, lx, x);
        // cexp(0 + I y) = cos(y) + I sin(y)
        if (((hx & 0x7fffffff) | lx) == 0)
            return Noa::Complex<double>(cos(y), sin(y));

        if (hy >= 0x7ff00000) {
            if (lx != 0 || (hx & 0x7fffffff) != 0x7ff00000) {
                return Noa::Complex<double>(y - y, y - y); // cexp(finite|NaN +- I Inf|NaN) = NaN + I NaN
            } else if (hx & 0x80000000) {
                return Noa::Complex<double>(0.0, 0.0); // cexp(-Inf +- I Inf|NaN) = 0 + I 0
            } else {
                return Noa::Complex<double>(x, y - y); // cexp(+Inf +- I Inf|NaN) = Inf + I NaN
            }
        }

        if (hx >= exp_ovfl && hx <= cexp_ovfl) {
            return ldexp_cexp(z, 0); // x is between 709.7 and 1454.3, so we must scale to avoid overflow in exp(x).
        } else {
            /*
             * Cases covered here:
             *  -  x < exp_ovfl and exp(x) won't overflow (common case)
             *  -  x > cexp_ovfl, so exp(x) * s overflows for all s > 0
             *  -  x = +-Inf (generated by exp())
             *  -  x = NaN (spurious inexact exception from y)
             */
            exp_x = Math::exp(x);
            return Noa::Complex<double>(exp_x * cos(y), exp_x * sin(y));
        }
    }
}

// Implementation for Math::exp(Complex<float>)
namespace Noa::Math::Details::Complex {
    NOA_IHD float frexp_expf(float x, int* expt) {
        const uint32_t k = 235;                 /* constant for reduction */
        const float kln2 = 162.88958740F;       /* k * ln2 */

        // should this be a double instead?
        float exp_x;
        uint32_t hx;

        exp_x = Math::exp(x - kln2);
        get_float_word(hx, exp_x);
        *expt = static_cast<int>((hx >> 23) - (0x7f + 127) + k);
        set_float_word(exp_x, (hx & 0x7fffff) | ((0x7f + 127) << 23));
        return exp_x;
    }

    NOA_IHD Noa::Complex<float> ldexp_cexpf(Noa::Complex<float> z, int expt) {
        float x, y, exp_x, scale1, scale2;
        int ex_expt, half_expt;

        x = z.real();
        y = z.imag();
        exp_x = frexp_expf(x, &ex_expt);
        expt += ex_expt;

        half_expt = expt / 2;
        set_float_word(scale1, static_cast<uint32_t>((0x7f + half_expt) << 23));
        half_expt = expt - half_expt;
        set_float_word(scale2, static_cast<uint32_t>((0x7f + half_expt) << 23));

        return Noa::Complex<float>(std::cos(y) * exp_x * scale1 * scale2,
                                   std::sin(y) * exp_x * scale1 * scale2);
    }

    NOA_IHD Noa::Complex<float> cexpf(const Noa::Complex<float>& z) {
        float x, y, exp_x;
        uint32_t hx, hy;

        const uint32_t exp_ovfl = 0x42b17218; // MAX_EXP * ln2 ~= 88.722839355
        const uint32_t cexp_ovfl = 0x43400074; // (MAX_EXP - MIN_DENORM_EXP) * ln2

        x = z.real();
        y = z.imag();

        get_float_word(hy, y);
        hy &= 0x7fffffff;

        /* cexp(x + I 0) = exp(x) + I 0 */
        if (hy == 0)
            return (Noa::Complex<float>(Math::exp(x), y));
        get_float_word(hx, x);
        /* cexp(0 + I y) = cos(y) + I sin(y) */
        if ((hx & 0x7fffffff) == 0) {
            return (Noa::Complex<float>(Math::cos(y), Math::sin(y)));
        }
        if (hy >= 0x7f800000) {
            if ((hx & 0x7fffffff) != 0x7f800000) {
                return (Noa::Complex<float>(y - y, y - y)); // cexp(finite|NaN +- I Inf|NaN) = NaN + I NaN
            } else if (hx & 0x80000000) {
                return (Noa::Complex<float>(0.0, 0.0)); // cexp(-Inf +- I Inf|NaN) = 0 + I 0
            } else {
                return (Noa::Complex<float>(x, y - y)); // cexp(+Inf +- I Inf|NaN) = Inf + I NaN
            }
        }

        if (hx >= exp_ovfl && hx <= cexp_ovfl) {
            return ldexp_cexpf(z, 0); // x is between 88.7 and 192, so we must scale to avoid overflow in expf(x).
        } else {
            /*
             * Cases covered here:
             *  -  x < exp_ovfl and exp(x) won't overflow (common case)
             *  -  x > cexp_ovfl, so exp(x) * s overflows for all s > 0
             *  -  x = +-Inf (generated by exp())
             *  -  x = NaN (spurious inexact exception from y)
             */
            exp_x = Math::exp(x);
            return (Noa::Complex<float>(exp_x * Math::cos(y), exp_x * Math::sin(y)));
        }
    }
}

namespace Noa::Math {
    NOA_FHD Complex<double> exp(const Complex<double>& z) {
        return Noa::Math::Details::Complex::cexp(z);
    }

    NOA_FHD Complex<float> exp(const Complex<float>& z) {
        return Noa::Math::Details::Complex::cexpf(z);
    }
}