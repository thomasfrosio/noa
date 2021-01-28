// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include "noa/util/Complex.h"
#include "noa/util/complex/math_private.h"

// Implementation for Math::sqrt(Complex<double>)
namespace Noa::Math::Details::Complex {
    NOA_DH inline Noa::Complex<double> csqrt(const Noa::Complex<double>& z) {
        Noa::Complex<double> result;
        double a, b;
        double t;
        int scale;

        /* We risk spurious overflow for components >= DBL_MAX / (1 + sqrt(2)). */
        const double THRESH = 7.446288774449766337959726e+307;

        a = z.real();
        b = z.imag();

        /* Handle special cases. */
        if (z == 0.0)
            return Noa::Complex<double>(0.0, b);
        if (Math::isInf(b))
            return Noa::Complex<double>(infinity<double>(), b);
        if (Math::isNaN(a)) {
            t = (b - b) / (b - b);    /* raise invalid if b is not a NaN */
            return Noa::Complex<double>(a, t);    /* return NaN + NaN i */
        }
        if (Math::isInf(a)) {
            /*
             * csqrt(inf + NaN i)  = inf +  NaN i
             * csqrt(inf + y i)    = inf +  0 i
             * csqrt(-inf + NaN i) = NaN +- inf i
             * csqrt(-inf + y i)   = 0   +  inf i
             */
            if (Math::signbit(a))
                return (Noa::Complex<double>(Math::abs(b - b), Math::copysign(a, b)));
            else
                return (Noa::Complex<double>(a, Math::copysign(b - b, b)));
        }
        /*
         * The remaining special case (b is NaN) is handled just fine by
         * the normal code path below.
         */

        // DBL_MIN*2
        const double low_thresh = 4.450147717014402766180465e-308;
        scale = 0;

        if (Math::abs(a) >= THRESH || Math::abs(b) >= THRESH) {
            /* Scale to avoid overflow. */
            a *= 0.25;
            b *= 0.25;
            scale = 1;
        } else if (Math::abs(a) <= low_thresh && Math::abs(b) <= low_thresh) {
            /* Scale to avoid underflow. */
            a *= 4.0;
            b *= 4.0;
            scale = 2;
        }

        /* Algorithm 312, CACM vol 10, Oct 1967. */
        if (a >= 0.0) {
            t = Math::sqrt((a + Math::hypot(a, b)) * 0.5);
            result = Noa::Complex<double>(t, b / (2 * t));
        } else {
            t = Math::sqrt((-a + Math::hypot(a, b)) * 0.5);
            result = Noa::Complex<double>(Math::abs(b) / (2 * t), Math::copysign(t, b));
        }

        /* Rescale. */
        if (scale == 1)
            return result * 2.0;
        else if (scale == 2)
            return result * 0.5;
        else
            return result;
    }
}

// Implementation for Math::sqrt(Complex<float>)
namespace Noa::Math::Details::Complex {
    NOA_DH inline Noa::Complex<float> csqrtf(const Noa::Complex<float>& z) {
        float a = z.real(), b = z.imag();
        float t;
        int scale;
        Noa::Complex<float> result;

        /* We risk spurious overflow for components >= FLT_MAX / (1 + sqrt(2)). */
        const float THRESH = 1.40949553037932e+38f;

        /* Handle special cases. */
        if (z == 0.0f)
            return Noa::Complex<float>(0, b);
        if (Math::isInf(b))
            return Noa::Complex<float>(infinity<float>(), b);
        if (Math::isNaN(a)) {
            t = (b - b) / (b - b);    /* raise invalid if b is not a NaN */
            return Noa::Complex<float>(a, t);    /* return NaN + NaN i */
        }
        if (Math::isInf(a)) {
            /*
             * csqrtf(inf + NaN i)  = inf +  NaN i
             * csqrtf(inf + y i)    = inf +  0 i
             * csqrtf(-inf + NaN i) = NaN +- inf i
             * csqrtf(-inf + y i)   = 0   +  inf i
             */
            if (Math::signbit(a))
                return Noa::Complex<float>(Math::abs(b - b), Math::copysign(a, b));
            else
                return Noa::Complex<float>(a, Math::copysign(b - b, b));
        }
        /*
         * The remaining special case (b is NaN) is handled just fine by
         * the normal code path below.
         */

        /*
         * Unlike in the FreeBSD code we'll avoid using double precision as
         * not all hardware supports it.
         */

        // FLT_MIN*2
        const float low_thresh = 2.35098870164458e-38f;
        scale = 0;

        if (Math::abs(a) >= THRESH || Math::abs(b) >= THRESH) {
            /* Scale to avoid overflow. */
            a *= 0.25f;
            b *= 0.25f;
            scale = 1;
        } else if (Math::abs(a) <= low_thresh && Math::abs(b) <= low_thresh) {
            /* Scale to avoid underflow. */
            a *= 4.f;
            b *= 4.f;
            scale = 2;
        }

        /* Algorithm 312, CACM vol 10, Oct 1967. */
        if (a >= 0.0f) {
            t = Math::sqrt((a + Math::hypot(a, b)) * 0.5f);
            result = Noa::Complex<float>(t, b / (2.0f * t));
        } else {
            t = Math::sqrt((-a + Math::hypot(a, b)) * 0.5f);
            result = Noa::Complex<float>(Math::abs(b) / (2.0f * t), Math::copysign(t, b));
        }

        /* Rescale. */
        if (scale == 1)
            return result * 2.0f;
        else if (scale == 2)
            return result * 0.5f;
        else
            return result;
    }
}

namespace Noa::Math {
    NOA_DH inline Complex<double> sqrt(const Complex<double>& z) {
        return Details::Complex::csqrt(z);
    }

    NOA_DH inline Complex<float> sqrt(const Complex<float>& z) {
        return Details::Complex::csqrtf(z);
    }
}
