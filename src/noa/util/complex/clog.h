// Adapted from thrust/complex.h.
// See licence/nvidia_thrust.txt.

#pragma once

#include "noa/util/complex/math_private.h"

// Implementation for Math::log(Complex<double>)
namespace Noa::Math::Details::Complex {

    /* round down to 18 = 54/3 bits */
    NOA_IHD double trim(double x) {
        uint32_t hi;
        get_high_word(hi, x);
        insert_words(x, hi & 0xfffffff8, 0);
        return x;
    }

    NOA_IHD Noa::Complex<double> clog(const Noa::Complex<double>& z) {
        // Adapted from FreeBSDs msun
        double x, y;
        double ax, ay;
        double x0, y0, x1, y1, x2, y2, t, hm1;
        double val[12];
        int i, sorted;
        const double e = 2.7182818284590452354;

        x = z.real();
        y = z.imag();

        // Handle NaNs using the general formula to mix them right.
        if (x != x || y != y)
            return Noa::Complex<double>(Math::log(Math::norm(z)), Math::atan2(y, x));

        ax = Math::abs(x);
        ay = Math::abs(y);
        if (ax < ay) {
            t = ax;
            ax = ay;
            ay = t;
        }

        /*
         * To avoid unnecessary overflow, if x and y are very large, divide x
         * and y by M_E, and then add 1 to the logarithm.  This depends on
         * M_E being larger than sqrt(2).
         * There is a potential loss of accuracy caused by dividing by M_E,
         * but this case should happen extremely rarely.
         */
        //    if (ay > 5e307){
        // For high values of ay -> hypotf(DBL_MAX,ay) = inf
        // We expect that for values at or below ay = 5e307 this should not happen
        if (ay > 5e307) {
            return Noa::Complex<double>(Math::log(Math::hypot(x / e, y / e)) + 1.0, Math::atan2(y, x));
        }
        if (ax == 1.) {
            if (ay < 1e-150)
                return Noa::Complex<double>((ay * 0.5) * ay, Math::atan2(y, x));
            return Noa::Complex<double>(Math::log1p(ay * ay) * 0.5, Math::atan2(y, x));
        }

        /*
         * Because atan2 and hypot conform to C99, this also covers all the
         * edge cases when x or y are 0 or infinite.
         */
        if (ax < 1e-50 || ay < 1e-50 || ax > 1e50 || ay > 1e50)
            return Noa::Complex<double>(Math::log(Math::hypot(x, y)), Math::atan2(y, x));

        /*
         * From this point on, we don't need to worry about underflow or
         * overflow in calculating ax*ax or ay*ay.
         */

        /* Some easy cases. */
        if (ax >= 1.0)
            return Noa::Complex<double>(Math::log1p((ax - 1) * (ax + 1) + ay * ay) * 0.5, Math::atan2(y, x));
        if (ax * ax + ay * ay <= 0.7)
            return Noa::Complex<double>(Math::log(ax * ax + ay * ay) * 0.5, Math::atan2(y, x));

        /*
         * Take extra care so that ULP of real part is small if hypot(x,y) is
         * moderately close to 1.
         */
        x0 = trim(ax);
        ax = ax - x0;
        x1 = trim(ax);
        x2 = ax - x1;
        y0 = trim(ay);
        ay = ay - y0;
        y1 = trim(ay);
        y2 = ay - y1;

        val[0] = x0 * x0;
        val[1] = y0 * y0;
        val[2] = 2 * x0 * x1;
        val[3] = 2 * y0 * y1;
        val[4] = x1 * x1;
        val[5] = y1 * y1;
        val[6] = 2 * x0 * x2;
        val[7] = 2 * y0 * y2;
        val[8] = 2 * x1 * x2;
        val[9] = 2 * y1 * y2;
        val[10] = x2 * x2;
        val[11] = y2 * y2;

        /* Bubble sort. */
        do {
            sorted = 1;
            for (i = 0; i < 11; i++) {
                if (val[i] < val[i + 1]) {
                    sorted = 0;
                    t = val[i];
                    val[i] = val[i + 1];
                    val[i + 1] = t;
                }
            }
        } while (!sorted);

        hm1 = -1;
        for (i = 0; i < 12; i++)
            hm1 += val[i];

        return Noa::Complex<double>(0.5 * Math::log1p(hm1), Math::atan2(y, x));
    }
}

// Implementation for Math::log(Complex<float>)
namespace Noa::Math::Details::Complex {
    /* round down to 8 = 24/3 bits */
    NOA_IHD float trim(float x) {
        uint32_t hx;
        get_float_word(hx, x);
        hx &= 0xffff0000;
        float ret;
        set_float_word(ret, hx);
        return ret;
    }

    NOA_IHD Noa::Complex<float> clogf(const Noa::Complex<float>& z) {
        // Adapted from FreeBSDs msun
        float x, y;
        float ax, ay;
        float x0, y0, x1, y1, x2, y2, t, hm1;
        float val[12];
        int i, sorted;
        const float e = 2.7182818284590452354f;

        x = z.real();
        y = z.imag();

        /* Handle NaNs using the general formula to mix them right. */
        if (x != x || y != y) {
            return Noa::Complex<float>(Math::log(Math::norm(z)), Math::atan2(y, x));
        }

        ax = Math::abs(x);
        ay = Math::abs(y);
        if (ax < ay) {
            t = ax;
            ax = ay;
            ay = t;
        }

        /*
         * To avoid unnecessary overflow, if x and y are very large, divide x
         * and y by M_E, and then add 1 to the logarithm.  This depends on
         * M_E being larger than sqrt(2).
         * There is a potential loss of accuracy caused by dividing by M_E,
         * but this case should happen extremely rarely.
         */
        // For high values of ay -> hypotf(FLT_MAX,ay) = inf
        // We expect that for values at or below ay = 1e34f this should not happen
        if (ay > 1e34f)
            return Noa::Complex<float>(Math::log(Math::hypot(x / e, y / e)) + 1.0f, Math::atan2(y, x));
        if (ax == 1.f) {
            if (ay < 1e-19f)
                return Noa::Complex<float>((ay * 0.5f) * ay, Math::atan2(y, x));
            return Noa::Complex<float>(Math::log1p(ay * ay) * 0.5f, Math::atan2(y, x));
        }

        /*
         * Because atan2 and hypot conform to C99, this also covers all the
         * edge cases when x or y are 0 or infinite.
         */
        if (ax < 1e-6f || ay < 1e-6f || ax > 1e6f || ay > 1e6f)
            return Noa::Complex<float>(Math::log(Math::hypot(x, y)), Math::atan2(y, x));

        /*
         * From this point on, we don't need to worry about underflow or
         * overflow in calculating ax*ax or ay*ay.
         */

        /* Some easy cases. */
        if (ax >= 1.0f)
            return Noa::Complex<float>(Math::log1p((ax - 1.f) * (ax + 1.f) + ay * ay) * 0.5f, Math::atan2(y, x));
        if (ax * ax + ay * ay <= 0.7f)
            return Noa::Complex<float>(Math::log(ax * ax + ay * ay) * 0.5f, Math::atan2(y, x));

        /*
         * Take extra care so that ULP of real part is small if hypot(x,y) is
         * moderately close to 1.
         */
        x0 = trim(ax);
        ax = ax - x0;
        x1 = trim(ax);
        x2 = ax - x1;
        y0 = trim(ay);
        ay = ay - y0;
        y1 = trim(ay);
        y2 = ay - y1;

        val[0] = x0 * x0;
        val[1] = y0 * y0;
        val[2] = 2 * x0 * x1;
        val[3] = 2 * y0 * y1;
        val[4] = x1 * x1;
        val[5] = y1 * y1;
        val[6] = 2 * x0 * x2;
        val[7] = 2 * y0 * y2;
        val[8] = 2 * x1 * x2;
        val[9] = 2 * y1 * y2;
        val[10] = x2 * x2;
        val[11] = y2 * y2;

        /* Bubble sort. */
        do {
            sorted = 1;
            for (i = 0; i < 11; i++) {
                if (val[i] < val[i + 1]) {
                    sorted = 0;
                    t = val[i];
                    val[i] = val[i + 1];
                    val[i + 1] = t;
                }
            }
        } while (!sorted);

        hm1 = -1;
        for (i = 0; i < 12; i++) {
            hm1 += val[i];
        }
        return Noa::Complex<float>(0.5f * Math::log1p(hm1), Math::atan2(y, x));
    }
}

namespace Noa::Math {
    NOA_IHD Complex<double> log(const Complex<double>& z) {
        return Details::Complex::clog(z);
    }

    NOA_IHD Complex<float> log(const Complex<float>& z) {
        return Details::Complex::clogf(z);
    }

    NOA_IHD Complex<double> log10(const Complex<double>& z) {
        return log(z) / 2.30258509299404568402;
    }

    NOA_IHD Complex<float> log10(const Complex<float>& z) {
        return log(z) / 2.30258509299404568402f;
    }
}
