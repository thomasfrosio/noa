/**
 * @file Math.h
 * @brief Various "math related functions".
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

#include <math.h> // I'm not sure cmath is entirely CUDA friendly.
#include <cstdint>
#include <limits>
#include <cfloat> // FLT_EPSILON, DBL_EPSILON

#include "noa/Define.h"
#include "noa/util/traits/BaseTypes.h"

namespace Noa {
    constexpr double PId = 3.1415926535897932384626433832795;
    constexpr float PIf = 3.1415926535897932384626433832795f;
    constexpr double PI2d = 6.283185307179586476925286766559;
    constexpr float PI2f = 6.283185307179586476925286766559f;
    constexpr double PIHALFd = 1.5707963267948966192313216916398;
    constexpr float PIHALFf = 1.5707963267948966192313216916398f;

#ifdef NOA_SINGLE_PRECISION
    constexpr float PI = PIf;
    constexpr float PI2 = PI2f;
    constexpr float PIHALF = PIHALFf;
#else
    constexpr double PI = PId;
    constexpr double PI2 = PI2d;
    constexpr double PIHALF = PIHALFd;
#endif
}

/*
 * math.h: The CUDA math library supports all overloaded versions required by the C++ standard.
 *         It includes the host’s math.h header file and the corresponding device code.
 */

namespace Noa::Math {
    /* --- Trigonometric functions --- */

    /**  Returns the cosine of an angle of x radians. */
    NOA_FHD double cos(double x) { return ::cos(x); }
    NOA_FHD float cos(float x) { return ::cosf(x); }

    /**  Returns the sine of an angle of x radians. */
    NOA_FHD double sin(double x) { return ::sin(x); }
    NOA_FHD float sin(float x) { return ::sinf(x); }

    /**  Returns the tangent of an angle of x radians. */
    NOA_FHD double tan(double x) { return ::tan(x); }
    NOA_FHD float tan(float x) { return ::tanf(x); }

    /** Returns the principal value of the arc cos of x, in radians. */
    NOA_FHD double acos(double x) { return ::acos(x); }
    NOA_FHD float acos(float x) { return ::acosf(x); }

    /** Returns the principal value of the arc sine of x, in radians. */
    NOA_FHD double asin(double x) { return ::asin(x); }
    NOA_FHD float asin(float x) { return ::asinf(x); }

    /** Returns the principal value of the arc tangent of x, in radians. */
    NOA_FHD double atan(double x) { return ::atan(x); }
    NOA_FHD float atan(float x) { return ::atanf(x); }

    /** Returns the principal value of the arc tangent of y/x, in radians. */
    NOA_FHD double atan2(double y, double x) { return ::atan2(y, x); }
    NOA_FHD float atan2(float y, float x) { return ::atan2f(y, x); }

    NOA_FHD constexpr double toDeg(double x) { return x * (180. / PId); }
    NOA_FHD constexpr float toDeg(float x) { return x * (180.f / PIf); }

    NOA_FHD constexpr double toRad(double x) { return x * (PId / 180.); }
    NOA_FHD constexpr float toRad(float x) { return x * (PIf / 180.f); }

    /* --- Hyperbolic functions --- */

    /**  Returns the hyperbolic cosine of x. */
    NOA_FHD double cosh(double x) { return ::cosh(x); }
    NOA_FHD float cosh(float x) { return ::coshf(x); }

    /**  Returns the hyperbolic sine of x. */
    NOA_FHD double sinh(double x) { return ::sinh(x); }
    NOA_FHD float sinh(float x) { return ::sinhf(x); }

    /**  Returns the hyperbolic tangent of x. */
    NOA_FHD double tanh(double x) { return ::tanh(x); }
    NOA_FHD float tanh(float x) { return ::tanhf(x); }

    /** Returns the non-negative area hyperbolic cosine of x. */
    NOA_FHD double acosh(double x) { return ::acosh(x); }
    NOA_FHD float acosh(float x) { return ::acoshf(x); }

    /** Returns the area hyperbolic sine of x. */
    NOA_FHD double asinh(double x) { return ::asinh(x); }
    NOA_FHD float asinh(float x) { return ::asinhf(x); }

    /** Returns the area hyperbolic tangent of x. */
    NOA_FHD double atanh(double x) { return ::atanh(x); }
    NOA_FHD float atanh(float x) { return ::atanhf(x); }

    /* --- Exponential and logarithmic functions --- */

    /** Returns the exponential of @a x. */
    NOA_FHD double exp(double x) { return ::exp(x); }
    NOA_FHD float exp(float x) { return ::expf(x); }

    /** Returns the natural logarithm of @a x. */
    NOA_FHD double log(double x) { return ::log(x); }
    NOA_FHD float log(float x) { return ::logf(x); }

    /** Returns the base 10 logarithm of @a x. */
    NOA_FHD double log10(double x) { return ::log10(x); }
    NOA_FHD float log10(float x) { return ::log10f(x); }

    /** Returns the natural logarithm of one plus @a x. */
    NOA_FHD double log1p(double x) { return ::log1p(x); }
    NOA_FHD float log1p(float x) { return ::log1pf(x); }

    /* --- Power functions --- */

    /** Returns the hypotenuse of a right-angled triangle whose legs are @a x and @a y. */
    NOA_FHD double hypot(double x, double y) { return ::hypot(x, y); }
    NOA_FHD float hypot(float x, float y) { return ::hypotf(x, y); }

    /**  Returns @a base raised to the power @a exponent. */
    NOA_FHD double pow(double base, double exponent) { return ::pow(base, exponent); }
    NOA_FHD float pow(float base, float exponent) { return ::powf(base, exponent); }

    /** Returns the square root of @a x. */
    NOA_FHD double sqrt(double x) { return ::sqrt(x); }
    NOA_FHD float sqrt(float v) { return ::sqrtf(v); }

    /** Returns 1. / sqrt(@a x). */
    NOA_FHD double rsqrt(double v) {
#ifdef __CUDA_ARCH__
        return ::rsqrt(v); // device code trajectory steered by nvcc
#else
        return 1. / ::sqrt(v);
#endif
    }

    NOA_FHD float rsqrt(float v) {
#ifdef __CUDA_ARCH__
        return ::rsqrtf(v); // device code trajectory steered by nvcc
#else
        return 1.f / ::sqrtf(v);
#endif
    }

    template<class T>
    NOA_FHD bool isPowerOf2(T value) { return (value & (value - 1)) == 0; }

    /* --- Rounding and remainder functions --- */

    /** Rounds @a v to nearest integral value. */
    NOA_FHD double round(double v) { return ::round(v); }
    NOA_FHD float round(float v) { return ::roundf(v); }

    /** Rounds @a v to integral value. Should be preferred to round a double to an integer. */
    NOA_FHD double rint(double v) { return ::rint(v); }
    NOA_FHD float rint(float v) { return ::rintf(v); }

    /** Rounds up @a v. */
    NOA_FHD double ceil(double v) { return ::ceil(v); }
    NOA_FHD float ceil(float v) { return ::ceilf(v); }

    /** Rounds down @a v. */
    NOA_FHD double floor(double v) { return ::floor(v); }
    NOA_FHD float floor(float v) { return ::floorf(v); }

    /** Truncates @a v. */
    NOA_FHD double trunc(double v) { return ::trunc(v); }
    NOA_FHD float trunc(float v) { return ::truncf(v); }

    /* --- Floating-point manipulation functions --- */

    /** Returns a value with the magnitude of x and the sign of y. */
    NOA_FHD constexpr double copysign(double x, double y) { return ::copysign(x, y); }
    NOA_FHD constexpr float copysign(float x, float y) { return ::copysign(x, y); }

    struct Limits {
        static constexpr float float_epsilon = FLT_EPSILON;
        static constexpr double double_epsilon = DBL_EPSILON;

        template<class FP>
        NOA_FHD static constexpr std::enable_if_t<std::is_same_v<FP, float> || std::is_same_v<FP, double>, FP> epsilon() {
            if constexpr (std::is_same_v<FP, float>) {
                return float_epsilon;
            } else {
                return double_epsilon;
            }
        }
    };

    /* --- Classification --- */

    /** Returns whether x is a NaN (Not-A-Number) value. */
    NOA_FHD bool isNaN(double v) { return ::isnan(v); }
    NOA_FHD bool isNaN(float v) { return ::isnan(v); }

    /** Returns whether x is an infinity value (either positive infinity or negative infinity). */
    NOA_FHD bool isInf(double v) { return ::isinf(v); }
    NOA_FHD bool isInf(float v) { return ::isinf(v); }

    /** Returns whether x is a finite value (i.e. neither inf nor NaN). */
    NOA_FHD bool isFinite(double v) { return ::isfinite(v); }
    NOA_FHD bool isFinite(float v) { return ::isfinite(v); }

    /** Returns whether x is a normal value (i.e. neither inf, NaN, zero or subnormal. */
    NOA_FHD bool isNormal(double v) { return ::isnormal(v); }
    NOA_FHD bool isNormal(float v) { return ::isnormal(v); }

    /** Returns whether the sign of x is negative. Can be also applied to inf, NaNs and 0s (unsigned is positive). */
    NOA_FHD bool signbit(double v) { return ::signbit(v); }
    NOA_FHD bool signbit(float v) { return ::signbit(v); }

    /* --- Other functions --- */

    /** Returns the absolute value of @a v. */
    NOA_FHD double abs(double v) { return ::abs(v); }
    NOA_FHD float abs(float v) { return ::abs(v); }
    NOA_FHD int8_t abs(int8_t x) { return static_cast<int8_t>(::abs(x)); }
    NOA_FHD int16_t abs(int16_t x) { return static_cast<int16_t>(::abs(x)); }
    NOA_FHD int32_t abs(int32_t x) { return ::abs(x); }
    NOA_FHD int64_t abs(int64_t x) { return ::abs(x); }

    NOA_FHD constexpr double min(double x, double y) { return (y < x) ? y : x; }
    NOA_FHD constexpr float min(float x, float y) { return (y < x) ? y : x; }
    NOA_FHD constexpr int8_t min(int8_t x, int8_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr int16_t min(int16_t x, int16_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr int32_t min(int32_t x, int32_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr int64_t min(int64_t x, int64_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr uint8_t min(uint8_t x, uint8_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr uint16_t min(uint16_t x, uint16_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr uint32_t min(uint32_t x, uint32_t y) { return (y < x) ? y : x; }
    NOA_FHD constexpr uint64_t min(uint64_t x, uint64_t y) { return (y < x) ? y : x; }

    NOA_FHD constexpr double max(double x, double y) { return (y > x) ? y : x; }
    NOA_FHD constexpr float max(float x, float y) { return (y > x) ? y : x; }
    NOA_FHD constexpr int8_t max(int8_t x, int8_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr int16_t max(int16_t x, int16_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr int32_t max(int32_t x, int32_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr int64_t max(int64_t x, int64_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr uint8_t max(uint8_t x, uint8_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr uint16_t max(uint16_t x, uint16_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr uint32_t max(uint32_t x, uint32_t y) { return (y > x) ? y : x; }
    NOA_FHD constexpr uint64_t max(uint64_t x, uint64_t y) { return (y > x) ? y : x; }
}

#define ULP 2
#define EPSILON 1e-6f

namespace Noa::Math {
    /**
     * Whether or not two floating-points are "significantly" equal.
     * @details For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
     *          the values used and multiplied by the desired precision in ULPs. The magnitude is
     *          often set as max(abs(x), abs(y)), but this function is setting the magnitude as
     *          abs(x+y), which is basically equivalent and is should be more efficient. Relative
     *          epsilons and Unit in the Last Place (ULPs) comparisons are usually meaningless for
     *          close-to-zero numbers, hence the absolute comparison with @a epsilon, acting as
     *          a safety net.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    NOA_IHD constexpr bool isEqual(T x, T y, T epsilon = EPSILON) {
        const auto diff = std::abs(x - y);
        if (!Math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (std::abs(x + y) * Limits::epsilon<T>() * ulp);
    }

    /**
     * Whether or not @a x is less or "significantly" equal than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    NOA_IHD constexpr bool isLessOrEqual(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = x - y;
        if (!Math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (std::abs(x + y) * Limits::epsilon<T>() * ulp);
    }

    /**
     * Whether or not @a x is greater or "significantly" equal than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    NOA_IHD constexpr bool isGreaterOrEqual(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = y - x;
        if (!Math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (std::abs(x + y) * Limits::epsilon<T>() * ulp);
    }

    /**
     * Whether or not @a x is "significantly" withing @a min and @a max.
     * @note    If one or all values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    NOA_FHD constexpr bool isWithin(T x, T min, T max, T epsilon = EPSILON) noexcept {
        return isGreaterOrEqual<ulp>(x, min, epsilon) && isLessOrEqual<ulp>(x, max, epsilon);
    }

    /**
     * Whether or not @a x is "significantly" less than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    NOA_IHD constexpr bool isLess(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = y - x;
        if (!Math::isFinite(diff))
            return false;

        return diff > epsilon || diff > (std::abs(x + y) * Limits::epsilon<T>() * ulp);
    }

    /**
     * Whether or not @a x is "significantly" greater than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ulp = ULP, typename T, typename = std::enable_if_t<Noa::Traits::is_float_v<T>>>
    NOA_IHD constexpr bool isGreater(T x, T y, T epsilon = EPSILON) noexcept {
        const auto diff = x - y;
        if (!Math::isFinite(diff))
            return false;

        return diff > epsilon || diff > (std::abs(x + y) * Limits::epsilon<T>() * ulp);
    }
}

#undef ULP
#undef EPSILON
