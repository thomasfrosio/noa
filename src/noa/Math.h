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

#include "noa/Definitions.h"
#include "noa/util/traits/BaseTypes.h"

namespace Noa::Math {
    /// Some constants.
    template<typename T>
    struct Constants {
        static constexpr T PI = static_cast<T>(3.1415926535897932384626433832795);
        static constexpr T PI2 = static_cast<T>(6.283185307179586476925286766559);
        static constexpr T PIHALF = static_cast<T>(1.5707963267948966192313216916398);
    };

    /// Some limits.
    template<typename T>
    struct Limits {
        NOA_FHD static constexpr T epsilon() {
            if constexpr (std::is_same_v<T, float>) {
                return FLT_EPSILON;
            } else if constexpr (std::is_same_v<T, double>) {
                return DBL_EPSILON;
            } else {
                static_assert(Noa::Traits::always_false_v<T>);
            }
        }
    };
}

/*
 * math.h: The CUDA math library supports most overloaded versions required by the C++ standard.
 *         It includes the host’s math.h header file and the corresponding device code.
 */

namespace Noa::Math {
    /* --- Trigonometric functions --- */

    /// Returns the cosine of an angle of x radians.
    template<typename T> NOA_HD T cos(T x);
    template<> NOA_FHD double cos<double>(double x) { return ::cos(x); }
    template<> NOA_FHD float cos<float>(float x) { return ::cosf(x); }

    /// Returns the sine of an angle of x radians.
    template<typename T> NOA_HD T sin(T x);
    template<> NOA_FHD double sin<double>(double x) { return ::sin(x); }
    template<> NOA_FHD float sin<float>(float x) { return ::sinf(x); }

    /// Returns the tangent of an angle of x radians.
    template<typename T> NOA_HD T tan(T x);
    template<> NOA_FHD double tan<double>(double x) { return ::tan(x); }
    template<> NOA_FHD float tan<float>(float x) { return ::tanf(x); }

    /// Returns the principal value of the arc cos of x, in radians.
    template<typename T> NOA_HD T acos(T x);
    template<> NOA_FHD double acos<double>(double x) { return ::acos(x); }
    template<> NOA_FHD float acos<float>(float x) { return ::acosf(x); }

    /// Returns the principal value of the arc sine of x, in radians.
    template<typename T> NOA_HD T asin(T x);
    template<> NOA_FHD double asin<double>(double x) { return ::asin(x); }
    template<> NOA_FHD float asin<float>(float x) { return ::asinf(x); }

    /// Returns the principal value of the arc tangent of x, in radians.
    template<typename T> NOA_HD T atan(T x);
    template<> NOA_FHD double atan<double>(double x) { return ::atan(x); }
    template<> NOA_FHD float atan<float>(float x) { return ::atanf(x); }

    /// Returns the principal value of the arc tangent of y/x, in radians.
    template<typename T> NOA_HD T atan2(T y, T x);
    template<> NOA_FHD double atan2<double>(double y, double x) { return ::atan2(y, x); }
    template<> NOA_FHD float atan2<float>(float y, float x) { return ::atan2f(y, x); }

    template<typename T> NOA_HD T toDeg(T x);
    template<> NOA_FHD constexpr double toDeg<double>(double x) { return x * (180. / Constants<double>::PI); }
    template<> NOA_FHD constexpr float toDeg<float>(float x) { return x * (180.f / Constants<float>::PI); }

    template<typename T> NOA_HD T toRad(T x);
    template<> NOA_FHD constexpr double toRad<double>(double x) { return x * (Constants<double>::PI / 180.); }
    template<> NOA_FHD constexpr float toRad<float>(float x) { return x * (Constants<float>::PI / 180.f); }

    /* --- Hyperbolic functions --- */

    ///  Returns the hyperbolic cosine of x.
    template<typename T> NOA_HD T cosh(T x);
    template<> NOA_FHD double cosh<double>(double x) { return ::cosh(x); }
    template<> NOA_FHD float cosh<float>(float x) { return ::coshf(x); }

    ///  Returns the hyperbolic sine of x.
    template<typename T> NOA_HD T sinh(T x);
    template<> NOA_FHD double sinh<double>(double x) { return ::sinh(x); }
    template<> NOA_FHD float sinh<float>(float x) { return ::sinhf(x); }

    ///  Returns the hyperbolic tangent of x.
    template<typename T> NOA_HD T tanh(T x);
    template<> NOA_FHD double tanh<double>(double x) { return ::tanh(x); }
    template<> NOA_FHD float tanh<float>(float x) { return ::tanhf(x); }

    /// Returns the non-negative area hyperbolic cosine of x.
    template<typename T> NOA_HD T acosh(T x);
    template<> NOA_FHD double acosh<double>(double x) { return ::acosh(x); }
    template<> NOA_FHD float acosh<float>(float x) { return ::acoshf(x); }

    /// Returns the area hyperbolic sine of x.
    template<typename T> NOA_HD T asinh(T x);
    template<> NOA_FHD double asinh<double>(double x) { return ::asinh(x); }
    template<> NOA_FHD float asinh<float>(float x) { return ::asinhf(x); }

    /// Returns the area hyperbolic tangent of x.
    template<typename T> NOA_HD T atanh(T x);
    template<> NOA_FHD double atanh<double>(double x) { return ::atanh(x); }
    template<> NOA_FHD float atanh<float>(float x) { return ::atanhf(x); }

    /* --- Exponential and logarithmic functions --- */

    /// Returns the exponential of @a x.
    template<typename T> NOA_HD T exp(T x);
    template<> NOA_FHD double exp<double>(double x) { return ::exp(x); }
    template<> NOA_FHD float exp<float>(float x) { return ::expf(x); }

    /// Returns the natural logarithm of @a x.
    template<typename T> NOA_HD T log(T x);
    template<> NOA_FHD double log<double>(double x) { return ::log(x); }
    template<> NOA_FHD float log<float>(float x) { return ::logf(x); }

    /// Returns the base 10 logarithm of @a x.
    template<typename T> NOA_HD T log10(T x);
    template<> NOA_FHD double log10<double>(double x) { return ::log10(x); }
    template<> NOA_FHD float log10<float>(float x) { return ::log10f(x); }

    /// Returns the natural logarithm of one plus @a x.
    template<typename T> NOA_HD T log1p(T x);
    template<> NOA_FHD double log1p<double>(double x) { return ::log1p(x); }
    template<> NOA_FHD float log1p<float>(float x) { return ::log1pf(x); }

    /* --- Power functions --- */

    /// Returns the hypotenuse of a right-angled triangle whose legs are @a x and @a y.
    template<typename T> NOA_HD T hypot(T x, T y);
    template<> NOA_FHD double hypot<double>(double x, double y) { return ::hypot(x, y); }
    template<> NOA_FHD float hypot<float>(float x, float y) { return ::hypotf(x, y); }

    ///  Returns @a base raised to the power @a exponent.
    template<typename T> NOA_HD T pow(T base, T exponent);
    template<> NOA_FHD double pow<double>(double base, double exponent) { return ::pow(base, exponent); }
    template<> NOA_FHD float pow<float>(float base, float exponent) { return ::powf(base, exponent); }

    /// Returns the next power of 2.
    /// @warning If @a x is a power of 2 or is equal to 1, returns x.
    template<typename T>
    T nextPowerOf2(T x) {
        static_assert(std::is_integral_v<T>);
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    template<typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
    NOA_FHD constexpr T nextMultipleOf(T value, T base) { return (value + base - 1) / base * base; }

    template<class T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
    NOA_FHD constexpr bool isPowerOf2(T value) { return (value & (value - 1)) == 0; }

    /// Returns the square root of @a x.
    template<typename T> T NOA_HD sqrt(T x);
    template<> NOA_FHD double sqrt<double>(double x) { return ::sqrt(x); }
    template<> NOA_FHD float sqrt<float>(float x) { return ::sqrtf(x); }

    /// Returns 1. / sqrt(@a x).
    template<typename T> T NOA_HD rsqrt(T x);
    template<> NOA_FHD double rsqrt<double>(double x) {
#ifdef __CUDA_ARCH__
        return ::rsqrt(x);
#else
        return 1. / ::sqrt(x);
#endif
    }

    template<> NOA_FHD float rsqrt<float>(float x) {
#ifdef __CUDA_ARCH__
        return ::rsqrtf(x);
#else
        return 1.f / ::sqrtf(x);
#endif
    }

    /* --- Rounding and remainder functions --- */

    /// Rounds @a x to nearest integral value.
    template<typename T> NOA_HD T round(T x);
    template<> NOA_FHD double round<double>(double x) { return ::round(x); }
    template<> NOA_FHD float round<float>(float x) { return ::roundf(x); }

    /// Rounds @a x to integral value. Should be preferred to round a double to an integer.
    template<typename T> NOA_HD T rint(T x);
    template<> NOA_FHD double rint<double>(double x) { return ::rint(x); }
    template<> NOA_FHD float rint<float>(float x) { return ::rintf(x); }

    /// Rounds up @a x.
    template<typename T> NOA_HD T ceil(T x);
    template<> NOA_FHD double ceil<double>(double x) { return ::ceil(x); }
    template<> NOA_FHD float ceil<float>(float x) { return ::ceilf(x); }

    /// Rounds down @a x.
    template<typename T> NOA_HD T floor(T x);
    template<> NOA_FHD double floor<double>(double x) { return ::floor(x); }
    template<> NOA_FHD float floor<float>(float x) { return ::floorf(x); }

    /// Truncates @a x.
    template<typename T> NOA_HD T trunc(T x);
    template<> NOA_FHD double trunc<double>(double x) { return ::trunc(x); }
    template<> NOA_FHD float trunc<float>(float x) { return ::truncf(x); }

    /* --- Floating-point manipulation functions --- */

    /// Returns a value with the magnitude of x and the sign of y.
    template<typename T> NOA_HD T copysign(T x, T y);
    template<> NOA_FHD double copysign<double>(double x, double y) { return ::copysign(x, y); }
    template<> NOA_FHD float copysign<float>(float x, float y) { return ::copysign(x, y); }

    /* --- Classification --- */

    /// Returns whether x is a NaN (Not-A-Number) value.
    template<typename T> NOA_HD bool isNaN(T x);
    template<> NOA_FHD constexpr bool isNaN<double>(double x) { return ::isnan(x); }
    template<> NOA_FHD constexpr bool isNaN<float>(float x) { return ::isnan(x); }

    /// Returns whether x is an infinity value (either positive infinity or negative infinity).
    template<typename T> NOA_HD bool isInf(T x);
    template<> NOA_FHD constexpr bool isInf<double>(double x) { return ::isinf(x); }
    template<> NOA_FHD constexpr bool isInf<float>(float x) { return ::isinf(x); }

    /// Returns whether x is a finite value (i.e. neither inf nor NaN).
    template<typename T> NOA_HD bool isFinite(T x);
    template<> NOA_FHD constexpr bool isFinite<double>(double x) { return ::isfinite(x); }
    template<> NOA_FHD constexpr bool isFinite<float>(float x) { return ::isfinite(x); }

    /// Returns whether x is a normal value (i.e. neither inf, NaN, zero or subnormal.
    template<typename T> NOA_HOST bool isNormal(T x);
    template<> NOA_FH constexpr bool isNormal<double>(double x) { return ::isnormal(x); }
    template<> NOA_FH constexpr bool isNormal<float>(float x) { return ::isnormal(x); }
    // ::isnormal is not a device function, but constexpr __host__. Requires --expr-relaxed-constexpr.
    // Since it is not currently used, remove it from device code: NOA_FHD to NOA_FH.

    /// Returns whether the sign of x is negative. Can be also applied to inf, NaNs and 0s (unsigned is positive).
    template<typename T> NOA_HD bool signbit(T x);
    template<> NOA_FHD constexpr bool signbit<double>(double x) { return ::signbit(x); }
    template<> NOA_FHD constexpr bool signbit<float>(float x) { return ::signbit(x); }

    /* --- Other functions --- */

    /// Returns the absolute value of @a v.
    template<typename T> NOA_FHD T abs(T x) { return ::abs(x); }
    template<> [[nodiscard]] NOA_FHD int8_t abs<int8_t>(int8_t x) { return static_cast<int8_t>(::abs(x)); }
    template<> [[nodiscard]] NOA_FHD int16_t abs<int16_t>(int16_t x) { return static_cast<int16_t>(::abs(x)); }

    template<typename T> [[nodiscard]] NOA_FHD constexpr T min(T x, T y) { return (y < x) ? y : x; }
    template<typename T> [[nodiscard]] NOA_FHD constexpr T max(T x, T y) { return (y > x) ? y : x; }
    template<typename T> [[nodiscard]] NOA_FHD constexpr T clamp(T val, T low, T high) {
        return min(high, max(val, low));
    }

    /// Returns the centered index of the corresponding non-centered @a idx. Should be within 0 <= idx < dim.
    template<typename T, typename = std::enable_if_t<Noa::Traits::is_int_v<T>>>
    [[nodiscard]] NOA_FHD constexpr T FFTShift(T idx, T dim) {
        return (idx < (dim + 1) / 2) ? idx + dim / 2 : idx - (dim + 1) / 2; // or (idx + dim / 2) % dim
    }

    /// Returns the non-centered index of the corresponding centered @a idx. Should be within 0 <= idx < dim.
    template<typename T, typename = std::enable_if_t<Noa::Traits::is_int_v<T>>>
    [[nodiscard]] NOA_FHD constexpr T iFFTShift(T idx, T dim) {
        return (idx < dim / 2) ? idx + (dim + 1) / 2 : idx - dim / 2; // or (idx + (dim + 1) / 2) % dim
    }

    /* --- Floating-point comparisons --- */

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
    template<uint32_t ULP, typename T>
    NOA_IHD constexpr bool isEqual(T x, T y, T epsilon) {
        static_assert(Noa::Traits::is_float_v<T>);
        const auto diff = std::abs(x - y);
        if (!Math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (std::abs(x + y) * Limits<T>::epsilon() * ULP);
    }

    template<typename T>
    NOA_IHD constexpr bool isEqual(T x, T y) { return isEqual<4>(x, y, 1e-6f); }

    /**
     * Whether or not @a x is less or "significantly" equal than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ULP, typename T>
    NOA_IHD constexpr bool isLessOrEqual(T x, T y, T epsilon) noexcept {
        static_assert(Noa::Traits::is_float_v<T>);
        const auto diff = x - y;
        if (!Math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (std::abs(x + y) * Limits<T>::epsilon() * ULP);
    }

    template<typename T>
    NOA_IHD constexpr bool isLessOrEqual(T x, T y) { return isLessOrEqual<4>(x, y, 1e-6f); }

    /**
     * Whether or not @a x is greater or "significantly" equal than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ULP, typename T>
    NOA_IHD constexpr bool isGreaterOrEqual(T x, T y, T epsilon) noexcept {
        static_assert(Noa::Traits::is_float_v<T>);
        const auto diff = y - x;
        if (!Math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (std::abs(x + y) * Limits<T>::epsilon() * ULP);
    }

    template<typename T>
    NOA_IHD constexpr bool isGreaterOrEqual(T x, T y) { return isGreaterOrEqual<4>(x, y, 1e-6f); }

    /**
     * Whether or not @a x is "significantly" withing @a min and @a max.
     * @note    If one or all values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ULP, typename T>
    NOA_FHD constexpr bool isWithin(T x, T min, T max, T epsilon) noexcept {
        static_assert(Noa::Traits::is_float_v<T>);
        return isGreaterOrEqual<ULP>(x, min, epsilon) && isLessOrEqual<ULP>(x, max, epsilon);
    }

    template<typename T>
    NOA_FHD constexpr bool isWithin(T x, T min, T max) noexcept { return isWithin<4>(x, min, max, 1e-6f); }

    /**
     * Whether or not @a x is "significantly" less than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ULP, typename T>
    NOA_IHD constexpr bool isLess(T x, T y, T epsilon) noexcept {
        static_assert(Noa::Traits::is_float_v<T>);
        const auto diff = y - x;
        if (!Math::isFinite(diff))
            return false;

        return diff > epsilon || diff > (std::abs(x + y) * Limits<T>::epsilon() * ULP);
    }

    template<typename T>
    NOA_FHD constexpr bool isLess(T x, T y) noexcept { return isLess<4>(x, y, 1e-6f); }

    /**
     * Whether or not @a x is "significantly" greater than @a y.
     * @note    If one or both values are NaN and|or +/-Inf, returns false.
     */
    template<uint32_t ULP, typename T>
    NOA_IHD constexpr bool isGreater(T x, T y, T epsilon) noexcept {
        static_assert(Noa::Traits::is_float_v<T>);
        const auto diff = x - y;
        if (!Math::isFinite(diff))
            return false;

        return diff > epsilon || diff > (std::abs(x + y) * Limits<T>::epsilon() * ULP);
    }

    template<typename T>
    NOA_FHD constexpr bool isGreater(T x, T y) noexcept { return isGreater<4>(x, y, 1e-6f); }
}
