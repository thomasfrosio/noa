#pragma once

#include <math.h> // I'm not sure cmath is entirely CUDA friendly.
#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/math/Constant.h"
#include "noa/common/math/Generic.h"

namespace noa::math {
    /// Returns whether x is a NaN (Not-A-Number) value.
    NOA_FHD constexpr bool isNaN(double x) { return ::isnan(x); }
    NOA_FHD constexpr bool isNaN(float x) { return ::isnan(x); }

    /// Returns whether x is an infinity value (either positive infinity or negative infinity).
    NOA_FHD constexpr bool isInf(double x) { return ::isinf(x); }
    NOA_FHD constexpr bool isInf(float x) { return ::isinf(x); }

    /// Returns whether x is a finite value (i.e. neither inf nor NaN).
    NOA_FHD constexpr bool isFinite(double x) { return ::isfinite(x); }
    NOA_FHD constexpr bool isFinite(float x) { return ::isfinite(x); }

    /// Returns whether x is a normal value (i.e. neither inf, NaN, zero or subnormal.
    NOA_FH constexpr bool isNormal(double x) { return ::isnormal(x); }
    NOA_FH constexpr bool isNormal(float x) { return ::isnormal(x); }
    // ::isnormal is not a device function, but constexpr __host__. Requires --expr-relaxed-constexpr.
    // Since it is not currently used, remove it from device code.

    template<typename T>
    NOA_FHD constexpr T min(T x, T y) { return (y < x) ? y : x; }
    template<typename T>
    NOA_IH constexpr T min(std::initializer_list<T> list) { return std::min(list); }
    template<typename T>
    NOA_FHD constexpr T max(T x, T y) { return (y > x) ? y : x; }
    template<typename T>
    NOA_IH constexpr T max(std::initializer_list<T> list) { return std::max(list); }
    template<typename T>
    NOA_FHD constexpr T clamp(T val, T low, T high) {
    #ifdef __CUDA_ARCH__
        return min(high, max(val, low));
    #else
        return std::clamp(val, low, high);
    #endif
    }

    /// Whether or not two floating-points are "significantly" equal.
    /// \details For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
    ///          the values used and multiplied by the desired precision in ULPs. Relative epsilons
    ///          and Unit in the Last Place (ULPs) comparisons are usually meaningless for close-to-zero
    ///          numbers, hence the absolute comparison with \a epsilon, acting as a safety net.
    /// \note    If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename T>
    NOA_IHD constexpr bool isEqual(T x, T y, T epsilon) {
        static_assert(noa::traits::is_float_v<T>);
        const T diff(math::abs(x - y));
        if (!math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (math::max(math::abs(x), math::abs(y)) * Limits<T>::epsilon() * T(ULP));
    }

    template<typename T>
    NOA_IHD constexpr bool isEqual(T x, T y) { return isEqual<4>(x, y, static_cast<T>(1e-6)); }

    /// Whether or not \a x is less or "significantly" equal than \a y.
    /// \note    If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename T>
    NOA_IHD constexpr bool isLessOrEqual(T x, T y, T epsilon) noexcept {
        static_assert(noa::traits::is_float_v<T>);
        const T diff(x - y);
        if (!math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (math::max(math::abs(x), math::abs(y)) * Limits<T>::epsilon() * T(ULP));
    }

    template<typename T>
    NOA_IHD constexpr bool isLessOrEqual(T x, T y) { return isLessOrEqual<4>(x, y, static_cast<T>(1e-6)); }

    /// Whether or not \a x is greater or "significantly" equal than \a y.
    /// \note    If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename T>
    NOA_IHD constexpr bool isGreaterOrEqual(T x, T y, T epsilon) noexcept {
        static_assert(noa::traits::is_float_v<T>);
        const T diff(y - x);
        if (!math::isFinite(diff))
            return false;

        return diff <= epsilon || diff <= (math::max(math::abs(x), math::abs(y)) * Limits<T>::epsilon() * T(ULP));
    }

    template<typename T>
    NOA_IHD constexpr bool isGreaterOrEqual(T x, T y) { return isGreaterOrEqual<4>(x, y, static_cast<T>(1e-6)); }

    /// Whether or not \a x is "significantly" withing \a min and \a max.
    /// \note    If one or all values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename T>
    NOA_FHD constexpr bool isWithin(T x, T min, T max, T epsilon) noexcept {
        static_assert(noa::traits::is_float_v<T>);
        return isGreaterOrEqual<ULP>(x, min, epsilon) && isLessOrEqual<ULP>(x, max, epsilon);
    }

    template<typename T>
    NOA_FHD constexpr bool isWithin(T x, T min, T max) noexcept {
        return isWithin<4>(x, min, max, static_cast<T>(1e-6));
    }

    /// Whether or not \a x is "significantly" less than \a y.
    /// \note    If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename T>
    NOA_IHD constexpr bool isLess(T x, T y, T epsilon) noexcept {
        static_assert(noa::traits::is_float_v<T>);
        const T diff(y - x);
        if (!math::isFinite(diff))
            return false;

        return diff > epsilon || diff > (math::max(math::abs(x), math::abs(y)) * Limits<T>::epsilon() * T(ULP));
    }

    template<typename T>
    NOA_FHD constexpr bool isLess(T x, T y) noexcept { return isLess<4>(x, y, static_cast<T>(1e-6)); }

    /// Whether or not \a x is "significantly" greater than \a y.
    /// \note    If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename T>
    NOA_IHD constexpr bool isGreater(T x, T y, T epsilon) noexcept {
        static_assert(noa::traits::is_float_v<T>);
        const T diff(x - y);
        if (!math::isFinite(diff))
            return false;

        return diff > epsilon || diff > (math::max(math::abs(x), math::abs(y)) * Limits<T>::epsilon() * T(ULP));
    }

    template<typename T>
    NOA_FHD constexpr bool isGreater(T x, T y) noexcept { return isGreater<4>(x, y, static_cast<T>(1e-6)); }
}
