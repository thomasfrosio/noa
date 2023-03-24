#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa::math {
    [[nodiscard]] NOA_FHD constexpr bool is_nan(double x) { return ::isnan(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_nan(float x) { return ::isnan(x); }

    [[nodiscard]] NOA_FHD constexpr bool is_inf(double x) { return ::isinf(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_inf(float x) { return ::isinf(x); }

    [[nodiscard]] NOA_FHD constexpr bool is_finite(double x) { return ::isfinite(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_finite(float x) { return ::isfinite(x); }

    [[nodiscard]] NOA_FH constexpr bool is_normal(double x) { return ::isnormal(x); }
    [[nodiscard]] NOA_FH constexpr bool is_normal(float x) { return ::isnormal(x); }
    // ::isnormal is not a device function, but constexpr __host__. Requires --expr-relaxed-constexpr.
    // Since it is not currently used, remove it from device code.

    template<typename T>
    constexpr bool is_valid_min_max_v =
            noa::traits::is_scalar_v<T> ||
            noa::traits::is_detected_convertible_v<bool, noa::traits::has_greater_operator, T>;

    template<typename T, std::enable_if_t<is_valid_min_max_v<T>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr T min(T x, T y) { return (y < x) ? y : x; }

    template<typename T, std::enable_if_t<is_valid_min_max_v<T>, bool> = true>
    [[nodiscard]] NOA_IH constexpr T min(std::initializer_list<T> list) { return std::min(list); }

    template<typename T, std::enable_if_t<is_valid_min_max_v<T>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr T max(T x, T y) { return (y > x) ? y : x; }

    template<typename T, std::enable_if_t<is_valid_min_max_v<T>, bool> = true>
    [[nodiscard]] NOA_IH constexpr T max(std::initializer_list<T> list) { return std::max(list); }

    template<typename T, std::enable_if_t<is_valid_min_max_v<T>, bool> = true>
    [[nodiscard]] NOA_FHD constexpr T clamp(T val, T low, T high) {
    #ifdef __CUDA_ARCH__
        return min(high, max(val, low));
    #else
        return std::clamp(val, low, high);
    #endif
    }

    // Whether two floating-points are "significantly" equal.
    // For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
    // the values used and multiplied by the desired precision in ULPs. Relative epsilons
    // and Unit in the Last Place (ULPs) comparisons are usually meaningless for close-to-zero
    // numbers, hence the absolute comparison with epsilon, acting as a safety net.
    // If one or both values are NaN and|or +/-Inf, returns false.
    template<int32_t ULP, typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool are_almost_equal(Real x, Real y, Real epsilon) {
        const Real diff(math::abs(x - y));
        if (!math::is_finite(diff))
            return false;

        constexpr auto THRESHOLD = Limits<Real>::epsilon() * static_cast<Real>(ULP);
        return diff <= epsilon || diff <= (math::max(math::abs(x), math::abs(y)) * THRESHOLD);
    }

    template<typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool are_almost_equal(Real x, Real y) {
        return are_almost_equal<4>(x, y, static_cast<Real>(1e-6));
    }

    // Whether x is less or "significantly" equal than y.
    // If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool is_almost_leq(Real x, Real y, Real epsilon) noexcept {
        const Real diff(x - y);
        if (!math::is_finite(diff))
            return false;

        constexpr auto THRESHOLD = Limits<Real>::epsilon() * static_cast<Real>(ULP);
        return diff <= epsilon || diff <= (math::max(math::abs(x), math::abs(y)) * THRESHOLD);
    }

    template<typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool is_almost_leq(Real x, Real y) {
        return is_almost_leq<4>(x, y, static_cast<Real>(1e-6));
    }

    // Whether x is greater or "significantly" equal than y.
    // If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool is_almost_geq(Real x, Real y, Real epsilon) noexcept {
        const Real diff(y - x);
        if (!math::is_finite(diff))
            return false;

        constexpr auto THRESHOLD = Limits<Real>::epsilon() * static_cast<Real>(ULP);
        return diff <= epsilon || diff <= (math::max(math::abs(x), math::abs(y)) * THRESHOLD);
    }

    template<typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool is_almost_geq(Real x, Real y) {
        return is_almost_geq<4>(x, y, static_cast<Real>(1e-6));
    }

    // Whether x is "significantly" withing min and max.
    // If one or all values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_FHD constexpr bool is_almost_within(Real x, Real min, Real max, Real epsilon) noexcept {
        return is_almost_geq<ULP>(x, min, epsilon) && is_almost_leq<ULP>(x, max, epsilon);
    }

    template<typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_FHD constexpr bool is_almost_within(Real x, Real min, Real max) noexcept {
        return is_almost_within<4>(x, min, max, static_cast<Real>(1e-6));
    }

    // Whether x is "significantly" less than y.
    // If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool is_almost_less(Real x, Real y, Real epsilon) noexcept {
        const Real diff(y - x);
        if (!math::is_finite(diff))
            return false;

        constexpr auto THRESHOLD = Limits<Real>::epsilon() * static_cast<Real>(ULP);
        return diff > epsilon || diff > (math::max(math::abs(x), math::abs(y)) * THRESHOLD);
    }

    template<typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_FHD constexpr bool is_almost_less(Real x, Real y) noexcept {
        return is_almost_less<4>(x, y, static_cast<Real>(1e-6));
    }

    // Whether x is "significantly" greater than y.
    // If one or both values are NaN and|or +/-Inf, returns false.
    template<uint ULP, typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool is_almost_greater(Real x, Real y, Real epsilon) noexcept {
        const Real diff(x - y);
        if (!math::is_finite(diff))
            return false;

        constexpr auto THRESHOLD = Limits<Real>::epsilon() * static_cast<Real>(ULP);
        return diff > epsilon || diff > (math::max(math::abs(x), math::abs(y)) * THRESHOLD);
    }

    template<typename Real, typename = std::enable_if_t<noa::traits::is_real_v<Real>>>
    [[nodiscard]] NOA_FHD constexpr bool is_almost_greater(Real x, Real y) noexcept {
        return is_almost_greater<4>(x, y, static_cast<Real>(1e-6));
    }
}
