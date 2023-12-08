#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Constant.hpp"
#include "noa/core/math/Generic.hpp"

#if defined(NOA_IS_OFFLINE)
#include <algorithm> // clamp
#endif

namespace noa {
    [[nodiscard]] NOA_FHD constexpr bool is_nan(double x) { return std::isnan(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_nan(float x) { return std::isnan(x); }

    [[nodiscard]] NOA_FHD constexpr bool is_inf(double x) { return std::isinf(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_inf(float x) { return std::isinf(x); }

    [[nodiscard]] NOA_FHD constexpr bool is_finite(double x) { return std::isfinite(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_finite(float x) { return std::isfinite(x); }

    [[nodiscard]] NOA_FH constexpr bool is_normal(double x) { return std::isnormal(x); }
    [[nodiscard]] NOA_FH constexpr bool is_normal(float x) { return std::isnormal(x); }
    // ::isnormal is not a device function, but constexpr __host__. Requires --expr-relaxed-constexpr.
    // Since it is not currently used, remove it from device code.

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T min(T x, T y) { return (y < x) ? y : x; }

    template<typename T>
    [[nodiscard]] NOA_IH constexpr T min(std::initializer_list<T> list) { return std::min(list); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T max(T x, T y) { return (y > x) ? y : x; }

    template<typename T>
    [[nodiscard]] NOA_IH constexpr T max(std::initializer_list<T> list) { return std::max(list); }

    template<typename T>
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
    template<int32_t ULP, typename Real, typename = std::enable_if_t<nt::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool allclose(Real x, Real y, Real epsilon) {
        const Real diff(abs(x - y));
        if (!is_finite(diff))
            return false;

        constexpr auto THRESHOLD = std::numeric_limits<Real>::epsilon() * static_cast<Real>(ULP);
        return diff <= epsilon || diff <= (max(abs(x), abs(y)) * THRESHOLD);
    }

    template<typename Real, typename = std::enable_if_t<nt::is_real_v<Real>>>
    [[nodiscard]] NOA_IHD constexpr bool allclose(Real x, Real y) {
        return allclose<4>(x, y, static_cast<Real>(1e-6));
    }
}
