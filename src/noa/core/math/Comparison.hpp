#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"

#ifdef NOA_IS_OFFLINE
#include <algorithm> // clamp
#endif

namespace noa {
    [[nodiscard]] NOA_FHD constexpr bool is_nan(double x) noexcept { return std::isnan(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_nan(float x) noexcept { return std::isnan(x); }

    [[nodiscard]] NOA_FHD constexpr bool is_inf(double x) noexcept { return std::isinf(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_inf(float x) noexcept { return std::isinf(x); }

    [[nodiscard]] NOA_FHD constexpr bool is_finite(double x) noexcept { return std::isfinite(x); }
    [[nodiscard]] NOA_FHD constexpr bool is_finite(float x) noexcept { return std::isfinite(x); }

    [[nodiscard]] NOA_FH constexpr bool is_normal(double x) noexcept { return std::isnormal(x); }
    [[nodiscard]] NOA_FH constexpr bool is_normal(float x) noexcept { return std::isnormal(x); }

    template<nt::scalar T>
    [[nodiscard]] NOA_FHD constexpr T min(T x, T y) noexcept { return (y < x) ? y : x; }

    template<typename T>
    [[nodiscard]] NOA_IH constexpr T min(std::initializer_list<T> list) noexcept { return std::min(list); }

    template<nt::scalar T>
    [[nodiscard]] NOA_FHD constexpr T max(T x, T y) noexcept { return (y > x) ? y : x; }

    template<typename T>
    [[nodiscard]] NOA_IH constexpr T max(std::initializer_list<T> list) noexcept { return std::max(list); }

    template<nt::scalar T>
    [[nodiscard]] NOA_FHD constexpr T clamp(T val, std::type_identity_t<T> low, std::type_identity_t<T> high) noexcept {
    #ifdef __CUDA_ARCH__
        return min(high, max(val, low));
    #else
        return std::clamp(val, low, high);
    #endif
    }

    /// Whether two floating-points are "almost" equal.
    /// For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
    /// the values used and multiplied by the desired precision in ULPs. Relative epsilons
    /// and Unit in the Last Place (ULPs) comparisons are usually meaningless for close-to-zero
    /// numbers, hence the absolute comparison with epsilon, acting as a safety net.
    /// If one or both values are NaN and|or +/-Inf, returns false.
    template<i32 ULP = 2, nt::real T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(T x, T y, std::type_identity_t<T> epsilon) noexcept {
        const T diff(abs(x - y));
        if (not is_finite(diff))
            return false;

        constexpr auto THRESHOLD = std::numeric_limits<T>::epsilon() * static_cast<T>(ULP);
        return diff <= epsilon or diff <= (max(abs(x), abs(y)) * THRESHOLD);
    }

    template<nt::real T>
    [[nodiscard]] NOA_IHD constexpr bool allclose(T x, T y) noexcept {
        return allclose<2>(x, y, static_cast<T>(1e-6));
    }
}
