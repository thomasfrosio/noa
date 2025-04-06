#pragma once

#include <algorithm> // clamp

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa {
    [[nodiscard]] NOA_FHD constexpr auto is_nan(double x) noexcept -> bool { return std::isnan(x); }
    [[nodiscard]] NOA_FHD constexpr auto is_nan(float x) noexcept -> bool { return std::isnan(x); }

    [[nodiscard]] NOA_FHD constexpr auto is_inf(double x) noexcept -> bool { return std::isinf(x); }
    [[nodiscard]] NOA_FHD constexpr auto is_inf(float x) noexcept -> bool { return std::isinf(x); }

    [[nodiscard]] NOA_FHD constexpr auto is_finite(double x) noexcept -> bool { return std::isfinite(x); }
    [[nodiscard]] NOA_FHD constexpr auto is_finite(float x) noexcept -> bool { return std::isfinite(x); }

    [[nodiscard]] NOA_FH constexpr auto is_normal(double x) noexcept -> bool { return std::isnormal(x); }
    [[nodiscard]] NOA_FH constexpr auto is_normal(float x) noexcept -> bool { return std::isnormal(x); }

    template<nt::numeric T>
    [[nodiscard]] NOA_FHD constexpr auto min(T x, T y) noexcept -> T { return (y < x) ? y : x; }

    template<typename T>
    [[nodiscard]] NOA_IH constexpr auto min(std::initializer_list<T> list) noexcept -> T { return std::min(list); }

    template<nt::numeric T>
    [[nodiscard]] NOA_FHD constexpr auto max(T x, T y) noexcept -> T { return (y > x) ? y : x; }

    template<typename T>
    [[nodiscard]] NOA_IH constexpr auto max(std::initializer_list<T> list) noexcept -> T { return std::max(list); }

    template<nt::numeric T>
    [[nodiscard]] NOA_FHD constexpr auto clamp(T val, std::type_identity_t<T> low, std::type_identity_t<T> high) noexcept -> T {
    #ifdef __CUDA_ARCH__
        return min(high, max(val, low));
    #else
        return std::clamp(val, low, high);
    #endif
    }

    /// Whether two numerics are equal or almost equal to each other.
    /// \details Floating-points:
    ///     For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
    ///     the values used and multiplied by the desired precision in ULPs. Relative epsilons
    ///     and Unit in the Last Place (ULPs) comparisons are usually meaningless for close-to-zero
    ///     numbers, hence the absolute comparison with epsilon, acting as a safety net.
    ///     If one or both values are NaN and|or +/-Inf, returns false.
    /// \details Complex floating-points:
    ///     Computes the allclose for the real and imaginary components. Returns true if both are allclose.
    /// \details Integers:
    ///     ULP is ignored, and this functions simply checks whether x and y are within +/- epsilon of each other.
    template<i32 ULP = 2, nt::numeric T, typename U = nt::value_type_t<T>>
    [[nodiscard]] NOA_IHD constexpr auto allclose(
        T x, T y,
        std::type_identity_t<U> epsilon = static_cast<U>(1e-6)
    ) noexcept -> bool {
        if constexpr (nt::integer<T>) {
            T diff = static_cast<T>(abs(x - y));
            return diff <= epsilon;
        } else if constexpr (nt::complex<T>) {
            return allclose<ULP>(x.real, y.real, epsilon) and
                   allclose<ULP>(x.imag, y.imag, epsilon);
        } else {
            const T diff = abs(x - y);
            if (not is_finite(diff))
                return false;
            constexpr auto THRESHOLD = std::numeric_limits<T>::epsilon() * static_cast<T>(ULP);
            return diff <= epsilon or diff <= (max(abs(x), abs(y)) * THRESHOLD);
        }
    }
}
