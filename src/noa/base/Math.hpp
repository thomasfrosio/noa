#pragma once

#include <algorithm> // clamp
#include <cstdlib>
#include <cmath>

#include "noa/base/Config.hpp"
#include "noa/base/Traits.hpp"

namespace noa {
    template<nt::real T>
    struct Constant {
        static constexpr T PI = static_cast<T>(3.1415926535897932384626433832795);
        static constexpr T PLANCK = static_cast<T>(6.62607015e-34); // J.Hz-1
        static constexpr T SPEED_OF_LIGHT = static_cast<T>(299792458.0); // m.s
        static constexpr T ELECTRON_MASS = static_cast<T>(9.1093837015e-31); // kg
        static constexpr T ELEMENTARY_CHARGE = static_cast<T>(1.602176634e-19); // Coulombs
    };
}

namespace noa {
    [[nodiscard]] NOA_FHD auto cos(double x) noexcept -> double { return std::cos(x); }
    [[nodiscard]] NOA_FHD auto cos(float x) noexcept -> float { return std::cos(x); }

    [[nodiscard]] NOA_FHD auto sin(double x) noexcept -> double { return std::sin(x); }
    [[nodiscard]] NOA_FHD auto sin(float x) noexcept -> float { return std::sin(x); }

    [[nodiscard]] NOA_FHD auto sinc(double x) noexcept -> double { return x == 0 ? 1 : sin(x) / x; }
    [[nodiscard]] NOA_FHD auto sinc(float x) noexcept -> float { return x == 0 ? 1 : sin(x) / x; }

    NOA_FHD void sincos(double x, double* s, double* c) noexcept {
        #ifdef __CUDA_ARCH__
        ::sincos(x, s, c); // included by nvcc/nvrtc
        #else
        *s = std::sin(x);
        *c = std::cos(x); // gcc calls its sincos
        #endif
    }

    NOA_FHD void sincos(float x, float* s, float* c) noexcept {
        #ifdef __CUDA_ARCH__
        ::sincosf(x, s, c); // included by nvcc/nvrtc
        #else
        *s = std::sin(x);
        *c = std::cos(x); // gcc calls its sincos
        #endif
    }

    [[nodiscard]] NOA_FHD auto tan(double x) noexcept -> double { return std::tan(x); }
    [[nodiscard]] NOA_FHD auto tan(float x) noexcept -> float { return std::tan(x); }

    [[nodiscard]] NOA_FHD auto acos(double x) noexcept -> double { return std::acos(x); }
    [[nodiscard]] NOA_FHD auto acos(float x) noexcept -> float { return std::acos(x); }

    [[nodiscard]] NOA_FHD auto asin(double x) noexcept -> double { return std::asin(x); }
    [[nodiscard]] NOA_FHD auto asin(float x) noexcept -> float { return std::asin(x); }

    [[nodiscard]] NOA_FHD auto atan(double x) noexcept -> double { return std::atan(x); }
    [[nodiscard]] NOA_FHD auto atan(float x) noexcept -> float { return std::atan(x); }

    [[nodiscard]] NOA_FHD auto atan2(double y, double x) noexcept -> double { return std::atan2(y, x); }
    [[nodiscard]] NOA_FHD auto atan2(float y, float x) noexcept -> float { return std::atan2(y, x); }

    [[nodiscard]] NOA_FHD auto constexpr rad2deg(double x) noexcept -> double { return x * (180. / Constant<double>::PI); }
    [[nodiscard]] NOA_FHD auto constexpr rad2deg(float x) noexcept -> float { return x * (180.f / Constant<float>::PI); }

    [[nodiscard]] NOA_FHD auto constexpr deg2rad(double x) noexcept -> double { return x * (Constant<double>::PI / 180.); }
    [[nodiscard]] NOA_FHD auto constexpr deg2rad(float x) noexcept -> float { return x * (Constant<float>::PI / 180.f); }

    [[nodiscard]] NOA_FHD auto cosh(double x) noexcept -> double { return std::cosh(x); }
    [[nodiscard]] NOA_FHD auto cosh(float x) noexcept -> float { return std::cosh(x); }

    [[nodiscard]] NOA_FHD auto sinh(double x) noexcept -> double { return std::sinh(x); }
    [[nodiscard]] NOA_FHD auto sinh(float x) noexcept -> float { return std::sinh(x); }

    [[nodiscard]] NOA_FHD auto tanh(double x) noexcept -> double { return std::tanh(x); }
    [[nodiscard]] NOA_FHD auto tanh(float x) noexcept -> float { return std::tanh(x); }

    [[nodiscard]] NOA_FHD auto acosh(double x) noexcept -> double { return std::acosh(x); }
    [[nodiscard]] NOA_FHD auto acosh(float x) noexcept -> float { return std::acosh(x); }

    [[nodiscard]] NOA_FHD auto asinh(double x) noexcept -> double { return std::asinh(x); }
    [[nodiscard]] NOA_FHD auto asinh(float x) noexcept -> float { return std::asinh(x); }

    [[nodiscard]] NOA_FHD auto atanh(double x) noexcept -> double { return std::atanh(x); }
    [[nodiscard]] NOA_FHD auto atanh(float x) noexcept -> float { return std::atanh(x); }

    [[nodiscard]] NOA_FHD auto exp(double x) noexcept -> double { return std::exp(x); }
    [[nodiscard]] NOA_FHD auto exp(float x) noexcept -> float { return std::exp(x); }

    [[nodiscard]] NOA_FHD auto log(double x) noexcept -> double { return std::log(x); }
    [[nodiscard]] NOA_FHD auto log(float x) noexcept -> float { return std::log(x); }

    [[nodiscard]] NOA_FHD auto log10(double x) noexcept -> double { return std::log10(x); }
    [[nodiscard]] NOA_FHD auto log10(float x) noexcept -> float { return std::log10(x); }

    [[nodiscard]] NOA_FHD auto log1p(double x) noexcept -> double { return std::log1p(x); }
    [[nodiscard]] NOA_FHD auto log1p(float x) noexcept -> float { return std::log1p(x); }

    [[nodiscard]] NOA_FHD auto hypot(double x, double y) noexcept -> double { return std::hypot(x, y); }
    [[nodiscard]] NOA_FHD auto hypot(float x, float y) noexcept -> float { return std::hypot(x, y); }

    [[nodiscard]] NOA_FHD auto pow(double base, double exponent) noexcept -> double { return std::pow(base, exponent); }
    [[nodiscard]] NOA_FHD auto pow(float base, float exponent) noexcept -> float { return std::pow(base, exponent); }

    [[nodiscard]] NOA_FHD auto sqrt(double x) noexcept -> double { return std::sqrt(x); }
    [[nodiscard]] NOA_FHD auto sqrt(float x) noexcept -> float { return std::sqrt(x); }

    [[nodiscard]] NOA_FHD auto rsqrt(double x) noexcept -> double {
        #ifdef __CUDA_ARCH__
        return ::rsqrt(x);
        #else
        return 1. / sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD auto rsqrt(float x) noexcept -> float {
        #ifdef __CUDA_ARCH__
        return ::rsqrtf(x);
        #else
        return 1.f / sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD auto round(double x) noexcept -> double { return std::round(x); }
    [[nodiscard]] NOA_FHD auto round(float x) noexcept -> float { return std::round(x); }

    [[nodiscard]] NOA_FHD auto rint(double x) noexcept -> double { return std::rint(x); }
    [[nodiscard]] NOA_FHD auto rint(float x) noexcept -> float { return std::rint(x); }

    [[nodiscard]] NOA_FHD auto ceil(double x) noexcept -> double { return std::ceil(x); }
    [[nodiscard]] NOA_FHD auto ceil(float x) noexcept -> float { return std::ceil(x); }

    [[nodiscard]] NOA_FHD auto floor(double x) noexcept -> double { return std::floor(x); }
    [[nodiscard]] NOA_FHD auto floor(float x) noexcept -> float { return std::floor(x); }

    [[nodiscard]] NOA_FHD auto trunc(double x) noexcept -> double { return std::trunc(x); }
    [[nodiscard]] NOA_FHD auto trunc(float x) noexcept -> float { return std::trunc(x); }

    [[nodiscard]] NOA_FHD auto copysign(double x, double y) noexcept -> double { return std::copysign(x, y); }
    [[nodiscard]] NOA_FHD auto copysign(float x, float y) noexcept -> float { return std::copysign(x, y); }

    [[nodiscard]] NOA_FHD constexpr auto signbit(double x) noexcept -> bool { return std::signbit(x); }
    [[nodiscard]] NOA_FHD constexpr auto signbit(float x) noexcept -> bool { return std::signbit(x); }

    [[nodiscard]] NOA_FHD auto fma(double x, double y, double z) noexcept -> double { return std::fma(x, y, z); }
    [[nodiscard]] NOA_FHD auto fma(float x, float y, float z) noexcept -> float { return std::fma(x, y, z); }
}


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

    template<nt::numeric T>
    [[nodiscard]] NOA_FHD constexpr auto max(T x, T y) noexcept -> T { return (y > x) ? y : x; }

    template<nt::numeric T>
    [[nodiscard]] NOA_FHD constexpr auto clamp(T val, std::type_identity_t<T> low, std::type_identity_t<T> high) noexcept -> T {
    #ifdef __CUDA_ARCH__
        return min(high, max(val, low));
    #else
        return std::clamp(val, low, high);
    #endif
    }

    /// Whether two floating-points are equal or almost equal to each other.
    /// \details For the relative epsilon, the machine epsilon has to be scaled to the magnitude of
    ///          the values used and multiplied by the desired precision in ULPs. Relative epsilons
    ///          and Unit in the Last Place (ULPs) comparisons are usually meaningless for close-to-zero
    ///          numbers, hence the absolute comparison with epsilon, acting as a safety net.
    ///          If one or both values are NaN and|or +/-Inf, returns false.
    template<i32 ULP = 2, nt::real T>
    [[nodiscard]] NOA_IHD constexpr auto allclose(
        T x, T y,
        std::type_identity_t<T> epsilon = static_cast<T>(1e-6)
    ) noexcept -> bool {
        const T diff = abs(x - y);
        if (not is_finite(diff))
            return false;
        constexpr auto THRESHOLD = std::numeric_limits<T>::epsilon() * static_cast<T>(ULP);
        return diff <= epsilon or diff <= (max(abs(x), abs(y)) * THRESHOLD);
    }

    /// Whether two integers are equal or almost equal to each other.
    /// ULP is ignored, and this function simply checks whether x and y are within +/- epsilon of each other.
    template<i32 ULP = 2, nt::integer T>
    [[nodiscard]] NOA_IHD constexpr auto allclose(T x, T y, T epsilon = 0) noexcept -> bool {
        T diff = static_cast<T>(abs(x - y));
        return diff <= epsilon;
    }
}

namespace noa {
    /// Returns the next power of 2.
    /// If x is a power of 2 or is equal to 1, returns x.
    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto next_power_of_2(T x) noexcept -> T {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    /// Rounds up to the nearest multiple of a number.
    /// \warning This should only be used for positive numbers, and the base should be greater than zero.
    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto next_multiple_of(T value, std::type_identity_t<T> base) noexcept -> T {
        return ((value + base - 1) / base) * base;
    }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto is_multiple_of(T value, std::type_identity_t<T> base) noexcept -> bool { return (value % base) == 0; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto is_even(T value) noexcept -> bool { return !(value % 2); }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto is_odd(T value) noexcept -> bool { return value % 2; }

    template<nt::uinteger T>
    [[nodiscard]] NOA_FHD constexpr auto is_power_of_2(T value) noexcept -> bool { return (value & (value - 1)) == 0; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr auto divide_up(T dividend, T divisor) noexcept -> T { return (dividend + divisor - 1) / divisor; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD auto round(T x) noexcept -> T { return x; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD auto rint(T x) noexcept -> T { return x; }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto sign(T x) noexcept -> T { return x >= 0 ? 1 : -1; }

    template<typename T> requires (nt::integer<T> or nt::any_of<T, f32, f64>)
    [[nodiscard]] NOA_FHD constexpr auto abs(T x) noexcept -> T {
        if constexpr (nt::uinteger<T>) {
            return x;
        } else if constexpr (nt::integer<T>) {
            if constexpr (nt::almost_same_as<T, long>)
                return std::labs(x);
            else if constexpr (nt::almost_same_as<T, long long>)
                return std::llabs(x);
            else if constexpr (nt::almost_same_as<T, i8>)
                return static_cast<i8>(::abs(x));
            else if constexpr (nt::almost_same_as<T, i16>)
                return static_cast<i16>(::abs(x));
            else
                return std::abs(x);
        } else {
            return std::abs(x);
        }
    }
    template<nt::scalar T>
    [[nodiscard]] NOA_FHD constexpr auto abs_squared(T x) noexcept {
        return x * x;
    }

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr auto cast_or_abs_squared(const U& value) noexcept -> T {
        if constexpr (nt::complex<U> and nt::real<T>)
            return static_cast<T>(abs_squared(value));
        else
            return static_cast<T>(value);
    }

    template<nt::real_or_complex T>
    NOA_IHD constexpr void kahan_sum(T value, T& sum, T& error) noexcept {
        auto sum_value = value + sum;
        if constexpr (nt::real<T>) {
            error += abs(sum) >= abs(value) ?
                     (sum - sum_value) + value :
                     (value - sum_value) + sum;
        } else if constexpr (nt::complex<T>) {
            for (isize i = 0; i < 2; ++i) {
                error[i] += abs(sum[i]) >= abs(value[i]) ?
                            (sum[i] - sum_value[i]) + value[i] :
                            (value[i] - sum_value[i]) + sum[i];
            }
        }
        sum = sum_value;
    }

    template<typename T, typename U>
    requires ((nt::scalar<T, U> and nt::same_as<T, U>) or (nt::real_or_complex<T, U> and nt::same_value_type<T, U>))
    constexpr auto divide_safe(const T& lhs, const U& rhs) noexcept {
        if constexpr (nt::real_or_complex<T, U>) {
            constexpr auto epsilon = std::numeric_limits<nt::value_type_t<U>>::epsilon();
            if constexpr (nt::complex<U>)
                return abs(rhs.real) < epsilon or abs(rhs.imag) < epsilon ? U{} : lhs / rhs;
            else
                return abs(rhs) < epsilon ? T{} : lhs / rhs;
        } else if constexpr (nt::integer<T, U>) {
            return rhs == 0 ? T{} : lhs / rhs;
        } else {
            static_assert(nt::always_false<T>);
        }
    }
}
