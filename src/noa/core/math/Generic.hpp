#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/math/Constant.hpp"

#ifdef NOA_IS_OFFLINE
#include <cstdlib>
#include <cmath>
#else
#include <cuda/std/cstdlib>
#include <cuda/std/cmath>
#endif

namespace noa {
    [[nodiscard]] NOA_FHD double cos(double x) noexcept { return std::cos(x); }
    [[nodiscard]] NOA_FHD float cos(float x) noexcept { return std::cos(x); }

    [[nodiscard]] NOA_FHD double sin(double x) noexcept { return std::sin(x); }
    [[nodiscard]] NOA_FHD float sin(float x) noexcept { return std::sin(x); }

    [[nodiscard]] NOA_FHD double sinc(double x) noexcept { return x == 0 ? 1 : sin(x) / x; }
    [[nodiscard]] NOA_FHD float sinc(float x) noexcept { return x == 0 ? 1 : sin(x) / x; }

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

    [[nodiscard]] NOA_FHD double tan(double x) noexcept { return std::tan(x); }
    [[nodiscard]] NOA_FHD float tan(float x) noexcept { return std::tan(x); }

    [[nodiscard]] NOA_FHD double acos(double x) noexcept { return std::acos(x); }
    [[nodiscard]] NOA_FHD float acos(float x) noexcept { return std::acos(x); }

    [[nodiscard]] NOA_FHD double asin(double x) noexcept { return std::asin(x); }
    [[nodiscard]] NOA_FHD float asin(float x) noexcept { return std::asin(x); }

    [[nodiscard]] NOA_FHD double atan(double x) noexcept { return std::atan(x); }
    [[nodiscard]] NOA_FHD float atan(float x) noexcept { return std::atan(x); }

    [[nodiscard]] NOA_FHD double atan2(double y, double x) noexcept { return std::atan2(y, x); }
    [[nodiscard]] NOA_FHD float atan2(float y, float x) noexcept { return std::atan2(y, x); }

    [[nodiscard]] NOA_FHD constexpr double rad2deg(double x) noexcept { return x * (180. / Constant<double>::PI); }
    [[nodiscard]] NOA_FHD constexpr float rad2deg(float x) noexcept { return x * (180.f / Constant<float>::PI); }

    [[nodiscard]] NOA_FHD constexpr double deg2rad(double x) noexcept { return x * (Constant<double>::PI / 180.); }
    [[nodiscard]] NOA_FHD constexpr float deg2rad(float x) noexcept { return x * (Constant<float>::PI / 180.f); }

    [[nodiscard]] NOA_FHD double cosh(double x) noexcept { return std::cosh(x); }
    [[nodiscard]] NOA_FHD float cosh(float x) noexcept { return std::cosh(x); }

    [[nodiscard]] NOA_FHD double sinh(double x) noexcept { return std::sinh(x); }
    [[nodiscard]] NOA_FHD float sinh(float x) noexcept { return std::sinh(x); }

    [[nodiscard]] NOA_FHD double tanh(double x) noexcept { return std::tanh(x); }
    [[nodiscard]] NOA_FHD float tanh(float x) noexcept { return std::tanh(x); }

    [[nodiscard]] NOA_FHD double acosh(double x) noexcept { return std::acosh(x); }
    [[nodiscard]] NOA_FHD float acosh(float x) noexcept { return std::acosh(x); }

    [[nodiscard]] NOA_FHD double asinh(double x) noexcept { return std::asinh(x); }
    [[nodiscard]] NOA_FHD float asinh(float x) noexcept { return std::asinh(x); }

    [[nodiscard]] NOA_FHD double atanh(double x) noexcept { return std::atanh(x); }
    [[nodiscard]] NOA_FHD float atanh(float x) noexcept { return std::atanh(x); }

    [[nodiscard]] NOA_FHD double exp(double x) noexcept { return std::exp(x); }
    [[nodiscard]] NOA_FHD float exp(float x) noexcept { return std::exp(x); }

    [[nodiscard]] NOA_FHD double log(double x) noexcept { return std::log(x); }
    [[nodiscard]] NOA_FHD float log(float x) noexcept { return std::log(x); }

    [[nodiscard]] NOA_FHD double log10(double x) noexcept { return std::log10(x); }
    [[nodiscard]] NOA_FHD float log10(float x) noexcept { return std::log10(x); }

    [[nodiscard]] NOA_FHD double log1p(double x) noexcept { return std::log1p(x); }
    [[nodiscard]] NOA_FHD float log1p(float x) noexcept { return std::log1p(x); }

    [[nodiscard]] NOA_FHD double hypot(double x, double y) noexcept { return std::hypot(x, y); }
    [[nodiscard]] NOA_FHD float hypot(float x, float y) noexcept { return std::hypot(x, y); }

    [[nodiscard]] NOA_FHD double pow(double base, double exponent) noexcept { return std::pow(base, exponent); }
    [[nodiscard]] NOA_FHD float pow(float base, float exponent) noexcept { return std::pow(base, exponent); }

    // Returns the next power of 2. If x is a power of 2 or is equal to 1, returns x.
    template<nt::integer Int>
    [[nodiscard]] NOA_FHD constexpr Int next_power_of_2(Int x) noexcept {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    template<nt::uinteger T>
    [[nodiscard]] NOA_FHD constexpr T next_multiple_of(T value, std::type_identity_t<T> base) noexcept { return (value + base - 1) / base * base; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr bool is_multiple_of(T value, std::type_identity_t<T> base) noexcept { return (value % base) == 0; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr bool is_even(T value) noexcept { return !(value % 2); }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr bool is_odd(T value) noexcept { return value % 2; }

    template<nt::uinteger T>
    [[nodiscard]] NOA_FHD constexpr bool is_power_of_2(T value) noexcept { return (value & (value - 1)) == 0; }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD constexpr T divide_up(T dividend, T divisor) noexcept { return (dividend + divisor - 1) / divisor; }

    [[nodiscard]] NOA_FHD double sqrt(double x) noexcept { return std::sqrt(x); }
    [[nodiscard]] NOA_FHD float sqrt(float x) noexcept { return std::sqrt(x); }

    [[nodiscard]] NOA_FHD double rsqrt(double x) noexcept {
        #ifdef __CUDA_ARCH__
        return ::rsqrt(x);
        #else
        return 1. / sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD float rsqrt(float x) noexcept {
        #ifdef __CUDA_ARCH__
        return ::rsqrtf(x);
        #else
        return 1.f / sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD double round(double x) noexcept { return std::round(x); }
    [[nodiscard]] NOA_FHD float round(float x) noexcept { return std::round(x); }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD T round(T x) noexcept { return x; }

    [[nodiscard]] NOA_FHD double rint(double x) noexcept { return std::rint(x); }
    [[nodiscard]] NOA_FHD float rint(float x) noexcept { return std::rint(x); }

    template<nt::integer T>
    [[nodiscard]] NOA_FHD T rint(T x) noexcept { return x; }

    [[nodiscard]] NOA_FHD double ceil(double x) noexcept { return std::ceil(x); }
    [[nodiscard]] NOA_FHD float ceil(float x) noexcept { return std::ceil(x); }

    [[nodiscard]] NOA_FHD double floor(double x) noexcept { return std::floor(x); }
    [[nodiscard]] NOA_FHD float floor(float x) noexcept { return std::floor(x); }

    [[nodiscard]] NOA_FHD double trunc(double x) noexcept { return std::trunc(x); }
    [[nodiscard]] NOA_FHD float trunc(float x) noexcept { return std::trunc(x); }

    [[nodiscard]] NOA_FHD double copysign(double x, double y) noexcept { return std::copysign(x, y); }
    [[nodiscard]] NOA_FHD float copysign(float x, float y) noexcept { return std::copysign(x, y); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sign(T x) noexcept { return x >= 0 ? 1 : -1; }

    [[nodiscard]] NOA_FHD constexpr bool signbit(double x) noexcept { return std::signbit(x); }
    [[nodiscard]] NOA_FHD constexpr bool signbit(float x) noexcept { return std::signbit(x); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T abs(T x) noexcept {
        if constexpr (nt::uinteger<T>) {
            return x;
        } else if constexpr (nt::integer<T>) {
            if constexpr (nt::almost_same_as<T, long>)
                return std::labs(x);
            else if constexpr (nt::almost_same_as<T, long long>)
                return std::llabs(x);
            else if constexpr (nt::almost_same_as<T, int8_t>)
                return static_cast<int8_t>(::abs(x));
            else if constexpr (nt::almost_same_as<T, int16_t>)
                return static_cast<int16_t>(::abs(x));
            else
                return std::abs(x);
        } else {
            return std::abs(x);
        }
    }
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr auto abs_squared(T x) noexcept {
        auto t = abs(x);
        return t * t;
    }

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr T cast_or_abs_squared(const U& value) noexcept {
        if constexpr (nt::complex<U> and nt::real<T>)
            return static_cast<T>(abs_squared(value));
        else
            return static_cast<T>(value);
    }

    [[nodiscard]] NOA_FHD double fma(double x, double y, double z) noexcept { return std::fma(x, y, z); }
    [[nodiscard]] NOA_FHD float fma(float x, float y, float z) noexcept { return std::fma(x, y, z); }

    template<nt::real_or_complex T>
    NOA_IHD constexpr void kahan_sum(T value, T& sum, T& error) noexcept {
        auto sum_value = value + sum;
        if constexpr (nt::real<T>) {
            error += abs(sum) >= abs(value) ?
                     (sum - sum_value) + value :
                     (value - sum_value) + sum;
        } else if constexpr (nt::complex<T>) {
            for (i64 i = 0; i < 2; ++i) {
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
