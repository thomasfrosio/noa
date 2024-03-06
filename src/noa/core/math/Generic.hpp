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
    [[nodiscard]] NOA_FHD double cos(double x) { return std::cos(x); }
    [[nodiscard]] NOA_FHD float cos(float x) { return std::cos(x); }

    [[nodiscard]] NOA_FHD double sin(double x) { return std::sin(x); }
    [[nodiscard]] NOA_FHD float sin(float x) { return std::sin(x); }

    [[nodiscard]] NOA_FHD double sinc(double x) { return x == 0 ? 1 : sin(x) / x; }
    [[nodiscard]] NOA_FHD float sinc(float x) { return x == 0 ? 1 : sin(x) / x; }

    NOA_FHD void sincos(double x, double* s, double* c) {
        #ifdef __CUDA_ARCH__
        ::sincos(x, s, c); // included by nvcc/nvrtc
        #else
        *s = std::sin(x);
        *c = std::cos(x); // gcc calls its sincos
        #endif
    }

    NOA_FHD void sincos(float x, float* s, float* c) {
        #ifdef __CUDA_ARCH__
        ::sincosf(x, s, c); // included by nvcc/nvrtc
        #else
        *s = std::sin(x);
        *c = std::cos(x); // gcc calls its sincos
        #endif
    }

    [[nodiscard]] NOA_FHD double tan(double x) { return std::tan(x); }
    [[nodiscard]] NOA_FHD float tan(float x) { return std::tan(x); }

    [[nodiscard]] NOA_FHD double acos(double x) { return std::acos(x); }
    [[nodiscard]] NOA_FHD float acos(float x) { return std::acos(x); }

    [[nodiscard]] NOA_FHD double asin(double x) { return std::asin(x); }
    [[nodiscard]] NOA_FHD float asin(float x) { return std::asin(x); }

    [[nodiscard]] NOA_FHD double atan(double x) { return std::atan(x); }
    [[nodiscard]] NOA_FHD float atan(float x) { return std::atan(x); }

    [[nodiscard]] NOA_FHD double atan2(double y, double x) { return std::atan2(y, x); }
    [[nodiscard]] NOA_FHD float atan2(float y, float x) { return std::atan2(y, x); }

    [[nodiscard]] NOA_FHD constexpr double rad2deg(double x) noexcept { return x * (180. / Constant<double>::PI); }
    [[nodiscard]] NOA_FHD constexpr float rad2deg(float x) noexcept { return x * (180.f / Constant<float>::PI); }

    [[nodiscard]] NOA_FHD constexpr double deg2rad(double x) noexcept { return x * (Constant<double>::PI / 180.); }
    [[nodiscard]] NOA_FHD constexpr float deg2rad(float x) noexcept { return x * (Constant<float>::PI / 180.f); }

    [[nodiscard]] NOA_FHD double cosh(double x) { return std::cosh(x); }
    [[nodiscard]] NOA_FHD float cosh(float x) { return std::cosh(x); }

    [[nodiscard]] NOA_FHD double sinh(double x) { return std::sinh(x); }
    [[nodiscard]] NOA_FHD float sinh(float x) { return std::sinh(x); }

    [[nodiscard]] NOA_FHD double tanh(double x) { return std::tanh(x); }
    [[nodiscard]] NOA_FHD float tanh(float x) { return std::tanh(x); }

    [[nodiscard]] NOA_FHD double acosh(double x) { return std::acosh(x); }
    [[nodiscard]] NOA_FHD float acosh(float x) { return std::acosh(x); }

    [[nodiscard]] NOA_FHD double asinh(double x) { return std::asinh(x); }
    [[nodiscard]] NOA_FHD float asinh(float x) { return std::asinh(x); }

    [[nodiscard]] NOA_FHD double atanh(double x) { return std::atanh(x); }
    [[nodiscard]] NOA_FHD float atanh(float x) { return std::atanh(x); }

    [[nodiscard]] NOA_FHD double exp(double x) { return std::exp(x); }
    [[nodiscard]] NOA_FHD float exp(float x) { return std::exp(x); }

    [[nodiscard]] NOA_FHD double log(double x) { return std::log(x); }
    [[nodiscard]] NOA_FHD float log(float x) { return std::log(x); }

    [[nodiscard]] NOA_FHD double log10(double x) { return std::log10(x); }
    [[nodiscard]] NOA_FHD float log10(float x) { return std::log10(x); }

    [[nodiscard]] NOA_FHD double log1p(double x) { return std::log1p(x); }
    [[nodiscard]] NOA_FHD float log1p(float x) { return std::log1p(x); }

    [[nodiscard]] NOA_FHD double hypot(double x, double y) { return std::hypot(x, y); }
    [[nodiscard]] NOA_FHD float hypot(float x, float y) { return std::hypot(x, y); }

    [[nodiscard]] NOA_FHD double pow(double base, double exponent) { return std::pow(base, exponent); }
    [[nodiscard]] NOA_FHD float pow(float base, float exponent) { return std::pow(base, exponent); }

    // Returns the next power of 2. If x is a power of 2 or is equal to 1, returns x.
    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr Int next_power_of_2(Int x) {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    template<typename UInt> requires std::is_unsigned_v<UInt>
    [[nodiscard]] NOA_FHD constexpr UInt next_multiple_of(UInt value, UInt base) { return (value + base - 1) / base * base; }

    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr bool is_multiple_of(Int value, Int base) { return (value % base) == 0; }

    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr bool is_even(Int value) { return !(value % 2); }

    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr bool is_odd(Int value) { return value % 2; }

    template<class UInt> requires std::is_unsigned_v<UInt>
    [[nodiscard]] NOA_FHD constexpr bool is_power_of_2(UInt value) { return (value & (value - 1)) == 0; }

    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD constexpr Int divide_up(Int dividend, Int divisor) { return (dividend + divisor - 1) / divisor; }

    [[nodiscard]] NOA_FHD double sqrt(double x) { return std::sqrt(x); }
    [[nodiscard]] NOA_FHD float sqrt(float x) { return std::sqrt(x); }

    [[nodiscard]] NOA_FHD double rsqrt(double x) {
        #ifdef __CUDA_ARCH__
        return ::rsqrt(x);
        #else
        return 1. / sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD float rsqrt(float x) {
        #ifdef __CUDA_ARCH__
        return ::rsqrtf(x);
        #else
        return 1.f / sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD double round(double x) { return std::round(x); }
    [[nodiscard]] NOA_FHD float round(float x) { return std::round(x); }

    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD Int round(Int x) { return x; }

    [[nodiscard]] NOA_FHD double rint(double x) { return std::rint(x); }
    [[nodiscard]] NOA_FHD float rint(float x) { return std::rint(x); }

    template<typename Int> requires std::is_integral_v<Int>
    [[nodiscard]] NOA_FHD Int rint(Int x) { return x; }

    [[nodiscard]] NOA_FHD double ceil(double x) { return std::ceil(x); }
    [[nodiscard]] NOA_FHD float ceil(float x) { return std::ceil(x); }

    [[nodiscard]] NOA_FHD double floor(double x) { return std::floor(x); }
    [[nodiscard]] NOA_FHD float floor(float x) { return std::floor(x); }

    [[nodiscard]] NOA_FHD double trunc(double x) { return std::trunc(x); }
    [[nodiscard]] NOA_FHD float trunc(float x) { return std::trunc(x); }

    [[nodiscard]] NOA_FHD double copysign(double x, double y) { return std::copysign(x, y); }
    [[nodiscard]] NOA_FHD float copysign(float x, float y) { return std::copysign(x, y); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sign(T x) { return x >= 0 ? 1 : -1; }

    [[nodiscard]] NOA_FHD constexpr bool signbit(double x) { return std::signbit(x); }
    [[nodiscard]] NOA_FHD constexpr bool signbit(float x) { return std::signbit(x); }

    template<typename T>
    [[nodiscard]] NOA_FHD T abs(T x) {
        if constexpr (nt::is_uint_v<T>) {
            return x;
        } else if constexpr (nt::is_int_v<T>) {
            if constexpr (nt::is_almost_same_v<T, long>)
                return std::labs(x);
            else if constexpr (nt::is_almost_same_v<T, long long>)
                return std::llabs(x);
            else if constexpr (nt::is_almost_same_v<T, int8_t>)
                return static_cast<int8_t>(::abs(x));
            else if constexpr (nt::is_almost_same_v<T, int16_t>)
                return static_cast<int16_t>(::abs(x));
            return std::abs(x);
        } else {
            return std::abs(x);
        }
    }
    template<typename T>
    [[nodiscard]] NOA_FHD auto abs_squared(T x) {
        auto t = abs(x);
        return t * t;
    }

    [[nodiscard]] NOA_FHD double fma(double x, double y, double z) { return std::fma(x, y, z); }
    [[nodiscard]] NOA_FHD float fma(float x, float y, float z) { return std::fma(x, y, z); }

    template<typename T> requires nt::is_real_or_complex_v<T>
    NOA_IHD void kahan_sum(T value, T& sum, T& error) {
        auto sum_value = value + sum;
        if constexpr (nt::is_real_v<T>) {
            error += abs(sum) >= abs(value) ?
                     (sum - sum_value) + value :
                     (value - sum_value) + sum;
        } else if constexpr (nt::is_complex_v<T>) {
            for (i64 i = 0; i < 2; ++i) {
                error[i] += abs(sum[i]) >= abs(value[i]) ?
                            (sum[i] - sum_value[i]) + value[i] :
                            (value[i] - sum_value[i]) + sum[i];
            }
        }
        sum = sum_value;
    }
}
