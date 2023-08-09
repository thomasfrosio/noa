#pragma once

#include <math.h> // not sure cmath is entirely CUDA friendly
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cfloat> // FLT_EPSILON, DBL_EPSILON

#include "noa/core/Definitions.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/math/Constant.hpp"

namespace noa::math {
    [[nodiscard]] NOA_FHD double cos(double x) { return ::cos(x); }
    [[nodiscard]] NOA_FHD float cos(float x) { return ::cosf(x); }

    [[nodiscard]] NOA_FHD double sin(double x) { return ::sin(x); }
    [[nodiscard]] NOA_FHD float sin(float x) { return ::sinf(x); }

    [[nodiscard]] NOA_FHD double sinc(double x) { return x == 0 ? 1 : sin(x) / x; }
    [[nodiscard]] NOA_FHD float sinc(float x) { return x == 0 ? 1 : sin(x) / x; }

    NOA_FHD void sincos(double x, double* s, double* c) {
        #ifdef __CUDA_ARCH__
        ::sincos(x, s, c);
        #else
        *s = ::sin(x);
        *c = ::cos(x); // gcc calls its sincos
        #endif
    }

    NOA_FHD void sincos(float x, float* s, float* c) {
        #ifdef __CUDA_ARCH__
        ::sincosf(x, s, c);
        #else
        *s = ::sinf(x);
        *c = ::cosf(x); // gcc calls its sincos
        #endif
    }

    [[nodiscard]] NOA_FHD double tan(double x) { return ::tan(x); }
    [[nodiscard]] NOA_FHD float tan(float x) { return ::tanf(x); }

    [[nodiscard]] NOA_FHD double acos(double x) { return ::acos(x); }
    [[nodiscard]] NOA_FHD float acos(float x) { return ::acosf(x); }

    [[nodiscard]] NOA_FHD double asin(double x) { return ::asin(x); }
    [[nodiscard]] NOA_FHD float asin(float x) { return ::asinf(x); }

    [[nodiscard]] NOA_FHD double atan(double x) { return ::atan(x); }
    [[nodiscard]] NOA_FHD float atan(float x) { return ::atanf(x); }

    [[nodiscard]] NOA_FHD double atan2(double y, double x) { return ::atan2(y, x); }
    [[nodiscard]] NOA_FHD float atan2(float y, float x) { return ::atan2f(y, x); }

    [[nodiscard]] NOA_FHD constexpr double rad2deg(double x) noexcept { return x * (180. / Constant<double>::PI); }
    [[nodiscard]] NOA_FHD constexpr float rad2deg(float x) noexcept { return x * (180.f / Constant<float>::PI); }

    [[nodiscard]] NOA_FHD constexpr double deg2rad(double x) noexcept { return x * (Constant<double>::PI / 180.); }
    [[nodiscard]] NOA_FHD constexpr float deg2rad(float x) noexcept { return x * (Constant<float>::PI / 180.f); }

    [[nodiscard]] NOA_FHD double cosh(double x) { return ::cosh(x); }
    [[nodiscard]] NOA_FHD float cosh(float x) { return ::coshf(x); }

    [[nodiscard]] NOA_FHD double sinh(double x) { return ::sinh(x); }
    [[nodiscard]] NOA_FHD float sinh(float x) { return ::sinhf(x); }

    [[nodiscard]] NOA_FHD double tanh(double x) { return ::tanh(x); }
    [[nodiscard]] NOA_FHD float tanh(float x) { return ::tanhf(x); }

    [[nodiscard]] NOA_FHD double acosh(double x) { return ::acosh(x); }
    [[nodiscard]] NOA_FHD float acosh(float x) { return ::acoshf(x); }

    [[nodiscard]] NOA_FHD double asinh(double x) { return ::asinh(x); }
    [[nodiscard]] NOA_FHD float asinh(float x) { return ::asinhf(x); }

    [[nodiscard]] NOA_FHD double atanh(double x) { return ::atanh(x); }
    [[nodiscard]] NOA_FHD float atanh(float x) { return ::atanhf(x); }

    [[nodiscard]] NOA_FHD double exp(double x) { return ::exp(x); }
    [[nodiscard]] NOA_FHD float exp(float x) { return ::expf(x); }

    [[nodiscard]] NOA_FHD double log(double x) { return ::log(x); }
    [[nodiscard]] NOA_FHD float log(float x) { return ::logf(x); }

    [[nodiscard]] NOA_FHD double log10(double x) { return ::log10(x); }
    [[nodiscard]] NOA_FHD float log10(float x) { return ::log10f(x); }

    [[nodiscard]] NOA_FHD double log1p(double x) { return ::log1p(x); }
    [[nodiscard]] NOA_FHD float log1p(float x) { return ::log1pf(x); }

    [[nodiscard]] NOA_FHD double hypot(double x, double y) { return ::hypot(x, y); }
    [[nodiscard]] NOA_FHD float hypot(float x, float y) { return ::hypotf(x, y); }

    [[nodiscard]] NOA_FHD double pow(double base, double exponent) { return ::pow(base, exponent); }
    [[nodiscard]] NOA_FHD float pow(float base, float exponent) { return ::powf(base, exponent); }

    // Returns the next power of 2. If x is a power of 2 or is equal to 1, returns x.
    template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
    [[nodiscard]] NOA_FHD constexpr Int next_power_of_2(Int x) {
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    template<typename UInt, typename = std::enable_if_t<std::is_unsigned_v<UInt>>>
    [[nodiscard]] NOA_FHD constexpr UInt next_multiple_of(UInt value, UInt base) { return (value + base - 1) / base * base; }

    template<class UInt, typename = std::enable_if_t<std::is_unsigned_v<UInt>>>
    [[nodiscard]] NOA_FHD constexpr bool is_power_of_2(UInt value) { return (value & (value - 1)) == 0; }

    template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
    [[nodiscard]] NOA_FHD constexpr Int divide_up(Int dividend, Int divisor) { return (dividend + divisor - 1) / divisor; }

    [[nodiscard]] NOA_FHD double sqrt(double x) { return ::sqrt(x); }
    [[nodiscard]] NOA_FHD float sqrt(float x) { return ::sqrtf(x); }

    [[nodiscard]] NOA_FHD double rsqrt(double x) {
        #ifdef __CUDA_ARCH__
        return ::rsqrt(x);
        #else
        return 1. / ::sqrt(x);
        #endif
    }

    [[nodiscard]] NOA_FHD float rsqrt(float x) {
        #ifdef __CUDA_ARCH__
        return ::rsqrtf(x);
        #else
        return 1.f / ::sqrtf(x);
        #endif
    }

    [[nodiscard]] NOA_FHD double round(double x) { return ::round(x); }
    [[nodiscard]] NOA_FHD float round(float x) { return ::roundf(x); }

    template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
    [[nodiscard]] NOA_FHD Int round(Int x) { return x; }

    [[nodiscard]] NOA_FHD double rint(double x) { return ::rint(x); }
    [[nodiscard]] NOA_FHD float rint(float x) { return ::rintf(x); }

    template<typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
    [[nodiscard]] NOA_FHD Int rint(Int x) { return x; }

    [[nodiscard]] NOA_FHD double ceil(double x) { return ::ceil(x); }
    [[nodiscard]] NOA_FHD float ceil(float x) { return ::ceilf(x); }

    [[nodiscard]] NOA_FHD double floor(double x) { return ::floor(x); }
    [[nodiscard]] NOA_FHD float floor(float x) { return ::floorf(x); }

    [[nodiscard]] NOA_FHD double trunc(double x) { return ::trunc(x); }
    [[nodiscard]] NOA_FHD float trunc(float x) { return ::truncf(x); }

    [[nodiscard]] NOA_FHD double copysign(double x, double y) { return ::copysign(x, y); }
    [[nodiscard]] NOA_FHD float copysign(float x, float y) { return ::copysign(x, y); }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sign(T x) { return x >= 0 ? 1 : -1; }

    [[nodiscard]] NOA_FHD constexpr bool signbit(double x) { return ::signbit(x); }
    [[nodiscard]] NOA_FHD constexpr bool signbit(float x) { return ::signbit(x); }

    template<typename T>
    [[nodiscard]] NOA_FHD T abs(T x) {
        if constexpr (nt::is_uint_v<T>) {
            return x;
        } else if constexpr (nt::is_int_v<T>) {
            if constexpr (nt::is_almost_same_v<T, long>)
                return ::labs(x);
            else if constexpr (nt::is_almost_same_v<T, long long>)
                return ::llabs(x);
            else if constexpr (nt::is_almost_same_v<T, int8_t>)
                return static_cast<int8_t>(::abs(x));
            else if constexpr (nt::is_almost_same_v<T, int16_t>)
                return static_cast<int16_t>(::abs(x));
            return ::abs(x);
        } else {
            return ::abs(x);
        }
    }

    [[nodiscard]] NOA_FHD double fma(double x, double y, double z) { return ::fma(x, y, z); }
    [[nodiscard]] NOA_FHD float fma(float x, float y, float z) { return ::fmaf(x, y, z); }
}
