#pragma once

#include <math.h> // I'm not sure cmath is entirely CUDA friendly.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cfloat> // FLT_EPSILON, DBL_EPSILON

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/math/Constant.h"

namespace noa::math {
    /// Returns the cosine of an angle of x radians.
    NOA_FHD double cos(double x) { return ::cos(x); }
    NOA_FHD float cos(float x) { return ::cosf(x); }

    /// Returns the sine of an angle of x radians.
    NOA_FHD double sin(double x) { return ::sin(x); }
    NOA_FHD float sin(float x) { return ::sinf(x); }

    /// Returns the sinus cardinal of index Pi.
    NOA_FHD double sinc(double x) { return x == 0 ? 1 : sin(x) / x; }
    NOA_FHD float sinc(float x) { return x == 0 ? 1 : sin(x) / x; }

    /// Returns the sine in \p s and cosine in \p c of an angle of x radians.
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
        *c = ::cosf(x); // gcc calls its sincosf
    #endif
    }

    /// Returns the tangent of an angle of x radians.
    NOA_FHD double tan(double x) { return ::tan(x); }
    NOA_FHD float tan(float x) { return ::tanf(x); }

    /// Returns the principal value of the arc cos of x, in radians.
    NOA_FHD double acos(double x) { return ::acos(x); }
    NOA_FHD float acos(float x) { return ::acosf(x); }

    /// Returns the principal value of the arc sine of x, in radians.
    NOA_FHD double asin(double x) { return ::asin(x); }
    NOA_FHD float asin(float x) { return ::asinf(x); }

    /// Returns the principal value of the arc tangent of x, in radians.
    NOA_FHD double atan(double x) { return ::atan(x); }
    NOA_FHD float atan(float x) { return ::atanf(x); }

    /// Returns the principal value of the arc tangent of y/x, in radians.
    NOA_FHD double atan2(double y, double x) { return ::atan2(y, x); }
    NOA_FHD float atan2(float y, float x) { return ::atan2f(y, x); }

    NOA_FHD constexpr double toDeg(double x) { return x * (180. / Constants<double>::PI); }
    NOA_FHD constexpr float toDeg(float x) { return x * (180.f / Constants<float>::PI); }
    NOA_FHD constexpr double rad2deg(double x) { return x * (180. / Constants<double>::PI); }
    NOA_FHD constexpr float rad2deg(float x) { return x * (180.f / Constants<float>::PI); }

    NOA_FHD constexpr double toRad(double x) { return x * (Constants<double>::PI / 180.); }
    NOA_FHD constexpr float toRad(float x) { return x * (Constants<float>::PI / 180.f); }
    NOA_FHD constexpr double deg2rad(double x) { return x * (Constants<double>::PI / 180.); }
    NOA_FHD constexpr float deg2rad(float x) { return x * (Constants<float>::PI / 180.f); }

    ///  Returns the hyperbolic cosine of x.
    NOA_FHD double cosh(double x) { return ::cosh(x); }
    NOA_FHD float cosh(float x) { return ::coshf(x); }

    ///  Returns the hyperbolic sine of x.
    NOA_FHD double sinh(double x) { return ::sinh(x); }
    NOA_FHD float sinh(float x) { return ::sinhf(x); }

    ///  Returns the hyperbolic tangent of x.
    NOA_FHD double tanh(double x) { return ::tanh(x); }
    NOA_FHD float tanh(float x) { return ::tanhf(x); }

    /// Returns the non-negative area hyperbolic cosine of x.
    NOA_FHD double acosh(double x) { return ::acosh(x); }
    NOA_FHD float acosh(float x) { return ::acoshf(x); }

    /// Returns the area hyperbolic sine of x.
    NOA_FHD double asinh(double x) { return ::asinh(x); }
    NOA_FHD float asinh(float x) { return ::asinhf(x); }

    /// Returns the area hyperbolic tangent of x.
    NOA_FHD double atanh(double x) { return ::atanh(x); }
    NOA_FHD float atanh(float x) { return ::atanhf(x); }

    /// Returns the exponential of x.
    NOA_FHD double exp(double x) { return ::exp(x); }
    NOA_FHD float exp(float x) { return ::expf(x); }

    /// Returns the natural logarithm of x.
    NOA_FHD double log(double x) { return ::log(x); }
    NOA_FHD float log(float x) { return ::logf(x); }

    /// Returns the base 10 logarithm of x.
    NOA_FHD double log10(double x) { return ::log10(x); }
    NOA_FHD float log10(float x) { return ::log10f(x); }

    /// Returns the natural logarithm of one plus x.
    NOA_FHD double log1p(double x) { return ::log1p(x); }
    NOA_FHD float log1p(float x) { return ::log1pf(x); }

    /// Returns the hypotenuse of a right-angled triangle whose legs are x and y.
    NOA_FHD double hypot(double x, double y) { return ::hypot(x, y); }
    NOA_FHD float hypot(float x, float y) { return ::hypotf(x, y); }

    ///  Returns @a base raised to the power exponent.
    NOA_FHD double pow(double base, double exponent) { return ::pow(base, exponent); }
    NOA_FHD float pow(float base, float exponent) { return ::powf(base, exponent); }

    /// Returns the next power of 2.
    /// \note If x is a power of 2 or is equal to 1, returns x.
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

    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    NOA_FHD constexpr T divideUp(T dividend, T divisor) { return (dividend + divisor - 1) / divisor; }

    /// Returns the square root of @a x.
    NOA_FHD double sqrt(double x) { return ::sqrt(x); }
    NOA_FHD float sqrt(float x) { return ::sqrtf(x); }

    /// Returns 1. / sqrt(@a x).
    NOA_FHD double rsqrt(double x) {
    #ifdef __CUDA_ARCH__
        return ::rsqrt(x);
    #else
        return 1. / ::sqrt(x);
    #endif
    }

    NOA_FHD float rsqrt(float x) {
    #ifdef __CUDA_ARCH__
        return ::rsqrtf(x);
    #else
        return 1.f / ::sqrtf(x);
    #endif
    }

    /// Rounds x to nearest integral value.
    NOA_FHD double round(double x) { return ::round(x); }
    NOA_FHD float round(float x) { return ::roundf(x); }

    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    NOA_FHD T round(T x) { return x; }

    /// Rounds x to integral value. Should be preferred to round a double to an integer.
    NOA_FHD double rint(double x) { return ::rint(x); }
    NOA_FHD float rint(float x) { return ::rintf(x); }

    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    NOA_FHD T rint(T x) { return x; }

    /// Rounds up x.
    NOA_FHD double ceil(double x) { return ::ceil(x); }
    NOA_FHD float ceil(float x) { return ::ceilf(x); }

    /// Rounds down x.
    NOA_FHD double floor(double x) { return ::floor(x); }
    NOA_FHD float floor(float x) { return ::floorf(x); }

    /// Truncates x.
    NOA_FHD double trunc(double x) { return ::trunc(x); }
    NOA_FHD float trunc(float x) { return ::truncf(x); }

    /// Returns a value with the magnitude of x and the sign of y.
    NOA_FHD double copysign(double x, double y) { return ::copysign(x, y); }
    NOA_FHD float copysign(float x, float y) { return ::copysign(x, y); }

    /// Returns the sign x (1 or -1). If x == 0, return 1.
    template<typename T>
    NOA_FHD constexpr T sign(T x) { return x >= 0 ? 1 : -1; }

    /// Returns whether the sign of x is negative. Can be also applied to inf, NaNs and 0s (unsigned is positive).
    NOA_FHD constexpr bool signbit(double x) { return ::signbit(x); }
    NOA_FHD constexpr bool signbit(float x) { return ::signbit(x); }

    /// Returns the absolute value of x.
    template<typename T>
    NOA_FHD T abs(T x) { return ::abs(x); }
    template<>
    NOA_FHD int8_t abs<int8_t>(int8_t x) { return static_cast<int8_t>(::abs(x)); }
    template<>
    NOA_FHD int16_t abs<int16_t>(int16_t x) { return static_cast<int16_t>(::abs(x)); }

    NOA_FHD double fma(double x, double y, double z) { return ::fma(x, y, z); }
    NOA_FHD float fma(float x, float y, float z) { return ::fmaf(x, y, z); }

    /// Returns the centered index of the corresponding non-centered idx. Should be within `0 <= idx < dim`.
    template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
    [[nodiscard]] NOA_FHD constexpr T FFTShift(T idx, T dim) {
        return (idx < (dim + 1) / 2) ? idx + dim / 2 : idx - (dim + 1) / 2; // or (idx + dim / 2) % dim
    }

    /// Returns the non-centered index of the corresponding centered idx. Should be within `0 <= idx < dim`.
    template<typename T, typename = std::enable_if_t<noa::traits::is_int_v<T>>>
    [[nodiscard]] NOA_FHD constexpr T iFFTShift(T idx, T dim) {
        return (idx < dim / 2) ? idx + (dim + 1) / 2 : idx - dim / 2; // or (idx + (dim + 1) / 2) % dim
    }

    /// Returns the value at a particular index within an evenly spaced range within a given interval.
    template<typename T>
    T linspace(size_t elements, size_t index, T start, T stop, bool endpoint = true) {
        if (elements <= start) {
            return start;
        } else if (endpoint && index == elements - 1) {
            return stop;
        } else {
            const size_t count = elements - static_cast<size_t>(endpoint);
            const T delta = stop - start;
            const T step = delta / static_cast<T>(count);
            return start + static_cast<T>(index) * step;
        }
    }

    /// Returns the step of an evenly spaced range within a given interval.
    template<typename T>
    T linspace(size_t elements, T start, T stop, bool endpoint = true) {
        const size_t count = elements - static_cast<size_t>(endpoint);
        const T delta = stop - start;
        return delta / static_cast<T>(count);
    }
}
