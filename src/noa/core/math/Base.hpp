#pragma once

#include <cstdlib>
#include <cmath>

#include "noa/core/Config.hpp"
#include "noa/core/math/Constant.hpp"

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
