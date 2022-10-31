/// \file noa/common/types/Float3.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 3 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/ArrayTypes.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool3.h"
#include "noa/common/types/ClampCast.h"
#include "noa/common/types/Half.h"
#include "noa/common/types/SafeCast.h"
#include "noa/common/utils/Sort.h"

namespace noa {
    template<typename>
    class Int3;

    template<typename T>
    class Float3 {
    public:
        using value_type = T;

    public: // Default Constructors
        constexpr Float3() noexcept = default;
        constexpr Float3(const Float3&) noexcept = default;
        constexpr Float3(Float3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z,
                 typename = std::enable_if_t<traits::is_scalar_v<X> &&
                                             traits::is_scalar_v<Y> &&
                                             traits::is_scalar_v<Z>>>
        NOA_HD constexpr Float3(X a0, Y a1, Z a2) noexcept
                : m_data{static_cast<T>(a0), static_cast<T>(a1), static_cast<T>(a2)} {
            NOA_ASSERT(isSafeCast<T>(a0) && isSafeCast<T>(a1) && isSafeCast<T>(a2));
        }

        template<typename U>
        NOA_HD constexpr explicit Float3(U x) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(x), static_cast<T>(x)} {
            NOA_ASSERT(isSafeCast<T>(x));
        }

        template<typename U>
        NOA_HD constexpr explicit Float3(Float3<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {
            NOA_ASSERT(isSafeCast<T>(v[0]) && isSafeCast<T>(v[1]) && isSafeCast<T>(v[2]));
        }

        NOA_HD constexpr explicit Float3(Bool3 v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Int3<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {
            NOA_ASSERT(isSafeCast<T>(v[0]) && isSafeCast<T>(v[1]) && isSafeCast<T>(v[2]));
        }

        template<typename U>
        NOA_HD constexpr explicit Float3(U* ptr) noexcept {
            NOA_ASSERT(ptr != nullptr);
            for (size_t i = 0; i < COUNT; ++i) {
                NOA_ASSERT(isSafeCast<T>(ptr[i]));
                m_data[i] = static_cast<T>(ptr[i]);
            }
        }

    public: // Assignment operators
        constexpr Float3& operator=(const Float3& v) noexcept = default;
        constexpr Float3& operator=(Float3&& v) noexcept = default;

        NOA_HD constexpr Float3& operator=(T v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            m_data[2] = v;
            return *this;
        }

        NOA_HD constexpr Float3& operator+=(Float3 rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator-=(Float3 rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator*=(Float3 rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator/=(Float3 rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator+=(T rhs) noexcept {
            *this = *this + rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator-=(T rhs) noexcept {
            *this = *this - rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator*=(T rhs) noexcept {
            *this = *this * rhs;
            return *this;
        }

        NOA_HD constexpr Float3& operator/=(T rhs) noexcept {
            *this = *this / rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Float3 operator+(Float3 v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator-(Float3 v) noexcept {
            return {-v[0], -v[1], -v[2]};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Float3 operator+(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator+(T lhs, Float3 rhs) noexcept {
            return {lhs + rhs[0], lhs + rhs[1], lhs + rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator+(Float3 lhs, T rhs) noexcept {
            return {lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator-(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator-(T lhs, Float3 rhs) noexcept {
            return {lhs - rhs[0], lhs - rhs[1], lhs - rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator-(Float3 lhs, T rhs) noexcept {
            return {lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator*(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator*(T lhs, Float3 rhs) noexcept {
            return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator*(Float3 lhs, T rhs) noexcept {
            return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator/(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator/(T lhs, Float3 rhs) noexcept {
            return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Float3 operator/(Float3 lhs, T rhs) noexcept {
            return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs};
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr Bool3 operator>(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] > rhs[0], lhs[1] > rhs[1], lhs[2] > rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator>(Float3 lhs, T rhs) noexcept {
            return {lhs[0] > rhs, lhs[1] > rhs, lhs[2] > rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator>(T lhs, Float3 rhs) noexcept {
            return {lhs > rhs[0], lhs > rhs[1], lhs > rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator<(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] < rhs[0], lhs[1] < rhs[1], lhs[2] < rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator<(Float3 lhs, T rhs) noexcept {
            return {lhs[0] < rhs, lhs[1] < rhs, lhs[2] < rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator<(T lhs, Float3 rhs) noexcept {
            return {lhs < rhs[0], lhs < rhs[1], lhs < rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator>=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1], lhs[2] >= rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator>=(Float3 lhs, T rhs) noexcept {
            return {lhs[0] >= rhs, lhs[1] >= rhs, lhs[2] >= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator>=(T lhs, Float3 rhs) noexcept {
            return {lhs >= rhs[0], lhs >= rhs[1], lhs >= rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator<=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1], lhs[2] <= rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator<=(Float3 lhs, T rhs) noexcept {
            return {lhs[0] <= rhs, lhs[1] <= rhs, lhs[2] <= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator<=(T lhs, Float3 rhs) noexcept {
            return {lhs <= rhs[0], lhs <= rhs[1], lhs <= rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator==(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator==(Float3 lhs, T rhs) noexcept {
            return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator==(T lhs, Float3 rhs) noexcept {
            return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator!=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator!=(Float3 lhs, T rhs) noexcept {
            return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool3 operator!=(T lhs, Float3 rhs) noexcept {
            return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 3;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]]NOA_HD constexpr T& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]]NOA_HD constexpr const T& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_data; }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const T* get(I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr T* get(I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        [[nodiscard]] NOA_HD constexpr Float3 flip() const noexcept { return {m_data[2], m_data[1], m_data[0]}; }

    private:
        static_assert(noa::traits::is_float_v<T>);
        T m_data[3]{};
    };

    template<typename T>
    struct traits::proclaim_is_float3<Float3<T>> : std::true_type {};

    using half3_t = Float3<half_t>;
    using float3_t = Float3<float>;
    using double3_t = Float3<double>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 3> toArray(Float3<T> v) noexcept {
        return {v[0], v[1], v[2]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<half3_t>() { return "half3"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<float3_t>() { return "float3"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<double3_t>() { return "double3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Float3<T> v) {
        os << string::format("({:.3f},{:.3f},{:.3f})", v[0], v[1], v[2]);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Float3<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Float3<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[1], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[2], ctx);
            *out = ')';
            return out;
        }
    };
}

namespace noa {
    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_float3_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Float3<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return {clamp_cast<value_t>(src[0]), clamp_cast<value_t>(src[1]), clamp_cast<value_t>(src[2])};
    }

    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_float3_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr bool isSafeCast(const Float3<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return isSafeCast<value_t>(src[0]) && isSafeCast<value_t>(src[1]) && isSafeCast<value_t>(src[2]);
    }
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> cos(Float3<T> v) {
        return Float3<T>(cos(v[0]), cos(v[1]), cos(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> sin(Float3<T> v) {
        return Float3<T>(sin(v[0]), sin(v[1]), sin(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> sinc(Float3<T> v) {
        return Float3<T>(sinc(v[0]), sinc(v[1]), sinc(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> tan(Float3<T> v) {
        return Float3<T>(tan(v[0]), tan(v[1]), tan(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> acos(Float3<T> v) {
        return Float3<T>(acos(v[0]), acos(v[1]), acos(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> asin(Float3<T> v) {
        return Float3<T>(asin(v[0]), asin(v[1]), asin(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> atan(Float3<T> v) {
        return Float3<T>(atan(v[0]), atan(v[1]), atan(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> cosh(Float3<T> v) {
        return Float3<T>(cosh(v[0]), cosh(v[1]), cosh(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> sinh(Float3<T> v) {
        return Float3<T>(sinh(v[0]), sinh(v[1]), sinh(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> tanh(Float3<T> v) {
        return Float3<T>(tanh(v[0]), tanh(v[1]), tanh(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> acosh(Float3<T> v) {
        return Float3<T>(acosh(v[0]), acosh(v[1]), acosh(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> asinh(Float3<T> v) {
        return Float3<T>(asinh(v[0]), asinh(v[1]), asinh(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> atanh(Float3<T> v) {
        return Float3<T>(atanh(v[0]), atanh(v[1]), atanh(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> deg2rad(Float3<T> v) noexcept {
        return Float3<T>(deg2rad(v[0]), deg2rad(v[1]), deg2rad(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> rad2deg(Float3<T> v) noexcept {
        return Float3<T>(rad2deg(v[0]), rad2deg(v[1]), rad2deg(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> exp(Float3<T> v) {
        return Float3<T>(exp(v[0]), exp(v[1]), exp(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> log(Float3<T> v) {
        return Float3<T>(log(v[0]), log(v[1]), log(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> log10(Float3<T> v) {
        return Float3<T>(log10(v[0]), log10(v[1]), log10(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> log1p(Float3<T> v) {
        return Float3<T>(log1p(v[0]), log1p(v[1]), log1p(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> sqrt(Float3<T> v) {
        return Float3<T>(sqrt(v[0]), sqrt(v[1]), sqrt(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> rsqrt(Float3<T> v) {
        return Float3<T>(rsqrt(v[0]), rsqrt(v[1]), rsqrt(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> round(Float3<T> v) noexcept {
        return Float3<T>(round(v[0]), round(v[1]), round(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> rint(Float3<T> v) noexcept {
        return Float3<T>(rint(v[0]), rint(v[1]), rint(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> ceil(Float3<T> v) noexcept {
        return Float3<T>(ceil(v[0]), ceil(v[1]), ceil(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> floor(Float3<T> v) noexcept {
        return Float3<T>(floor(v[0]), floor(v[1]), floor(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> abs(Float3<T> v) noexcept {
        return Float3<T>(abs(v[0]), abs(v[1]), abs(v[2]));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sum(Float3<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(sum(Float3<HALF_ARITHMETIC_TYPE>(v)));
        return v[0] + v[1] + v[2];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T prod(Float3<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(prod(Float3<HALF_ARITHMETIC_TYPE>(v)));
        return v[0] * v[1] * v[2];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T dot(Float3<T> a, Float3<T> b) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return static_cast<T>(dot(Float3<HALF_ARITHMETIC_TYPE>(a), Float3<HALF_ARITHMETIC_TYPE>(b)));
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T norm(Float3<T> v) noexcept {
        if constexpr (std::is_same_v<T, half_t>) {
            auto tmp = Float3<HALF_ARITHMETIC_TYPE>(v);
            return static_cast<T>(sqrt(dot(tmp, tmp)));
        }
        return sqrt(dot(v, v));
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T length(Float3<T> v) noexcept {
        return norm(v);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> normalize(Float3<T> v) noexcept {
        return v / norm(v);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> cross(Float3<T> a, Float3<T> b) noexcept {
        if constexpr (std::is_same_v<T, half_t>)
            return Float3<T>(cross(Float3<HALF_ARITHMETIC_TYPE>(a), Float3<HALF_ARITHMETIC_TYPE>(b)));
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T min(Float3<T> v) noexcept {
        return (v[0] < v[1]) ? min(v[0], v[2]) : min(v[1], v[2]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> min(Float3<T> lhs, Float3<T> rhs) noexcept {
        return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1]), min(lhs[2], rhs[2])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> min(Float3<T> lhs, T rhs) noexcept {
        return {min(lhs[0], rhs), min(lhs[1], rhs), min(lhs[2], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> min(T lhs, Float3<T> rhs) noexcept {
        return {min(lhs, rhs[0]), min(lhs, rhs[1]), min(lhs, rhs[2])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T max(Float3<T> v) noexcept {
        return (v[0] > v[1]) ? max(v[0], v[2]) : max(v[1], v[2]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> max(Float3<T> lhs, Float3<T> rhs) noexcept {
        return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1]), max(lhs[2], rhs[2])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> max(Float3<T> lhs, T rhs) noexcept {
        return {max(lhs[0], rhs), max(lhs[1], rhs), max(lhs[2], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> max(T lhs, Float3<T> rhs) noexcept {
        return {max(lhs, rhs[0]), max(lhs, rhs[1]), max(lhs, rhs[2])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> clamp(Float3<T> lhs, Float3<T> low, Float3<T> high) noexcept {
        return min(max(lhs, low), high);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> clamp(Float3<T> lhs, T low, T high) noexcept {
        return min(max(lhs, low), high);
    }

    #define NOA_ULP_ 2
    #define NOA_EPSILON_ 1e-6f

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool3 isEqual(Float3<T> a, Float3<T> b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a[0], b[0], e), isEqual<ULP>(a[1], b[1], e), isEqual<ULP>(a[2], b[2], e)};
    }

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool3 isEqual(Float3<T> a, T b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a[0], b, e), isEqual<ULP>(a[1], b, e), isEqual<ULP>(a[2], b, e)};
    }

    template<uint ULP = NOA_ULP_, typename T>
    [[nodiscard]] NOA_FHD constexpr Bool3 isEqual(T a, Float3<T> b, T e = NOA_EPSILON_) noexcept {
        return {isEqual<ULP>(a, b[0], e), isEqual<ULP>(a, b[1], e), isEqual<ULP>(a, b[2], e)};
    }

    #undef NOA_ULP_
    #undef NOA_EPSILON_

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr Float3<T> sort(Float3<T> v, U&& comp) noexcept {
        smallStableSort<3>(v.get(), std::forward<U>(comp));
        return v;
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> sort(Float3<T> v) noexcept {
        return sort(v, [](const T& a, const T& b) { return a < b; });
    }
}
