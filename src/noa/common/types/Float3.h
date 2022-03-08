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
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool3.h"
#include "noa/common/types/Half.h"

namespace noa {
    template<typename>
    class Int3;

    template<typename T>
    class Float3 {
    public:
        typedef T value_type;

    public: // Default Constructors
        constexpr Float3() noexcept = default;
        constexpr Float3(const Float3&) noexcept = default;
        constexpr Float3(Float3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z>
        NOA_HD constexpr Float3(X x, Y y, Z z) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(y), static_cast<T>(z)} {}

        template<typename U>
        NOA_HD constexpr explicit Float3(U x) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(x), static_cast<T>(x)} {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Float3<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Int3<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2])} {}

        template<typename U>
        NOA_HD constexpr explicit Float3(U* ptr) noexcept
                : m_data{static_cast<T>(ptr[0]), static_cast<T>(ptr[1]), static_cast<T>(ptr[2])} {}

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
        friend NOA_HD constexpr Float3 operator+(Float3 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Float3 operator-(Float3 v) noexcept {
            return {-v[0], -v[1], -v[2]};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Float3 operator+(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator+(T lhs, Float3 rhs) noexcept {
            return {lhs + rhs[0], lhs + rhs[1], lhs + rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator+(Float3 lhs, T rhs) noexcept {
            return {lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs};
        }

        friend NOA_HD constexpr Float3 operator-(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator-(T lhs, Float3 rhs) noexcept {
            return {lhs - rhs[0], lhs - rhs[1], lhs - rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator-(Float3 lhs, T rhs) noexcept {
            return {lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs};
        }

        friend NOA_HD constexpr Float3 operator*(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator*(T lhs, Float3 rhs) noexcept {
            return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator*(Float3 lhs, T rhs) noexcept {
            return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs};
        }

        friend NOA_HD constexpr Float3 operator/(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator/(T lhs, Float3 rhs) noexcept {
            return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2]};
        }

        friend NOA_HD constexpr Float3 operator/(Float3 lhs, T rhs) noexcept {
            return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool3 operator>(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] > rhs[0], lhs[1] > rhs[1], lhs[2] > rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator>(Float3 lhs, T rhs) noexcept {
            return {lhs[0] > rhs, lhs[1] > rhs, lhs[2] > rhs};
        }

        friend NOA_HD constexpr Bool3 operator>(T lhs, Float3 rhs) noexcept {
            return {lhs > rhs[0], lhs > rhs[1], lhs > rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] < rhs[0], lhs[1] < rhs[1], lhs[2] < rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<(Float3 lhs, T rhs) noexcept {
            return {lhs[0] < rhs, lhs[1] < rhs, lhs[2] < rhs};
        }

        friend NOA_HD constexpr Bool3 operator<(T lhs, Float3 rhs) noexcept {
            return {lhs < rhs[0], lhs < rhs[1], lhs < rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator>=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1], lhs[2] >= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator>=(Float3 lhs, T rhs) noexcept {
            return {lhs[0] >= rhs, lhs[1] >= rhs, lhs[2] >= rhs};
        }

        friend NOA_HD constexpr Bool3 operator>=(T lhs, Float3 rhs) noexcept {
            return {lhs >= rhs[0], lhs >= rhs[1], lhs >= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1], lhs[2] <= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator<=(Float3 lhs, T rhs) noexcept {
            return {lhs[0] <= rhs, lhs[1] <= rhs, lhs[2] <= rhs};
        }

        friend NOA_HD constexpr Bool3 operator<=(T lhs, Float3 rhs) noexcept {
            return {lhs <= rhs[0], lhs <= rhs[1], lhs <= rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator==(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator==(Float3 lhs, T rhs) noexcept {
            return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs};
        }

        friend NOA_HD constexpr Bool3 operator==(T lhs, Float3 rhs) noexcept {
            return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator!=(Float3 lhs, Float3 rhs) noexcept {
            return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2]};
        }

        friend NOA_HD constexpr Bool3 operator!=(Float3 lhs, T rhs) noexcept {
            return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs};
        }

        friend NOA_HD constexpr Bool3 operator!=(T lhs, Float3 rhs) noexcept {
            return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 3;

        NOA_HD constexpr T& operator[](size_t i) noexcept {
            NOA_ASSERT(i < COUNT);
            return m_data[i];
        }

        NOA_HD constexpr const T& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr Float3 flip() const noexcept { return {m_data[2], m_data[1], m_data[0]}; }

    private:
        static_assert(noa::traits::is_float_v<T>);
        T m_data[3]{};
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr Float3<T> toRad(Float3<T> v) noexcept {
            return Float3<T>(toRad(v[0]), toRad(v[1]), toRad(v[2]));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> toDeg(Float3<T> v) noexcept {
            return Float3<T>(toDeg(v[0]), toDeg(v[1]), toDeg(v[2]));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> floor(Float3<T> v) noexcept {
            return Float3<T>(floor(v[0]), floor(v[1]), floor(v[2]));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> ceil(Float3<T> v) noexcept {
            return Float3<T>(ceil(v[0]), ceil(v[1]), ceil(v[2]));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> abs(Float3<T> v) noexcept {
            return Float3<T>(abs(v[0]), abs(v[1]), abs(v[2]));
        }

        template<typename T>
        NOA_FHD constexpr T sum(Float3<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(sum(Float3<HALF_ARITHMETIC_TYPE>(v)));
            return v[0] + v[1] + v[2];
        }

        template<typename T>
        NOA_FHD constexpr T prod(Float3<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(prod(Float3<HALF_ARITHMETIC_TYPE>(v)));
            return v[0] * v[1] * v[2];
        }

        template<typename T>
        NOA_FHD constexpr T dot(Float3<T> a, Float3<T> b) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return static_cast<T>(dot(Float3<HALF_ARITHMETIC_TYPE>(a), Float3<HALF_ARITHMETIC_TYPE>(b)));
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(Float3<T> a, Float3<T> b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(Float3<T> v) noexcept {
            if constexpr (std::is_same_v<T, half_t>) {
                auto tmp = Float3<HALF_ARITHMETIC_TYPE>(v);
                return static_cast<T>(sqrt(dot(tmp, tmp)));
            }
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(Float3<T> v) noexcept {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> normalize(Float3<T> v) noexcept {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> cross(Float3<T> a, Float3<T> b) noexcept {
            if constexpr (std::is_same_v<T, half_t>)
                return Float3<T>(cross(Float3<HALF_ARITHMETIC_TYPE>(a), Float3<HALF_ARITHMETIC_TYPE>(b)));
            return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
        }

        template<typename T>
        NOA_FHD constexpr T min(Float3<T> v) noexcept {
            return (v[0] < v[1]) ? min(v[0], v[2]) : min(v[1], v[2]);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(Float3<T> lhs, Float3<T> rhs) noexcept {
            return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1]), min(lhs[2], rhs[2])};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(Float3<T> lhs, T rhs) noexcept {
            return {min(lhs[0], rhs), min(lhs[1], rhs), min(lhs[2], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(T lhs, Float3<T> rhs) noexcept {
            return {min(lhs, rhs[0]), min(lhs, rhs[1]), min(lhs, rhs[2])};
        }

        template<typename T>
        NOA_FHD constexpr T max(Float3<T> v) noexcept {
            return (v[0] > v[1]) ? max(v[0], v[2]) : max(v[1], v[2]);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(Float3<T> lhs, Float3<T> rhs) noexcept {
            return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1]), max(lhs[2], rhs[2])};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(Float3<T> lhs, T rhs) noexcept {
            return {max(lhs[0], rhs), max(lhs[1], rhs), max(lhs[2], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(T lhs, Float3<T> rhs) noexcept {
            return {max(lhs, rhs[0]), max(lhs, rhs[1]), max(lhs, rhs[2])};
        }

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(Float3<T> a, Float3<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a[0], b[0], e), isEqual<ULP>(a[1], b[1], e), isEqual<ULP>(a[2], b[2], e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(Float3<T> a, T b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a[0], b, e), isEqual<ULP>(a[1], b, e), isEqual<ULP>(a[2], b, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(T a, Float3<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a, b[0], e), isEqual<ULP>(a, b[1], e), isEqual<ULP>(a, b[2], e)};
        }

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T>
        struct p_is_float3 : std::false_type {};
        template<typename T>
        struct p_is_float3<noa::Float3<T>> : std::true_type {};
        template<typename T> using is_float3 = std::bool_constant<p_is_float3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float3_v = is_float3<T>::value;

        template<typename T>
        struct proclaim_is_floatX<noa::Float3<T>> : std::true_type {};
    }

    using half3_t = Float3<half_t>;
    using float3_t = Float3<float>;
    using double3_t = Float3<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(Float3<T> v) noexcept {
        return {v[0], v[1], v[2]};
    }

    template<>
    NOA_IH std::string string::typeName<half3_t>() { return "half3"; }
    template<>
    NOA_IH std::string string::typeName<float3_t>() { return "float3"; }
    template<>
    NOA_IH std::string string::typeName<double3_t>() { return "double3"; }

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
