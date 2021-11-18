/// \file noa/common/types/Float2.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 2 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool2.h"

namespace noa {
    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Float2 {
    public:
        static_assert(noa::traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{};

    public: // Component accesses
        static constexpr size_t COUNT = 2;

        NOA_HD constexpr T& operator[](size_t i) noexcept {
            NOA_ASSERT(i < this->COUNT);
            if (i == 1)
                return this->y;
            else
                return this->x;
        }

        NOA_HD constexpr const T& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < this->COUNT);
            if (i == 1)
                return this->y;
            else
                return this->x;
        }

    public: // Default constructors
        constexpr Float2() noexcept = default;
        constexpr Float2(const Float2&) noexcept = default;
        constexpr Float2(Float2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y>
        NOA_HD constexpr Float2(X xi, Y yi) noexcept
                : x(static_cast<T>(xi)),
                  y(static_cast<T>(yi)) {}

        template<typename U>
        NOA_HD constexpr explicit Float2(U v) noexcept
                : x(static_cast<T>(v)),
                  y(static_cast<T>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Float2(Float2<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)) {}

        template<typename U>
        NOA_HD constexpr explicit Float2(Int2<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)) {}

        template<typename U>
        NOA_HD constexpr explicit Float2(U* ptr) noexcept
                : x(static_cast<T>(ptr[0])),
                  y(static_cast<T>(ptr[1])) {}

    public: // Assignment operators
        constexpr Float2& operator=(const Float2& v) noexcept = default;
        constexpr Float2& operator=(Float2&& v) noexcept = default;

        NOA_HD constexpr Float2& operator=(T v) noexcept {
            this->x = v;
            this->y = v;
            return *this;
        }

        NOA_HD constexpr Float2& operator=(T* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            return *this;
        }

        NOA_HD constexpr Float2& operator+=(Float2 rhs) noexcept {
            this->x += rhs.x;
            this->y += rhs.y;
            return *this;
        }

        NOA_HD constexpr Float2& operator-=(Float2 rhs) noexcept {
            this->x -= rhs.x;
            this->y -= rhs.y;
            return *this;
        }

        NOA_HD constexpr Float2& operator*=(Float2 rhs) noexcept {
            this->x *= rhs.x;
            this->y *= rhs.y;
            return *this;
        }

        NOA_HD constexpr Float2& operator/=(Float2 rhs) noexcept {
            this->x /= rhs.x;
            this->y /= rhs.y;
            return *this;
        }

        NOA_HD constexpr Float2& operator+=(T rhs) noexcept {
            this->x += rhs;
            this->y += rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator-=(T rhs) noexcept {
            this->x -= rhs;
            this->y -= rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator*=(T rhs) noexcept {
            this->x *= rhs;
            this->y *= rhs;
            return *this;
        }

        NOA_HD constexpr Float2& operator/=(T rhs) noexcept {
            this->x /= rhs;
            this->y /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Float2 operator+(Float2 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Float2 operator-(Float2 v) noexcept {
            return {-v.x, -v.y};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Float2 operator+(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x + rhs.x, lhs.y + rhs.y};
        }

        friend NOA_HD constexpr Float2 operator+(T lhs, Float2 rhs) noexcept {
            return {lhs + rhs.x, lhs + rhs.y};
        }

        friend NOA_HD constexpr Float2 operator+(Float2 lhs, T rhs) noexcept {
            return {lhs.x + rhs, lhs.y + rhs};
        }

        friend NOA_HD constexpr Float2 operator-(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x - rhs.x, lhs.y - rhs.y};
        }

        friend NOA_HD constexpr Float2 operator-(T lhs, Float2 rhs) noexcept {
            return {lhs - rhs.x, lhs - rhs.y};
        }

        friend NOA_HD constexpr Float2 operator-(Float2 lhs, T rhs) noexcept {
            return {lhs.x - rhs, lhs.y - rhs};
        }

        friend NOA_HD constexpr Float2 operator*(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x * rhs.x, lhs.y * rhs.y};
        }

        friend NOA_HD constexpr Float2 operator*(T lhs, Float2 rhs) noexcept {
            return {lhs * rhs.x, lhs * rhs.y};
        }

        friend NOA_HD constexpr Float2 operator*(Float2 lhs, T rhs) noexcept {
            return {lhs.x * rhs, lhs.y * rhs};
        }

        friend NOA_HD constexpr Float2 operator/(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x / rhs.x, lhs.y / rhs.y};
        }

        friend NOA_HD constexpr Float2 operator/(T lhs, Float2 rhs) noexcept {
            return {lhs / rhs.x, lhs / rhs.y};
        }

        friend NOA_HD constexpr Float2 operator/(Float2 lhs, T rhs) noexcept {
            return {lhs.x / rhs, lhs.y / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool2 operator>(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x > rhs.x, lhs.y > rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator>(Float2 lhs, T rhs) noexcept {
            return {lhs.x > rhs, lhs.y > rhs};
        }

        friend NOA_HD constexpr Bool2 operator>(T lhs, Float2 rhs) noexcept {
            return {lhs > rhs.x, lhs > rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x < rhs.x, lhs.y < rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<(Float2 lhs, T rhs) noexcept {
            return {lhs.x < rhs, lhs.y < rhs};
        }

        friend NOA_HD constexpr Bool2 operator<(T lhs, Float2 rhs) noexcept {
            return {lhs < rhs.x, lhs < rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator>=(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x >= rhs.x, lhs.y >= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator>=(Float2 lhs, T rhs) noexcept {
            return {lhs.x >= rhs, lhs.y >= rhs};
        }

        friend NOA_HD constexpr Bool2 operator>=(T lhs, Float2 rhs) noexcept {
            return {lhs >= rhs.x, lhs >= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<=(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x <= rhs.x, lhs.y <= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<=(Float2 lhs, T rhs) noexcept {
            return {lhs.x <= rhs, lhs.y <= rhs};
        }

        friend NOA_HD constexpr Bool2 operator<=(T lhs, Float2 rhs) noexcept {
            return {lhs <= rhs.x, lhs <= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator==(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x == rhs.x, lhs.y == rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator==(Float2 lhs, T rhs) noexcept {
            return {lhs.x == rhs, lhs.y == rhs};
        }

        friend NOA_HD constexpr Bool2 operator==(T lhs, Float2 rhs) noexcept {
            return {lhs == rhs.x, lhs == rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator!=(Float2 lhs, Float2 rhs) noexcept {
            return {lhs.x != rhs.x, lhs.y != rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator!=(Float2 lhs, T rhs) noexcept {
            return {lhs.x != rhs, lhs.y != rhs};
        }

        friend NOA_HD constexpr Bool2 operator!=(T lhs, Float2 rhs) noexcept {
            return {lhs != rhs.x, lhs != rhs.y};
        }
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr Float2<T> floor(Float2<T> v) noexcept {
            return Float2<T>(floor(v.x), floor(v.y));
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> ceil(Float2<T> v) noexcept {
            return Float2<T>(ceil(v.x), ceil(v.y));
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> abs(Float2<T> v) noexcept {
            return Float2<T>(abs(v.x), abs(v.y));
        }

        template<typename T>
        NOA_FHD constexpr T dot(Float2<T> a, Float2<T> b) noexcept {
            return a.x * b.x + a.y * b.y;
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(Float2<T> a, Float2<T> b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(Float2<T> v) noexcept {
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(Float2<T> v) noexcept {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> normalize(Float2<T> v) noexcept {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr T sum(Float2<T> v) noexcept {
            return v.x + v.y;
        }

        template<typename T>
        NOA_FHD constexpr T prod(Float2<T> v) noexcept {
            return v.x * v.y;
        }

        template<typename T>
        NOA_FHD constexpr T min(Float2<T> v) noexcept {
            return min(v.x, v.y);
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> min(Float2<T> lhs, Float2<T> rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> min(Float2<T> lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> min(T lhs, Float2<T> rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr T max(Float2<T> v) noexcept {
            return max(v.x, v.y);
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> max(Float2<T> lhs, Float2<T> rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> max(Float2<T> lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> max(T lhs, Float2<T> rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y)};
        }

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(Float2<T> a, Float2<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(Float2<T> a, T b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a.x, b, e), isEqual<ULP>(a.y, b, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(T a, Float2<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e)};
        }

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T>
        struct p_is_float2 : std::false_type {};
        template<typename T>
        struct p_is_float2<noa::Float2<T>> : std::true_type {};
        template<typename T> using is_float2 = std::bool_constant<p_is_float2<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float2_v = is_float2<T>::value;

        template<typename T>
        struct proclaim_is_floatX<noa::Float2<T>> : std::true_type {};
    }

    using float2_t = Float2<float>;
    using double2_t = Float2<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(Float2<T> v) noexcept {
        return {v.x, v.y};
    }

    template<>
    NOA_IH std::string string::typeName<float2_t>() { return "float2"; }
    template<>
    NOA_IH std::string string::typeName<double2_t>() { return "double2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Float2<T> v) {
        os << string::format("({:.3f},{:.3f})", v.x, v.y);
        return os;
    }
}
