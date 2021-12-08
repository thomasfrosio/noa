/// \file noa/common/types/Float4.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 4 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool4.h"

namespace noa {
    template<typename>
    class Int4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Float4 {
    public:
        static_assert(noa::traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{}, z{}, w{};

    public: // Component accesses
        static constexpr size_t COUNT = 4;

        NOA_HD constexpr T& operator[](size_t i) noexcept {
            NOA_ASSERT(i < this->COUNT);
            switch (i) {
                default:
                case 0:
                    return this->x;
                case 1:
                    return this->y;
                case 2:
                    return this->z;
                case 3:
                    return this->w;
            }
        }

        NOA_HD constexpr const T& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < this->COUNT);
            switch (i) {
                default:
                case 0:
                    return this->x;
                case 1:
                    return this->y;
                case 2:
                    return this->z;
                case 3:
                    return this->w;
            }
        }

    public: // Default Constructors
        constexpr Float4() noexcept = default;
        constexpr Float4(const Float4&) noexcept = default;
        constexpr Float4(Float4&&) noexcept = default;

    public: // Conversion constructors
        template<class X, class Y, class Z, class W>
        NOA_HD constexpr Float4(X xi, Y yi, Z zi, W wi) noexcept
                : x(static_cast<T>(xi)),
                  y(static_cast<T>(yi)),
                  z(static_cast<T>(zi)),
                  w(static_cast<T>(wi)) {}

        template<typename U>
        NOA_HD constexpr explicit Float4(U v) noexcept
                : x(static_cast<T>(v)),
                  y(static_cast<T>(v)),
                  z(static_cast<T>(v)),
                  w(static_cast<T>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Float4(Float4<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)),
                  w(static_cast<T>(v.w)) {}

        template<typename U>
        NOA_HD constexpr explicit Float4(Int4<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)),
                  w(static_cast<T>(v.w)) {}

        template<typename U>
        NOA_HD constexpr explicit Float4(U* ptr) noexcept
                : x(static_cast<T>(ptr[0])),
                  y(static_cast<T>(ptr[1])),
                  z(static_cast<T>(ptr[2])),
                  w(static_cast<T>(ptr[3])) {}

    public: // Assignment operators
        constexpr Float4& operator=(const Float4& v) noexcept = default;
        constexpr Float4& operator=(Float4&& v) noexcept = default;

        NOA_HD constexpr Float4& operator=(T v) noexcept {
            this->x = v;
            this->y = v;
            this->z = v;
            this->w = v;
            return *this;
        }

        NOA_HD constexpr Float4& operator=(T* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            this->z = ptr[2];
            this->w = ptr[3];
            return *this;
        }

        NOA_HD constexpr Float4& operator+=(Float4 rhs) noexcept {
            this->x += rhs.x;
            this->y += rhs.y;
            this->z += rhs.z;
            this->w += rhs.w;
            return *this;
        }

        NOA_HD constexpr Float4& operator-=(Float4 rhs) noexcept {
            this->x -= rhs.x;
            this->y -= rhs.y;
            this->z -= rhs.z;
            this->w -= rhs.w;
            return *this;
        }

        NOA_HD constexpr Float4& operator*=(Float4 rhs) noexcept {
            this->x *= rhs.x;
            this->y *= rhs.y;
            this->z *= rhs.z;
            this->w *= rhs.w;
            return *this;
        }

        NOA_HD constexpr Float4& operator/=(Float4 rhs) noexcept {
            this->x /= rhs.x;
            this->y /= rhs.y;
            this->z /= rhs.z;
            this->w /= rhs.w;
            return *this;
        }

        NOA_HD constexpr Float4& operator+=(T rhs) noexcept {
            this->x += rhs;
            this->y += rhs;
            this->z += rhs;
            this->w += rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator-=(T rhs) noexcept {
            this->x -= rhs;
            this->y -= rhs;
            this->z -= rhs;
            this->w -= rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator*=(T rhs) noexcept {
            this->x *= rhs;
            this->y *= rhs;
            this->z *= rhs;
            this->w *= rhs;
            return *this;
        }

        NOA_HD constexpr Float4& operator/=(T rhs) noexcept {
            this->x /= rhs;
            this->y /= rhs;
            this->z /= rhs;
            this->w /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Float4 operator+(Float4 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Float4 operator-(Float4 v) noexcept {
            return {-v.x, -v.y, -v.z, -v.w};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Float4 operator+(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
        }

        friend NOA_HD constexpr Float4 operator+(T lhs, Float4 rhs) noexcept {
            return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
        }

        friend NOA_HD constexpr Float4 operator+(Float4 lhs, T rhs) noexcept {
            return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
        }

        friend NOA_HD constexpr Float4 operator-(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
        }

        friend NOA_HD constexpr Float4 operator-(T lhs, Float4 rhs) noexcept {
            return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
        }

        friend NOA_HD constexpr Float4 operator-(Float4 lhs, T rhs) noexcept {
            return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
        }

        friend NOA_HD constexpr Float4 operator*(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
        }

        friend NOA_HD constexpr Float4 operator*(T lhs, Float4 rhs) noexcept {
            return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
        }

        friend NOA_HD constexpr Float4 operator*(Float4 lhs, T rhs) noexcept {
            return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
        }

        friend NOA_HD constexpr Float4 operator/(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
        }

        friend NOA_HD constexpr Float4 operator/(T lhs, Float4 rhs) noexcept {
            return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
        }

        friend NOA_HD constexpr Float4 operator/(Float4 lhs, T rhs) noexcept {
            return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool4 operator>(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator>(Float4 lhs, T rhs) noexcept {
            return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs, lhs.w > rhs};
        }

        friend NOA_HD constexpr Bool4 operator>(T lhs, Float4 rhs) noexcept {
            return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z, lhs > rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<(Float4 lhs, T rhs) noexcept {
            return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs, lhs.w < rhs};
        }

        friend NOA_HD constexpr Bool4 operator<(T lhs, Float4 rhs) noexcept {
            return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z, lhs < rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator>=(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator>=(Float4 lhs, T rhs) noexcept {
            return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs, lhs.w >= rhs};
        }

        friend NOA_HD constexpr Bool4 operator>=(T lhs, Float4 rhs) noexcept {
            return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z, lhs >= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<=(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<=(Float4 lhs, T rhs) noexcept {
            return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs, lhs.w <= rhs};
        }

        friend NOA_HD constexpr Bool4 operator<=(T lhs, Float4 rhs) noexcept {
            return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z, lhs <= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator==(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator==(Float4 lhs, T rhs) noexcept {
            return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs, lhs.w == rhs};
        }

        friend NOA_HD constexpr Bool4 operator==(T lhs, Float4 rhs) noexcept {
            return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z, lhs == rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator!=(Float4 lhs, Float4 rhs) noexcept {
            return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator!=(Float4 lhs, T rhs) noexcept {
            return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs, lhs.w != rhs};
        }

        friend NOA_HD constexpr Bool4 operator!=(T lhs, Float4 rhs) noexcept {
            return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z, lhs != rhs.w};
        }
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr Float4<T> floor(Float4<T> v) noexcept {
            return Float4<T>(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> ceil(Float4<T> v) noexcept {
            return Float4<T>(ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w));
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> abs(Float4<T> v) noexcept {
            return Float4<T>(abs(v.x), abs(v.y), abs(v.z), abs(v.w));
        }

        template<typename T>
        NOA_FHD constexpr T sum(Float4<T> v) noexcept {
            return v.x + v.y + v.z + v.w;
        }

        template<typename T>
        NOA_FHD constexpr T prod(Float4<T> v) noexcept {
            return v.x * v.y * v.z * v.w;
        }

        template<typename T>
        NOA_FHD constexpr T dot(Float4<T> a, Float4<T> b) noexcept {
            return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(Float4<T> a, Float4<T> b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(Float4<T> v) noexcept {
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(Float4<T> v) noexcept {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> normalize(Float4<T> v) noexcept {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr T min(Float4<T> v) noexcept {
            return min(min(v.x, v.y), min(v.z, v.w));
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> min(Float4<T> lhs, Float4<T> rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> min(Float4<T> lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> min(T lhs, Float4<T> rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
        }

        template<typename T>
        NOA_FHD constexpr T max(Float4<T> v) noexcept {
            return max(max(v.x, v.y), max(v.z, v.w));
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> max(Float4<T> lhs, Float4<T> rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> max(Float4<T> lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float4<T> max(T lhs, Float4<T> rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
        }

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(Float4<T> a, Float4<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e),
                    isEqual<ULP>(a.z, b.z, e), isEqual<ULP>(a.w, b.w, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(Float4<T> a, T b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(b, a.x, e), isEqual<ULP>(b, a.y, e),
                    isEqual<ULP>(b, a.z, e), isEqual<ULP>(b, a.w, e)};
        }

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(T a, Float4<T> b, T e = NOA_EPSILON_) noexcept {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e),
                    isEqual<ULP>(a, b.z, e), isEqual<ULP>(a, b.w, e)};
        }

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T>
        struct p_is_float4 : std::false_type {};
        template<typename T>
        struct p_is_float4<noa::Float4<T>> : std::true_type {};
        template<typename T> using is_float4 = std::bool_constant<p_is_float4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float4_v = is_float4<T>::value;

        template<typename T>
        struct proclaim_is_floatX<noa::Float4<T>> : std::true_type {};
    }

    using float4_t = Float4<float>;
    using double4_t = Float4<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(Float4<T> v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<>
    NOA_IH std::string string::typeName<float4_t>() { return "float4"; }
    template<>
    NOA_IH std::string string::typeName<double4_t>() { return "double4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Float4<T> v) {
        os << string::format("({:.3f},{:.3f},{:.3f},{:.3f})", v.x, v.y, v.z, v.w);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Float4<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Float4<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.x, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.y, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.z, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.w, ctx);
            *out = ')';
            return out;
        }
    };
}
