/// \file noa/common/types/Int4.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 4 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool4.h"

namespace noa {
    template<typename>
    class Float4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Int4 {
    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
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
        constexpr Int4() noexcept = default;
        constexpr Int4(const Int4&) noexcept = default;
        constexpr Int4(Int4&&) noexcept = default;

    public: // Conversion constructors
        template<class X, class Y, class Z, class W>
        NOA_HD constexpr Int4(X xi, Y yi, Z zi, W wi) noexcept
                : x(static_cast<T>(xi)),
                  y(static_cast<T>(yi)),
                  z(static_cast<T>(zi)),
                  w(static_cast<T>(wi)) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(U v) noexcept
                : x(static_cast<T>(v)),
                  y(static_cast<T>(v)),
                  z(static_cast<T>(v)),
                  w(static_cast<T>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Int4<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)),
                  w(static_cast<T>(v.w)) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Float4<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)),
                  z(static_cast<T>(v.z)),
                  w(static_cast<T>(v.w)) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(U* ptr) noexcept
                : x(static_cast<T>(ptr[0])),
                  y(static_cast<T>(ptr[1])),
                  z(static_cast<T>(ptr[2])),
                  w(static_cast<T>(ptr[3])) {}

    public: // Assignment operators
        constexpr Int4& operator=(const Int4& v) noexcept = default;
        constexpr Int4& operator=(Int4&& v) noexcept = default;

        NOA_HD constexpr Int4& operator=(T v) noexcept {
            this->x = v;
            this->y = v;
            this->z = v;
            this->w = v;
            return *this;
        }

        NOA_HD constexpr Int4& operator=(T* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            this->z = ptr[2];
            this->w = ptr[3];
            return *this;
        }

        NOA_HD constexpr Int4& operator+=(Int4 rhs) noexcept {
            this->x += rhs.x;
            this->y += rhs.y;
            this->z += rhs.z;
            this->w += rhs.w;
            return *this;
        }

        NOA_HD constexpr Int4& operator-=(Int4 rhs) noexcept {
            this->x -= rhs.x;
            this->y -= rhs.y;
            this->z -= rhs.z;
            this->w -= rhs.w;
            return *this;
        }

        NOA_HD constexpr Int4& operator*=(Int4 rhs) noexcept {
            this->x *= rhs.x;
            this->y *= rhs.y;
            this->z *= rhs.z;
            this->w *= rhs.w;
            return *this;
        }

        NOA_HD constexpr Int4& operator/=(Int4 rhs) noexcept {
            this->x /= rhs.x;
            this->y /= rhs.y;
            this->z /= rhs.z;
            this->w /= rhs.w;
            return *this;
        }

        NOA_HD constexpr Int4& operator+=(T rhs) noexcept {
            this->x += rhs;
            this->y += rhs;
            this->z += rhs;
            this->w += rhs;
            return *this;
        }

        NOA_HD constexpr Int4& operator-=(T rhs) noexcept {
            this->x -= rhs;
            this->y -= rhs;
            this->z -= rhs;
            this->w -= rhs;
            return *this;
        }

        NOA_HD constexpr Int4& operator*=(T rhs) noexcept {
            this->x *= rhs;
            this->y *= rhs;
            this->z *= rhs;
            this->w *= rhs;
            return *this;
        }

        NOA_HD constexpr Int4& operator/=(T rhs) noexcept {
            this->x /= rhs;
            this->y /= rhs;
            this->z /= rhs;
            this->w /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Int4 operator+(Int4 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Int4 operator-(Int4 v) noexcept {
            return {-v.x, -v.y, -v.z, -v.w};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Int4 operator+(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
        }

        friend NOA_HD constexpr Int4 operator+(T lhs, Int4 rhs) noexcept {
            return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
        }

        friend NOA_HD constexpr Int4 operator+(Int4 lhs, T rhs) noexcept {
            return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
        }

        friend NOA_HD constexpr Int4 operator-(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
        }

        friend NOA_HD constexpr Int4 operator-(T lhs, Int4 rhs) noexcept {
            return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
        }

        friend NOA_HD constexpr Int4 operator-(Int4 lhs, T rhs) noexcept {
            return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
        }

        friend NOA_HD constexpr Int4 operator*(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
        }

        friend NOA_HD constexpr Int4 operator*(T lhs, Int4 rhs) noexcept {
            return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
        }

        friend NOA_HD constexpr Int4 operator*(Int4 lhs, T rhs) noexcept {
            return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
        }

        friend NOA_HD constexpr Int4 operator/(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
        }

        friend NOA_HD constexpr Int4 operator/(T lhs, Int4 rhs) noexcept {
            return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
        }

        friend NOA_HD constexpr Int4 operator/(Int4 lhs, T rhs) noexcept {
            return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool4 operator>(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator>(Int4 lhs, T rhs) noexcept {
            return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs, lhs.w > rhs};
        }

        friend NOA_HD constexpr Bool4 operator>(T lhs, Int4 rhs) noexcept {
            return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z, lhs > rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<(Int4 lhs, T rhs) noexcept {
            return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs, lhs.w < rhs};
        }

        friend NOA_HD constexpr Bool4 operator<(T lhs, Int4 rhs) noexcept {
            return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z, lhs < rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator>=(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator>=(Int4 lhs, T rhs) noexcept {
            return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs, lhs.w >= rhs};
        }

        friend NOA_HD constexpr Bool4 operator>=(T lhs, Int4 rhs) noexcept {
            return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z, lhs >= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<=(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator<=(Int4 lhs, T rhs) noexcept {
            return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs, lhs.w <= rhs};
        }

        friend NOA_HD constexpr Bool4 operator<=(T lhs, Int4 rhs) noexcept {
            return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z, lhs <= rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator==(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator==(Int4 lhs, T rhs) noexcept {
            return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs, lhs.w == rhs};
        }

        friend NOA_HD constexpr Bool4 operator==(T lhs, Int4 rhs) noexcept {
            return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z, lhs == rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator!=(Int4 lhs, Int4 rhs) noexcept {
            return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
        }

        friend NOA_HD constexpr Bool4 operator!=(Int4 lhs, T rhs) noexcept {
            return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs, lhs.w != rhs};
        }

        friend NOA_HD constexpr Bool4 operator!=(T lhs, Int4 rhs) noexcept {
            return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z, lhs != rhs.w};
        }
    };

    template<typename T>
    NOA_FHD constexpr T elements(Int4<T> v) noexcept {
        return v.x * v.y * v.z * v.w;
    }

    template<typename T>
    NOA_FHD constexpr T elementsSlice(Int4<T> v) noexcept {
        return v.x * v.y;
    }

    template<typename T>
    NOA_FHD constexpr T elementsFFT(Int4<T> v) noexcept {
        return (v.x / 2 + 1) * v.y * v.z * v.w;
    }

    template<typename T>
    NOA_FHD constexpr Int4<T> shapeFFT(Int4<T> v) noexcept {
        return {v.x / 2 + 1, v.y, v.z, v.w};
    }

    template<typename T>
    NOA_FHD constexpr Int4<T> slice(Int4<T> v) noexcept {
        return {v.x, v.y, 1, 1};
    }

    namespace math {
        template<typename T>
        NOA_FHD constexpr T sum(Int4<T> v) noexcept {
            return v.x + v.y + v.z + v.w;
        }

        template<typename T>
        NOA_FHD constexpr T prod(Int4<T> v) noexcept {
            return v.x * v.y * v.z * v.w;
        }

        template<typename T>
        NOA_FHD constexpr T min(Int4<T> v) noexcept {
            return min(min(v.x, v.y), min(v.z, v.w));
        }

        template<typename T>
        NOA_FHD constexpr Int4<T> min(Int4<T> lhs, Int4<T> rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
        }

        template<typename T>
        NOA_FHD constexpr Int4<T> min(Int4<T> lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int4<T> min(T lhs, Int4<T> rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
        }

        template<typename T>
        NOA_FHD constexpr T max(Int4<T> v) noexcept {
            return max(max(v.x, v.y), max(v.z, v.w));
        }

        template<typename T>
        NOA_FHD constexpr Int4<T> max(Int4<T> lhs, Int4<T> rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
        }

        template<typename T>
        NOA_FHD constexpr Int4<T> max(Int4<T> lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int4<T> max(T lhs, Int4<T> rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
        }
    }

    namespace traits {
        template<typename>
        struct p_is_int4 : std::false_type {};
        template<typename T>
        struct p_is_int4<noa::Int4<T>> : std::true_type {};
        template<typename T> using is_int4 = std::bool_constant<p_is_int4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

        template<typename>
        struct p_is_uint4 : std::false_type {};
        template<typename T>
        struct p_is_uint4<noa::Int4<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

        template<typename T>
        struct proclaim_is_intX<noa::Int4<T>> : std::true_type {};
        template<typename T>
        struct proclaim_is_uintX<noa::Int4<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }

    using int4_t = Int4<int>;
    using uint4_t = Int4<uint>;
    using long4_t = Int4<int64_t>;
    using ulong4_t = Int4<uint64_t>;
    using size4_t = Int4<size_t>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(Int4<T> v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<>
    NOA_IH std::string string::typeName<int4_t>() { return "int4"; }
    template<>
    NOA_IH std::string string::typeName<uint4_t>() { return "uint4"; }
    template<>
    NOA_IH std::string string::typeName<long4_t>() { return "long4"; }
    template<>
    NOA_IH std::string string::typeName<ulong4_t>() { return "ulong4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Int4<T> v) {
        os << string::format("({},{},{},{})", v.x, v.y, v.z, v.w);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Int4<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Int4<T>& vec, FormatContext& ctx) {
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
