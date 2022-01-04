/// \file noa/common/types/Int2.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 2 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool2.h"

namespace noa {
    template<typename>
    class Float2;

    template<typename T>
    class alignas(sizeof(T) * 2) Int2 {
    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
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

    public: // Default Constructors
        constexpr Int2() noexcept = default;
        constexpr Int2(const Int2&) noexcept = default;
        constexpr Int2(Int2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y>
        NOA_HD constexpr Int2(X xi, Y yi) noexcept
                : x(static_cast<T>(xi)),
                  y(static_cast<T>(yi)) {}

        template<typename U>
        NOA_HD constexpr explicit Int2(U v) noexcept
                : x(static_cast<T>(v)),
                  y(static_cast<T>(v)) {}

        NOA_HD constexpr explicit Int2(Bool2 v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)) {}

        template<typename U>
        NOA_HD constexpr explicit Int2(Int2<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)) {}

        template<typename U>
        NOA_HD constexpr explicit Int2(Float2<U> v) noexcept
                : x(static_cast<T>(v.x)),
                  y(static_cast<T>(v.y)) {}

        template<typename U>
        NOA_HD constexpr explicit Int2(U* ptr) noexcept
                : x(static_cast<T>(ptr[0])),
                  y(static_cast<T>(ptr[1])) {}

    public: // Assignment operators
        constexpr Int2& operator=(const Int2& v) noexcept = default;
        constexpr Int2& operator=(Int2&& v) noexcept = default;

        NOA_HD constexpr Int2& operator=(T v) noexcept {
            this->x = v;
            this->y = v;
            return *this;
        }

        NOA_HD constexpr Int2& operator=(T* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            return *this;
        }

        NOA_HD constexpr Int2& operator+=(Int2 rhs) noexcept {
            this->x += rhs.x;
            this->y += rhs.y;
            return *this;
        }

        NOA_HD constexpr Int2& operator-=(Int2 rhs) noexcept {
            this->x -= rhs.x;
            this->y -= rhs.y;
            return *this;
        }

        NOA_HD constexpr Int2& operator*=(Int2 rhs) noexcept {
            this->x *= rhs.x;
            this->y *= rhs.y;
            return *this;
        }

        NOA_HD constexpr Int2& operator/=(Int2 rhs) noexcept {
            this->x /= rhs.x;
            this->y /= rhs.y;
            return *this;
        }

        NOA_HD constexpr Int2& operator+=(T rhs) noexcept {
            this->x += rhs;
            this->y += rhs;
            return *this;
        }

        NOA_HD constexpr Int2& operator-=(T rhs) noexcept {
            this->x -= rhs;
            this->y -= rhs;
            return *this;
        }

        NOA_HD constexpr Int2& operator*=(T rhs) noexcept {
            this->x *= rhs;
            this->y *= rhs;
            return *this;
        }

        NOA_HD constexpr Int2& operator/=(T rhs) noexcept {
            this->x /= rhs;
            this->y /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Int2 operator+(Int2 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Int2 operator-(Int2 v) noexcept {
            return {-v.x, -v.y};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Int2 operator+(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x + rhs.x, lhs.y + rhs.y};
        }

        friend NOA_HD constexpr Int2 operator+(T lhs, Int2 rhs) noexcept {
            return {lhs + rhs.x, lhs + rhs.y};
        }

        friend NOA_HD constexpr Int2 operator+(Int2 lhs, T rhs) noexcept {
            return {lhs.x + rhs, lhs.y + rhs};
        }

        friend NOA_HD constexpr Int2 operator-(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x - rhs.x, lhs.y - rhs.y};
        }

        friend NOA_HD constexpr Int2 operator-(T lhs, Int2 rhs) noexcept {
            return {lhs - rhs.x, lhs - rhs.y};
        }

        friend NOA_HD constexpr Int2 operator-(Int2 lhs, T rhs) noexcept {
            return {lhs.x - rhs, lhs.y - rhs};
        }

        friend NOA_HD constexpr Int2 operator*(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x * rhs.x, lhs.y * rhs.y};
        }

        friend NOA_HD constexpr Int2 operator*(T lhs, Int2 rhs) noexcept {
            return {lhs * rhs.x, lhs * rhs.y};
        }

        friend NOA_HD constexpr Int2 operator*(Int2 lhs, T rhs) noexcept {
            return {lhs.x * rhs, lhs.y * rhs};
        }

        friend NOA_HD constexpr Int2 operator/(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x / rhs.x, lhs.y / rhs.y};
        }

        friend NOA_HD constexpr Int2 operator/(T lhs, Int2 rhs) noexcept {
            return {lhs / rhs.x, lhs / rhs.y};
        }

        friend NOA_HD constexpr Int2 operator/(Int2 lhs, T rhs) noexcept {
            return {lhs.x / rhs, lhs.y / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool2 operator>(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x > rhs.x, lhs.y > rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator>(Int2 lhs, T rhs) noexcept {
            return {lhs.x > rhs, lhs.y > rhs};
        }

        friend NOA_HD constexpr Bool2 operator>(T lhs, Int2 rhs) noexcept {
            return {lhs > rhs.x, lhs > rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x < rhs.x, lhs.y < rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<(Int2 lhs, T rhs) noexcept {
            return {lhs.x < rhs, lhs.y < rhs};
        }

        friend NOA_HD constexpr Bool2 operator<(T lhs, Int2 rhs) noexcept {
            return {lhs < rhs.x, lhs < rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator>=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x >= rhs.x, lhs.y >= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator>=(Int2 lhs, T rhs) noexcept {
            return {lhs.x >= rhs, lhs.y >= rhs};
        }

        friend NOA_HD constexpr Bool2 operator>=(T lhs, Int2 rhs) noexcept {
            return {lhs >= rhs.x, lhs >= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x <= rhs.x, lhs.y <= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator<=(Int2 lhs, T rhs) noexcept {
            return {lhs.x <= rhs, lhs.y <= rhs};
        }

        friend NOA_HD constexpr Bool2 operator<=(T lhs, Int2 rhs) noexcept {
            return {lhs <= rhs.x, lhs <= rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator==(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x == rhs.x, lhs.y == rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator==(Int2 lhs, T rhs) noexcept {
            return {lhs.x == rhs, lhs.y == rhs};
        }

        friend NOA_HD constexpr Bool2 operator==(T lhs, Int2 rhs) noexcept {
            return {lhs == rhs.x, lhs == rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator!=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x != rhs.x, lhs.y != rhs.y};
        }

        friend NOA_HD constexpr Bool2 operator!=(Int2 lhs, T rhs) noexcept {
            return {lhs.x != rhs, lhs.y != rhs};
        }

        friend NOA_HD constexpr Bool2 operator!=(T lhs, Int2 rhs) noexcept {
            return {lhs != rhs.x, lhs != rhs.y};
        }

        // -- Other Operators --

        friend NOA_HD constexpr Int2 operator%(Int2 lhs, Int2 rhs) noexcept {
            return {lhs.x % rhs.x, lhs.y % rhs.y};
        }

        friend NOA_HD constexpr Int2 operator%(Int2 lhs, T rhs) noexcept {
            return {lhs.x % rhs, lhs.y % rhs};
        }

        friend NOA_HD constexpr Int2 operator%(T lhs, Int2 rhs) noexcept {
            return {lhs % rhs.x, lhs % rhs.y};
        }
    };

    template<typename T>
    NOA_FHD constexpr T elements(Int2<T> v) noexcept {
        return v.x * v.y;
    }

    template<typename T>
    NOA_FHD constexpr T elementsFFT(Int2<T> v) noexcept {
        return (v.x / 2 + 1) * v.y;
    }

    template<typename T>
    NOA_FHD constexpr Int2<T> shapeFFT(Int2<T> v) noexcept {
        return {v.x / 2 + 1, v.y};
    }

    namespace math {
        template<typename T>
        NOA_FHD constexpr T sum(Int2<T> v) noexcept {
            return v.x + v.y;
        }

        template<typename T>
        NOA_FHD constexpr T prod(Int2<T> v) noexcept {
            return v.x * v.y;
        }

        template<typename T>
        NOA_FHD constexpr T min(Int2<T> v) noexcept {
            return min(v.x, v.y);
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> min(Int2<T> lhs, Int2<T> rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> min(Int2<T> lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> min(T lhs, Int2<T> rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr T max(Int2<T> v) noexcept {
            return max(v.x, v.y);
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> max(Int2<T> lhs, Int2<T> rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> max(Int2<T> lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> max(T lhs, Int2<T> rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y)};
        }
    }

    using int2_t = Int2<int>;
    using uint2_t = Int2<uint>;
    using long2_t = Int2<int64_t>;
    using ulong2_t = Int2<uint64_t>;
    using size2_t = Int2<size_t>;

    namespace traits {
        template<typename>
        struct p_is_int2 : std::false_type {};
        template<typename T>
        struct p_is_int2<noa::Int2<T>> : std::true_type {};
        template<typename T> using is_int2 = std::bool_constant<p_is_int2<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int2_v = is_int2<T>::value;

        template<typename>
        struct p_is_uint2 : std::false_type {};
        template<typename T>
        struct p_is_uint2<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint2 = std::bool_constant<p_is_uint2<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint2_v = is_uint2<T>::value;

        template<typename T>
        struct proclaim_is_intX<noa::Int2<T>> : std::true_type {};
        template<typename T>
        struct proclaim_is_uintX<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }


    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(Int2<T> v) noexcept {
        return {v.x, v.y};
    }

    template<>
    NOA_IH std::string string::typeName<int2_t>() { return "int2"; }
    template<>
    NOA_IH std::string string::typeName<uint2_t>() { return "uint2"; }
    template<>
    NOA_IH std::string string::typeName<long2_t>() { return "long2"; }
    template<>
    NOA_IH std::string string::typeName<ulong2_t>() { return "ulong2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Int2<T> v) {
        os << string::format("({},{})", v.x, v.y);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Int2<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Int2<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.x, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec.y, ctx);
            *out = ')';
            return out;
        }
    };
}
