/**
 * @file noa/util/Float2.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 * Vector containing 2 floating-point numbers.
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"
#include "noa/util/Bool2.h"

namespace Noa {
    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Float2 {
    public:
        static_assert(Noa::Traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 2; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Float2() noexcept = default;
        template<typename X, typename Y> NOA_HD constexpr Float2(X xi, Y yi) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Float2<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator=(U* ptr);
        template<typename U> NOA_HD constexpr Float2<T>& operator=(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator=(const Int2<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Float2<T>& operator+=(const Float2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator-=(const Float2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator*=(const Float2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator/=(const Float2<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Float2<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Float2<T> operator+(const Float2<T>& v) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator-(const Float2<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_HD constexpr Float2<T> operator+(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator+(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator+(const Float2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float2<T> operator-(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator-(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator-(const Float2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float2<T> operator*(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator*(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator*(const Float2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float2<T> operator/(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator/(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float2<T> operator/(const Float2<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_HD constexpr Bool2 operator>(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator>(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator>(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool2 operator<(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator<(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator<(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool2 operator>=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator>=(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator>=(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool2 operator<=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator<=(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator<=(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool2 operator==(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator==(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator==(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool2 operator!=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator!=(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool2 operator!=(T lhs, const Float2<T>& rhs) noexcept;

    namespace Math {
        template<typename T> NOA_HD constexpr Float2<T> floor(const Float2<T>& v);
        template<typename T> NOA_HD constexpr Float2<T> ceil(const Float2<T>& v);
        template<typename T> NOA_HD constexpr T sum(const Float2<T>& v) noexcept;
        template<typename T> NOA_HD constexpr T prod(const Float2<T>& v) noexcept;
        template<typename T> NOA_HD constexpr T dot(const Float2<T>& a, const Float2<T>& b) noexcept;
        template<typename T> NOA_HD constexpr T innerProduct(const Float2<T>& a, const Float2<T>& b) noexcept;
        template<typename T> NOA_HD constexpr T norm(const Float2<T>& v) noexcept;
        template<typename T> NOA_HD constexpr T length(const Float2<T>& v);
        template<typename T> NOA_HD constexpr Float2<T> normalize(const Float2<T>& v);

        template<typename T> NOA_HD constexpr Float2<T> min(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
        template<typename T> NOA_HD constexpr Float2<T> min(const Float2<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_HD constexpr Float2<T> min(T lhs, const Float2<T>& rhs) noexcept;
        template<typename T> NOA_HD constexpr Float2<T> max(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
        template<typename T> NOA_HD constexpr Float2<T> max(const Float2<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_HD constexpr Float2<T> max(T lhs, const Float2<T>& rhs) noexcept;

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool2 isEqual(const Float2<T>& a, const Float2<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool2 isEqual(const Float2<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool2 isEqual(T a, const Float2<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace Traits {
        template<typename T> struct p_is_float2 : std::false_type {};
        template<typename T> struct p_is_float2<Noa::Float2<T>> : std::true_type {};
        template<typename T> using is_float2 = std::bool_constant<p_is_float2<Noa::Traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float2_v = is_float2<T>::value;

        template<typename T> struct proclaim_is_floatX<Noa::Float2<T>> : std::true_type {};
    }

    using float2_t = Float2<float>;
    using double2_t = Float2<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(const Float2<T>& v) noexcept {
        return {v.x, v.y};
    }

    template<> NOA_IH std::string String::typeName<float2_t>() { return "float2"; }
    template<> NOA_IH std::string String::typeName<double2_t>() { return "double2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Float2<T>& v) {
        os << String::format("({:.3f},{:.3f})", v.x, v.y);
        return os;
    }
}

namespace Noa {
    // -- Component accesses --

    template<typename T>
    NOA_HD constexpr T& Float2<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->elements());
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    template<typename T>
    NOA_HD constexpr const T& Float2<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->elements());
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y>
    NOA_HD constexpr Float2<T>::Float2(X xi, Y yi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>::Float2(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>::Float2(const Float2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>::Float2(const Int2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>::Float2(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator=(U* ptr) {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator=(const Float2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator=(const Int2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator+=(const Float2<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator-=(const Float2<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator*=(const Float2<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator/=(const Float2<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Float2<T>& Float2<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T>
    NOA_HD constexpr Float2<T> operator+(const Float2<T>& v) noexcept {
        return v;
    }

    template<typename T>
    NOA_HD constexpr Float2<T> operator-(const Float2<T>& v) noexcept {
        return {-v.x, -v.y};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    NOA_FHD constexpr Float2<T> operator+(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator+(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator+(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float2<T> operator-(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator-(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator-(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float2<T> operator*(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator*(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator*(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float2<T> operator/(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator/(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Float2<T> operator/(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    NOA_FHD constexpr Bool2 operator>(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator>(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator>(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y};
    }

    template<typename T>
    NOA_FHD constexpr Bool2 operator<(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator<(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator<(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y};
    }

    template<typename T>
    NOA_FHD constexpr Bool2 operator>=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator>=(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator>=(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y};
    }

    template<typename T>
    NOA_FHD constexpr Bool2 operator<=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator<=(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator<=(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y};
    }

    template<typename T>
    NOA_FHD constexpr Bool2 operator==(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator==(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator==(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    template<typename T>
    NOA_FHD constexpr Bool2 operator!=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator!=(const Float2<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool2 operator!=(T lhs, const Float2<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    namespace Math {
        template<typename T>
        NOA_FHD constexpr Float2<T> floor(const Float2<T>& v) {
            return Float2<T>(floor(v.x), floor(v.y));
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> ceil(const Float2<T>& v) {
            return Float2<T>(ceil(v.x), ceil(v.y));
        }

        template<typename T>
        NOA_FHD constexpr T dot(const Float2<T>& a, const Float2<T>& b) noexcept {
            return a.x * b.x + a.y * b.y;
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(const Float2<T>& a, const Float2<T>& b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(const Float2<T>& v) noexcept {
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(const Float2<T>& v) {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> normalize(const Float2<T>& v) {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr T sum(const Float2<T>& v) noexcept {
            return v.x + v.y;
        }

        template<typename T>
        NOA_FHD constexpr T prod(const Float2<T>& v) noexcept {
            return v.x * v.y;
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> min(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> min(const Float2<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> min(T lhs, const Float2<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> max(const Float2<T>& lhs, const Float2<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> max(const Float2<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float2<T> max(T lhs, const Float2<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Float2<T>& a, const Float2<T>& b, T e) {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Float2<T>& a, T b, T e) {
            return {isEqual<ULP>(a.x, b, e), isEqual<ULP>(a.y, b, e)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool2 isEqual(T a, const Float2<T>& b, T e) {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e)};
        }
    }
}
