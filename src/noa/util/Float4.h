/**
 * @file noa/util/Float4.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 * Vector containing 4 floating-point numbers.
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"
#include "noa/util/Bool4.h"

namespace Noa {
    template<typename>
    struct Int4;

    template<typename T>
    struct alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Float4 {
        static_assert(Noa::Traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{}, z{}, w{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 4; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Float4() noexcept = default;
        template<class X, class Y, class Z, class W> NOA_HD constexpr Float4(X xi, Y yi, Z zi, W wi) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(const Int4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Float4<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator=(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator=(const Int4<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Float4<T>& operator+=(const Float4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator-=(const Float4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator*=(const Float4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator/=(const Float4<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Float4<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Float4<T> operator+(const Float4<T>& v) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator-(const Float4<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_HD constexpr Float4<T> operator+(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator+(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator+(const Float4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float4<T> operator-(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator-(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator-(const Float4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float4<T> operator*(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator*(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator*(const Float4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float4<T> operator/(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator/(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float4<T> operator/(const Float4<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_HD constexpr Bool4 operator>(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator>(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator>(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool4 operator<(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator<(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator<(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool4 operator>=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator>=(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator>=(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool4 operator<=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator<=(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator<=(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool4 operator==(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator==(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator==(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool4 operator!=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator!=(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool4 operator!=(T lhs, const Float4<T>& rhs) noexcept;

    namespace Math {
        template<class T> NOA_HD constexpr Float4<T> floor(const Float4<T>& v);
        template<class T> NOA_HD constexpr Float4<T> ceil(const Float4<T>& v);
        template<class T> NOA_HD constexpr T lengthSq(const Float4<T>& v) noexcept;
        template<class T> NOA_HD constexpr T length(const Float4<T>& v);
        template<class T> NOA_HD constexpr Float4<T> normalize(const Float4<T>& v);
        template<class T> NOA_HD constexpr T sum(const Float4<T>& v) noexcept;
        template<class T> NOA_HD constexpr T prod(const Float4<T>& v) noexcept;

        template<class T> NOA_HD constexpr Float4<T> min(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
        template<class T> NOA_HD constexpr Float4<T> min(const Float4<T>& lhs, T rhs) noexcept;
        template<class T> NOA_HD constexpr Float4<T> min(T lhs, const Float4<T>& rhs) noexcept;
        template<class T> NOA_HD constexpr Float4<T> max(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
        template<class T> NOA_HD constexpr Float4<T> max(const Float4<T>& lhs, T rhs) noexcept;
        template<class T> NOA_HD constexpr Float4<T> max(T lhs, const Float4<T>& rhs) noexcept;

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool4 isEqual(const Float4<T>& a, const Float4<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool4 isEqual(const Float4<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool4 isEqual(T a, const Float4<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace Traits {
        template<typename T> struct p_is_float4 : std::false_type {};
        template<typename T> struct p_is_float4<Noa::Float4<T>> : std::true_type {};
        template<typename T> using is_float4 = std::bool_constant<p_is_float4<Noa::Traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float4_v = is_float4<T>::value;

        template<typename T> struct proclaim_is_floatX<Noa::Float4<T>> : std::true_type {};
    }

    using float4_t = Float4<float>;
    using double4_t = Float4<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(const Float4<T>& v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<> NOA_IH std::string String::typeName<float4_t>() { return "float4"; }
    template<> NOA_IH std::string String::typeName<double4_t>() { return "double4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Float4<T>& v) {
        os << String::format("({:.3f},{:.3f},{:.3f},{:.3f})", v.x, v.y, v.z, v.w);
        return os;
    }
}

namespace Noa {
    // -- Component accesses --

    template<typename T>
    NOA_HD constexpr T& Float4<T>::operator[](size_t i) {
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

    template<typename T>
    NOA_HD constexpr const T& Float4<T>::operator[](size_t i) const {
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

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y, typename Z, typename W>
    NOA_HD constexpr Float4<T>::Float4(X xi, Y yi, Z zi, W wi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)),
              w(static_cast<T>(wi)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>::Float4(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)),
              w(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>::Float4(const Float4<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)),
              w(static_cast<T>(v.w)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>::Float4(const Int4<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)),
              w(static_cast<T>(v.w)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>::Float4(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])),
              w(static_cast<T>(ptr[3])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        this->w = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        this->w = static_cast<T>(ptr[3]);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator=(const Float4<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        this->w = static_cast<T>(v.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator=(const Int4<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        this->w = static_cast<T>(v.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator+=(const Float4<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        this->w += static_cast<T>(rhs.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        this->w += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator-=(const Float4<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        this->w -= static_cast<T>(rhs.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        this->w -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator*=(const Float4<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        this->w *= static_cast<T>(rhs.w);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        this->w *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator/=(const Float4<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        this->w /= static_cast<T>(rhs.w);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Float4<T>& Float4<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        this->w /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Float4<T> operator+(const Float4<T>& v) noexcept {
        return v;
    }

    template<typename T> NOA_HD constexpr Float4<T> operator-(const Float4<T>& v) noexcept {
        return {-v.x, -v.y, -v.z, -v.w};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    NOA_FHD constexpr Float4<T> operator+(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator+(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator+(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float4<T> operator-(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator-(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator-(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float4<T> operator*(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator*(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator*(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float4<T> operator/(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator/(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Float4<T> operator/(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    NOA_FHD constexpr Bool4 operator>(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator>(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs, lhs.w > rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator>(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z, lhs > rhs.w};
    }

    template<typename T>
    NOA_FHD constexpr Bool4 operator<(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator<(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs, lhs.w < rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator<(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z, lhs < rhs.w};
    }

    template<typename T>
    NOA_FHD constexpr Bool4 operator>=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator>=(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs, lhs.w >= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator>=(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z, lhs >= rhs.w};
    }

    template<typename T>
    NOA_FHD constexpr Bool4 operator<=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator<=(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs, lhs.w <= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator<=(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z, lhs <= rhs.w};
    }

    template<typename T>
    NOA_FHD constexpr Bool4 operator==(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator==(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs, lhs.w == rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator==(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z, lhs == rhs.w};
    }

    template<typename T>
    NOA_FHD constexpr Bool4 operator!=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator!=(const Float4<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs, lhs.w != rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool4 operator!=(T lhs, const Float4<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z, lhs != rhs.w};
    }

    namespace Math {
        template<class T>
        NOA_FHD constexpr Float4<T> floor(const Float4<T>& v) {
            return Float4<T>(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
        }

        template<class T>
        NOA_FHD constexpr Float4<T> ceil(const Float4<T>& v) {
            return Float4<T>(ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w));
        }

        template<class T>
        NOA_FHD constexpr T lengthSq(const Float4<T>& v) noexcept {
            return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }

        template<class T>
        NOA_FHD constexpr T length(const Float4<T>& v) {
            return sqrt(lengthSq(v));
        }

        template<class T>
        NOA_FHD constexpr Float4<T> normalize(const Float4<T>& v) {
            return v / length(v);
        }

        template<class T>
        NOA_FHD constexpr T sum(const Float4<T>& v) noexcept {
            return v.x + v.y + v.z + v.w;
        }

        template<class T>
        NOA_FHD constexpr T prod(const Float4<T>& v) noexcept {
            return v.x * v.y * v.z * v.w;
        }

        template<class T>
        NOA_FHD constexpr Float4<T> min(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
        }

        template<class T>
        NOA_FHD constexpr Float4<T> min(const Float4<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
        }

        template<class T>
        NOA_FHD constexpr Float4<T> min(T lhs, const Float4<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
        }

        template<class T>
        NOA_FHD constexpr Float4<T> max(const Float4<T>& lhs, const Float4<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
        }

        template<class T>
        NOA_FHD constexpr Float4<T> max(const Float4<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
        }

        template<class T>
        NOA_FHD constexpr Float4<T> max(T lhs, const Float4<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool4 isEqual(const Float4<T>& a, const Float4<T>& b, T e) {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e),
                    isEqual<ULP>(a.z, b.z, e), isEqual<ULP>(a.w, b.w, e)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool4 isEqual(const Float4<T>& a, T b, T e) {
            return {isEqual<ULP>(b, a.x, e), isEqual<ULP>(b, a.y, e),
                    isEqual<ULP>(b, a.z, e), isEqual<ULP>(b, a.w, e)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool4 isEqual(T a, const Float4<T>& b, T e) {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e),
                    isEqual<ULP>(a, b.z, e), isEqual<ULP>(a, b.w, e)};
        }
    }
}
