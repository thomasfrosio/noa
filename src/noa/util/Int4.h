/// \file noa/util/Int4.h
/// \author Thomas - ffyr2w
/// \date 10/12/2020
/// Vector containing 4 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"
#include "noa/util/Bool4.h"

namespace noa {
    template<typename>
    class Float4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Int4 {
    public:
        static_assert(noa::traits::is_int_v<T>);
        typedef T value_type;
        T x{}, y{}, z{}, w{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 4; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Int4() noexcept = default;
        template<class X, class Y, class Z, class W> NOA_HD constexpr Int4(X xi, Y yi, Z zi, W wi) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(const Int4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Int4<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator=(const Int4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator=(const Float4<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Int4<T>& operator+=(const Int4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator-=(const Int4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator*=(const Int4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator/=(const Int4<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Int4<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Int4<T> operator+(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator-(const Int4<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Int4<T> operator+(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator+(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator+(const Int4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int4<T> operator-(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator-(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator-(const Int4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int4<T> operator*(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator*(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator*(const Int4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int4<T> operator/(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator/(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator/(const Int4<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool4 operator>(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator<(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator>=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>=(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>=(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator<=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<=(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<=(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator==(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator==(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator==(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator!=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator!=(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator!=(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr size_t getElements(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr size_t getElementsSlice(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr size_t getElementsFFT(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> getShapeSlice(const Int4<T>& v) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr T sum(const Int4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Int4<T>& v) noexcept;

        template<typename T> NOA_FHD constexpr T min(const Int4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> min(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> min(const Int4<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> min(T lhs, const Int4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Int4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> max(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> max(const Int4<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> max(T lhs, const Int4<T>& rhs) noexcept;
    }

    namespace traits {
        template<typename> struct p_is_int4 : std::false_type {};
        template<typename T> struct p_is_int4<noa::Int4<T>> : std::true_type {};
        template<typename T> using is_int4 = std::bool_constant<p_is_int4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

        template<typename> struct p_is_uint4 : std::false_type {};
        template<typename T> struct p_is_uint4<noa::Int4<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

        template<typename T> struct proclaim_is_intX<noa::Int4<T>> : std::true_type {};
        template<typename T> struct proclaim_is_uintX<noa::Int4<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }

    using int4_t = Int4<int>;
    using uint4_t = Int4<uint>;
    using long4_t = Int4<long long>;
    using ulong4_t = Int4<unsigned long long>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(const Int4<T>& v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<> NOA_IH std::string string::typeName<int4_t>() { return "int4"; }
    template<> NOA_IH std::string string::typeName<uint4_t>() { return "uint4"; }
    template<> NOA_IH std::string string::typeName<long4_t>() { return "long4"; }
    template<> NOA_IH std::string string::typeName<ulong4_t>() { return "ulong4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Int4<T>& v) {
        os << string::format("({},{},{},{})", v.x, v.y, v.z, v.w);
        return os;
    }
}

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Int4<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->elements());
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
    constexpr const T& Int4<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->elements());
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
    constexpr Int4<T>::Int4(X xi, Y yi, Z zi, W wi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)),
              w(static_cast<T>(wi)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)),
              w(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(const Int4<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)),
              w(static_cast<T>(v.w)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(const Float4<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)),
              w(static_cast<T>(v.w)) {}

    template<typename T>
    template<typename U>
    constexpr Int4<T>::Int4(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])),
              w(static_cast<T>(ptr[3])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        this->w = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        this->w = static_cast<T>(ptr[3]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(const Int4<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        this->w = static_cast<T>(v.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator=(const Float4<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        this->w = static_cast<T>(v.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator+=(const Int4<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        this->w += static_cast<T>(rhs.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        this->w += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator-=(const Int4<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        this->w -= static_cast<T>(rhs.w);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        this->w -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator*=(const Int4<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        this->w *= static_cast<T>(rhs.w);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        this->w *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator/=(const Int4<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        this->w /= static_cast<T>(rhs.w);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int4<T>& Int4<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        this->w /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> constexpr Int4<T> operator+(const Int4<T>& v) noexcept {
        return v;
    }

    template<typename T> constexpr Int4<T> operator-(const Int4<T>& v) noexcept {
        return {-v.x, -v.y, -v.z, -v.w};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Int4<T> operator+(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator+(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator+(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
    }

    template<typename T>
    constexpr Int4<T> operator-(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator-(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator-(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
    }

    template<typename T>
    constexpr Int4<T> operator*(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator*(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator*(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
    }

    template<typename T>
    constexpr Int4<T> operator/(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator/(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
    }
    template<typename T>
    constexpr Int4<T> operator/(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool4 operator>(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z, lhs.w > rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator>(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs, lhs.w > rhs};
    }
    template<typename T>
    constexpr Bool4 operator>(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z, lhs > rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator<(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z, lhs.w < rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator<(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs, lhs.w < rhs};
    }
    template<typename T>
    constexpr Bool4 operator<(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z, lhs < rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator>=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z, lhs.w >= rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator>=(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs, lhs.w >= rhs};
    }
    template<typename T>
    constexpr Bool4 operator>=(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z, lhs >= rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator<=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z, lhs.w <= rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator<=(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs, lhs.w <= rhs};
    }
    template<typename T>
    constexpr Bool4 operator<=(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z, lhs <= rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator==(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator==(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs, lhs.w == rhs};
    }
    template<typename T>
    constexpr Bool4 operator==(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z, lhs == rhs.w};
    }

    template<typename T>
    constexpr Bool4 operator!=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
    }
    template<typename T>
    constexpr Bool4 operator!=(const Int4<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs, lhs.w != rhs};
    }
    template<typename T>
    constexpr Bool4 operator!=(T lhs, const Int4<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z, lhs != rhs.w};
    }

    template<typename T>
    constexpr size_t getElements(const Int4<T>& v) noexcept {
        return static_cast<size_t>(v.x) *
               static_cast<size_t>(v.y) *
               static_cast<size_t>(v.z) *
               static_cast<size_t>(v.w);
    }

    template<typename T>
    constexpr size_t getElementsSlice(const Int4<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y);
    }

    template<typename T>
    constexpr size_t getElementsFFT(const Int4<T>& v) noexcept {
        return static_cast<size_t>(v.x / 2 + 1) *
               static_cast<size_t>(v.y) *
               static_cast<size_t>(v.z) *
               static_cast<size_t>(v.w);
    }

    template<typename T>
    constexpr Int4<T> getShapeSlice(const Int4<T>& v) noexcept {
        return {v.x, v.y, 1, 1};
    }

    namespace math {
        template<typename T>
        constexpr T sum(const Int4<T>& v) noexcept {
            return v.x + v.y + v.z + v.w;
        }

        template<typename T>
        constexpr T prod(const Int4<T>& v) noexcept {
            return v.x * v.y * v.z * v.w;
        }

        template<typename T>
        constexpr T min(const Int4<T>& v) noexcept {
            return min(min(v.x, v.y), min(v.z, v.w));
        }

        template<typename T>
        constexpr Int4<T> min(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
        }

        template<typename T>
        constexpr Int4<T> min(const Int4<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
        }

        template<typename T>
        constexpr Int4<T> min(T lhs, const Int4<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
        }

        template<typename T>
        constexpr T max(const Int4<T>& v) noexcept {
            return max(max(v.x, v.y), max(v.z, v.w));
        }

        template<typename T>
        constexpr Int4<T> max(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
        }

        template<typename T>
        constexpr Int4<T> max(const Int4<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
        }

        template<typename T>
        constexpr Int4<T> max(T lhs, const Int4<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
        }
    }
}
