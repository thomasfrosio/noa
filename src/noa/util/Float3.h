/**
 * @file noa/util/Float3.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 * Vector containing 3 floating-point numbers.
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"
#include "noa/util/Bool3.h"

namespace Noa {
    template<typename>
    struct Int3;

    template<typename T>
    struct Float3 {
        static_assert(Noa::Traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{}, z{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 3; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Float3() noexcept = default;
        template<typename X, typename Y, typename Z> NOA_HD constexpr Float3(X xi, Y yi, Z zi) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(const Int3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Float3<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator=(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator=(const Int3<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Float3<T>& operator+=(const Float3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator-=(const Float3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator*=(const Float3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator/=(const Float3<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Float3<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Float3<T> operator+(const Float3<T>& v) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator-(const Float3<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_HD constexpr Float3<T> operator+(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator+(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator+(const Float3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float3<T> operator-(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator-(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator-(const Float3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float3<T> operator*(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator*(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator*(const Float3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Float3<T> operator/(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator/(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Float3<T> operator/(const Float3<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_HD constexpr Bool3 operator>(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator<(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator>=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>=(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>=(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator<=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<=(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<=(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator==(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator==(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator==(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator!=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator!=(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator!=(T lhs, const Float3<T>& rhs) noexcept;

    namespace Math {
        template<typename T> NOA_HD constexpr Float3<T> floor(const Float3<T>& v);
        template<typename T> NOA_HD constexpr Float3<T> ceil(const Float3<T>& v);
        template<typename T> NOA_HD constexpr T sum(const Float3<T>& v) noexcept;
        template<typename T> NOA_HD constexpr T prod(const Float3<T>& v) noexcept;
        template<typename T> NOA_HD constexpr T dot(const Float3<T>& a, const Float3<T>& b) noexcept;
        template<typename T> NOA_HD constexpr T innerProduct(const Float3<T>& a, const Float3<T>& b) noexcept;
        template<typename T> NOA_HD constexpr T norm(const Float3<T>& v);
        template<typename T> NOA_HD constexpr T length(const Float3<T>& v);
        template<typename T> NOA_HD constexpr Float3<T> normalize(const Float3<T>& v);
        template<typename T> NOA_HD constexpr Float3<T> cross(const Float3<T>& a, const Float3<T>& b) noexcept;

        template<typename T> NOA_HD constexpr Float3<T> min(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
        template<typename T> NOA_HD constexpr Float3<T> min(const Float3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_HD constexpr Float3<T> min(T lhs, const Float3<T>& rhs) noexcept;
        template<typename T> NOA_HD constexpr Float3<T> max(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
        template<typename T> NOA_HD constexpr Float3<T> max(const Float3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_HD constexpr Float3<T> max(T lhs, const Float3<T>& rhs) noexcept;

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool3 isEqual(const Float3<T>& a, const Float3<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool3 isEqual(const Float3<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_HD constexpr Bool3 isEqual(T a, const Float3<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace Traits {
        template<typename T> struct p_is_float3 : std::false_type {};
        template<typename T> struct p_is_float3<Noa::Float3<T>> : std::true_type {};
        template<typename T> using is_float3 = std::bool_constant<p_is_float3<Noa::Traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float3_v = is_float3<T>::value;

        template<typename T> struct proclaim_is_floatX<Noa::Float3<T>> : std::true_type {};
    }

    using float3_t = Float3<float>;
    using double3_t = Float3<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(const Float3<T>& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string String::typeName<float3_t>() { return "float3"; }
    template<> NOA_IH std::string String::typeName<double3_t>() { return "double3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Float3<T>& v) {
        os << String::format("({:.3f},{:.3f},{:.3f})", v.x, v.y, v.z);
        return os;
    }
}

namespace Noa {
    // -- Component accesses --

    template<typename T>
    NOA_HD constexpr T& Float3<T>::operator[](size_t i) {
        switch (i) {
            default:
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
        }
    }

    template<typename T>
    NOA_HD constexpr const T& Float3<T>::operator[](size_t i) const {
        switch (i) {
            default:
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
        }
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y, typename Z>
    NOA_HD constexpr Float3<T>::Float3(X xi, Y yi, Z zi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>::Float3(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>::Float3(const Float3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>::Float3(const Int3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>::Float3(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator=(const Float3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator=(const Int3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator+=(const Float3<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator-=(const Float3<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator*=(const Float3<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator/=(const Float3<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Float3<T>& Float3<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Float3<T> operator+(const Float3<T>& v) noexcept {
        return v;
    }

    template<typename T> NOA_HD constexpr Float3<T> operator-(const Float3<T>& v) noexcept {
        return {-v.x, -v.y, -v.z};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    NOA_FHD constexpr Float3<T> operator+(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator+(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator+(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float3<T> operator-(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator-(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator-(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float3<T> operator*(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator*(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator*(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename T>
    NOA_FHD constexpr Float3<T> operator/(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator/(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Float3<T> operator/(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    NOA_FHD constexpr Bool3 operator>(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator<(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator>=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>=(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>=(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator<=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<=(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<=(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator==(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator==(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator==(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator!=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator!=(const Float3<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator!=(T lhs, const Float3<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    namespace Math {
        template<typename T>
        NOA_FHD constexpr Float3<T> floor(const Float3<T>& v) {
            return Float3<T>(floor(v.x), floor(v.y), floor(v.z));
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> ceil(const Float3<T>& v) {
            return Float3<T>(ceil(v.x), ceil(v.y), ceil(v.z));
        }

        template<typename T>
        NOA_FHD constexpr T sum(const Float3<T>& v) noexcept {
            return v.x + v.y + v.z;
        }

        template<typename T>
        NOA_FHD constexpr T prod(const Float3<T>& v) noexcept {
            return v.x * v.y * v.z;
        }

        template<typename T>
        NOA_FHD constexpr T dot(const Float3<T>& a, const Float3<T>& b) noexcept {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        template<typename T>
        NOA_FHD constexpr T innerProduct(const Float3<T>& a, const Float3<T>& b) noexcept {
            return dot(a, b);
        }

        template<typename T>
        NOA_FHD constexpr T norm(const Float3<T>& v) {
            return sqrt(dot(v, v));
        }

        template<typename T>
        NOA_FHD constexpr T length(const Float3<T>& v) {
            return norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> normalize(const Float3<T>& v) {
            return v / norm(v);
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> cross(const Float3<T>& a, const Float3<T>& b) noexcept {
            return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(const Float3<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> min(T lhs, const Float3<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(const Float3<T>& lhs, const Float3<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(const Float3<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Float3<T> max(T lhs, const Float3<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool3 isEqual(const Float3<T>& a, const Float3<T>& b, T e) {
            return {isEqual<ULP>(a.x, b.x, e), isEqual<ULP>(a.y, b.y, e), isEqual<ULP>(a.z, b.z, e)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool3 isEqual(const Float3<T>& a, T b, T e) {
            return {isEqual<ULP>(a.x, b, e), isEqual<ULP>(a.y, b, e), isEqual<ULP>(a.z, b, e)};
        }

        template<uint ULP, typename T>
        NOA_FHD constexpr Bool3 isEqual(T a, const Float3<T>& b, T e) {
            return {isEqual<ULP>(a, b.x, e), isEqual<ULP>(a, b.y, e), isEqual<ULP>(a, b.z, e)};
        }
    }
}
