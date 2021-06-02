/**
 * @file noa/util/Int3.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 * Vector containing 3 integers.
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
    class Float3;

    template<typename T>
    class Int3 {
    public:
        static_assert(Noa::Traits::is_int_v<T>);
        typedef T value_type;
        T x{}, y{}, z{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 3; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        NOA_HD constexpr Int3() noexcept = default;
        template<typename X, typename Y, typename Z> NOA_HD constexpr Int3(X xi, Y yi, Z zi) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(const Int3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Int3<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator=(const Int3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator=(const Float3<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Int3<T>& operator+=(const Int3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator-=(const Int3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator*=(const Int3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator/=(const Int3<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Int3<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Int3<T> operator+(const Int3<T>& v) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator-(const Int3<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_HD constexpr Int3<T> operator+(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator+(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator+(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Int3<T> operator-(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator-(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator-(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Int3<T> operator*(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator*(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator*(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_HD constexpr Int3<T> operator/(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator/(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Int3<T> operator/(const Int3<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_HD constexpr Bool3 operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator>=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator<=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator==(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator==(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_HD constexpr Bool3 operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator!=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_HD constexpr Bool3 operator!=(T lhs, const Int3<T>& rhs) noexcept;

    template<class T> NOA_FHD constexpr size_t getElements(const Int3<T>& v) noexcept;
    template<class T> NOA_FHD constexpr size_t getElementsSlice(const Int3<T>& v) noexcept;
    template<class T> NOA_FHD constexpr size_t getElementsFFT(const Int3<T>& v) noexcept;
    template<class T> NOA_FHD constexpr Int3<T> getShapeSlice(const Int3<T>& v) noexcept;

    namespace Math {
        template<class T> NOA_HD constexpr T sum(const Int3<T>& v) noexcept;
        template<class T> NOA_HD constexpr T prod(const Int3<T>& v) noexcept;

        template<class T> NOA_HD constexpr Int3<T> min(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
        template<class T> NOA_HD constexpr Int3<T> min(const Int3<T>& lhs, T rhs) noexcept;
        template<class T> NOA_HD constexpr Int3<T> min(T lhs, const Int3<T>& rhs) noexcept;
        template<class T> NOA_HD constexpr Int3<T> max(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
        template<class T> NOA_HD constexpr Int3<T> max(const Int3<T>& lhs, T rhs) noexcept;
        template<class T> NOA_HD constexpr Int3<T> max(T lhs, const Int3<T>& rhs) noexcept;
    }

    namespace Traits {
        template<typename> struct p_is_int3 : std::false_type {};
        template<typename T> struct p_is_int3<Noa::Int3<T>> : std::true_type {};
        template<typename T> using is_int3 = std::bool_constant<p_is_int3<Noa::Traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

        template<typename> struct p_is_uint3 : std::false_type {};
        template<typename T> struct p_is_uint3<Noa::Int3<T>> : std::bool_constant<Noa::Traits::is_uint_v<T>> {};
        template<typename T> using is_uint3 = std::bool_constant<p_is_uint3<Noa::Traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

        template<typename T> struct proclaim_is_intX<Noa::Int3<T>> : std::true_type {};
        template<typename T> struct proclaim_is_uintX<Noa::Int3<T>> : std::bool_constant<Noa::Traits::is_uint_v<T>> {};
    }

    using int3_t = Int3<int>;
    using uint3_t = Int3<uint>;
    using long3_t = Int3<long long>;
    using ulong3_t = Int3<unsigned long long>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(const Int3<T>& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string String::typeName<int3_t>() { return "int3"; }
    template<> NOA_IH std::string String::typeName<uint3_t>() { return "uint3"; }
    template<> NOA_IH std::string String::typeName<long3_t>() { return "long3"; }
    template<> NOA_IH std::string String::typeName<ulong3_t>() { return "ulong3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Int3<T>& v) {
        os << String::format("({},{},{})", v.x, v.y, v.z);
        return os;
    }
}

namespace Noa {
    // -- Component accesses --

    template<typename T>
    NOA_HD constexpr T& Int3<T>::operator[](size_t i) {
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
    NOA_HD constexpr const T& Int3<T>::operator[](size_t i) const {
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
    NOA_HD constexpr Int3<T>::Int3(X xi, Y yi, Z zi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>::Int3(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>::Int3(const Int3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>::Int3(const Float3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>::Int3(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator=(const Int3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator=(const Float3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator+=(const Int3<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator-=(const Int3<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator*=(const Int3<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator/=(const Int3<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    NOA_HD constexpr Int3<T>& Int3<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> NOA_HD constexpr Int3<T> operator+(const Int3<T>& v) noexcept {
        return v;
    }

    template<typename T> NOA_HD constexpr Int3<T> operator-(const Int3<T>& v) noexcept {
        return {-v.x, -v.y, -v.z};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    NOA_FHD constexpr Int3<T> operator+(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator+(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator+(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> operator-(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator-(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator-(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> operator*(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator*(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator*(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> operator/(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator/(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator/(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    NOA_FHD constexpr Bool3 operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator>=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator<=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator==(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator==(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    template<typename T>
    NOA_FHD constexpr Bool3 operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator!=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }
    template<typename T>
    NOA_FHD constexpr Bool3 operator!=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    template<class T>
    NOA_FHD constexpr size_t getElements(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y) * static_cast<size_t>(v.z);
    }

    template<class T>
    NOA_FHD constexpr size_t getElementsSlice(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y);
    }

    template<class T>
    NOA_FHD constexpr size_t getElementsFFT(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x / 2 + 1) * static_cast<size_t>(v.y) * static_cast<size_t>(v.z);
    }

    template<class T>
    NOA_FHD constexpr Int3<T> getShapeSlice(const Int3<T>& v) noexcept {
        return {v.x, v.y, 1};
    }

    namespace Math {
        template<class T>
        NOA_FHD constexpr T sum(const Int3<T>& v) noexcept {
            return v.x + v.y + v.z;
        }

        template<class T>
        NOA_FHD constexpr T prod(const Int3<T>& v) noexcept {
            return v.x * v.y * v.z;
        }

        template<class T>
        NOA_FHD constexpr Int3<T> min(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<class T>
        NOA_FHD constexpr Int3<T> min(const Int3<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<class T>
        NOA_FHD constexpr Int3<T> min(T lhs, const Int3<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<class T>
        NOA_FHD constexpr Int3<T> max(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<class T>
        NOA_FHD constexpr Int3<T> max(const Int3<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<class T>
        NOA_FHD constexpr Int3<T> max(T lhs, const Int3<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
        }
    }
}
