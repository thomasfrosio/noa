/// \file noa/common/types/Int3.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 3 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool3.h"

namespace noa {
    template<typename>
    class Float3;

    template<typename T>
    class Int3 {
    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        typedef T value_type;
        T x{}, y{}, z{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 3; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Int3() noexcept = default;
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

    template<typename T> NOA_FHD constexpr Int3<T> operator+(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator-(const Int3<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Int3<T> operator+(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator+(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator+(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int3<T> operator-(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator-(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator-(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int3<T> operator*(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator*(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator*(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int3<T> operator/(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator/(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator/(const Int3<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool3 operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator==(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator==(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator!=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator!=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr size_t getElements(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr size_t getElementsSlice(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr size_t getElementsFFT(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> getShapeSlice(const Int3<T>& v) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr T sum(const Int3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Int3<T>& v) noexcept;

        template<typename T> NOA_FHD constexpr T min(const Int3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> min(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> min(const Int3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> min(T lhs, const Int3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Int3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> max(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> max(const Int3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> max(T lhs, const Int3<T>& rhs) noexcept;
    }

    namespace traits {
        template<typename> struct p_is_int3 : std::false_type {};
        template<typename T> struct p_is_int3<noa::Int3<T>> : std::true_type {};
        template<typename T> using is_int3 = std::bool_constant<p_is_int3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

        template<typename> struct p_is_uint3 : std::false_type {};
        template<typename T> struct p_is_uint3<noa::Int3<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint3 = std::bool_constant<p_is_uint3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

        template<typename T> struct proclaim_is_intX<noa::Int3<T>> : std::true_type {};
        template<typename T> struct proclaim_is_uintX<noa::Int3<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }

    using int3_t = Int3<int>;
    using uint3_t = Int3<uint>;
    using long3_t = Int3<int64_t>;
    using ulong3_t = Int3<uint64_t>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(const Int3<T>& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string string::typeName<int3_t>() { return "int3"; }
    template<> NOA_IH std::string string::typeName<uint3_t>() { return "uint3"; }
    template<> NOA_IH std::string string::typeName<long3_t>() { return "long3"; }
    template<> NOA_IH std::string string::typeName<ulong3_t>() { return "ulong3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Int3<T>& v) {
        os << string::format("({},{},{})", v.x, v.y, v.z);
        return os;
    }
}

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Int3<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->elements());
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
    constexpr const T& Int3<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->elements());
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
    constexpr Int3<T>::Int3(X xi, Y yi, Z zi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)),
              z(static_cast<T>(zi)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)),
              z(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(const Int3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(const Float3<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)),
              z(static_cast<T>(v.z)) {}

    template<typename T>
    template<typename U>
    constexpr Int3<T>::Int3(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])),
              z(static_cast<T>(ptr[2])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        this->z = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        this->z = static_cast<T>(ptr[2]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(const Int3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator=(const Float3<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        this->z = static_cast<T>(v.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator+=(const Int3<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        this->z += static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        this->z += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator-=(const Int3<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        this->z -= static_cast<T>(rhs.z);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        this->z -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator*=(const Int3<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        this->z *= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        this->z *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator/=(const Int3<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        this->z /= static_cast<T>(rhs.z);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int3<T>& Int3<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        this->z /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> constexpr Int3<T> operator+(const Int3<T>& v) noexcept {
        return v;
    }

    template<typename T> constexpr Int3<T> operator-(const Int3<T>& v) noexcept {
        return {-v.x, -v.y, -v.z};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Int3<T> operator+(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator+(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator+(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename T>
    constexpr Int3<T> operator-(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator-(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator-(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename T>
    constexpr Int3<T> operator*(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator*(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator*(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename T>
    constexpr Int3<T> operator/(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator/(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename T>
    constexpr Int3<T> operator/(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool3 operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y, lhs.z > rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator>(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs, lhs.z > rhs};
    }
    template<typename T>
    constexpr Bool3 operator>(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y, lhs > rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y, lhs.z < rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator<(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs, lhs.z < rhs};
    }
    template<typename T>
    constexpr Bool3 operator<(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y, lhs < rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y, lhs.z >= rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator>=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs, lhs.z >= rhs};
    }
    template<typename T>
    constexpr Bool3 operator>=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y, lhs >= rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y, lhs.z <= rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator<=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs, lhs.z <= rhs};
    }
    template<typename T>
    constexpr Bool3 operator<=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y, lhs <= rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator==(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }
    template<typename T>
    constexpr Bool3 operator==(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    template<typename T>
    constexpr Bool3 operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }
    template<typename T>
    constexpr Bool3 operator!=(const Int3<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }
    template<typename T>
    constexpr Bool3 operator!=(T lhs, const Int3<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    template<typename T>
    constexpr size_t getElements(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y) * static_cast<size_t>(v.z);
    }

    template<typename T>
    constexpr size_t getElementsSlice(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y);
    }

    template<typename T>
    constexpr size_t getElementsFFT(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x / 2 + 1) * static_cast<size_t>(v.y) * static_cast<size_t>(v.z);
    }

    template<typename T>
    constexpr Int3<T> getShapeSlice(const Int3<T>& v) noexcept {
        return {v.x, v.y, 1};
    }

    namespace math {
        template<typename T>
        constexpr T sum(const Int3<T>& v) noexcept {
            return v.x + v.y + v.z;
        }

        template<typename T>
        constexpr T prod(const Int3<T>& v) noexcept {
            return v.x * v.y * v.z;
        }

        template<typename T>
        constexpr T min(const Int3<T>& v) noexcept {
            return (v.x < v.y) ? min(v.x, v.z) : min(v.y, v.z);
        }

        template<typename T>
        constexpr Int3<T> min(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
        }

        template<typename T>
        constexpr Int3<T> min(const Int3<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
        }

        template<typename T>
        constexpr Int3<T> min(T lhs, const Int3<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
        }

        template<typename T>
        constexpr T max(const Int3<T>& v) noexcept {
            return (v.x > v.y) ? max(v.x, v.z) : max(v.y, v.z);
        }

        template<typename T>
        constexpr Int3<T> max(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
        }

        template<typename T>
        constexpr Int3<T> max(const Int3<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
        }

        template<typename T>
        constexpr Int3<T> max(T lhs, const Int3<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
        }
    }
}
