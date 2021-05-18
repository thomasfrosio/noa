/**
 * @file noa/util/Int4.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

namespace Noa {
    template<typename>
    struct Float4;

    template<typename T>
    struct alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Int4 {
        std::enable_if_t<Noa::Traits::is_int_v<T>, T> x{}, y{}, z{}, w{};
        typedef T value_type;

        // Constructors.
        NOA_HD constexpr Int4() = default;
        NOA_HD constexpr Int4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}
        NOA_HD constexpr explicit Int4(T v) : x(v), y(v), z(v), w(v) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int4(U* ptr)
                : x(static_cast<T>(ptr[0])), y(static_cast<T>(ptr[1])),
                  z(static_cast<T>(ptr[2])), w(static_cast<T>(ptr[3])) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Int4<U> vec)
                : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)),
                  z(static_cast<T>(vec.z)), w(static_cast<T>(vec.w)) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Float4<U> vec)
                : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)),
                  z(static_cast<T>(vec.z)), w(static_cast<T>(vec.w)) {}

        // Assignment operators.
        NOA_HD constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr auto& operator=(U* v) noexcept {
            x = static_cast<T>(v[0]);
            y = static_cast<T>(v[1]);
            z = static_cast<T>(v[2]);
            w = static_cast<T>(v[3]);
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(Int4<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            w = static_cast<T>(vec.w);
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(Float4<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            w = static_cast<T>(vec.w);
            return *this;
        }

        NOA_HD constexpr Int4<T>& operator+=(const Int4<T>& rhs) noexcept;
        NOA_HD constexpr Int4<T>& operator-=(const Int4<T>& rhs) noexcept;
        NOA_HD constexpr Int4<T>& operator*=(const Int4<T>& rhs) noexcept;
        NOA_HD constexpr Int4<T>& operator/=(const Int4<T>& rhs) noexcept;

        NOA_HD constexpr Int4<T>& operator+=(T rhs) noexcept;
        NOA_HD constexpr Int4<T>& operator-=(T rhs) noexcept;
        NOA_HD constexpr Int4<T>& operator*=(T rhs) noexcept;
        NOA_HD constexpr Int4<T>& operator/=(T rhs) noexcept;

        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return 4; }
        [[nodiscard]] NOA_HD static constexpr size_t elements() noexcept { return size(); }
        [[nodiscard]] NOA_IH constexpr std::array<T, 4> toArray() const noexcept { return {x, y, z, w}; }
    };

    using int4_t = Int4<int>;
    using uint4_t = Int4<uint>;
    using long4_t = Int4<long long>;
    using ulong4_t = Int4<unsigned long long>;

    template<> NOA_IH std::string String::typeName<int4_t>() { return "int4"; }
    template<> NOA_IH std::string String::typeName<uint4_t>() { return "uint4"; }
    template<> NOA_IH std::string String::typeName<long4_t>() { return "long4"; }
    template<> NOA_IH std::string String::typeName<ulong4_t>() { return "ulong4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Noa::Int4<T>& v) {
        os << String::format("({},{},{},{})", v.x, v.y, v.z, v.w);
        return os;
    }

    /* --- Binary Arithmetic Operators --- */

    template<typename T>
    NOA_FHD constexpr Int4<T> operator+(Int4<T> lhs, Int4<T> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator+(T lhs, Int4<T> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator+(Int4<T> lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int4<T> operator-(Int4<T> lhs, Int4<T> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator-(T lhs, Int4<T> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator-(Int4<T> lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int4<T> operator*(Int4<T> lhs, Int4<T> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator*(T lhs, Int4<T> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator*(Int4<T> lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int4<T> operator/(Int4<T> lhs, Int4<T> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator/(T lhs, Int4<T> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
    }
    template<typename T>
    NOA_FHD constexpr Int4<T> operator/(Int4<T> lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator+=(const Int4<T>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator+=(T rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator-=(const Int4<T>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator-=(T rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator*=(const Int4<T>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator*=(T rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator/=(const Int4<T>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int4<T>& Int4<T>::operator/=(T rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename T>
    NOA_FHD constexpr bool operator>(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>(const Int4<T>& lhs, T rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs && lhs.z > rhs && lhs.w > rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>(T lhs, const Int4<T>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y && lhs > rhs.z && lhs > rhs.w;
    }

    template<typename T>
    NOA_FHD constexpr bool operator<(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<(const Int4<T>& lhs, T rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs && lhs.z < rhs && lhs.w < rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<(T lhs, const Int4<T>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y && lhs < rhs.z && lhs < rhs.w;
    }

    template<typename T>
    NOA_FHD constexpr bool operator>=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>=(const Int4<T>& lhs, T rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs && lhs.w >= rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>=(T lhs, const Int4<T>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z && lhs >= rhs.w;
    }

    template<typename T>
    NOA_FHD constexpr bool operator<=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<=(const Int4<T>& lhs, T rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs && lhs.w <= rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<=(T lhs, const Int4<T>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z && lhs <= rhs.w;
    }

    template<typename T>
    NOA_FHD constexpr bool operator==(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
    }
    template<typename T>
    NOA_FHD constexpr bool operator==(const Int4<T>& lhs, T rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs && lhs.z == rhs && lhs.w == rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator==(T lhs, const Int4<T>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y && lhs == rhs.z && lhs == rhs.w;
    }

    template<typename T>
    NOA_FHD constexpr bool operator!=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename T>
    NOA_FHD constexpr bool operator!=(const Int4<T>& lhs, T rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename T>
    NOA_FHD constexpr bool operator!=(T lhs, const Int4<T>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

namespace Noa::Math {
    template<class T>
    [[nodiscard]] NOA_FHD constexpr T sum(const Int4<T>& v) noexcept {
        return v.x + v.y + v.z + v.w;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T prod(const Int4<T>& v) noexcept {
        return v.x * v.y * v.z * v.w;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> min(Int4<T> lhs, Int4<T> rhs) {
        return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> min(Int4<T> lhs, T rhs) {
        return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> min(T lhs, Int4<T> rhs) {
        return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> max(Int4<T> lhs, Int4<T> rhs) {
        return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> max(Int4<T> lhs, T rhs) {
        return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> max(T lhs, Int4<T> rhs) {
        return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
    }
}

namespace Noa {
    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t getElements(const Int4<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y) *
               static_cast<size_t>(v.z) * static_cast<size_t>(v.w);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t getElementsSlice(const Int4<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t getElementsFFT(const Int4<T>& v) noexcept {
        return static_cast<size_t>(v.x / 2 + 1) * static_cast<size_t>(v.y) *
               static_cast<size_t>(v.z) * static_cast<size_t>(v.w);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> getShapeSlice(const Int4<T>& v) noexcept {
        return {v.x, v.y, 1, 1};
    }
}

namespace Noa::Traits {
    template<typename> struct p_is_int4 : std::false_type {};
    template<typename T> struct p_is_int4<Noa::Int4<T>> : std::true_type {};
    template<typename T> using is_int4 = std::bool_constant<p_is_int4<Noa::Traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

    template<typename> struct p_is_uint4 : std::false_type {};
    template<typename T> struct p_is_uint4<Noa::Int4<T>> : std::bool_constant<Noa::Traits::is_uint_v<T>> {};
    template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<Noa::Traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

    template<typename T> struct proclaim_is_intX<Noa::Int4<T>> : std::true_type {};
    template<typename T> struct proclaim_is_uintX<Noa::Int4<T>> : std::bool_constant<Noa::Traits::is_uint_v<T>> {};
}
