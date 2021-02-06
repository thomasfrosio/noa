/**
 * @file noa/util/Int4.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>
#include <spdlog/fmt/fmt.h>

#include "noa/Define.h"
#include "noa/util/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

namespace Noa {
    template<typename>
    struct Float4;

    template<typename T = int32_t>
    struct Int4 {
        std::enable_if_t<Noa::Traits::is_int_v<T>, T> x{0}, y{0}, z{0}, w{0};

        // Constructors.
        NOA_HD constexpr Int4() = default;
        NOA_HD constexpr Int4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        NOA_HD constexpr explicit Int4(T v) : x(v), y(v), z(v), w(v) {}
        NOA_HD constexpr explicit Int4(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int4(U* ptr)
                : x(T(ptr[0])), y(T(ptr[1])), z(T(ptr[2])), w(T(ptr[3])) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Int4<U> vec)
                : x(T(vec.x)), y(T(vec.y)), z(T(vec.z)), w(T(vec.w)) {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Float4<U> vec)
                : x(T(vec.x)), y(T(vec.y)), z(T(vec.z)), w(T(vec.w)) {}

        // Assignment operators.
        NOA_FHD constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }

        NOA_FHD constexpr auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            w = ptr[3];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_FHD constexpr auto& operator=(U* v) noexcept {
            x = T(v[0]);
            y = T(v[1]);
            z = T(v[2]);
            w = T(v[3]);
            return *this;
        }

        template<typename U>
        NOA_FHD constexpr auto& operator=(Int4<U> vec) noexcept {
            x = T(vec.x);
            y = T(vec.y);
            z = T(vec.z);
            w = T(vec.w);
            return *this;
        }

        template<typename U>
        NOA_FHD constexpr auto& operator=(Float4<U> vec) noexcept {
            x = T(vec.x);
            y = T(vec.y);
            z = T(vec.z);
            w = T(vec.w);
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

        [[nodiscard]] NOA_FHD static constexpr size_t size() noexcept { return 4U; }
        [[nodiscard]] NOA_IH constexpr std::array<T, 4U> toArray() const noexcept { return {x, y, z, w}; }
        [[nodiscard]] NOA_IH std::string toString() const { return String::format("({}, {}, {}, {})", x, y, z, w); }
    };

    /* --- Binary Arithmetic Operators --- */

    template<typename I>
    NOA_FHD constexpr Int4<I> operator+(Int4<I> lhs, Int4<I> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator+(I lhs, Int4<I> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator+(Int4<I> lhs, I rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
    }

    template<typename I>
    NOA_FHD constexpr Int4<I> operator-(Int4<I> lhs, Int4<I> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator-(I lhs, Int4<I> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator-(Int4<I> lhs, I rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
    }

    template<typename I>
    NOA_FHD constexpr Int4<I> operator*(Int4<I> lhs, Int4<I> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator*(I lhs, Int4<I> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator*(Int4<I> lhs, I rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
    }

    template<typename I>
    NOA_FHD constexpr Int4<I> operator/(Int4<I> lhs, Int4<I> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator/(I lhs, Int4<I> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
    }
    template<typename I>
    NOA_FHD constexpr Int4<I> operator/(Int4<I> lhs, I rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator+=(const Int4<I>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator+=(I rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator-=(const Int4<I>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator-=(I rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator*=(const Int4<I>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator*=(I rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator/=(const Int4<I>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Int4<I>& Int4<I>::operator/=(I rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename I>
    NOA_FHD constexpr bool operator>(const Int4<I>& lhs, const Int4<I>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>(const Int4<I>& lhs, I rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs && lhs.z > rhs && lhs.w > rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>(I lhs, const Int4<I>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y && lhs > rhs.z && lhs > rhs.w;
    }

    template<typename I>
    NOA_FHD constexpr bool operator<(const Int4<I>& lhs, const Int4<I>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<(const Int4<I>& lhs, I rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs && lhs.z < rhs && lhs.w < rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<(I lhs, const Int4<I>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y && lhs < rhs.z && lhs < rhs.w;
    }

    template<typename I>
    NOA_FHD constexpr bool operator>=(const Int4<I>& lhs, const Int4<I>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>=(const Int4<I>& lhs, I rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs && lhs.w >= rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>=(I lhs, const Int4<I>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z && lhs >= rhs.w;
    }

    template<typename I>
    NOA_FHD constexpr bool operator<=(const Int4<I>& lhs, const Int4<I>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<=(const Int4<I>& lhs, I rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs && lhs.w <= rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<=(I lhs, const Int4<I>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z && lhs <= rhs.w;
    }

    template<typename I>
    NOA_FHD constexpr bool operator==(const Int4<I>& lhs, const Int4<I>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
    }
    template<typename I>
    NOA_FHD constexpr bool operator==(const Int4<I>& lhs, I rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs && lhs.z == rhs && lhs.w == rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator==(I lhs, const Int4<I>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y && lhs == rhs.z && lhs == rhs.w;
    }

    template<typename I>
    NOA_FHD constexpr bool operator!=(const Int4<I>& lhs, const Int4<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_FHD constexpr bool operator!=(const Int4<I>& lhs, I rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_FHD constexpr bool operator!=(I lhs, const Int4<I>& rhs) noexcept {
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
    [[nodiscard]] NOA_FHD constexpr size_t elements(const Int4<T>& v) noexcept {
        return size_t(v.x) * size_t(v.y) * size_t(v.z) * size_t(v.w);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t elementsSlice(const Int4<T>& v) noexcept {
        return size_t(v.x) * size_t(v.y);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t elementsFFT(const Int4<T>& v) noexcept {
        return size_t(v.x / 2 + 1) * size_t(v.y) * size_t(v.z) * size_t(v.w);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int4<T> slice(const Int4<T>& v) noexcept {
        return {v.x, v.y, 1, 1};
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

//@CLION-formatter:off
namespace Noa::Traits {
    template<typename> struct p_is_int4 : std::false_type {};
    template<typename T> struct p_is_int4<Noa::Int4<T>> : std::true_type {};
    template<typename T> using is_int4 = std::bool_constant<p_is_int4<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

    template<typename> struct p_is_uint4 : std::false_type {};
    template<typename T> struct p_is_uint4<Noa::Int4<T>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

    template<typename T> struct proclaim_is_intX<Noa::Int4<T>> : std::true_type {};
    template<typename T> struct proclaim_is_uintX<Noa::Int4<T>> : std::bool_constant<is_uint_v<T>> {};
}
//@CLION-formatter:on

template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_int4_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& int4, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(int4.toString(), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Int4<T>& int4) {
    os << int4.toString();
    return os;
}
