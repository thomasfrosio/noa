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
    struct Int4;

    template<typename T>
    struct Float4 {
        std::enable_if_t<Noa::Traits::is_float_v<T>, T> x{0}, y{0}, z{0}, w{0};

        // Constructors.
        constexpr Float4() = default;
        constexpr Float4(T xi, T yi, T zi, T wi) : x(xi), y(yi), z(zi), w(wi) {}

        constexpr explicit Float4(T v) : x(v), y(v), z(v), w(v) {}
        constexpr explicit Float4(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]), w(ptr[3]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_float_v<U>>>
        constexpr explicit Float4(U* ptr)
                : x(T(ptr[0])), y(T(ptr[1])), z(T(ptr[2])), w(T(ptr[3])) {}

        template<typename U>
        constexpr explicit Float4(Float4<U> v)
                : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) {}

        template<typename U>
        constexpr explicit Float4(Int4<U> v)
                : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) {}

        // Assignment operators.
        inline constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            w = v;
            return *this;
        }

        inline constexpr auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            w = ptr[3];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        inline constexpr auto& operator=(U* ptr) noexcept {
            x = T(ptr[0]);
            y = T(ptr[1]);
            z = T(ptr[2]);
            w = T(ptr[3]);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Float4<U> v) noexcept {
            x = T(v.x);
            y = T(v.y);
            z = T(v.z);
            w = T(v.w);
            return *this;
        }

        template<typename U>
        inline constexpr auto& operator=(Int4<U> v) noexcept {
            x = T(v.x);
            y = T(v.y);
            z = T(v.z);
            w = T(v.w);
            return *this;
        }

        NOA_DH inline constexpr Float4<T>& operator+=(const Float4<T>& rhs) noexcept;
        NOA_DH inline constexpr Float4<T>& operator-=(const Float4<T>& rhs) noexcept;
        NOA_DH inline constexpr Float4<T>& operator*=(const Float4<T>& rhs) noexcept;
        NOA_DH inline constexpr Float4<T>& operator/=(const Float4<T>& rhs) noexcept;

        NOA_DH inline constexpr Float4<T>& operator+=(T rhs) noexcept;
        NOA_DH inline constexpr Float4<T>& operator-=(T rhs) noexcept;
        NOA_DH inline constexpr Float4<T>& operator*=(T rhs) noexcept;
        NOA_DH inline constexpr Float4<T>& operator/=(T rhs) noexcept;

        [[nodiscard]] static inline constexpr size_t size() noexcept { return 4; }
        [[nodiscard]] inline constexpr std::array<T, 4U> toArray() const noexcept { return {x, y, z, w}; }
        [[nodiscard]] inline std::string toString() const { return String::format("({}, {}, {}, {})", x, y, z, w); }
    };

    /* --- Binary Arithmetic Operators --- */

    template<typename I>
    NOA_DH inline constexpr Float4<I> operator+(Float4<I> lhs, Float4<I> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator+(I lhs, Float4<I> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z, lhs + rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator+(Float4<I> lhs, I rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs, lhs.w + rhs};
    }

    template<typename I>
    NOA_DH inline constexpr Float4<I> operator-(Float4<I> lhs, Float4<I> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator-(I lhs, Float4<I> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z, lhs - rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator-(Float4<I> lhs, I rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs, lhs.w - rhs};
    }

    template<typename I>
    NOA_DH inline constexpr Float4<I> operator*(Float4<I> lhs, Float4<I> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z, lhs.w * rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator*(I lhs, Float4<I> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z, lhs * rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator*(Float4<I> lhs, I rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs};
    }

    template<typename I>
    NOA_DH inline constexpr Float4<I> operator/(Float4<I> lhs, Float4<I> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z, lhs.w / rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator/(I lhs, Float4<I> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z, lhs / rhs.w};
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I> operator/(Float4<I> lhs, I rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator+=(const Float4<I>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator+=(I rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator-=(const Float4<I>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator-=(I rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator*=(const Float4<I>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator*=(I rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator/=(const Float4<I>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename I>
    NOA_DH inline constexpr Float4<I>& Float4<I>::operator/=(I rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename I>
    NOA_DH inline constexpr bool operator>(const Float4<I>& lhs, const Float4<I>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z && lhs.w > rhs.w;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator>(const Float4<I>& lhs, I rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs && lhs.z > rhs && lhs.w > rhs;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator>(I lhs, const Float4<I>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y && lhs > rhs.z && lhs > rhs.w;
    }

    template<typename I>
    NOA_DH inline constexpr bool operator<(const Float4<I>& lhs, const Float4<I>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z && lhs.w < rhs.w;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator<(const Float4<I>& lhs, I rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs && lhs.z < rhs && lhs.w < rhs;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator<(I lhs, const Float4<I>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y && lhs < rhs.z && lhs < rhs.w;
    }

    template<typename I>
    NOA_DH inline constexpr bool operator>=(const Float4<I>& lhs, const Float4<I>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z && lhs.w >= rhs.w;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator>=(const Float4<I>& lhs, I rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs && lhs.w >= rhs;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator>=(I lhs, const Float4<I>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z && lhs >= rhs.w;
    }

    template<typename I>
    NOA_DH inline constexpr bool operator<=(const Float4<I>& lhs, const Float4<I>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z && lhs.w <= rhs.w;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator<=(const Float4<I>& lhs, I rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs && lhs.w <= rhs;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator<=(I lhs, const Float4<I>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z && lhs <= rhs.w;
    }

    template<typename I>
    NOA_DH inline constexpr bool operator==(const Float4<I>& lhs, const Float4<I>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator==(const Float4<I>& lhs, I rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs && lhs.z == rhs && lhs.w == rhs;
    }
    template<typename I>
    NOA_DH inline constexpr bool operator==(I lhs, const Float4<I>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y && lhs == rhs.z && lhs == rhs.w;
    }

    template<typename I>
    NOA_DH inline constexpr bool operator!=(const Float4<I>& lhs, const Float4<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_DH inline constexpr bool operator!=(const Float4<I>& lhs, I rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_DH inline constexpr bool operator!=(I lhs, const Float4<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

#define ULP 2
#define EPSILON 1e-6f

namespace Noa::Math {
    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> floor(const Float4<T>& v) {
        return Float4<T>(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> ceil(const Float4<T>& v) {
        return Float4<T>(ceil(v.x), ceil(v.y), ceil(v.z), ceil(v.w));
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr T lengthSq(const Float4<T>& v) noexcept {
        return v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr T length(const Float4<T>& v) {
        return sqrt(lengthSq(v));
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> normalize(const Float4<T>& v) {
        return v / length(v);
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr T sum(const Float4<T>& v) noexcept {
        return v.x + v.y + v.z + v.w;
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr T prod(const Float4<T>& v) noexcept {
        return v.x * v.y * v.z * v.w;
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> min(Float4<T> lhs, Float4<T> rhs) {
        return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z), min(lhs.w, rhs.w)};
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> min(Float4<T> lhs, T rhs) {
        return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs), min(lhs.w, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> min(T lhs, Float4<T> rhs) {
        return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z), min(lhs, rhs.w)};
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> max(Float4<T> lhs, Float4<T> rhs) {
        return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z), max(lhs.w, rhs.w)};
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> max(Float4<T> lhs, T rhs) {
        return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs), max(lhs.w, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_DH inline constexpr Float4<T> max(T lhs, Float4<T> rhs) {
        return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z), max(lhs, rhs.w)};
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] inline constexpr bool isEqual(const Float4<T>& a, const Float4<T>& b, T e = EPSILON) {
        return isEqual<ulp>(a.x, b.x, e) && isEqual<ulp>(a.y, b.y, e) &&
               isEqual<ulp>(a.z, b.z, e) && isEqual<ulp>(a.w, b.w, e);
    }

    template<uint32_t ulp = ULP, typename Float>
    [[nodiscard]] inline constexpr bool isEqual(const Float4<Float>& a, Float b, Float e = EPSILON) {
        return isEqual<ulp>(b, a.x, e) && isEqual<ulp>(b, a.y, e) &&
               isEqual<ulp>(b, a.z, e) && isEqual<ulp>(b, a.w, e);
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] inline constexpr bool isEqual(T a, const Float4<T>& b, T e = EPSILON) {
        return isEqual<ulp>(a, b.x, e) && isEqual<ulp>(a, b.y, e) &&
               isEqual<ulp>(a, b.z, e) && isEqual<ulp>(a, b.w, e);
    }
}

#undef ULP
#undef EPSILON

//@CLION-formatter:off
namespace Noa::Traits {
    template<typename T> struct p_is_float4 : std::false_type {};
    template<typename T> struct p_is_float4<Noa::Float4<T>> : std::true_type {};
    template<typename T> using is_float4 = std::bool_constant<p_is_float4<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float4_v = is_float4<T>::value;

    template<typename T> struct proclaim_is_floatX<Noa::Float4<T>> : std::true_type {};
}
//@CLION-formatter:on

template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_float4_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& float4, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(float4.toString(), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Float4<T>& float4) {
    os << float4.toString();
    return os;
}
