/**
 * @file noa/util/Float2.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>
#include <spdlog/fmt/fmt.h>

#include "noa/Definitions.h"
#include "noa/util/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

namespace Noa {
    template<typename>
    struct Int2;

    template<typename T>
    struct Float2 {
        std::enable_if_t<Traits::is_float_v < T>, T> x{ 0 }, y{ 0 };

        // Constructors.
        NOA_HD constexpr Float2() = default;
        NOA_HD constexpr Float2(T xi, T yi) : x(xi), y(yi) {}

        NOA_HD constexpr explicit Float2(T v) : x(v), y(v) {}
        NOA_HD constexpr explicit Float2(T* ptr) : x(ptr[0]), y(ptr[1]) {}

        template<typename U, typename = std::enable_if_t<Traits::is_float_v < U>>>
        NOA_HD constexpr explicit Float2(U* ptr) : x(T(ptr[0])), y(T(ptr[1])) {}

        template<typename U>
        NOA_HD constexpr explicit Float2(Float2<U> v) : x(T(v.x)), y(T(v.y)) {}

        template<typename U>
        NOA_HD constexpr explicit Float2(Int2<U> v) : x(T(v.x)), y(T(v.y)) {}

        // Assignment operators.
        NOA_FHD constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            return *this;
        }

        NOA_FHD constexpr auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Traits::is_scalar_v < U>>>
        NOA_FHD constexpr auto& operator=(U* ptr) noexcept {
            x = T(ptr[0]);
            y = T(ptr[1]);
            return *this;
        }

        template<typename U>
        NOA_FHD constexpr auto& operator=(Float2<U> v) noexcept {
            x = T(v.x);
            y = T(v.y);
            return *this;
        }

        template<typename U>
        NOA_FHD constexpr auto& operator=(Int2<U> v) noexcept {
            x = T(v.x);
            y = T(v.y);
            return *this;
        }

        NOA_HD constexpr Float2<T>& operator+=(const Float2<T>& rhs) noexcept;
        NOA_HD constexpr Float2<T>& operator-=(const Float2<T>& rhs) noexcept;
        NOA_HD constexpr Float2<T>& operator*=(const Float2<T>& rhs) noexcept;
        NOA_HD constexpr Float2<T>& operator/=(const Float2<T>& rhs) noexcept;

        NOA_HD constexpr Float2<T>& operator+=(T rhs) noexcept;
        NOA_HD constexpr Float2<T>& operator-=(T rhs) noexcept;
        NOA_HD constexpr Float2<T>& operator*=(T rhs) noexcept;
        NOA_HD constexpr Float2<T>& operator/=(T rhs) noexcept;

        [[nodiscard]] NOA_FHD static constexpr size_t size() noexcept { return 2U; }
        [[nodiscard]] NOA_IH constexpr std::array<T, 2U> toArray() const noexcept { return {x, y}; }
        [[nodiscard]] NOA_IH std::string toString() const { return String::format("({},{})", x, y); }
    };

    template<typename T>
    [[nodiscard]] NOA_IH std::string toString(const Float2<T>& v) { return v.toString(); }

    /* --- Binary Arithmetic Operators --- */

    template<typename I>
    NOA_FHD constexpr Float2<I> operator+(Float2<I> lhs, Float2<I> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator+(I lhs, Float2<I> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator+(Float2<I> lhs, I rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs};
    }

    template<typename I>
    NOA_FHD constexpr Float2<I> operator-(Float2<I> lhs, Float2<I> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator-(I lhs, Float2<I> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator-(Float2<I> lhs, I rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs};
    }

    template<typename I>
    NOA_FHD constexpr Float2<I> operator*(Float2<I> lhs, Float2<I> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator*(I lhs, Float2<I> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator*(Float2<I> lhs, I rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs};
    }

    template<typename I>
    NOA_FHD constexpr Float2<I> operator/(Float2<I> lhs, Float2<I> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator/(I lhs, Float2<I> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y};
    }
    template<typename I>
    NOA_FHD constexpr Float2<I> operator/(Float2<I> lhs, I rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator+=(const Float2<I>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator+=(I rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator-=(const Float2<I>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator-=(I rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator*=(const Float2<I>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator*=(I rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator/=(const Float2<I>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float2<I>& Float2<I>::operator/=(I rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename I>
    NOA_FHD constexpr bool operator>(const Float2<I>& lhs, const Float2<I>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>(const Float2<I>& lhs, I rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>(I lhs, const Float2<I>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y;
    }

    template<typename I>
    NOA_FHD constexpr bool operator<(const Float2<I>& lhs, const Float2<I>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<(const Float2<I>& lhs, I rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<(I lhs, const Float2<I>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y;
    }

    template<typename I>
    NOA_FHD constexpr bool operator>=(const Float2<I>& lhs, const Float2<I>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>=(const Float2<I>& lhs, I rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>=(I lhs, const Float2<I>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y;
    }

    template<typename I>
    NOA_FHD constexpr bool operator<=(const Float2<I>& lhs, const Float2<I>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<=(const Float2<I>& lhs, I rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<=(I lhs, const Float2<I>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y;
    }

    template<typename I>
    NOA_FHD constexpr bool operator==(const Float2<I>& lhs, const Float2<I>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y;
    }
    template<typename I>
    NOA_FHD constexpr bool operator==(const Float2<I>& lhs, I rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator==(I lhs, const Float2<I>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y;
    }

    template<typename I>
    NOA_FHD constexpr bool operator!=(const Float2<I>& lhs, const Float2<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_FHD constexpr bool operator!=(const Float2<I>& lhs, I rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_FHD constexpr bool operator!=(I lhs, const Float2<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

#define ULP 2
#define EPSILON 1e-6f

namespace Noa::Math {
    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> floor(const Float2<T>& v) {
        return Float2<T>(floor(v.x), floor(v.y));
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> ceil(const Float2<T>& v) {
        return Float2<T>(ceil(v.x), ceil(v.y));
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T lengthSq(const Float2<T>& v) noexcept {
        return v.x * v.x + v.y * v.y;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T length(const Float2<T>& v) {
        return sqrt(lengthSq(v));
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> normalize(const Float2<T>& v) {
        return v / length(v);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T sum(const Float2<T>& v) noexcept {
        return v.x + v.y;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T prod(const Float2<T>& v) noexcept {
        return v.x * v.y;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T dot(Float2<T> a, Float2<T> b) noexcept {
        return a.x * b.x + a.y * b.y;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> min(Float2<T> lhs, Float2<T> rhs) {
        return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> min(Float2<T> lhs, T rhs) {
        return {min(lhs.x, rhs), min(lhs.y, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> min(T lhs, Float2<T> rhs) {
        return {min(lhs, rhs.x), min(lhs, rhs.y)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> max(Float2<T> lhs, Float2<T> rhs) {
        return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> max(Float2<T> lhs, T rhs) {
        return {max(lhs.x, rhs), max(lhs.y, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float2<T> max(T lhs, Float2<T> rhs) {
        return {max(lhs, rhs.x), max(lhs, rhs.y)};
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] NOA_FHD constexpr bool isEqual(const Float2<T>& a, const Float2<T>& b, T e = EPSILON) {
        return isEqual<ulp>(a.x, b.x, e) && isEqual<ulp>(a.y, b.y, e);
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] NOA_FHD constexpr bool isEqual(const Float2<T>& a, T b, T e = EPSILON) {
        return isEqual<ulp>(a.x, b, e) && isEqual<ulp>(a.y, b, e);
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] NOA_FHD constexpr bool isEqual(T a, const Float2<T>& b, T e = EPSILON) {
        return isEqual<ulp>(a, b.x, e) && isEqual<ulp>(a, b.y, e);
    }
}

#undef ULP
#undef EPSILON

//@CLION-formatter:off
namespace Noa::Traits {
    template<typename T> struct p_is_float2 : std::false_type {};
    template<typename T> struct p_is_float2<Noa::Float2<T>> : std::true_type {};
    template<typename T> using is_float2 = std::bool_constant<p_is_float2<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float2_v = is_float2<T>::value;

    template<typename T> struct proclaim_is_floatX<Noa::Float2<T>> : std::true_type {};
}
//@CLION-formatter:on

template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_float2_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& float2, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(float2.toString(), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Float2<T>& float2) {
    os << float2.toString();
    return os;
}
