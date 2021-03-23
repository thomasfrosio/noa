/**
 * @file noa/util/Int3.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>
#include <spdlog/fmt/fmt.h>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

namespace Noa {
    template<typename>
    struct Int3;

    template<typename T>
    struct Float3 {
        std::enable_if_t<Noa::Traits::is_float_v<T>, T> x{0}, y{0}, z{0};

        // Constructors.
        NOA_HD constexpr Float3() = default;
        NOA_HD constexpr Float3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        NOA_HD constexpr explicit Float3(T v) : x(v), y(v), z(v) {}
        NOA_HD constexpr explicit Float3(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Float3(U* ptr) : x(T(ptr[0])), y(T(ptr[1])), z(T(ptr[2])) {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Float3<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Float3(Int3<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) {}

        // Assignment operators.
        NOA_HD constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }

        NOA_HD constexpr auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr auto& operator=(U* ptr) noexcept {
            x = T(ptr[0]);
            y = T(ptr[1]);
            z = T(ptr[2]);
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(Float3<U> v) noexcept {
            x = T(v.x);
            y = T(v.y);
            z = T(v.z);
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(Int3<U> v) noexcept {
            x = T(v.x);
            y = T(v.y);
            z = T(v.z);
            return *this;
        }

        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return 3U; }
        [[nodiscard]] NOA_HOST constexpr std::array<T, 3U> toArray() const noexcept { return {x, y, z}; }
        [[nodiscard]] NOA_HOST std::string toString() const {
            return String::format("({:.3f},{:.3f},{:.3f})", x, y, z);
        }

        NOA_HD constexpr Float3<T>& operator+=(const Float3<T>& rhs) noexcept;
        NOA_HD constexpr Float3<T>& operator-=(const Float3<T>& rhs) noexcept;
        NOA_HD constexpr Float3<T>& operator*=(const Float3<T>& rhs) noexcept;
        NOA_HD constexpr Float3<T>& operator/=(const Float3<T>& rhs) noexcept;

        NOA_HD constexpr Float3<T>& operator+=(T rhs) noexcept;
        NOA_HD constexpr Float3<T>& operator-=(T rhs) noexcept;
        NOA_HD constexpr Float3<T>& operator*=(T rhs) noexcept;
        NOA_HD constexpr Float3<T>& operator/=(T rhs) noexcept;
    };

    using float3_t = Float3<float>;
    using double3_t = Float3<double>;

    template<typename T>
    [[nodiscard]] NOA_IH std::string toString(const Float3<T>& v) { return v.toString(); }

    /* --- Binary Arithmetic Operators --- */

    template<typename I>
    NOA_FHD constexpr Float3<I> operator+(Float3<I> lhs, Float3<I> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator+(I lhs, Float3<I> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator+(Float3<I> lhs, I rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename I>
    NOA_FHD constexpr Float3<I> operator-(Float3<I> lhs, Float3<I> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator-(I lhs, Float3<I> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator-(Float3<I> lhs, I rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename I>
    NOA_FHD constexpr Float3<I> operator*(Float3<I> lhs, Float3<I> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator*(I lhs, Float3<I> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator*(Float3<I> lhs, I rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename I>
    NOA_FHD constexpr Float3<I> operator/(Float3<I> lhs, Float3<I> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator/(I lhs, Float3<I> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename I>
    NOA_FHD constexpr Float3<I> operator/(Float3<I> lhs, I rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator+=(const Float3<I>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator+=(I rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator-=(const Float3<I>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator-=(I rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator*=(const Float3<I>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator*=(I rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator/=(const Float3<I>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename I>
    NOA_FHD constexpr Float3<I>& Float3<I>::operator/=(I rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename I>
    NOA_FHD constexpr bool operator>(const Float3<I>& lhs, const Float3<I>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>(const Float3<I>& lhs, I rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs && lhs.z > rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>(I lhs, const Float3<I>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y && lhs > rhs.z;
    }

    template<typename I>
    NOA_FHD constexpr bool operator<(const Float3<I>& lhs, const Float3<I>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<(const Float3<I>& lhs, I rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs && lhs.z < rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<(I lhs, const Float3<I>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y && lhs < rhs.z;
    }

    template<typename I>
    NOA_FHD constexpr bool operator>=(const Float3<I>& lhs, const Float3<I>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>=(const Float3<I>& lhs, I rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator>=(I lhs, const Float3<I>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z;
    }

    template<typename I>
    NOA_FHD constexpr bool operator<=(const Float3<I>& lhs, const Float3<I>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<=(const Float3<I>& lhs, I rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator<=(I lhs, const Float3<I>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z;
    }

    template<typename I>
    NOA_FHD constexpr bool operator==(const Float3<I>& lhs, const Float3<I>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
    template<typename I>
    NOA_FHD constexpr bool operator==(const Float3<I>& lhs, I rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs && lhs.z == rhs;
    }
    template<typename I>
    NOA_FHD constexpr bool operator==(I lhs, const Float3<I>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y && lhs == rhs.z;
    }

    template<typename I>
    NOA_FHD constexpr bool operator!=(const Float3<I>& lhs, const Float3<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_FHD constexpr bool operator!=(const Float3<I>& lhs, I rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_FHD constexpr bool operator!=(I lhs, const Float3<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

#define ULP 2
#define EPSILON 1e-6f

namespace Noa::Math {
    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> floor(const Float3<T>& v) {
        return Float3<T>(floor(v.x), floor(v.y), floor(v.z));
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> ceil(const Float3<T>& v) {
        return Float3<T>(ceil(v.x), ceil(v.y), ceil(v.z));
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T lengthSq(const Float3<T>& v) noexcept {
        return v.x * v.x + v.y * v.y + v.z * v.z;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T length(const Float3<T>& v) {
        return sqrt(lengthSq(v));
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> normalize(const Float3<T>& v) {
        return v / length(v);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T sum(const Float3<T>& v) noexcept {
        return v.x + v.y + v.z;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T prod(const Float3<T>& v) noexcept {
        return v.x * v.y * v.z;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T dot(const Float3<T>& a, const Float3<T>& b) noexcept {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> cross(const Float3<T>& a, const Float3<T>& b) noexcept {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> min(Float3<T> lhs, Float3<T> rhs) {
        return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> min(Float3<T> lhs, T rhs) {
        return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> min(T lhs, Float3<T> rhs) {
        return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> max(Float3<T> lhs, Float3<T> rhs) {
        return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> max(Float3<T> lhs, T rhs) {
        return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Float3<T> max(T lhs, Float3<T> rhs) {
        return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] NOA_FHD constexpr bool isEqual(const Float3<T>& a, const Float3<T>& b, T e = EPSILON) {
        return isEqual<ulp>(a.x, b.x, e) && isEqual<ulp>(a.y, b.y, e) && isEqual<ulp>(a.z, b.z, e);
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] NOA_FHD constexpr bool isEqual(const Float3<T>& a, T b, T e = EPSILON) {
        return isEqual<ulp>(a.x, b, e) && isEqual<ulp>(a.y, b, e) && isEqual<ulp>(a.z, b, e);
    }

    template<uint32_t ulp = ULP, typename T>
    [[nodiscard]] NOA_FHD constexpr bool isEqual(T a, const Float3<T>& b, T e = EPSILON) {
        return isEqual<ulp>(a, b.x, e) && isEqual<ulp>(a, b.y, e) && isEqual<ulp>(a, b.z, e);
    }
}

#undef ULP
#undef EPSILON

namespace Noa::Traits {
    template<typename T> struct p_is_float3 : std::false_type {};
    template<typename T> struct p_is_float3<Noa::Float3<T>> : std::true_type {};
    template<typename T> using is_float3 = std::bool_constant<p_is_float3<Noa::Traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float3_v = is_float3<T>::value;

    template<typename T> struct proclaim_is_floatX<Noa::Float3<T>> : std::true_type {};
}

template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_float3_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& float3, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(float3.toString(), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Float3<T>& float3) {
    os << float3.toString();
    return os;
}
