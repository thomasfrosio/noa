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
#include "noa/util/Math.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Format.h"

namespace Noa {
    template<typename>
    struct Float3;

    template<typename T>
    struct Int3 {
        std::enable_if_t<Noa::Traits::is_int_v<T>, T> x{0}, y{0}, z{0};

        // Constructors.
        NOA_HD constexpr Int3() = default;
        NOA_HD constexpr Int3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}

        NOA_HD constexpr explicit Int3(T v) : x(v), y(v), z(v) {}
        NOA_HD constexpr explicit Int3(T* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int3(U* ptr) : x(T(ptr[0])), y(T(ptr[1])), z(T(ptr[2])) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Int3<U> vec) : x(T(vec.x)), y(T(vec.y)), z(T(vec.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Float3<U> vec) : x(T(vec.x)), y(T(vec.y)), z(T(vec.z)) {}

        // Assignment operators.
        NOA_IHD constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }

        NOA_IHD constexpr auto& operator=(T* ptr) noexcept {
            x = ptr[0];
            y = ptr[1];
            z = ptr[2];
            return *this;
        }

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_IHD constexpr auto& operator=(U* ptr) noexcept {
            x = T(ptr[0]);
            y = T(ptr[1]);
            z = T(ptr[2]);
            return *this;
        }

        template<typename U>
        NOA_IHD constexpr auto& operator=(Int3<U> vec) noexcept {
            x = T(vec.x);
            y = T(vec.y);
            z = T(vec.z);
            return *this;
        }

        template<typename U>
        NOA_IHD constexpr auto& operator=(Float3<U> vec) noexcept {
            x = T(vec.x);
            y = T(vec.y);
            z = T(vec.z);
            return *this;
        }

        NOA_IHD constexpr Int3<T>& operator+=(const Int3<T>& rhs) noexcept;
        NOA_IHD constexpr Int3<T>& operator-=(const Int3<T>& rhs) noexcept;
        NOA_IHD constexpr Int3<T>& operator*=(const Int3<T>& rhs) noexcept;
        NOA_IHD constexpr Int3<T>& operator/=(const Int3<T>& rhs) noexcept;

        NOA_IHD constexpr Int3<T>& operator+=(T rhs) noexcept;
        NOA_IHD constexpr Int3<T>& operator-=(T rhs) noexcept;
        NOA_IHD constexpr Int3<T>& operator*=(T rhs) noexcept;
        NOA_IHD constexpr Int3<T>& operator/=(T rhs) noexcept;

        [[nodiscard]] NOA_IHD static constexpr size_t size() noexcept { return 3U; }
        [[nodiscard]] NOA_IH constexpr std::array<T, 3U> toArray() const noexcept { return {x, y, z}; }
        [[nodiscard]] NOA_IH std::string toString() const { return String::format("({},{},{})", x, y, z); }
    };

    template<typename T>
    [[nodiscard]] NOA_IH std::string toString(const Int3<T>& v) { return v.toString(); }

    /* --- Binary Arithmetic Operators --- */

    template<typename I>
    NOA_HD constexpr Int3<I> operator+(Int3<I> lhs, Int3<I> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator+(I lhs, Int3<I> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator+(Int3<I> lhs, I rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename I>
    NOA_HD constexpr Int3<I> operator-(Int3<I> lhs, Int3<I> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator-(I lhs, Int3<I> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator-(Int3<I> lhs, I rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename I>
    NOA_HD constexpr Int3<I> operator*(Int3<I> lhs, Int3<I> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator*(I lhs, Int3<I> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator*(Int3<I> lhs, I rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename I>
    NOA_HD constexpr Int3<I> operator/(Int3<I> lhs, Int3<I> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator/(I lhs, Int3<I> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename I>
    NOA_HD constexpr Int3<I> operator/(Int3<I> lhs, I rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator+=(const Int3<I>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator+=(I rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator-=(const Int3<I>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator-=(I rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator*=(const Int3<I>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator*=(I rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator/=(const Int3<I>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename I>
    NOA_HD constexpr Int3<I>& Int3<I>::operator/=(I rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename I>
    NOA_HD constexpr bool operator>(const Int3<I>& lhs, const Int3<I>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z;
    }
    template<typename I>
    NOA_HD constexpr bool operator>(const Int3<I>& lhs, I rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs && lhs.z > rhs;
    }
    template<typename I>
    NOA_HD constexpr bool operator>(I lhs, const Int3<I>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y && lhs > rhs.z;
    }

    template<typename I>
    NOA_HD constexpr bool operator<(const Int3<I>& lhs, const Int3<I>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z;
    }
    template<typename I>
    NOA_HD constexpr bool operator<(const Int3<I>& lhs, I rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs && lhs.z < rhs;
    }
    template<typename I>
    NOA_HD constexpr bool operator<(I lhs, const Int3<I>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y && lhs < rhs.z;
    }

    template<typename I>
    NOA_HD constexpr bool operator>=(const Int3<I>& lhs, const Int3<I>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z;
    }
    template<typename I>
    NOA_HD constexpr bool operator>=(const Int3<I>& lhs, I rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs;
    }
    template<typename I>
    NOA_HD constexpr bool operator>=(I lhs, const Int3<I>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z;
    }

    template<typename I>
    NOA_HD constexpr bool operator<=(const Int3<I>& lhs, const Int3<I>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z;
    }
    template<typename I>
    NOA_HD constexpr bool operator<=(const Int3<I>& lhs, I rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs;
    }
    template<typename I>
    NOA_HD constexpr bool operator<=(I lhs, const Int3<I>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z;
    }

    template<typename I>
    NOA_HD constexpr bool operator==(const Int3<I>& lhs, const Int3<I>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
    template<typename I>
    NOA_HD constexpr bool operator==(const Int3<I>& lhs, I rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs && lhs.z == rhs;
    }
    template<typename I>
    NOA_HD constexpr bool operator==(I lhs, const Int3<I>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y && lhs == rhs.z;
    }

    template<typename I>
    NOA_HD constexpr bool operator!=(const Int3<I>& lhs, const Int3<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_HD constexpr bool operator!=(const Int3<I>& lhs, I rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename I>
    NOA_HD constexpr bool operator!=(I lhs, const Int3<I>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

namespace Noa::Math {
    template<class T>
    [[nodiscard]] NOA_HD constexpr T sum(const Int3<T>& v) noexcept {
        return v.x + v.y + v.z;
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr T prod(const Int3<T>& v) noexcept {
        return v.x * v.y * v.z;
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> min(Int3<T> lhs, Int3<T> rhs) {
        return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> min(Int3<T> lhs, T rhs) {
        return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> min(T lhs, Int3<T> rhs) {
        return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> max(Int3<T> lhs, Int3<T> rhs) {
        return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> max(Int3<T> lhs, T rhs) {
        return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> max(T lhs, Int3<T> rhs) {
        return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
    }
}

namespace Noa {
    template<class T>
    [[nodiscard]] NOA_HD constexpr size_t getElements(const Int3<T>& v) noexcept {
        return size_t(v.x) * size_t(v.y) * size_t(v.z);
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr size_t getElementsSlice(const Int3<T>& v) noexcept {
        return size_t(v.x) * size_t(v.y);
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr size_t getElementsFFT(const Int3<T>& v) noexcept {
        return size_t(v.x / 2 + 1) * size_t(v.y) * size_t(v.z);
    }

    template<class T>
    [[nodiscard]] NOA_HD constexpr Int3<T> getShapeSlice(const Int3<T>& v) noexcept {
        return {v.x, v.y, 1};
    }
}

//@CLION-formatter:off
namespace Noa::Traits {
    template<typename> struct p_is_int3 : std::false_type {};
    template<typename T> struct p_is_int3<Noa::Int3<T>> : std::true_type {};
    template<typename T> using is_int3 = std::bool_constant<p_is_int3<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

    template<typename> struct p_is_uint3 : std::false_type {};
    template<typename T> struct p_is_uint3<Noa::Int3<T>> : std::bool_constant<is_uint_v<T>> {};
    template<typename T> using is_uint3 = std::bool_constant<p_is_uint3<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

    template<typename T> struct proclaim_is_intX<Noa::Int3<T>> : std::true_type {};
    template<typename T> struct proclaim_is_uintX<Noa::Int3<T>> : std::bool_constant<is_uint_v<T>> {};
}
//@CLION-formatter:on

template<typename T>
struct fmt::formatter<T, std::enable_if_t<Noa::Traits::is_int3_v<T>, char>>
        : fmt::formatter<std::string> {
    template<typename FormatCtx>
    auto format(const T& int3, FormatCtx& ctx) {
        return fmt::formatter<std::string>::format(int3.toString(), ctx);
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Int3<T>& int3) {
    os << int3.toString();
    return os;
}
