/**
 * @file noa/util/Int3.h
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
    struct Float3;

    template<typename T>
    struct Int3 {
        std::enable_if_t<Noa::Traits::is_int_v<T>, T> x{}, y{}, z{};

        // Constructors.
        NOA_HD constexpr Int3() = default;
        NOA_HD constexpr Int3(T xi, T yi, T zi) : x(xi), y(yi), z(zi) {}
        NOA_HD constexpr explicit Int3(T v) : x(v), y(v), z(v) {}

        template<typename U, typename = std::enable_if_t<Noa::Traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int3(U* ptr)
                : x(static_cast<T>(ptr[0])), y(static_cast<T>(ptr[1])), z(static_cast<T>(ptr[2])) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Int3<U> vec)
                : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)), z(static_cast<T>(vec.z)) {}

        template<typename U>
        NOA_HD constexpr explicit Int3(Float3<U> vec)
                : x(static_cast<T>(vec.x)), y(static_cast<T>(vec.y)), z(static_cast<T>(vec.z)) {}

        // Assignment operators.
        NOA_HD constexpr auto& operator=(T v) noexcept {
            x = v;
            y = v;
            z = v;
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(U* ptr) noexcept {
            static_assert(Noa::Traits::is_scalar_v<U>);
            x = static_cast<T>(ptr[0]);
            y = static_cast<T>(ptr[1]);
            z = static_cast<T>(ptr[2]);
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(Int3<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            return *this;
        }

        template<typename U>
        NOA_HD constexpr auto& operator=(Float3<U> vec) noexcept {
            x = static_cast<T>(vec.x);
            y = static_cast<T>(vec.y);
            z = static_cast<T>(vec.z);
            return *this;
        }

        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return 3; }
        [[nodiscard]] NOA_HOST constexpr std::array<T, 3> toArray() const noexcept { return {x, y, z}; }
        [[nodiscard]] NOA_HOST std::string toString() const { return String::format("({},{},{})", x, y, z); }

        NOA_HD constexpr Int3<T>& operator+=(const Int3<T>& rhs) noexcept;
        NOA_HD constexpr Int3<T>& operator-=(const Int3<T>& rhs) noexcept;
        NOA_HD constexpr Int3<T>& operator*=(const Int3<T>& rhs) noexcept;
        NOA_HD constexpr Int3<T>& operator/=(const Int3<T>& rhs) noexcept;

        NOA_HD constexpr Int3<T>& operator+=(T rhs) noexcept;
        NOA_HD constexpr Int3<T>& operator-=(T rhs) noexcept;
        NOA_HD constexpr Int3<T>& operator*=(T rhs) noexcept;
        NOA_HD constexpr Int3<T>& operator/=(T rhs) noexcept;
    };

    using int3_t = Int3<int>;
    using uint3_t = Int3<uint>;
    using long3_t = Int3<long long>;
    using ulong3_t = Int3<unsigned long long>;

    template<typename T>
    [[nodiscard]] NOA_IH std::string toString(const Int3<T>& v) { return v.toString(); }

    /* --- Binary Arithmetic Operators --- */

    template<typename T>
    NOA_FHD constexpr Int3<T> operator+(Int3<T> lhs, Int3<T> rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator+(T lhs, Int3<T> rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y, lhs + rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator+(Int3<T> lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs, lhs.z + rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> operator-(Int3<T> lhs, Int3<T> rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator-(T lhs, Int3<T> rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y, lhs - rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator-(Int3<T> lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs, lhs.z - rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> operator*(Int3<T> lhs, Int3<T> rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator*(T lhs, Int3<T> rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y, lhs * rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator*(Int3<T> lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs, lhs.z * rhs};
    }

    template<typename T>
    NOA_FHD constexpr Int3<T> operator/(Int3<T> lhs, Int3<T> rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y, lhs.z / rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator/(T lhs, Int3<T> rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y, lhs / rhs.z};
    }
    template<typename T>
    NOA_FHD constexpr Int3<T> operator/(Int3<T> lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs, lhs.z / rhs};
    }

    /* --- Binary Arithmetic Assignment Operators --- */

    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator+=(const Int3<T>& rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator+=(T rhs) noexcept {
        *this = *this + rhs;
        return *this;
    }

    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator-=(const Int3<T>& rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator-=(T rhs) noexcept {
        *this = *this - rhs;
        return *this;
    }

    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator*=(const Int3<T>& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator*=(T rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator/=(const Int3<T>& rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }
    template<typename T>
    NOA_FHD constexpr Int3<T>& Int3<T>::operator/=(T rhs) noexcept {
        *this = *this / rhs;
        return *this;
    }

    /* --- Comparison Operators --- */

    template<typename T>
    NOA_FHD constexpr bool operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return lhs.x > rhs.x && lhs.y > rhs.y && lhs.z > rhs.z;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>(const Int3<T>& lhs, T rhs) noexcept {
        return lhs.x > rhs && lhs.y > rhs && lhs.z > rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>(T lhs, const Int3<T>& rhs) noexcept {
        return lhs > rhs.x && lhs > rhs.y && lhs > rhs.z;
    }

    template<typename T>
    NOA_FHD constexpr bool operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return lhs.x < rhs.x && lhs.y < rhs.y && lhs.z < rhs.z;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<(const Int3<T>& lhs, T rhs) noexcept {
        return lhs.x < rhs && lhs.y < rhs && lhs.z < rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<(T lhs, const Int3<T>& rhs) noexcept {
        return lhs < rhs.x && lhs < rhs.y && lhs < rhs.z;
    }

    template<typename T>
    NOA_FHD constexpr bool operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return lhs.x >= rhs.x && lhs.y >= rhs.y && lhs.z >= rhs.z;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>=(const Int3<T>& lhs, T rhs) noexcept {
        return lhs.x >= rhs && lhs.y >= rhs && lhs.z >= rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator>=(T lhs, const Int3<T>& rhs) noexcept {
        return lhs >= rhs.x && lhs >= rhs.y && lhs >= rhs.z;
    }

    template<typename T>
    NOA_FHD constexpr bool operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return lhs.x <= rhs.x && lhs.y <= rhs.y && lhs.z <= rhs.z;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<=(const Int3<T>& lhs, T rhs) noexcept {
        return lhs.x <= rhs && lhs.y <= rhs && lhs.z <= rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator<=(T lhs, const Int3<T>& rhs) noexcept {
        return lhs <= rhs.x && lhs <= rhs.y && lhs <= rhs.z;
    }

    template<typename T>
    NOA_FHD constexpr bool operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
    }
    template<typename T>
    NOA_FHD constexpr bool operator==(const Int3<T>& lhs, T rhs) noexcept {
        return lhs.x == rhs && lhs.y == rhs && lhs.z == rhs;
    }
    template<typename T>
    NOA_FHD constexpr bool operator==(T lhs, const Int3<T>& rhs) noexcept {
        return lhs == rhs.x && lhs == rhs.y && lhs == rhs.z;
    }

    template<typename T>
    NOA_FHD constexpr bool operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename T>
    NOA_FHD constexpr bool operator!=(const Int3<T>& lhs, T rhs) noexcept {
        return !(lhs == rhs);
    }
    template<typename T>
    NOA_FHD constexpr bool operator!=(T lhs, const Int3<T>& rhs) noexcept {
        return !(lhs == rhs);
    }
}

namespace Noa::Math {
    template<class T>
    [[nodiscard]] NOA_FHD constexpr T sum(const Int3<T>& v) noexcept {
        return v.x + v.y + v.z;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr T prod(const Int3<T>& v) noexcept {
        return v.x * v.y * v.z;
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> min(Int3<T> lhs, Int3<T> rhs) {
        return {min(lhs.x, rhs.x), min(lhs.y, rhs.y), min(lhs.z, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> min(Int3<T> lhs, T rhs) {
        return {min(lhs.x, rhs), min(lhs.y, rhs), min(lhs.z, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> min(T lhs, Int3<T> rhs) {
        return {min(lhs, rhs.x), min(lhs, rhs.y), min(lhs, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> max(Int3<T> lhs, Int3<T> rhs) {
        return {max(lhs.x, rhs.x), max(lhs.y, rhs.y), max(lhs.z, rhs.z)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> max(Int3<T> lhs, T rhs) {
        return {max(lhs.x, rhs), max(lhs.y, rhs), max(lhs.z, rhs)};
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> max(T lhs, Int3<T> rhs) {
        return {max(lhs, rhs.x), max(lhs, rhs.y), max(lhs, rhs.z)};
    }
}

namespace Noa {
    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t getElements(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y) * static_cast<size_t>(v.z);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t getElementsSlice(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr size_t getElementsFFT(const Int3<T>& v) noexcept {
        return static_cast<size_t>(v.x / 2 + 1) * static_cast<size_t>(v.y) * static_cast<size_t>(v.z);
    }

    template<class T>
    [[nodiscard]] NOA_FHD constexpr Int3<T> getShapeSlice(const Int3<T>& v) noexcept {
        return {v.x, v.y, 1};
    }
}

namespace Noa::Traits {
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

template<typename T>
std::ostream& operator<<(std::ostream& os, const Noa::Int3<T>& int3) {
    os << int3.toString();
    return os;
}
