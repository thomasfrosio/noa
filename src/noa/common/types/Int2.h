/// \file noa/common/types/Int2.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 2 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool2.h"

namespace noa {
    template<typename>
    class Float2;

    template<typename T>
    class alignas(sizeof(T) * 2) Int2 {
    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        typedef T value_type;
        T x{}, y{};

    public: // Component accesses
        NOA_HD static constexpr size_t elements() noexcept { return 2; }
        NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Int2() noexcept = default;
        template<typename X, typename Y> NOA_HD constexpr Int2(X xi, Y yi) noexcept;
        template<typename U> NOA_HD constexpr explicit Int2(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int2(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int2(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int2(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Int2<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator=(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator=(const Float2<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Int2<T>& operator+=(const Int2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator-=(const Int2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator*=(const Int2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator/=(const Int2<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Int2<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int2<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Int2<T> operator+(const Int2<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator-(const Int2<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Int2<T> operator+(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator+(T lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator+(const Int2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int2<T> operator-(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator-(T lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator-(const Int2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int2<T> operator*(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator*(T lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator*(const Int2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int2<T> operator/(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator/(T lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> operator/(const Int2<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool2 operator>(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>(const Int2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>(T lhs, const Int2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator<(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<(const Int2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<(T lhs, const Int2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator>=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>=(const Int2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>=(T lhs, const Int2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator<=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<=(const Int2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<=(T lhs, const Int2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator==(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator==(const Int2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator==(T lhs, const Int2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator!=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator!=(const Int2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator!=(T lhs, const Int2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr size_t getElements(const Int2<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr size_t getElementsFFT(const Int2<T>& v) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr T sum(const Int2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Int2<T>& v) noexcept;

        template<typename T> NOA_FHD constexpr T min(const Int2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int2<T> min(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int2<T> min(const Int2<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int2<T> min(T lhs, const Int2<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Int2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int2<T> max(const Int2<T>& lhs, const Int2<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int2<T> max(const Int2<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int2<T> max(T lhs, const Int2<T>& rhs) noexcept;
    }

    using int2_t = Int2<int>;
    using uint2_t = Int2<uint>;
    using long2_t = Int2<int64_t>;
    using ulong2_t = Int2<uint64_t>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(const Int2<T>& v) noexcept {
        return {v.x, v.y};
    }

    template<> NOA_IH std::string string::typeName<int2_t>() { return "int2"; }
    template<> NOA_IH std::string string::typeName<uint2_t>() { return "uint2"; }
    template<> NOA_IH std::string string::typeName<long2_t>() { return "long2"; }
    template<> NOA_IH std::string string::typeName<ulong2_t>() { return "ulong2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Int2<T>& v) {
        os << string::format("({},{})", v.x, v.y);
        return os;
    }
}

namespace noa {
    // -- Component accesses --

    template<typename T>
    constexpr T& Int2<T>::operator[](size_t i) {
        NOA_ASSERT(i < this->elements());
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    template<typename T>
    constexpr const T& Int2<T>::operator[](size_t i) const {
        NOA_ASSERT(i < this->elements());
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    // -- (Conversion) Constructors --

    template<typename T>
    template<typename X, typename Y>
    constexpr Int2<T>::Int2(X xi, Y yi) noexcept
            : x(static_cast<T>(xi)),
              y(static_cast<T>(yi)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(U v) noexcept
            : x(static_cast<T>(v)),
              y(static_cast<T>(v)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(const Int2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(const Float2<U>& v) noexcept
            : x(static_cast<T>(v.x)),
              y(static_cast<T>(v.y)) {}

    template<typename T>
    template<typename U>
    constexpr Int2<T>::Int2(U* ptr)
            : x(static_cast<T>(ptr[0])),
              y(static_cast<T>(ptr[1])) {}

    // -- Assignment operators --

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(U v) noexcept {
        this->x = static_cast<T>(v);
        this->y = static_cast<T>(v);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(U* ptr) noexcept {
        this->x = static_cast<T>(ptr[0]);
        this->y = static_cast<T>(ptr[1]);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(const Int2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator=(const Float2<U>& v) noexcept {
        this->x = static_cast<T>(v.x);
        this->y = static_cast<T>(v.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator+=(const Int2<U>& rhs) noexcept {
        this->x += static_cast<T>(rhs.x);
        this->y += static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator+=(U rhs) noexcept {
        this->x += static_cast<T>(rhs);
        this->y += static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator-=(const Int2<U>& rhs) noexcept {
        this->x -= static_cast<T>(rhs.x);
        this->y -= static_cast<T>(rhs.y);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator-=(U rhs) noexcept {
        this->x -= static_cast<T>(rhs);
        this->y -= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator*=(const Int2<U>& rhs) noexcept {
        this->x *= static_cast<T>(rhs.x);
        this->y *= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator*=(U rhs) noexcept {
        this->x *= static_cast<T>(rhs);
        this->y *= static_cast<T>(rhs);
        return *this;
    }

    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator/=(const Int2<U>& rhs) noexcept {
        this->x /= static_cast<T>(rhs.x);
        this->y /= static_cast<T>(rhs.y);
        return *this;
    }
    template<typename T>
    template<typename U>
    constexpr Int2<T>& Int2<T>::operator/=(U rhs) noexcept {
        this->x /= static_cast<T>(rhs);
        this->y /= static_cast<T>(rhs);
        return *this;
    }

    // -- Unary operators --

    template<typename T> constexpr Int2<T> operator+(const Int2<T>& v) noexcept {
        return v;
    }

    template<typename T> constexpr Int2<T> operator-(const Int2<T>& v) noexcept {
        return {-v.x, -v.y};
    }

    // -- Binary Arithmetic Operators --

    template<typename T>
    constexpr Int2<T> operator+(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x + rhs.x, lhs.y + rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator+(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs + rhs.x, lhs + rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator+(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x + rhs, lhs.y + rhs};
    }

    template<typename T>
    constexpr Int2<T> operator-(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x - rhs.x, lhs.y - rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator-(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs - rhs.x, lhs - rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator-(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x - rhs, lhs.y - rhs};
    }

    template<typename T>
    constexpr Int2<T> operator*(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x * rhs.x, lhs.y * rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator*(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs * rhs.x, lhs * rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator*(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x * rhs, lhs.y * rhs};
    }

    template<typename T>
    constexpr Int2<T> operator/(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x / rhs.x, lhs.y / rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator/(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs / rhs.x, lhs / rhs.y};
    }
    template<typename T>
    constexpr Int2<T> operator/(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x / rhs, lhs.y / rhs};
    }

    // -- Comparison Operators --

    template<typename T>
    constexpr Bool2 operator>(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x > rhs.x, lhs.y > rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator>(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x > rhs, lhs.y > rhs};
    }
    template<typename T>
    constexpr Bool2 operator>(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs > rhs.x, lhs > rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator<(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x < rhs.x, lhs.y < rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator<(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x < rhs, lhs.y < rhs};
    }
    template<typename T>
    constexpr Bool2 operator<(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs < rhs.x, lhs < rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator>=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x >= rhs.x, lhs.y >= rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator>=(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x >= rhs, lhs.y >= rhs};
    }
    template<typename T>
    constexpr Bool2 operator>=(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs >= rhs.x, lhs >= rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator<=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x <= rhs.x, lhs.y <= rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator<=(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x <= rhs, lhs.y <= rhs};
    }
    template<typename T>
    constexpr Bool2 operator<=(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs <= rhs.x, lhs <= rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator==(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator==(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }
    template<typename T>
    constexpr Bool2 operator==(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    template<typename T>
    constexpr Bool2 operator!=(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }
    template<typename T>
    constexpr Bool2 operator!=(const Int2<T>& lhs, T rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }
    template<typename T>
    constexpr Bool2 operator!=(T lhs, const Int2<T>& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    template<typename T>
    constexpr size_t getElements(const Int2<T>& v) noexcept {
        return static_cast<size_t>(v.x) * static_cast<size_t>(v.y);
    }

    template<typename T>
    constexpr size_t getElementsFFT(const Int2<T>& v) noexcept {
        return static_cast<size_t>(v.x / 2 + 1) * static_cast<size_t>(v.y);
    }

    namespace math {
        template<typename T>
        constexpr T sum(const Int2<T>& v) noexcept {
            return v.x + v.y;
        }

        template<typename T>
        constexpr T prod(const Int2<T>& v) noexcept {
            return v.x * v.y;
        }

        template<typename T>
        constexpr T min(const Int2<T>& v) noexcept {
            return min(v.x, v.y);
        }

        template<typename T>
        constexpr Int2<T> min(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
            return {min(lhs.x, rhs.x), min(lhs.y, rhs.y)};
        }

        template<typename T>
        constexpr Int2<T> min(const Int2<T>& lhs, T rhs) noexcept {
            return {min(lhs.x, rhs), min(lhs.y, rhs)};
        }

        template<typename T>
        constexpr Int2<T> min(T lhs, const Int2<T>& rhs) noexcept {
            return {min(lhs, rhs.x), min(lhs, rhs.y)};
        }

        template<typename T>
        constexpr T max(const Int2<T>& v) noexcept {
            return max(v.x, v.y);
        }

        template<typename T>
        constexpr Int2<T> max(const Int2<T>& lhs, const Int2<T>& rhs) noexcept {
            return {max(lhs.x, rhs.x), max(lhs.y, rhs.y)};
        }

        template<typename T>
        constexpr Int2<T> max(const Int2<T>& lhs, T rhs) noexcept {
            return {max(lhs.x, rhs), max(lhs.y, rhs)};
        }

        template<typename T>
        constexpr Int2<T> max(T lhs, const Int2<T>& rhs) noexcept {
            return {max(lhs, rhs.x), max(lhs, rhs.y)};
        }
    }
}

namespace noa::traits {
    template<typename> struct p_is_int2 : std::false_type {};
    template<typename T> struct p_is_int2<noa::Int2<T>> : std::true_type {};
    template<typename T> using is_int2 = std::bool_constant<p_is_int2<noa::traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int2_v = is_int2<T>::value;

    template<typename> struct p_is_uint2 : std::false_type {};
    template<typename T> struct p_is_uint2<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    template<typename T> using is_uint2 = std::bool_constant<p_is_uint2<noa::traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint2_v = is_uint2<T>::value;

    template<typename T> struct proclaim_is_intX<noa::Int2<T>> : std::true_type {};
    template<typename T> struct proclaim_is_uintX<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
}
