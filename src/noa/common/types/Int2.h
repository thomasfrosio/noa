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
        static constexpr size_t COUNT = 2;
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

    template<typename T> NOA_FHD constexpr T elements(const Int2<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr T elementsFFT(const Int2<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int2<T> shapeFFT(const Int2<T>& v) noexcept;

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
    using size2_t = Int2<size_t>;

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

#define NOA_INCLUDE_INT2_
#include "noa/common/types/details/Int2.inl"
#undef NOA_INCLUDE_INT2_
