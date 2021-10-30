/// \file noa/common/types/Float2.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 2 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool2.h"

namespace noa {
    template<typename>
    class Int2;

    template<typename T>
    class alignas(sizeof(T) * 2) Float2 {
    public:
        static_assert(noa::traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{};

    public: // Component accesses
        static constexpr size_t COUNT = 2;
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Float2() noexcept = default;
        template<typename X, typename Y> NOA_HD constexpr Float2(X xi, Y yi) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(const Int2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float2(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Float2<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator=(U* ptr);
        template<typename U> NOA_HD constexpr Float2<T>& operator=(const Float2<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator=(const Int2<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Float2<T>& operator+=(const Float2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator-=(const Float2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator*=(const Float2<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator/=(const Float2<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Float2<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float2<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Float2<T> operator+(const Float2<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator-(const Float2<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Float2<T> operator+(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator+(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator+(const Float2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float2<T> operator-(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator-(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator-(const Float2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float2<T> operator*(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator*(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator*(const Float2<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float2<T> operator/(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator/(T lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float2<T> operator/(const Float2<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool2 operator>(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator<(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator>=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>=(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator>=(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator<=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<=(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator<=(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator==(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator==(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator==(T lhs, const Float2<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool2 operator!=(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator!=(const Float2<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool2 operator!=(T lhs, const Float2<T>& rhs) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr Float2<T> floor(const Float2<T>& v);
        template<typename T> NOA_FHD constexpr Float2<T> ceil(const Float2<T>& v);
        template<typename T> NOA_FHD constexpr Float2<T> abs(const Float2<T>& v);
        template<typename T> NOA_FHD constexpr T sum(const Float2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Float2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T dot(const Float2<T>& a, const Float2<T>& b) noexcept;
        template<typename T> NOA_FHD constexpr T innerProduct(const Float2<T>& a, const Float2<T>& b) noexcept;
        template<typename T> NOA_FHD constexpr T norm(const Float2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T length(const Float2<T>& v);
        template<typename T> NOA_FHD constexpr Float2<T> normalize(const Float2<T>& v);

        template<typename T> NOA_FHD constexpr T min(const Float2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Float2<T> min(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float2<T> min(const Float2<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float2<T> min(T lhs, const Float2<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Float2<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Float2<T> max(const Float2<T>& lhs, const Float2<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float2<T> max(const Float2<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float2<T> max(T lhs, const Float2<T>& rhs) noexcept;

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Float2<T>& a, const Float2<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(const Float2<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool2 isEqual(T a, const Float2<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T> struct p_is_float2 : std::false_type {};
        template<typename T> struct p_is_float2<noa::Float2<T>> : std::true_type {};
        template<typename T> using is_float2 = std::bool_constant<p_is_float2<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float2_v = is_float2<T>::value;

        template<typename T> struct proclaim_is_floatX<noa::Float2<T>> : std::true_type {};
    }

    using float2_t = Float2<float>;
    using double2_t = Float2<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(const Float2<T>& v) noexcept {
        return {v.x, v.y};
    }

    template<> NOA_IH std::string string::typeName<float2_t>() { return "float2"; }
    template<> NOA_IH std::string string::typeName<double2_t>() { return "double2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Float2<T>& v) {
        os << string::format("({:.3f},{:.3f})", v.x, v.y);
        return os;
    }
}

#define NOA_INCLUDE_FLOAT2_
#include "noa/common/types/details/Float2.inl"
#undef NOA_INCLUDE_FLOAT2_
