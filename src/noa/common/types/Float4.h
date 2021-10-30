/// \file noa/common/types/Float4.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 4 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool4.h"

namespace noa {
    template<typename>
    class Int4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Float4 {
    public:
        static_assert(noa::traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{}, z{}, w{};

    public: // Component accesses
        static constexpr size_t COUNT = 4;
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Float4() noexcept = default;
        template<class X, class Y, class Z, class W> NOA_HD constexpr Float4(X xi, Y yi, Z zi, W wi) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(const Int4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float4(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Float4<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator=(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator=(const Int4<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Float4<T>& operator+=(const Float4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator-=(const Float4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator*=(const Float4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator/=(const Float4<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Float4<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float4<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Float4<T> operator+(const Float4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator-(const Float4<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Float4<T> operator+(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator+(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator+(const Float4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float4<T> operator-(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator-(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator-(const Float4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float4<T> operator*(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator*(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator*(const Float4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float4<T> operator/(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator/(T lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float4<T> operator/(const Float4<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool4 operator>(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator<(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator>=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>=(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>=(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator<=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<=(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<=(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator==(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator==(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator==(T lhs, const Float4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator!=(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator!=(const Float4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator!=(T lhs, const Float4<T>& rhs) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr Float4<T> floor(const Float4<T>& v);
        template<typename T> NOA_FHD constexpr Float4<T> ceil(const Float4<T>& v);
        template<typename T> NOA_FHD constexpr Float4<T> abs(const Float4<T>& v);
        template<typename T> NOA_FHD constexpr T sum(const Float4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Float4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T dot(const Float4<T>& a, const Float4<T>& b) noexcept;
        template<typename T> NOA_FHD constexpr T innerProduct(const Float4<T>& a, const Float4<T>& b) noexcept;
        template<typename T> NOA_FHD constexpr T norm(const Float4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T length(const Float4<T>& v);
        template<typename T> NOA_FHD constexpr Float4<T> normalize(const Float4<T>& v);

        template<typename T> NOA_FHD constexpr T min(const Float4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Float4<T> min(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float4<T> min(const Float4<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float4<T> min(T lhs, const Float4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Float4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Float4<T> max(const Float4<T>& lhs, const Float4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float4<T> max(const Float4<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float4<T> max(T lhs, const Float4<T>& rhs) noexcept;

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(const Float4<T>& a, const Float4<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(const Float4<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool4 isEqual(T a, const Float4<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T> struct p_is_float4 : std::false_type {};
        template<typename T> struct p_is_float4<noa::Float4<T>> : std::true_type {};
        template<typename T> using is_float4 = std::bool_constant<p_is_float4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float4_v = is_float4<T>::value;

        template<typename T> struct proclaim_is_floatX<noa::Float4<T>> : std::true_type {};
    }

    using float4_t = Float4<float>;
    using double4_t = Float4<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(const Float4<T>& v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<> NOA_IH std::string string::typeName<float4_t>() { return "float4"; }
    template<> NOA_IH std::string string::typeName<double4_t>() { return "double4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Float4<T>& v) {
        os << string::format("({:.3f},{:.3f},{:.3f},{:.3f})", v.x, v.y, v.z, v.w);
        return os;
    }
}

#define NOA_INCLUDE_FLOAT4_
#include "noa/common/types/details/Float4.inl"
#undef NOA_INCLUDE_FLOAT4_
