/// \file noa/common/types/Float3.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 3 floating-point numbers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/string/Format.h"
#include "noa/common/types/Bool3.h"

namespace noa {
    template<typename>
    class Int3;

    template<typename T>
    class Float3 {
    public:
        static_assert(noa::traits::is_float_v<T>);
        typedef T value_type;
        T x{}, y{}, z{};

    public: // Component accesses
        static constexpr size_t COUNT = 3;
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Float3() noexcept = default;
        template<typename X, typename Y, typename Z> NOA_HD constexpr Float3(X xi, Y yi, Z zi) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(const Int3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Float3(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Float3<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator=(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator=(const Int3<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Float3<T>& operator+=(const Float3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator-=(const Float3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator*=(const Float3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator/=(const Float3<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Float3<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Float3<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Float3<T> operator+(const Float3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator-(const Float3<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Float3<T> operator+(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator+(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator+(const Float3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float3<T> operator-(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator-(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator-(const Float3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float3<T> operator*(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator*(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator*(const Float3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Float3<T> operator/(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator/(T lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Float3<T> operator/(const Float3<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool3 operator>(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator<(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator>=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>=(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>=(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator<=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<=(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<=(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator==(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator==(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator==(T lhs, const Float3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator!=(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator!=(const Float3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator!=(T lhs, const Float3<T>& rhs) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr Float3<T> toRad(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr Float3<T> toDeg(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr Float3<T> floor(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr Float3<T> ceil(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr Float3<T> abs(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr T sum(const Float3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Float3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T dot(const Float3<T>& a, const Float3<T>& b) noexcept;
        template<typename T> NOA_FHD constexpr T innerProduct(const Float3<T>& a, const Float3<T>& b) noexcept;
        template<typename T> NOA_FHD constexpr T norm(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr T length(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr Float3<T> normalize(const Float3<T>& v);
        template<typename T> NOA_FHD constexpr Float3<T> cross(const Float3<T>& a, const Float3<T>& b) noexcept;

        template<typename T> NOA_FHD constexpr T min(const Float3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Float3<T> min(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float3<T> min(const Float3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float3<T> min(T lhs, const Float3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Float3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Float3<T> max(const Float3<T>& lhs, const Float3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float3<T> max(const Float3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Float3<T> max(T lhs, const Float3<T>& rhs) noexcept;

        #define NOA_ULP_ 2
        #define NOA_EPSILON_ 1e-6f

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(const Float3<T>& a, const Float3<T>& b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(const Float3<T>& a, T b, T e = NOA_EPSILON_);

        template<uint ULP = NOA_ULP_, typename T>
        NOA_FHD constexpr Bool3 isEqual(T a, const Float3<T>& b, T e = NOA_EPSILON_);

        #undef NOA_ULP_
        #undef NOA_EPSILON_
    }

    namespace traits {
        template<typename T> struct p_is_float3 : std::false_type {};
        template<typename T> struct p_is_float3<noa::Float3<T>> : std::true_type {};
        template<typename T> using is_float3 = std::bool_constant<p_is_float3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_float3_v = is_float3<T>::value;

        template<typename T> struct proclaim_is_floatX<noa::Float3<T>> : std::true_type {};
    }

    using float3_t = Float3<float>;
    using double3_t = Float3<double>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(const Float3<T>& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string string::typeName<float3_t>() { return "float3"; }
    template<> NOA_IH std::string string::typeName<double3_t>() { return "double3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Float3<T>& v) {
        os << string::format("({:.3f},{:.3f},{:.3f})", v.x, v.y, v.z);
        return os;
    }
}

#define NOA_INCLUDE_FLOAT3_
#include "noa/common/types/details/Float3.inl"
#undef NOA_INCLUDE_FLOAT3_
