/// \file noa/common/types/Int4.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 4 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool4.h"

namespace noa {
    template<typename>
    class Float4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Int4 {
    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        typedef T value_type;
        T x{}, y{}, z{}, w{};

    public: // Component accesses
        static constexpr size_t COUNT = 4;
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Int4() noexcept = default;
        template<class X, class Y, class Z, class W> NOA_HD constexpr Int4(X xi, Y yi, Z zi, W wi) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(const Int4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(const Float4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int4(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Int4<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator=(const Int4<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator=(const Float4<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Int4<T>& operator+=(const Int4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator-=(const Int4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator*=(const Int4<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator/=(const Int4<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Int4<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int4<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Int4<T> operator+(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator-(const Int4<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Int4<T> operator+(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator+(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator+(const Int4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int4<T> operator-(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator-(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator-(const Int4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int4<T> operator*(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator*(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator*(const Int4<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int4<T> operator/(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator/(T lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> operator/(const Int4<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool4 operator>(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator<(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator>=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>=(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator>=(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator<=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<=(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator<=(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator==(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator==(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator==(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool4 operator!=(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator!=(const Int4<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool4 operator!=(T lhs, const Int4<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr T elements(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr T elementsSlice(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr T elementsFFT(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> shapeFFT(const Int4<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int4<T> slice(const Int4<T>& v) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr T sum(const Int4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Int4<T>& v) noexcept;

        template<typename T> NOA_FHD constexpr T min(const Int4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> min(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> min(const Int4<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> min(T lhs, const Int4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Int4<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> max(const Int4<T>& lhs, const Int4<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> max(const Int4<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int4<T> max(T lhs, const Int4<T>& rhs) noexcept;
    }

    namespace traits {
        template<typename> struct p_is_int4 : std::false_type {};
        template<typename T> struct p_is_int4<noa::Int4<T>> : std::true_type {};
        template<typename T> using is_int4 = std::bool_constant<p_is_int4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

        template<typename> struct p_is_uint4 : std::false_type {};
        template<typename T> struct p_is_uint4<noa::Int4<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

        template<typename T> struct proclaim_is_intX<noa::Int4<T>> : std::true_type {};
        template<typename T> struct proclaim_is_uintX<noa::Int4<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }

    using int4_t = Int4<int>;
    using uint4_t = Int4<uint>;
    using long4_t = Int4<int64_t>;
    using ulong4_t = Int4<uint64_t>;
    using size4_t = Int4<size_t>;

    template<typename T>
    NOA_IH constexpr std::array<T, 4> toArray(const Int4<T>& v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<> NOA_IH std::string string::typeName<int4_t>() { return "int4"; }
    template<> NOA_IH std::string string::typeName<uint4_t>() { return "uint4"; }
    template<> NOA_IH std::string string::typeName<long4_t>() { return "long4"; }
    template<> NOA_IH std::string string::typeName<ulong4_t>() { return "ulong4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Int4<T>& v) {
        os << string::format("({},{},{},{})", v.x, v.y, v.z, v.w);
        return os;
    }
}

#define NOA_INCLUDE_INT4_
#include "noa/common/types/details/Int4.inl"
#undef NOA_INCLUDE_INT4_
