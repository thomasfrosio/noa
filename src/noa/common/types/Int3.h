/// \file noa/common/types/Int3.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 3 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool3.h"

namespace noa {
    template<typename>
    class Float3;

    template<typename T>
    class Int3 {
    public:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        typedef T value_type;
        T x{}, y{}, z{};

    public: // Component accesses
        static constexpr size_t COUNT = 3;
        NOA_HD constexpr T& operator[](size_t i);
        NOA_HD constexpr const T& operator[](size_t i) const;

    public: // (Conversion) Constructors
        constexpr Int3() noexcept = default;
        template<typename X, typename Y, typename Z> NOA_HD constexpr Int3(X xi, Y yi, Z zi) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(const Int3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(const Float3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr explicit Int3(U* ptr);
        template<typename U, typename V> NOA_HD constexpr Int3(const Int2<U>& v, V oz) noexcept;

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Int3<T>& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator=(U* ptr) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator=(const Int3<U>& v) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator=(const Float3<U>& v) noexcept;

        template<typename U> NOA_HD constexpr Int3<T>& operator+=(const Int3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator-=(const Int3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator*=(const Int3<U>& rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator/=(const Int3<U>& rhs) noexcept;

        template<typename U> NOA_HD constexpr Int3<T>& operator+=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator-=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator*=(U rhs) noexcept;
        template<typename U> NOA_HD constexpr Int3<T>& operator/=(U rhs) noexcept;
    };

    // -- Unary operators --

    template<typename T> NOA_FHD constexpr Int3<T> operator+(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator-(const Int3<T>& v) noexcept;

    // -- Binary operators --

    template<typename T> NOA_FHD constexpr Int3<T> operator+(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator+(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator+(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int3<T> operator-(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator-(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator-(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int3<T> operator*(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator*(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator*(const Int3<T>& lhs, T rhs) noexcept;

    template<typename T> NOA_FHD constexpr Int3<T> operator/(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator/(T lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> operator/(const Int3<T>& lhs, T rhs) noexcept;

    // -- Boolean operators --

    template<typename T> NOA_FHD constexpr Bool3 operator>(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator<(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator>=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator>=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator<=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator<=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator==(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator==(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator==(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr Bool3 operator!=(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator!=(const Int3<T>& lhs, T rhs) noexcept;
    template<typename T> NOA_FHD constexpr Bool3 operator!=(T lhs, const Int3<T>& rhs) noexcept;

    template<typename T> NOA_FHD constexpr T elements(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr T elementsSlice(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr T elementsFFT(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> shapeFFT(const Int3<T>& v) noexcept;
    template<typename T> NOA_FHD constexpr Int3<T> slice(const Int3<T>& v) noexcept;

    namespace math {
        template<typename T> NOA_FHD constexpr T sum(const Int3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr T prod(const Int3<T>& v) noexcept;

        template<typename T> NOA_FHD constexpr T min(const Int3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> min(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> min(const Int3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> min(T lhs, const Int3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr T max(const Int3<T>& v) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> max(const Int3<T>& lhs, const Int3<T>& rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> max(const Int3<T>& lhs, T rhs) noexcept;
        template<typename T> NOA_FHD constexpr Int3<T> max(T lhs, const Int3<T>& rhs) noexcept;
    }

    namespace traits {
        template<typename> struct p_is_int3 : std::false_type {};
        template<typename T> struct p_is_int3<noa::Int3<T>> : std::true_type {};
        template<typename T> using is_int3 = std::bool_constant<p_is_int3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

        template<typename> struct p_is_uint3 : std::false_type {};
        template<typename T> struct p_is_uint3<noa::Int3<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint3 = std::bool_constant<p_is_uint3<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

        template<typename T> struct proclaim_is_intX<noa::Int3<T>> : std::true_type {};
        template<typename T> struct proclaim_is_uintX<noa::Int3<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }

    using int3_t = Int3<int>;
    using uint3_t = Int3<uint>;
    using long3_t = Int3<int64_t>;
    using ulong3_t = Int3<uint64_t>;

    template<typename T>
    NOA_IH constexpr std::array<T, 3> toArray(const Int3<T>& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string string::typeName<int3_t>() { return "int3"; }
    template<> NOA_IH std::string string::typeName<uint3_t>() { return "uint3"; }
    template<> NOA_IH std::string string::typeName<long3_t>() { return "long3"; }
    template<> NOA_IH std::string string::typeName<ulong3_t>() { return "ulong3"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, const Int3<T>& v) {
        os << string::format("({},{},{})", v.x, v.y, v.z);
        return os;
    }
}

#define NOA_INCLUDE_INT3_
#include "noa/common/types/details/Int3.inl"
#undef NOA_INCLUDE_INT3_
