/// \file noa/common/types/Int2.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 2 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Assert.h"
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
        typedef T value_type;

    public: // Default Constructors
        constexpr Int2() noexcept = default;
        constexpr Int2(const Int2&) noexcept = default;
        constexpr Int2(Int2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y>
        NOA_HD constexpr Int2(X x, Y y) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(y)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int2(U x) noexcept
                : m_data{static_cast<T>(x), static_cast<T>(x)} {}

        NOA_HD constexpr explicit Int2(Bool2 v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Int2(Int2<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1])} {}

        template<typename U>
        NOA_HD constexpr explicit Int2(Float2<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1])} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int2(const U* ptr) noexcept
                : m_data{static_cast<T>(ptr[0]), static_cast<T>(ptr[1])} {}

    public: // Assignment operators
        constexpr Int2& operator=(const Int2& v) noexcept = default;
        constexpr Int2& operator=(Int2&& v) noexcept = default;

        NOA_HD constexpr Int2& operator=(T v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            return *this;
        }

        NOA_HD constexpr Int2& operator+=(Int2 rhs) noexcept {
            m_data[0] += rhs[0];
            m_data[1] += rhs[1];
            return *this;
        }

        NOA_HD constexpr Int2& operator-=(Int2 rhs) noexcept {
            m_data[0] -= rhs[0];
            m_data[1] -= rhs[1];
            return *this;
        }

        NOA_HD constexpr Int2& operator*=(Int2 rhs) noexcept {
            m_data[0] *= rhs[0];
            m_data[1] *= rhs[1];
            return *this;
        }

        NOA_HD constexpr Int2& operator/=(Int2 rhs) noexcept {
            m_data[0] /= rhs[0];
            m_data[1] /= rhs[1];
            return *this;
        }

        NOA_HD constexpr Int2& operator+=(T rhs) noexcept {
            m_data[0] += rhs;
            m_data[1] += rhs;
            return *this;
        }

        NOA_HD constexpr Int2& operator-=(T rhs) noexcept {
            m_data[0] -= rhs;
            m_data[1] -= rhs;
            return *this;
        }

        NOA_HD constexpr Int2& operator*=(T rhs) noexcept {
            m_data[0] *= rhs;
            m_data[1] *= rhs;
            return *this;
        }

        NOA_HD constexpr Int2& operator/=(T rhs) noexcept {
            m_data[0] /= rhs;
            m_data[1] /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        friend NOA_HD constexpr Int2 operator+(Int2 v) noexcept {
            return v;
        }

        friend NOA_HD constexpr Int2 operator-(Int2 v) noexcept {
            return {-v[0], -v[1]};
        }

        // -- Binary Arithmetic Operators --
        friend NOA_HD constexpr Int2 operator+(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] + rhs[0], lhs[1] + rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator+(T lhs, Int2 rhs) noexcept {
            return {lhs + rhs[0], lhs + rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator+(Int2 lhs, T rhs) noexcept {
            return {lhs[0] + rhs, lhs[1] + rhs};
        }

        friend NOA_HD constexpr Int2 operator-(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] - rhs[0], lhs[1] - rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator-(T lhs, Int2 rhs) noexcept {
            return {lhs - rhs[0], lhs - rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator-(Int2 lhs, T rhs) noexcept {
            return {lhs[0] - rhs, lhs[1] - rhs};
        }

        friend NOA_HD constexpr Int2 operator*(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] * rhs[0], lhs[1] * rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator*(T lhs, Int2 rhs) noexcept {
            return {lhs * rhs[0], lhs * rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator*(Int2 lhs, T rhs) noexcept {
            return {lhs[0] * rhs, lhs[1] * rhs};
        }

        friend NOA_HD constexpr Int2 operator/(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] / rhs[0], lhs[1] / rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator/(T lhs, Int2 rhs) noexcept {
            return {lhs / rhs[0], lhs / rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator/(Int2 lhs, T rhs) noexcept {
            return {lhs[0] / rhs, lhs[1] / rhs};
        }

        // -- Comparison Operators --
        friend NOA_HD constexpr Bool2 operator>(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] > rhs[0], lhs[1] > rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator>(Int2 lhs, T rhs) noexcept {
            return {lhs[0] > rhs, lhs[1] > rhs};
        }

        friend NOA_HD constexpr Bool2 operator>(T lhs, Int2 rhs) noexcept {
            return {lhs > rhs[0], lhs > rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator<(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] < rhs[0], lhs[1] < rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator<(Int2 lhs, T rhs) noexcept {
            return {lhs[0] < rhs, lhs[1] < rhs};
        }

        friend NOA_HD constexpr Bool2 operator<(T lhs, Int2 rhs) noexcept {
            return {lhs < rhs[0], lhs < rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator>=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator>=(Int2 lhs, T rhs) noexcept {
            return {lhs[0] >= rhs, lhs[1] >= rhs};
        }

        friend NOA_HD constexpr Bool2 operator>=(T lhs, Int2 rhs) noexcept {
            return {lhs >= rhs[0], lhs >= rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator<=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator<=(Int2 lhs, T rhs) noexcept {
            return {lhs[0] <= rhs, lhs[1] <= rhs};
        }

        friend NOA_HD constexpr Bool2 operator<=(T lhs, Int2 rhs) noexcept {
            return {lhs <= rhs[0], lhs <= rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator==(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] == rhs[0], lhs[1] == rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator==(Int2 lhs, T rhs) noexcept {
            return {lhs[0] == rhs, lhs[1] == rhs};
        }

        friend NOA_HD constexpr Bool2 operator==(T lhs, Int2 rhs) noexcept {
            return {lhs == rhs[0], lhs == rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator!=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] != rhs[0], lhs[1] != rhs[1]};
        }

        friend NOA_HD constexpr Bool2 operator!=(Int2 lhs, T rhs) noexcept {
            return {lhs[0] != rhs, lhs[1] != rhs};
        }

        friend NOA_HD constexpr Bool2 operator!=(T lhs, Int2 rhs) noexcept {
            return {lhs != rhs[0], lhs != rhs[1]};
        }

        // -- Other Operators --
        friend NOA_HD constexpr Int2 operator%(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] % rhs[0], lhs[1] % rhs[1]};
        }

        friend NOA_HD constexpr Int2 operator%(Int2 lhs, T rhs) noexcept {
            return {lhs[0] % rhs, lhs[1] % rhs};
        }

        friend NOA_HD constexpr Int2 operator%(T lhs, Int2 rhs) noexcept {
            return {lhs % rhs[0], lhs % rhs[1]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 2;

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr T& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr const T& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr Int2 flip() const noexcept { return {m_data[1], m_data[0]}; }

        [[nodiscard]] NOA_HD constexpr T ndim() const noexcept {
            NOA_ASSERT(all(*this >= T{1}));
            return m_data[0] > 1 ? 2 : 1;
        }

        [[nodiscard]] NOA_HD constexpr Int2 strides() const noexcept {
            return {m_data[1], 1};
        }

        [[nodiscard]] NOA_HD constexpr T pitches() const noexcept {
            return m_data[0];
        }

        [[nodiscard]] NOA_HD constexpr T elements() const noexcept {
            return m_data[0] * m_data[1];
        }

        [[nodiscard]] NOA_HD constexpr Int2 fft() const noexcept {
            return {m_data[0], m_data[1] / 2 + 1};
        }

    private:
        static_assert(noa::traits::is_int_v<T> && !noa::traits::is_bool_v<T>);
        T m_data[2]{};
    };

    namespace math {
        template<typename T>
        NOA_FHD constexpr T sum(Int2<T> v) noexcept {
            return v[0] + v[1];
        }

        template<typename T>
        NOA_FHD constexpr T prod(Int2<T> v) noexcept {
            return v[0] * v[1];
        }

        template<typename T>
        NOA_FHD constexpr T min(Int2<T> v) noexcept {
            return min(v[0], v[1]);
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> min(Int2<T> lhs, Int2<T> rhs) noexcept {
            return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1])};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> min(Int2<T> lhs, T rhs) noexcept {
            return {min(lhs[0], rhs), min(lhs[1], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> min(T lhs, Int2<T> rhs) noexcept {
            return {min(lhs, rhs[0]), min(lhs, rhs[1])};
        }

        template<typename T>
        NOA_FHD constexpr T max(Int2<T> v) noexcept {
            return max(v[0], v[1]);
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> max(Int2<T> lhs, Int2<T> rhs) noexcept {
            return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1])};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> max(Int2<T> lhs, T rhs) noexcept {
            return {max(lhs[0], rhs), max(lhs[1], rhs)};
        }

        template<typename T>
        NOA_FHD constexpr Int2<T> max(T lhs, Int2<T> rhs) noexcept {
            return {max(lhs, rhs[0]), max(lhs, rhs[1])};
        }
    }

    using int2_t = Int2<int>;
    using uint2_t = Int2<uint>;
    using long2_t = Int2<int64_t>;
    using ulong2_t = Int2<uint64_t>;
    using size2_t = Int2<size_t>;

    namespace traits {
        template<typename>
        struct p_is_int2 : std::false_type {};
        template<typename T>
        struct p_is_int2<noa::Int2<T>> : std::true_type {};
        template<typename T> using is_int2 = std::bool_constant<p_is_int2<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int2_v = is_int2<T>::value;

        template<typename>
        struct p_is_uint2 : std::false_type {};
        template<typename T>
        struct p_is_uint2<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
        template<typename T> using is_uint2 = std::bool_constant<p_is_uint2<noa::traits::remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint2_v = is_uint2<T>::value;

        template<typename T>
        struct proclaim_is_intX<noa::Int2<T>> : std::true_type {};
        template<typename T>
        struct proclaim_is_uintX<noa::Int2<T>> : std::bool_constant<noa::traits::is_uint_v<T>> {};
    }


    template<typename T>
    NOA_IH constexpr std::array<T, 2> toArray(Int2<T> v) noexcept {
        return {v[0], v[1]};
    }

    template<>
    NOA_IH std::string string::typeName<int2_t>() { return "int2"; }
    template<>
    NOA_IH std::string string::typeName<uint2_t>() { return "uint2"; }
    template<>
    NOA_IH std::string string::typeName<long2_t>() { return "long2"; }
    template<>
    NOA_IH std::string string::typeName<ulong2_t>() { return "ulong2"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Int2<T> v) {
        os << string::format("({},{})", v[0], v[1]);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Int2<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Int2<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[1], ctx);
            *out = ')';
            return out;
        }
    };
}
