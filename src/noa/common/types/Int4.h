/// \file noa/common/types/Int4.h
/// \author Thomas - ffyr2w
/// \date 10 Dec 2020
/// Vector containing 4 integers.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Bool4.h"
#include "noa/common/utils/Sort.h"

namespace noa {
    template<typename>
    class Float4;

    template<typename T>
    class alignas(sizeof(T) * 4 >= 16 ? 16 : sizeof(T) * 4) Int4 {
    public:
        using value_type = T;

    public: // Default Constructors
        constexpr Int4() noexcept = default;
        constexpr Int4(const Int4&) noexcept = default;
        constexpr Int4(Int4&&) noexcept = default;

    public: // Conversion constructors
        template<class X, class Y, class Z, class W>
        NOA_HD constexpr Int4(X a0, Y a1, Z a2, W a3) noexcept
                : m_data{static_cast<T>(a0), static_cast<T>(a1), static_cast<T>(a2), static_cast<T>(a3)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int4(U v) noexcept
                : m_data{static_cast<T>(v), static_cast<T>(v), static_cast<T>(v), static_cast<T>(v)} {}

        NOA_HD constexpr explicit Int4(Bool4 v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Int4<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {}

        template<typename U>
        NOA_HD constexpr explicit Int4(Float4<U> v) noexcept
                : m_data{static_cast<T>(v[0]), static_cast<T>(v[1]), static_cast<T>(v[2]), static_cast<T>(v[3])} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Int4(const U* ptr) noexcept
                : m_data{static_cast<T>(ptr[0]), static_cast<T>(ptr[1]),
                         static_cast<T>(ptr[2]), static_cast<T>(ptr[3])} {}

    public: // Assignment operators
        constexpr Int4& operator=(const Int4& v) noexcept = default;
        constexpr Int4& operator=(Int4&& v) noexcept = default;

        NOA_HD constexpr Int4& operator=(T v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            m_data[2] = v;
            m_data[3] = v;
            return *this;
        }

        NOA_HD constexpr Int4& operator+=(Int4 rhs) noexcept {
            m_data[0] += rhs[0];
            m_data[1] += rhs[1];
            m_data[2] += rhs[2];
            m_data[3] += rhs[3];
            return *this;
        }

        NOA_HD constexpr Int4& operator-=(Int4 rhs) noexcept {
            m_data[0] -= rhs[0];
            m_data[1] -= rhs[1];
            m_data[2] -= rhs[2];
            m_data[3] -= rhs[3];
            return *this;
        }

        NOA_HD constexpr Int4& operator*=(Int4 rhs) noexcept {
            m_data[0] *= rhs[0];
            m_data[1] *= rhs[1];
            m_data[2] *= rhs[2];
            m_data[3] *= rhs[3];
            return *this;
        }

        NOA_HD constexpr Int4& operator/=(Int4 rhs) noexcept {
            m_data[0] /= rhs[0];
            m_data[1] /= rhs[1];
            m_data[2] /= rhs[2];
            m_data[3] /= rhs[3];
            return *this;
        }

        NOA_HD constexpr Int4& operator+=(T rhs) noexcept {
            m_data[0] += rhs;
            m_data[1] += rhs;
            m_data[2] += rhs;
            m_data[3] += rhs;
            return *this;
        }

        NOA_HD constexpr Int4& operator-=(T rhs) noexcept {
            m_data[0] -= rhs;
            m_data[1] -= rhs;
            m_data[2] -= rhs;
            m_data[3] -= rhs;
            return *this;
        }

        NOA_HD constexpr Int4& operator*=(T rhs) noexcept {
            m_data[0] *= rhs;
            m_data[1] *= rhs;
            m_data[2] *= rhs;
            m_data[3] *= rhs;
            return *this;
        }

        NOA_HD constexpr Int4& operator/=(T rhs) noexcept {
            m_data[0] /= rhs;
            m_data[1] /= rhs;
            m_data[2] /= rhs;
            m_data[3] /= rhs;
            return *this;
        }

    public: // Non-member functions
        // -- Unary operators --
        [[nodiscard]] friend NOA_HD constexpr Int4 operator+(Int4 v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator-(Int4 v) noexcept {
            return {-v[0], -v[1], -v[2], -v[3]};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Int4 operator+(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator+(T lhs, Int4 rhs) noexcept {
            return {lhs + rhs[0], lhs + rhs[1], lhs + rhs[2], lhs + rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator+(Int4 lhs, T rhs) noexcept {
            return {lhs[0] + rhs, lhs[1] + rhs, lhs[2] + rhs, lhs[3] + rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator-(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator-(T lhs, Int4 rhs) noexcept {
            return {lhs - rhs[0], lhs - rhs[1], lhs - rhs[2], lhs - rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator-(Int4 lhs, T rhs) noexcept {
            return {lhs[0] - rhs, lhs[1] - rhs, lhs[2] - rhs, lhs[3] - rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator*(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2], lhs[3] * rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator*(T lhs, Int4 rhs) noexcept {
            return {lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator*(Int4 lhs, T rhs) noexcept {
            return {lhs[0] * rhs, lhs[1] * rhs, lhs[2] * rhs, lhs[3] * rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator/(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] / rhs[0], lhs[1] / rhs[1], lhs[2] / rhs[2], lhs[3] / rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator/(T lhs, Int4 rhs) noexcept {
            return {lhs / rhs[0], lhs / rhs[1], lhs / rhs[2], lhs / rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator/(Int4 lhs, T rhs) noexcept {
            return {lhs[0] / rhs, lhs[1] / rhs, lhs[2] / rhs, lhs[3] / rhs};
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] > rhs[0], lhs[1] > rhs[1], lhs[2] > rhs[2], lhs[3] > rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>(Int4 lhs, T rhs) noexcept {
            return {lhs[0] > rhs, lhs[1] > rhs, lhs[2] > rhs, lhs[3] > rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>(T lhs, Int4 rhs) noexcept {
            return {lhs > rhs[0], lhs > rhs[1], lhs > rhs[2], lhs > rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] < rhs[0], lhs[1] < rhs[1], lhs[2] < rhs[2], lhs[3] < rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<(Int4 lhs, T rhs) noexcept {
            return {lhs[0] < rhs, lhs[1] < rhs, lhs[2] < rhs, lhs[3] < rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<(T lhs, Int4 rhs) noexcept {
            return {lhs < rhs[0], lhs < rhs[1], lhs < rhs[2], lhs < rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>=(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1], lhs[2] >= rhs[2], lhs[3] >= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>=(Int4 lhs, T rhs) noexcept {
            return {lhs[0] >= rhs, lhs[1] >= rhs, lhs[2] >= rhs, lhs[3] >= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator>=(T lhs, Int4 rhs) noexcept {
            return {lhs >= rhs[0], lhs >= rhs[1], lhs >= rhs[2], lhs >= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<=(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1], lhs[2] <= rhs[2], lhs[3] <= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<=(Int4 lhs, T rhs) noexcept {
            return {lhs[0] <= rhs, lhs[1] <= rhs, lhs[2] <= rhs, lhs[3] <= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator<=(T lhs, Int4 rhs) noexcept {
            return {lhs <= rhs[0], lhs <= rhs[1], lhs <= rhs[2], lhs <= rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator==(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2], lhs[3] == rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator==(Int4 lhs, T rhs) noexcept {
            return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs, lhs[3] == rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator==(T lhs, Int4 rhs) noexcept {
            return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2], lhs == rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator!=(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2], lhs[3] != rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator!=(Int4 lhs, T rhs) noexcept {
            return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs, lhs[3] != rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool4 operator!=(T lhs, Int4 rhs) noexcept {
            return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2], lhs != rhs[3]};
        }

        // -- Other Operators --

        [[nodiscard]] friend NOA_HD constexpr Int4 operator%(Int4 lhs, Int4 rhs) noexcept {
            return {lhs[0] % rhs[0], lhs[1] % rhs[1], lhs[2] % rhs[2], lhs[3] % rhs[3]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator%(Int4 lhs, T rhs) noexcept {
            return {lhs[0] % rhs, lhs[1] % rhs, lhs[2] % rhs, lhs[3] % rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int4 operator%(T lhs, Int4 rhs) noexcept {
            return {lhs % rhs[0], lhs % rhs[1], lhs % rhs[2], lhs % rhs[3]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 4;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr T& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const T& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const T* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr T* get() noexcept { return m_data; }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const T* get(I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr T* get(I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

    public:
        [[nodiscard]] NOA_HD constexpr Int4 flip() const noexcept {
            return {m_data[3], m_data[2], m_data[1], m_data[0]};
        }

        /// Returns the logical number of dimensions assuming *this is a shape in the BDHW convention.
        /// Note that both row and column vectors are considered to be 1D.
        [[nodiscard]] NOA_HD constexpr T ndim() const noexcept {
            NOA_ASSERT(all(*this >= T{1}));
            return m_data[0] > 1 ? 4 :
                   m_data[1] > 1 ? 3 :
                   m_data[2] > 1 && m_data[3] > 1 ? 2 : 1;
        }

        /// Returns the strides, in elements, assuming *this is a shape in the BDHW convention.
        /// If the Height and Width dimensions are empty, 'C' and 'F' returns the same strides.
        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr Int4 strides() const noexcept {
            if constexpr (ORDER == 'C' || ORDER == 'c') {
                return {m_data[3] * m_data[2] * m_data[1],
                        m_data[3] * m_data[2],
                        m_data[3],
                        1};
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                return {m_data[3] * m_data[2] * m_data[1],
                        m_data[3] * m_data[2],
                        1,
                        m_data[2]};
            } else {
                static_assert(traits::always_false_v<T>);
            }
        }

        /// Returns the DHW pitches (i.e. physical shape), in elements,
        /// assuming *this are strides in the BDHW convention.
        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr Int3<T> pitches() const noexcept {
            NOA_ASSERT(all(*this != 0) && "Cannot recover pitches from broadcast strides");
            if constexpr (ORDER == 'C' || ORDER == 'c') {
                return {m_data[0] / m_data[1],
                        m_data[1] / m_data[2],
                        m_data[2]};
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                return {m_data[0] / m_data[1],
                        m_data[3],
                        m_data[1] / m_data[3]};
            } else {
                static_assert(traits::always_false_v<T>);
            }
        }

        /// Returns the number of elements in an array with *this as its shape.
        [[nodiscard]] NOA_HD constexpr T elements() const noexcept {
            return m_data[0] * m_data[1] * m_data[2] * m_data[3];
        }

        /// Returns the shape of the non-redundant FFT, in elements,
        /// assuming *this is the logical shape in the BDHW convention.
        [[nodiscard]] NOA_HD constexpr Int4 fft() const noexcept {
            return {m_data[0], m_data[1], m_data[2], m_data[3] / 2 + 1};
        }

    public:
        static_assert(traits::is_int_v<T> && !traits::is_bool_v<T>);
        T m_data[4]{};
    };

    namespace math {
        template<typename T>
        [[nodiscard]] NOA_FHD constexpr T sum(Int4<T> v) noexcept {
            return v[0] + v[1] + v[2] + v[3];
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr T prod(Int4<T> v) noexcept {
            return v[0] * v[1] * v[2] * v[3];
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> abs(Int4<T> v) noexcept {
            return {abs(v[0]), abs(v[1]), abs(v[2]), abs(v[3])};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr T min(Int4<T> v) noexcept {
            return min(min(v[0], v[1]), min(v[2], v[3]));
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> min(Int4<T> lhs, Int4<T> rhs) noexcept {
            return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1]), min(lhs[2], rhs[2]), min(lhs[3], rhs[3])};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> min(Int4<T> lhs, T rhs) noexcept {
            return {min(lhs[0], rhs), min(lhs[1], rhs), min(lhs[2], rhs), min(lhs[3], rhs)};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> min(T lhs, Int4<T> rhs) noexcept {
            return {min(lhs, rhs[0]), min(lhs, rhs[1]), min(lhs, rhs[2]), min(lhs, rhs[3])};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr T max(Int4<T> v) noexcept {
            return max(max(v[0], v[1]), max(v[2], v[3]));
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> max(Int4<T> lhs, Int4<T> rhs) noexcept {
            return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1]), max(lhs[2], rhs[2]), max(lhs[3], rhs[3])};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> max(Int4<T> lhs, T rhs) noexcept {
            return {max(lhs[0], rhs), max(lhs[1], rhs), max(lhs[2], rhs), max(lhs[3], rhs)};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> max(T lhs, Int4<T> rhs) noexcept {
            return {max(lhs, rhs[0]), max(lhs, rhs[1]), max(lhs, rhs[2]), max(lhs, rhs[3])};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> clamp(Int4<T> lhs, Int4<T> low, Int4<T> high) noexcept {
            return {clamp(lhs[0], low[0], high[0]), clamp(lhs[1], low[1], high[1]),
                    clamp(lhs[2], low[2], high[2]), clamp(lhs[3], low[3], high[3])};
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> clamp(Int4<T> lhs, T low, T high) noexcept {
            return {clamp(lhs[0], low, high), clamp(lhs[1], low, high),
                    clamp(lhs[2], low, high), clamp(lhs[3], low, high)};
        }

        template<typename T, typename U>
        [[nodiscard]] NOA_FHD constexpr Int4<T> sort(Int4<T> v, U&& comp) noexcept {
            smallStableSort<4>(v.get(), std::forward<U>(comp));
            return v;
        }

        template<typename T>
        [[nodiscard]] NOA_FHD constexpr Int4<T> sort(Int4<T> v) noexcept {
            return sort(v, [](const T& a, const T& b) { return a < b; });
        }
    }

    namespace traits {
        template<typename>
        struct p_is_int4 : std::false_type {};
        template<typename T>
        struct p_is_int4<Int4<T>> : std::true_type {};
        template<typename T> using is_int4 = std::bool_constant<p_is_int4<remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

        template<typename>
        struct p_is_uint4 : std::false_type {};
        template<typename T>
        struct p_is_uint4<Int4<T>> : std::bool_constant<is_uint_v<T>> {};
        template<typename T> using is_uint4 = std::bool_constant<p_is_uint4<remove_ref_cv_t<T>>::value>;
        template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

        template<typename T>
        struct proclaim_is_intX<Int4<T>> : std::true_type {};
        template<typename T>
        struct proclaim_is_uintX<Int4<T>> : std::bool_constant<is_uint_v<T>> {};
    }

    using int4_t = Int4<int>;
    using uint4_t = Int4<uint>;
    using long4_t = Int4<int64_t>;
    using ulong4_t = Int4<uint64_t>;
    using size4_t = Int4<size_t>;

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 4> toArray(Int4<T> v) noexcept {
        return {v[0], v[1], v[2], v[3]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<int4_t>() { return "int4"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<uint4_t>() { return "uint4"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<long4_t>() { return "long4"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<ulong4_t>() { return "ulong4"; }

    template<typename T>
    NOA_IH std::ostream& operator<<(std::ostream& os, Int4<T> v) {
        os << string::format("({},{},{},{})", v[0], v[1], v[2], v[3]);
        return os;
    }
}

namespace fmt {
    template<typename T>
    struct formatter<noa::Int4<T>> : formatter<T> {
        template<typename FormatContext>
        auto format(const noa::Int4<T>& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[1], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[2], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<T>::format(vec[3], ctx);
            *out = ')';
            return out;
        }
    };
}
