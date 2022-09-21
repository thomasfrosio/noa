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
#include "noa/common/types/ClampCast.h"
#include "noa/common/types/SafeCast.h"
#include "noa/common/utils/Sort.h"

namespace noa {
    template<typename>
    class Float2;

    template<typename T>
    class alignas(sizeof(T) * 2) Int2 {
    public:
        using value_type = T;

    public: // Default Constructors
        constexpr Int2() noexcept = default;
        constexpr Int2(const Int2&) noexcept = default;
        constexpr Int2(Int2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y>
        NOA_HD constexpr Int2(X a0, Y a1) noexcept
                : m_data{static_cast<T>(a0), static_cast<T>(a1)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
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

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
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
        [[nodiscard]] friend NOA_HD constexpr Int2 operator+(Int2 v) noexcept {
            return v;
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator-(Int2 v) noexcept {
            return {-v[0], -v[1]};
        }

        // -- Binary Arithmetic Operators --
        [[nodiscard]] friend NOA_HD constexpr Int2 operator+(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] + rhs[0], lhs[1] + rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator+(T lhs, Int2 rhs) noexcept {
            return {lhs + rhs[0], lhs + rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator+(Int2 lhs, T rhs) noexcept {
            return {lhs[0] + rhs, lhs[1] + rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator-(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] - rhs[0], lhs[1] - rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator-(T lhs, Int2 rhs) noexcept {
            return {lhs - rhs[0], lhs - rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator-(Int2 lhs, T rhs) noexcept {
            return {lhs[0] - rhs, lhs[1] - rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator*(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] * rhs[0], lhs[1] * rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator*(T lhs, Int2 rhs) noexcept {
            return {lhs * rhs[0], lhs * rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator*(Int2 lhs, T rhs) noexcept {
            return {lhs[0] * rhs, lhs[1] * rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator/(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] / rhs[0], lhs[1] / rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator/(T lhs, Int2 rhs) noexcept {
            return {lhs / rhs[0], lhs / rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator/(Int2 lhs, T rhs) noexcept {
            return {lhs[0] / rhs, lhs[1] / rhs};
        }

        // -- Comparison Operators --
        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] > rhs[0], lhs[1] > rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>(Int2 lhs, T rhs) noexcept {
            return {lhs[0] > rhs, lhs[1] > rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>(T lhs, Int2 rhs) noexcept {
            return {lhs > rhs[0], lhs > rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] < rhs[0], lhs[1] < rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<(Int2 lhs, T rhs) noexcept {
            return {lhs[0] < rhs, lhs[1] < rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<(T lhs, Int2 rhs) noexcept {
            return {lhs < rhs[0], lhs < rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] >= rhs[0], lhs[1] >= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>=(Int2 lhs, T rhs) noexcept {
            return {lhs[0] >= rhs, lhs[1] >= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator>=(T lhs, Int2 rhs) noexcept {
            return {lhs >= rhs[0], lhs >= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] <= rhs[0], lhs[1] <= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<=(Int2 lhs, T rhs) noexcept {
            return {lhs[0] <= rhs, lhs[1] <= rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator<=(T lhs, Int2 rhs) noexcept {
            return {lhs <= rhs[0], lhs <= rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator==(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] == rhs[0], lhs[1] == rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator==(Int2 lhs, T rhs) noexcept {
            return {lhs[0] == rhs, lhs[1] == rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator==(T lhs, Int2 rhs) noexcept {
            return {lhs == rhs[0], lhs == rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator!=(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] != rhs[0], lhs[1] != rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator!=(Int2 lhs, T rhs) noexcept {
            return {lhs[0] != rhs, lhs[1] != rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Bool2 operator!=(T lhs, Int2 rhs) noexcept {
            return {lhs != rhs[0], lhs != rhs[1]};
        }

        // -- Other Operators --
        [[nodiscard]] friend NOA_HD constexpr Int2 operator%(Int2 lhs, Int2 rhs) noexcept {
            return {lhs[0] % rhs[0], lhs[1] % rhs[1]};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator%(Int2 lhs, T rhs) noexcept {
            return {lhs[0] % rhs, lhs[1] % rhs};
        }

        [[nodiscard]] friend NOA_HD constexpr Int2 operator%(T lhs, Int2 rhs) noexcept {
            return {lhs % rhs[0], lhs % rhs[1]};
        }

    public: // Component accesses
        static constexpr size_t COUNT = 2;

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
        [[nodiscard]] NOA_HD constexpr Int2 flip() const noexcept { return {m_data[1], m_data[0]}; }

        /// Returns the logical number of dimensions assuming *this is a shape in the HW convention.
        /// Note that both row and column vectors are considered to be 1D.
        [[nodiscard]] NOA_HD constexpr T ndim() const noexcept {
            NOA_ASSERT(all(*this >= T{1}));
            return m_data[0] > 1 && m_data[1] > 1 ? 2 : 1;
        }

        /// Returns the strides, in elements, assuming *this is a shape in the HW convention.
        /// If the Height and Width dimensions are empty, 'C' and 'F' returns the same strides.
        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr Int2 strides() const noexcept {
            if constexpr (ORDER == 'C' || ORDER == 'c') {
                return {m_data[1], 1};
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                return {1, m_data[0]};
            } else {
                static_assert(traits::always_false_v<T>);
            }
        }

        /// Returns the pitch, i.e. physical size of the rows (if 'C') or of the columns (if 'F'),
        /// in elements, assuming *this are strides in the HW convention.
        template<char ORDER = 'C'>
        [[nodiscard]] NOA_HD constexpr T pitch() const noexcept {
            NOA_ASSERT(all(*this != 0) && "Cannot recover pitch from broadcast strides");
            if constexpr (ORDER == 'C' || ORDER == 'c') {
                return m_data[0];
            } else if constexpr (ORDER == 'F' || ORDER == 'f') {
                return m_data[1];
            } else {
                static_assert(traits::always_false_v<T>);
            }
        }

        /// Returns the number of elements in an array with *this as its shape.
        [[nodiscard]] NOA_HD constexpr T elements() const noexcept {
            return m_data[0] * m_data[1];
        }

        /// Returns the shape of the non-redundant FFT, in elements,
        /// assuming *this is the logical shape in the HW convention.
        [[nodiscard]] NOA_HD constexpr Int2 fft() const noexcept {
            return {m_data[0], m_data[1] / 2 + 1};
        }

    private:
        static_assert(traits::is_int_v<T> && !traits::is_bool_v<T>);
        T m_data[2]{};
    };

    using int2_t = Int2<int>;
    using uint2_t = Int2<uint>;
    using long2_t = Int2<int64_t>;
    using ulong2_t = Int2<uint64_t>;
    using size2_t = Int2<size_t>;

    template<typename T> struct traits::proclaim_is_int2<Int2<T>> : std::true_type {};
    template<typename T> struct traits::proclaim_is_uint2<Int2<T>> : std::bool_constant<is_uint_v<T>> {};

    template<typename T>
    [[nodiscard]] NOA_IH constexpr std::array<T, 2> toArray(Int2<T> v) noexcept {
        return {v[0], v[1]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<int2_t>() { return "int2"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<uint2_t>() { return "uint2"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<long2_t>() { return "long2"; }
    template<>
    [[nodiscard]] NOA_IH std::string string::human<ulong2_t>() { return "ulong2"; }

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

namespace noa {
    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_int2_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr TTo clamp_cast(const Int2<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return {clamp_cast<value_t>(src[0]), clamp_cast<value_t>(src[1])};
    }

    template<typename TTo, typename TFrom, typename = std::enable_if_t<traits::is_int2_v<TTo>>>
    [[nodiscard]] NOA_FHD constexpr bool isSafeCast(const Int2<TFrom>& src) noexcept {
        using value_t = traits::value_type_t<TTo>;
        return isSafeCast<value_t>(src[0]) && isSafeCast<value_t>(src[1]);
    }
}

namespace noa::math {
    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T sum(Int2<T> v) noexcept {
        return v[0] + v[1];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T prod(Int2<T> v) noexcept {
        return v[0] * v[1];
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> abs(Int2<T> v) noexcept {
        return {abs(v[0]), abs(v[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T min(Int2<T> v) noexcept {
        return min(v[0], v[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> min(Int2<T> lhs, Int2<T> rhs) noexcept {
        return {min(lhs[0], rhs[0]), min(lhs[1], rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> min(Int2<T> lhs, T rhs) noexcept {
        return {min(lhs[0], rhs), min(lhs[1], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> min(T lhs, Int2<T> rhs) noexcept {
        return {min(lhs, rhs[0]), min(lhs, rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr T max(Int2<T> v) noexcept {
        return max(v[0], v[1]);
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> max(Int2<T> lhs, Int2<T> rhs) noexcept {
        return {max(lhs[0], rhs[0]), max(lhs[1], rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> max(Int2<T> lhs, T rhs) noexcept {
        return {max(lhs[0], rhs), max(lhs[1], rhs)};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> max(T lhs, Int2<T> rhs) noexcept {
        return {max(lhs, rhs[0]), max(lhs, rhs[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> clamp(Int2<T> lhs, Int2<T> low, Int2<T> high) noexcept {
        return {clamp(lhs[0], low[0], high[0]), clamp(lhs[1], low[1], high[1])};
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> clamp(Int2<T> lhs, T low, T high) noexcept {
        return {clamp(lhs[0], low, high), clamp(lhs[1], low, high)};
    }

    template<typename T, typename U>
    [[nodiscard]] NOA_FHD constexpr Int2<T> sort(Int2<T> v, U&& comp) noexcept {
        smallStableSort<2>(v.get(), std::forward<U>(comp));
        return v;
    }

    template<typename T>
    [[nodiscard]] NOA_FHD constexpr Int2<T> sort(Int2<T> v) noexcept {
        return sort(v, [](const T& a, const T& b) { return a < b; });
    }
}
