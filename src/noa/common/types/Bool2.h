/// \file noa/common/types/Bool2.h
/// \author Thomas - ffyr2w
/// \date 25 May 2021
/// Vector containing 2 booleans.

#pragma once

#include <string>
#include <array>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa {
    class Bool2 {
    public:
        typedef bool value_type;

    public: // Default constructors
        constexpr Bool2() noexcept = default;
        constexpr Bool2(const Bool2&) noexcept = default;
        constexpr Bool2(Bool2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y>
        NOA_HD constexpr Bool2(X x, Y y) noexcept
                : m_data{static_cast<bool>(x), static_cast<bool>(y)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Bool2(U x) noexcept
                : m_data{static_cast<bool>(x), static_cast<bool>(x)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Bool2(const U* ptr)
                : m_data{static_cast<bool>(ptr[0]), static_cast<bool>(ptr[1])} {}

    public: // Assignment operators
        constexpr Bool2& operator=(const Bool2& v) noexcept = default;
        constexpr Bool2& operator=(Bool2&& v) noexcept = default;

        NOA_HD constexpr Bool2& operator=(bool v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            return *this;
        }

    public: // Component accesses
        static constexpr size_t COUNT = 2;

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr bool& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const bool& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        [[nodiscard]] NOA_HD constexpr const bool* get() const noexcept { return m_data; }
        [[nodiscard]] NOA_HD constexpr bool* get() noexcept { return m_data; }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr const bool* get(I i) const noexcept { return m_data + i; }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr bool* get(I i) noexcept { return m_data + i; }

        [[nodiscard]] NOA_HD constexpr Bool2 flip() const noexcept { return {m_data[1], m_data[0]}; }

    private:
        bool m_data[2]{};
    };

    // -- Boolean operators --
    [[nodiscard]] NOA_FHD constexpr Bool2 operator!(Bool2 rhs) noexcept {
        return {!rhs[0], !rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator==(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs[0] == rhs[0], lhs[1] == rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator==(Bool2 lhs, bool rhs) noexcept {
        return {lhs[0] == rhs, lhs[1] == rhs};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator==(bool lhs, Bool2 rhs) noexcept {
        return {lhs == rhs[0], lhs == rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator!=(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs[0] != rhs[0], lhs[1] != rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator!=(Bool2 lhs, bool rhs) noexcept {
        return {lhs[0] != rhs, lhs[1] != rhs};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator!=(bool lhs, Bool2 rhs) noexcept {
        return {lhs != rhs[0], lhs != rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator&&(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs[0] && rhs[0], lhs[1] && rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool2 operator||(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs[0] || rhs[0], lhs[1] || rhs[1]};
    }

    [[nodiscard]] NOA_FHD constexpr bool any(Bool2 v) noexcept {
        return v[0] || v[1];
    }

    [[nodiscard]] NOA_FHD constexpr bool all(Bool2 v) noexcept {
        return v[0] && v[1];
    }

    using bool2_t = Bool2;

    [[nodiscard]] NOA_IH constexpr std::array<bool, 2> toArray(Bool2 v) noexcept {
        return {v[0], v[1]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<Bool2>() {
        return "bool2";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, Bool2 v) {
        os << string::format("({},{})", v[0], v[1]);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool2> : std::true_type {};
}

namespace fmt {
    template<>
    struct formatter<noa::Bool2> : formatter<bool> {
        template<typename FormatContext>
        auto format(const noa::Bool2& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec[1], ctx);
            *out = ')';
            return out;
        }
    };
}
