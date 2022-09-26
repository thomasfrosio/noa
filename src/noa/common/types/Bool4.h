/// \file noa/common/types/Bool4.h
/// \author Thomas - ffyr2w
/// \date 25 May 2021
/// Vector containing 4 booleans.

#pragma once

#include <string>
#include <array>

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/traits/ArrayTypes.h"

namespace noa {
    struct Bool4 {
        using value_type = bool;

    public: // Default constructors
        constexpr Bool4() noexcept = default;
        constexpr Bool4(const Bool4&) noexcept = default;
        constexpr Bool4(Bool4&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z, typename W>
        NOA_HD constexpr Bool4(X a0, Y a1, Z a2, W a3) noexcept
                : m_data{static_cast<bool>(a0), static_cast<bool>(a1), static_cast<bool>(a2), static_cast<bool>(a3)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Bool4(U x) noexcept
                : m_data{static_cast<bool>(x), static_cast<bool>(x), static_cast<bool>(x), static_cast<bool>(x)} {}

        template<typename U, typename = std::enable_if_t<traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Bool4(const U* ptr) noexcept {
            NOA_ASSERT(ptr != nullptr);
            for (size_t i = 0; i < COUNT; ++i)
                m_data[i] = static_cast<bool>(ptr[i]);
        }

    public: // Assignment operators
        constexpr Bool4& operator=(const Bool4& v) noexcept = default;
        constexpr Bool4& operator=(Bool4&& v) noexcept = default;

        NOA_HD constexpr Bool4& operator=(bool v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            m_data[2] = v;
            m_data[3] = v;
            return *this;
        }

    public: // Component accesses
        static constexpr size_t COUNT = 4;

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
        [[nodiscard]] NOA_HD constexpr const bool* get(I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        template<typename I, typename = std::enable_if_t<traits::is_int_v<I>>>
        [[nodiscard]] NOA_HD constexpr bool* get(I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) <= COUNT);
            return m_data + i;
        }

        [[nodiscard]] NOA_HD constexpr Bool4 flip() const noexcept {
            return {m_data[3], m_data[2], m_data[1], m_data[0]};
        }

    private:
        bool m_data[4]{};
    };

    // -- Boolean operators --
    [[nodiscard]] NOA_FHD constexpr Bool4 operator!(Bool4 rhs) noexcept {
        return {!rhs[0], !rhs[1], !rhs[2], !rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator==(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2], lhs[3] == rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator==(Bool4 lhs, bool rhs) noexcept {
        return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs, lhs[3] == rhs};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator==(bool lhs, Bool4 rhs) noexcept {
        return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2], lhs == rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator!=(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2], lhs[3] != rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator!=(Bool4 lhs, bool rhs) noexcept {
        return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs, lhs[3] != rhs};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator!=(bool lhs, Bool4 rhs) noexcept {
        return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2], lhs != rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator&&(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs[0] && rhs[0], lhs[1] && rhs[1], lhs[2] && rhs[2], lhs[3] && rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr Bool4 operator||(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs[0] || rhs[0], lhs[1] || rhs[1], lhs[2] || rhs[2], lhs[3] || rhs[3]};
    }

    [[nodiscard]] NOA_FHD constexpr bool any(Bool4 v) noexcept {
        return v[0] || v[1] || v[2] || v[3];
    }

    [[nodiscard]] NOA_FHD constexpr bool all(Bool4 v) noexcept {
        return v[0] && v[1] && v[2] && v[3];
    }

    using bool4_t = Bool4;

    [[nodiscard]] NOA_IH constexpr std::array<bool, 4> toArray(Bool4 v) noexcept {
        return {v[0], v[1], v[2], v[3]};
    }

    template<>
    [[nodiscard]] NOA_IH std::string string::human<Bool4>() {
        return "bool4";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, Bool4 v) {
        os << string::format("({},{},{},{})", v[0], v[1], v[2], v[3]);
        return os;
    }

    template<>
    struct traits::proclaim_is_bool4<Bool4> : std::true_type {};
}

namespace fmt {
    template<>
    struct formatter<noa::Bool4> : formatter<bool> {
        template<typename FormatContext>
        auto format(const noa::Bool4& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec[0], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec[1], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec[2], ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec[3], ctx);
            *out = ')';
            return out;
        }
    };
}
