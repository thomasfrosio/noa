/// \file noa/common/types/Bool3.h
/// \author Thomas - ffyr2w
/// \date 25 May 2021
/// Vector containing 3 booleans.

#pragma once

#include <string>
#include <array>

#include "noa/common/Assert.h"
#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa {
    struct Bool3 {
        typedef bool value_type;

    public: // Default constructors
        constexpr Bool3() noexcept = default;
        constexpr Bool3(const Bool3&) noexcept = default;
        constexpr Bool3(Bool3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z>
        NOA_HD constexpr Bool3(X x, Y y, Z z) noexcept
                : m_data{static_cast<bool>(x), static_cast<bool>(y), static_cast<bool>(z)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Bool3(U x) noexcept
                : m_data{static_cast<bool>(x), static_cast<bool>(x), static_cast<bool>(x)} {}

        template<typename U, typename = std::enable_if_t<noa::traits::is_scalar_v<U>>>
        NOA_HD constexpr explicit Bool3(const U* ptr)
                : m_data{static_cast<bool>(ptr[0]), static_cast<bool>(ptr[1]), static_cast<bool>(ptr[2])} {}

    public: // Assignment operators
        constexpr Bool3& operator=(const Bool3& v) noexcept = default;
        constexpr Bool3& operator=(Bool3&& v) noexcept = default;

        NOA_HD constexpr Bool3& operator=(bool v) noexcept {
            m_data[0] = v;
            m_data[1] = v;
            m_data[2] = v;
            return *this;
        }

    public: // Component accesses
        static constexpr size_t COUNT = 3;

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr bool& operator[](I i) noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD constexpr const bool& operator[](I i) const noexcept {
            NOA_ASSERT(static_cast<size_t>(i) < COUNT);
            return m_data[i];
        }

        NOA_HD [[nodiscard]] constexpr const bool* get() const noexcept { return m_data; }
        NOA_HD [[nodiscard]] constexpr bool* get() noexcept { return m_data; }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD [[nodiscard]] constexpr const bool* get(I i) const noexcept { return m_data + i; }

        template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
        NOA_HD [[nodiscard]] constexpr bool* get(I i) noexcept { return m_data + i; }

        NOA_HD [[nodiscard]] constexpr Bool3 flip() const noexcept { return {m_data[2], m_data[1], m_data[0]}; }

    private:
        bool m_data[3]{};
    };

    // -- Boolean operators --
    NOA_FHD constexpr Bool3 operator!(Bool3 rhs) noexcept {
        return {!rhs[0], !rhs[1], !rhs[2]};
    }

    NOA_FHD constexpr Bool3 operator==(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs[0] == rhs[0], lhs[1] == rhs[1], lhs[2] == rhs[2]};
    }

    NOA_FHD constexpr Bool3 operator==(Bool3 lhs, bool rhs) noexcept {
        return {lhs[0] == rhs, lhs[1] == rhs, lhs[2] == rhs};
    }

    NOA_FHD constexpr Bool3 operator==(bool lhs, Bool3 rhs) noexcept {
        return {lhs == rhs[0], lhs == rhs[1], lhs == rhs[2]};
    }

    NOA_FHD constexpr Bool3 operator!=(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs[0] != rhs[0], lhs[1] != rhs[1], lhs[2] != rhs[2]};
    }

    NOA_FHD constexpr Bool3 operator!=(Bool3 lhs, bool rhs) noexcept {
        return {lhs[0] != rhs, lhs[1] != rhs, lhs[2] != rhs};
    }

    NOA_FHD constexpr Bool3 operator!=(bool lhs, Bool3 rhs) noexcept {
        return {lhs != rhs[0], lhs != rhs[1], lhs != rhs[2]};
    }

    NOA_FHD constexpr Bool3 operator&&(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs[0] && rhs[0], lhs[1] && rhs[1], lhs[2] && rhs[2]};
    }

    NOA_FHD constexpr Bool3 operator||(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs[0] || rhs[0], lhs[1] || rhs[1], lhs[2] || rhs[2]};
    }

    NOA_FHD constexpr bool any(Bool3 v) noexcept {
        return v[0] || v[1] || v[2];
    }

    NOA_FHD constexpr bool all(Bool3 v) noexcept {
        return v[0] && v[1] && v[2];
    }

    using bool3_t = Bool3;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 3> toArray(Bool3 v) noexcept {
        return {v[0], v[1], v[2]};
    }

    template<>
    NOA_IH std::string string::human<Bool3>() {
        return "bool3";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, Bool3 v) {
        os << string::format("({},{},{})", v[0], v[1], v[2]);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool3> : std::true_type {};
}

namespace fmt {
    template<>
    struct formatter<noa::Bool3> : formatter<bool> {
        template<typename FormatContext>
        auto format(const noa::Bool3& vec, FormatContext& ctx) {
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
            *out = ')';
            return out;
        }
    };
}
