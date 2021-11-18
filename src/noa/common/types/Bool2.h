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
    struct Bool2 {
        typedef bool value_type;
        bool x{}, y{};

    public: // Component accesses
        static constexpr size_t COUNT = 2;

        NOA_HD constexpr bool& operator[](size_t i) noexcept {
            NOA_ASSERT(i < this->COUNT);
            if (i == 1)
                return this->y;
            else
                return this->x;
        }

        NOA_HD constexpr const bool& operator[](size_t i) const noexcept {
            NOA_ASSERT(i < this->COUNT);
            if (i == 1)
                return this->y;
            else
                return this->x;
        }

    public: // Default constructors
        constexpr Bool2() noexcept = default;
        constexpr Bool2(const Bool2&) noexcept = default;
        constexpr Bool2(Bool2&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y>
        NOA_HD constexpr Bool2(X xi, Y yi) noexcept
                : x(static_cast<bool>(xi)),
                  y(static_cast<bool>(yi)) {}

        template<typename U>
        NOA_HD constexpr explicit Bool2(U v) noexcept
                : x(static_cast<bool>(v)),
                  y(static_cast<bool>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Bool2(U* ptr)
                : x(static_cast<bool>(ptr[0])),
                  y(static_cast<bool>(ptr[1])) {}

    public: // Assignment operators
        constexpr Bool2& operator=(const Bool2& v) noexcept = default;
        constexpr Bool2& operator=(Bool2&& v) noexcept = default;

        NOA_HD constexpr Bool2& operator=(bool v) noexcept {
            this->x = v;
            this->y = v;
            return *this;
        }

        NOA_HD constexpr Bool2& operator=(const bool* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            return *this;
        }
    };

    // -- Boolean operators --
    NOA_FHD constexpr Bool2 operator==(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }

    NOA_FHD constexpr Bool2 operator==(Bool2 lhs, bool rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }

    NOA_FHD constexpr Bool2 operator==(bool lhs, Bool2 rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    NOA_FHD constexpr Bool2 operator!=(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }

    NOA_FHD constexpr Bool2 operator!=(Bool2 lhs, bool rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }

    NOA_FHD constexpr Bool2 operator!=(bool lhs, Bool2 rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    NOA_FHD constexpr Bool2 operator&&(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs.x && rhs.x, lhs.y && rhs.y};
    }

    NOA_FHD constexpr Bool2 operator||(Bool2 lhs, Bool2 rhs) noexcept {
        return {lhs.x || rhs.x, lhs.y || rhs.y};
    }

    NOA_FHD constexpr bool any(Bool2 v) noexcept {
        return v.x || v.y;
    }

    NOA_FHD constexpr bool all(Bool2 v) noexcept {
        return v.x && v.y;
    }

    using bool2_t = Bool2;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 2> toArray(Bool2 v) noexcept {
        return {v.x, v.y};
    }

    template<>
    NOA_IH std::string string::typeName<Bool2>() {
        return "bool2";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, Bool2 v) {
        os << string::format("({},{})", v.x, v.y);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool2> : std::true_type {};
}
