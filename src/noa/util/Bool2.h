/**
 * @file noa/util/Bool2.h
 * @author Thomas - ffyr2w
 * @date 10/12/2020
 * Vector containing 2 booleans.
 */
#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Assert.h"
#include "noa/util/string/Format.h"
#include "noa/util/traits/BaseTypes.h"

namespace Noa {
    struct Bool2 {
        typedef bool value_type;
        bool x{}, y{};

    public: // Component accesses
        [[nodiscard]] NOA_HD static constexpr size_t elements() noexcept { return 2; }
        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr bool& operator[](size_t i) noexcept;
        NOA_HD constexpr const bool& operator[](size_t i) const noexcept;

    public: // (Conversion) Constructors
        NOA_HD constexpr Bool2() noexcept = default;
        template<typename X, typename Y> NOA_HD constexpr Bool2(X xi, Y yi) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool2(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool2(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Bool2& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Bool2& operator=(U* ptr) noexcept;
    };

    // -- Boolean operators --

    NOA_HD constexpr Bool2 operator==(const Bool2& lhs, const Bool2& rhs) noexcept;
    NOA_HD constexpr Bool2 operator==(const Bool2& lhs, bool rhs) noexcept;
    NOA_HD constexpr Bool2 operator==(bool lhs, const Bool2& rhs) noexcept;

    NOA_HD constexpr Bool2 operator!=(const Bool2& lhs, const Bool2& rhs) noexcept;
    NOA_HD constexpr Bool2 operator!=(const Bool2& lhs, bool rhs) noexcept;
    NOA_HD constexpr Bool2 operator!=(bool lhs, const Bool2& rhs) noexcept;

    NOA_HD constexpr Bool2 operator&&(const Bool2& lhs, const Bool2& rhs) noexcept;
    NOA_HD constexpr Bool2 operator||(const Bool2& lhs, const Bool2& rhs) noexcept;

    NOA_HD constexpr bool any(const Bool2& v) noexcept;
    NOA_HD constexpr bool all(const Bool2& v) noexcept;

    using bool2_t = Bool2;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 2> toArray(const Bool2& v) noexcept {
        return {v.x, v.y};
    }

    template<> NOA_IH std::string String::typeName<Bool2>() { return "bool2"; }

    NOA_IH std::ostream& operator<<(std::ostream& os, const Bool2& v) {
        os << String::format("({},{})", v.x, v.y);
        return os;
    }

    template<> struct Traits::proclaim_is_boolX<Bool2> : std::true_type {};
}

namespace Noa {
    // -- Component accesses --

    NOA_HD constexpr bool& Bool2::operator[](size_t i) noexcept {
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    NOA_HD constexpr const bool& Bool2::operator[](size_t i) const noexcept {
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    // -- (Conversion) Constructors --

    template<typename X, typename Y>
    NOA_HD constexpr Bool2::Bool2(X xi, Y yi) noexcept
            : x(static_cast<bool>(xi)),
              y(static_cast<bool>(yi)) {}

    template<typename U>
    NOA_HD constexpr Bool2::Bool2(U v) noexcept
            : x(static_cast<bool>(v)),
              y(static_cast<bool>(v)) {}

    template<typename U>
    NOA_HD constexpr Bool2::Bool2(U* ptr)
            : x(static_cast<bool>(ptr[0])),
              y(static_cast<bool>(ptr[1])) {}

    // -- Assignment operators --

    template<typename U>
    NOA_HD constexpr Bool2& Bool2::operator=(U v) noexcept {
        this->x = static_cast<bool>(v);
        this->y = static_cast<bool>(v);
        return *this;
    }

    template<typename U>
    NOA_HD constexpr Bool2& Bool2::operator=(U* ptr) noexcept {
        this->x = static_cast<bool>(ptr[0]);
        this->y = static_cast<bool>(ptr[1]);
        return *this;
    }

    NOA_FHD constexpr Bool2 operator==(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }
    NOA_FHD constexpr Bool2 operator==(const Bool2& lhs, bool rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }
    NOA_FHD constexpr Bool2 operator==(bool lhs, const Bool2& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    NOA_FHD constexpr Bool2 operator!=(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }
    NOA_FHD constexpr Bool2 operator!=(const Bool2& lhs, bool rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }
    NOA_FHD constexpr Bool2 operator!=(bool lhs, const Bool2& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    NOA_FHD constexpr Bool2 operator&&(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x && rhs.x, lhs.y && rhs.y};
    }
    NOA_FHD constexpr Bool2 operator||(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x || rhs.x, lhs.y || rhs.y};
    }

    NOA_FHD constexpr bool any(const Bool2& v) noexcept {
        return v.x || v.y;
    }
    NOA_FHD constexpr bool all(const Bool2& v) noexcept {
        return v.x && v.y;
    }
}
