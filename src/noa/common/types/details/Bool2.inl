#ifndef NOA_INCLUDE_BOOL2_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    constexpr bool& Bool2::operator[](size_t i) noexcept {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    constexpr const bool& Bool2::operator[](size_t i) const noexcept {
        NOA_ASSERT(i < this->COUNT);
        if (i == 1)
            return this->y;
        else
            return this->x;
    }

    // -- (Conversion) Constructors --

    template<typename X, typename Y>
    constexpr Bool2::Bool2(X xi, Y yi) noexcept
            : x(static_cast<bool>(xi)),
              y(static_cast<bool>(yi)) {}

    template<typename U>
    constexpr Bool2::Bool2(U v) noexcept
            : x(static_cast<bool>(v)),
              y(static_cast<bool>(v)) {}

    template<typename U>
    constexpr Bool2::Bool2(U* ptr)
            : x(static_cast<bool>(ptr[0])),
              y(static_cast<bool>(ptr[1])) {}

    // -- Assignment operators --

    template<typename U>
    constexpr Bool2& Bool2::operator=(U v) noexcept {
        this->x = static_cast<bool>(v);
        this->y = static_cast<bool>(v);
        return *this;
    }

    template<typename U>
    constexpr Bool2& Bool2::operator=(U* ptr) noexcept {
        this->x = static_cast<bool>(ptr[0]);
        this->y = static_cast<bool>(ptr[1]);
        return *this;
    }

    constexpr Bool2 operator==(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y};
    }
    constexpr Bool2 operator==(const Bool2& lhs, bool rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs};
    }
    constexpr Bool2 operator==(bool lhs, const Bool2& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y};
    }

    constexpr Bool2 operator!=(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y};
    }
    constexpr Bool2 operator!=(const Bool2& lhs, bool rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs};
    }
    constexpr Bool2 operator!=(bool lhs, const Bool2& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y};
    }

    constexpr Bool2 operator&&(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x && rhs.x, lhs.y && rhs.y};
    }
    constexpr Bool2 operator||(const Bool2& lhs, const Bool2& rhs) noexcept {
        return {lhs.x || rhs.x, lhs.y || rhs.y};
    }

    constexpr bool any(const Bool2& v) noexcept {
        return v.x || v.y;
    }
    constexpr bool all(const Bool2& v) noexcept {
        return v.x && v.y;
    }
}
