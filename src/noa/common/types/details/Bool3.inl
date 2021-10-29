#ifndef NOA_INCLUDE_BOOL3_
#error "This should not be directly included"
#endif

#include "noa/common/Assert.h"

namespace noa {
    // -- Component accesses --

    constexpr bool& Bool3::operator[](size_t i) noexcept {
        NOA_ASSERT(i < this->COUNT);
        switch (i) {
            default:
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
        }
    }

    constexpr const bool& Bool3::operator[](size_t i) const noexcept {
        NOA_ASSERT(i < this->COUNT);
        switch (i) {
            default:
            case 0:
                return this->x;
            case 1:
                return this->y;
            case 2:
                return this->z;
        }
    }

    // -- (Conversion) Constructors --

    template<typename X, typename Y, typename Z>
    constexpr Bool3::Bool3(X xi, Y yi, Z zi) noexcept
            : x(static_cast<bool>(xi)),
              y(static_cast<bool>(yi)),
              z(static_cast<bool>(zi)) {}

    template<typename U>
    constexpr Bool3::Bool3(U v) noexcept
            : x(static_cast<bool>(v)),
              y(static_cast<bool>(v)),
              z(static_cast<bool>(v)) {}

    template<typename U>
    constexpr Bool3::Bool3(U* ptr)
            : x(static_cast<bool>(ptr[0])),
              y(static_cast<bool>(ptr[1])),
              z(static_cast<bool>(ptr[2])) {}

    // -- Assignment operators --

    template<typename U>
    constexpr Bool3& Bool3::operator=(U v) noexcept {
        this->x = static_cast<bool>(v);
        this->y = static_cast<bool>(v);
        this->z = static_cast<bool>(v);
        return *this;
    }

    template<typename U>
    constexpr Bool3& Bool3::operator=(U* ptr) noexcept {
        this->x = static_cast<bool>(ptr[0]);
        this->y = static_cast<bool>(ptr[1]);
        this->z = static_cast<bool>(ptr[2]);
        return *this;
    }

    constexpr Bool3 operator==(const Bool3& lhs, const Bool3& rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }
    constexpr Bool3 operator==(const Bool3& lhs, bool rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }
    constexpr Bool3 operator==(bool lhs, const Bool3& rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    constexpr Bool3 operator!=(const Bool3& lhs, const Bool3& rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }
    constexpr Bool3 operator!=(const Bool3& lhs, bool rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }
    constexpr Bool3 operator!=(bool lhs, const Bool3& rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    constexpr Bool3 operator&&(const Bool3& lhs, const Bool3& rhs) noexcept {
        return {lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z};
    }
    constexpr Bool3 operator||(const Bool3& lhs, const Bool3& rhs) noexcept {
        return {lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z};
    }

    constexpr bool any(const Bool3& v) noexcept {
        return v.x || v.y || v.z;
    }
    constexpr bool all(const Bool3& v) noexcept {
        return v.x && v.y && v.z;
    }
}
