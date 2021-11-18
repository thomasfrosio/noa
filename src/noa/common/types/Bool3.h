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
        bool x{}, y{}, z{};

    public: // Component accesses
        static constexpr size_t COUNT = 3;

        NOA_HD constexpr bool& operator[](size_t i) noexcept {
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

        NOA_HD constexpr const bool& operator[](size_t i) const noexcept {
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

    public: // Default constructors
        constexpr Bool3() noexcept = default;
        constexpr Bool3(const Bool3&) noexcept = default;
        constexpr Bool3(Bool3&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z>
        NOA_HD constexpr Bool3(X xi, Y yi, Z zi) noexcept
                : x(static_cast<bool>(xi)),
                  y(static_cast<bool>(yi)),
                  z(static_cast<bool>(zi)) {}

        template<typename U>
        NOA_HD constexpr explicit Bool3(U v) noexcept
                : x(static_cast<bool>(v)),
                  y(static_cast<bool>(v)),
                  z(static_cast<bool>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Bool3(U* ptr)
                : x(static_cast<bool>(ptr[0])),
                  y(static_cast<bool>(ptr[1])),
                  z(static_cast<bool>(ptr[2])) {}

    public: // Assignment operators
        constexpr Bool3& operator=(const Bool3& v) noexcept = default;
        constexpr Bool3& operator=(Bool3&& v) noexcept = default;

        NOA_HD constexpr Bool3& operator=(bool v) noexcept {
            this->x = v;
            this->y = v;
            this->z = v;
            return *this;
        }

        NOA_HD constexpr Bool3& operator=(const bool* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            this->z = ptr[2];
            return *this;
        }
    };

    // -- Boolean operators --

    NOA_FHD constexpr Bool3 operator==(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z};
    }

    NOA_FHD constexpr Bool3 operator==(Bool3 lhs, bool rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs};
    }

    NOA_FHD constexpr Bool3 operator==(bool lhs, Bool3 rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z};
    }

    NOA_FHD constexpr Bool3 operator!=(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z};
    }

    NOA_FHD constexpr Bool3 operator!=(Bool3 lhs, bool rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs};
    }

    NOA_FHD constexpr Bool3 operator!=(bool lhs, Bool3 rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z};
    }

    NOA_FHD constexpr Bool3 operator&&(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z};
    }

    NOA_FHD constexpr Bool3 operator||(Bool3 lhs, Bool3 rhs) noexcept {
        return {lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z};
    }

    NOA_FHD constexpr bool any(Bool3 v) noexcept {
        return v.x || v.y || v.z;
    }

    NOA_FHD constexpr bool all(Bool3 v) noexcept {
        return v.x && v.y && v.z;
    }

    using bool3_t = Bool3;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 3> toArray(Bool3 v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<>
    NOA_IH std::string string::typeName<Bool3>() {
        return "bool3";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, Bool3 v) {
        os << string::format("({},{},{})", v.x, v.y, v.z);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool3> : std::true_type {};
}
