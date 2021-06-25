/// \file noa/util/Bool3.h
/// \author Thomas - ffyr2w
/// \date 10/12/2020
/// Vector containing 3 booleans.

#pragma once

#include <string>
#include <array>
#include <type_traits>

#include "noa/Definitions.h"
#include "noa/Assert.h"
#include "noa/util/string/Format.h"
#include "noa/util/traits/BaseTypes.h"

namespace noa {
    struct Bool3 {
        typedef bool value_type;
        bool x{}, y{}, z{};

    public: // Component accesses
        [[nodiscard]] NOA_HD static constexpr size_t elements() noexcept { return 3; }
        [[nodiscard]] NOA_HD static constexpr size_t size() noexcept { return elements(); }
        NOA_HD constexpr bool& operator[](size_t i) noexcept;
        NOA_HD constexpr const bool& operator[](size_t i) const noexcept;

    public: // (Conversion) Constructors
        constexpr Bool3() noexcept = default;
        template<typename X, typename Y, typename Z> NOA_HD constexpr Bool3(X xi, Y yi, Z zi) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool3(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool3(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Bool3& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Bool3& operator=(U* ptr) noexcept;
    };

    // -- Boolean operators --

    NOA_FHD constexpr Bool3 operator==(const Bool3& lhs, const Bool3& rhs) noexcept;
    NOA_FHD constexpr Bool3 operator==(const Bool3& lhs, bool rhs) noexcept;
    NOA_FHD constexpr Bool3 operator==(bool lhs, const Bool3& rhs) noexcept;

    NOA_FHD constexpr Bool3 operator!=(const Bool3& lhs, const Bool3& rhs) noexcept;
    NOA_FHD constexpr Bool3 operator!=(const Bool3& lhs, bool rhs) noexcept;
    NOA_FHD constexpr Bool3 operator!=(bool lhs, const Bool3& rhs) noexcept;

    NOA_FHD constexpr Bool3 operator&&(const Bool3& lhs, const Bool3& rhs) noexcept;
    NOA_FHD constexpr Bool3 operator||(const Bool3& lhs, const Bool3& rhs) noexcept;

    NOA_FHD constexpr bool any(const Bool3& v) noexcept;
    NOA_FHD constexpr bool all(const Bool3& v) noexcept;

    using bool3_t = Bool3;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 3> toArray(const Bool3& v) noexcept {
        return {v.x, v.y, v.z};
    }

    template<> NOA_IH std::string string::typeName<Bool3>() { return "bool3"; }

    NOA_IH std::ostream& operator<<(std::ostream& os, const Bool3& v) {
        os << string::format("({},{},{})", v.x, v.y, v.z);
        return os;
    }

    template<> struct traits::proclaim_is_boolX<Bool3> : std::true_type {};
}

namespace noa {
    // -- Component accesses --

    constexpr bool& Bool3::operator[](size_t i) noexcept {
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
