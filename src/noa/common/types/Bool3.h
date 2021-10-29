/// \file noa/common/types/Bool3.h
/// \author Thomas - ffyr2w
/// \date 25 May 2021
/// Vector containing 3 booleans.

#pragma once

#include <string>
#include <array>

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa {
    struct Bool3 {
        typedef bool value_type;
        bool x{}, y{}, z{};

    public: // Component accesses
        static constexpr size_t COUNT = 3;
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

    template<>
    NOA_IH std::string string::typeName<Bool3>() {
        return "bool3";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, const Bool3& v) {
        os << string::format("({},{},{})", v.x, v.y, v.z);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool3> : std::true_type {};
}

#define NOA_INCLUDE_BOOL3_
#include "noa/common/types/details/Bool3.inl"
#undef NOA_INCLUDE_BOOL3_
