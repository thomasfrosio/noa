/// \file noa/common/types/Bool2.h
/// \author Thomas - ffyr2w
/// \date 25 May 2021
/// Vector containing 2 booleans.

#pragma once

#include <string>
#include <array>

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa {
    struct Bool2 {
        typedef bool value_type;
        bool x{}, y{};

    public: // Component accesses
        static constexpr size_t COUNT = 2;
        NOA_HD constexpr bool& operator[](size_t i) noexcept;
        NOA_HD constexpr const bool& operator[](size_t i) const noexcept;

    public: // (Conversion) Constructors
        constexpr Bool2() noexcept = default;
        template<typename X, typename Y> NOA_HD constexpr Bool2(X xi, Y yi) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool2(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool2(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Bool2& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Bool2& operator=(U* ptr) noexcept;
    };

    // -- Boolean operators --

    NOA_FHD constexpr Bool2 operator==(const Bool2& lhs, const Bool2& rhs) noexcept;
    NOA_FHD constexpr Bool2 operator==(const Bool2& lhs, bool rhs) noexcept;
    NOA_FHD constexpr Bool2 operator==(bool lhs, const Bool2& rhs) noexcept;

    NOA_FHD constexpr Bool2 operator!=(const Bool2& lhs, const Bool2& rhs) noexcept;
    NOA_FHD constexpr Bool2 operator!=(const Bool2& lhs, bool rhs) noexcept;
    NOA_FHD constexpr Bool2 operator!=(bool lhs, const Bool2& rhs) noexcept;

    NOA_FHD constexpr Bool2 operator&&(const Bool2& lhs, const Bool2& rhs) noexcept;
    NOA_FHD constexpr Bool2 operator||(const Bool2& lhs, const Bool2& rhs) noexcept;

    NOA_FHD constexpr bool any(const Bool2& v) noexcept;
    NOA_FHD constexpr bool all(const Bool2& v) noexcept;

    using bool2_t = Bool2;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 2> toArray(const Bool2& v) noexcept {
        return {v.x, v.y};
    }

    template<>
    NOA_IH std::string string::typeName<Bool2>() {
        return "bool2";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, const Bool2& v) {
        os << string::format("({},{})", v.x, v.y);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool2> : std::true_type {};
}

#define NOA_INCLUDE_BOOL2_
#include "noa/common/types/details/Bool2.inl"
#undef NOA_INCLUDE_BOOL2_
