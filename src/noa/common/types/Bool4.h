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

namespace noa {
    struct Bool4 {
        typedef bool value_type;
        bool x{}, y{}, z{}, w{};

    public: // Component accesses
        static constexpr size_t COUNT = 4;
        NOA_HD constexpr bool& operator[](size_t i) noexcept;
        NOA_HD constexpr const bool& operator[](size_t i) const noexcept;

    public: // (Conversion) Constructors
        constexpr Bool4() noexcept = default;
        template<typename X, typename Y, typename Z, typename W> NOA_HD constexpr Bool4(X xi, Y yi, Z zi, W wi) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool4(U v) noexcept;
        template<typename U> NOA_HD constexpr explicit Bool4(U* ptr);

    public: // Assignment operators
        template<typename U> NOA_HD constexpr Bool4& operator=(U v) noexcept;
        template<typename U> NOA_HD constexpr Bool4& operator=(U* ptr) noexcept;
    };

    // -- Boolean operators --

    NOA_FHD constexpr Bool4 operator==(const Bool4& lhs, const Bool4& rhs) noexcept;
    NOA_FHD constexpr Bool4 operator==(const Bool4& lhs, bool rhs) noexcept;
    NOA_FHD constexpr Bool4 operator==(bool lhs, const Bool4& rhs) noexcept;

    NOA_FHD constexpr Bool4 operator!=(const Bool4& lhs, const Bool4& rhs) noexcept;
    NOA_FHD constexpr Bool4 operator!=(const Bool4& lhs, bool rhs) noexcept;
    NOA_FHD constexpr Bool4 operator!=(bool lhs, const Bool4& rhs) noexcept;

    NOA_FHD constexpr Bool4 operator&&(const Bool4& lhs, const Bool4& rhs) noexcept;
    NOA_FHD constexpr Bool4 operator||(const Bool4& lhs, const Bool4& rhs) noexcept;

    NOA_FHD constexpr bool any(const Bool4& v) noexcept;
    NOA_FHD constexpr bool all(const Bool4& v) noexcept;

    using bool4_t = Bool4;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 4> toArray(const Bool4& v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<>
    NOA_IH std::string string::typeName<Bool4>() {
        return "bool4";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, const Bool4& v) {
        os << string::format("({},{},{},{})", v.x, v.y, v.z, v.w);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool4> : std::true_type {};
}

#define NOA_INCLUDE_BOOL4_
#include "noa/common/types/details/Bool4.inl"
#undef NOA_INCLUDE_BOOL4_
