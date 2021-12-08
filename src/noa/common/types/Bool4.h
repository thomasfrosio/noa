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
                case 3:
                    return this->w;
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
                case 3:
                    return this->w;
            }
        }

    public: // Default constructors
        constexpr Bool4() noexcept = default;
        constexpr Bool4(const Bool4&) noexcept = default;
        constexpr Bool4(Bool4&&) noexcept = default;

    public: // Conversion constructors
        template<typename X, typename Y, typename Z, typename W>
        NOA_HD constexpr Bool4(X xi, Y yi, Z zi, W wi) noexcept
                : x(static_cast<bool>(xi)),
                  y(static_cast<bool>(yi)),
                  z(static_cast<bool>(zi)),
                  w(static_cast<bool>(wi)) {}

        template<typename U>
        NOA_HD constexpr explicit Bool4(U v) noexcept
                : x(static_cast<bool>(v)),
                  y(static_cast<bool>(v)),
                  z(static_cast<bool>(v)),
                  w(static_cast<bool>(v)) {}

        template<typename U>
        NOA_HD constexpr explicit Bool4(U* ptr)
                : x(static_cast<bool>(ptr[0])),
                  y(static_cast<bool>(ptr[1])),
                  z(static_cast<bool>(ptr[2])),
                  w(static_cast<bool>(ptr[3])) {}

    public: // Assignment operators
        constexpr Bool4& operator=(const Bool4& v) noexcept = default;
        constexpr Bool4& operator=(Bool4&& v) noexcept = default;

        NOA_HD constexpr Bool4& operator=(bool v) noexcept {
            this->x = v;
            this->y = v;
            this->z = v;
            this->w = v;
            return *this;
        }

        NOA_HD constexpr Bool4& operator=(const bool* ptr) noexcept {
            this->x = ptr[0];
            this->y = ptr[1];
            this->z = ptr[2];
            this->z = ptr[3];
            return *this;
        }
    };

    // -- Boolean operators --

    NOA_FHD constexpr Bool4 operator==(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs.x == rhs.x, lhs.y == rhs.y, lhs.z == rhs.z, lhs.w == rhs.w};
    }

    NOA_FHD constexpr Bool4 operator==(Bool4 lhs, bool rhs) noexcept {
        return {lhs.x == rhs, lhs.y == rhs, lhs.z == rhs, lhs.w == rhs};
    }

    NOA_FHD constexpr Bool4 operator==(bool lhs, Bool4 rhs) noexcept {
        return {lhs == rhs.x, lhs == rhs.y, lhs == rhs.z, lhs == rhs.w};
    }

    NOA_FHD constexpr Bool4 operator!=(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs.x != rhs.x, lhs.y != rhs.y, lhs.z != rhs.z, lhs.w != rhs.w};
    }

    NOA_FHD constexpr Bool4 operator!=(Bool4 lhs, bool rhs) noexcept {
        return {lhs.x != rhs, lhs.y != rhs, lhs.z != rhs, lhs.w != rhs};
    }

    NOA_FHD constexpr Bool4 operator!=(bool lhs, Bool4 rhs) noexcept {
        return {lhs != rhs.x, lhs != rhs.y, lhs != rhs.z, lhs != rhs.w};
    }

    NOA_FHD constexpr Bool4 operator&&(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs.x && rhs.x, lhs.y && rhs.y, lhs.z && rhs.z, lhs.w && rhs.w};
    }

    NOA_FHD constexpr Bool4 operator||(Bool4 lhs, Bool4 rhs) noexcept {
        return {lhs.x || rhs.x, lhs.y || rhs.y, lhs.z || rhs.z, lhs.w || rhs.w};
    }

    NOA_FHD constexpr bool any(Bool4 v) noexcept {
        return v.x || v.y || v.z || v.w;
    }

    NOA_FHD constexpr bool all(Bool4 v) noexcept {
        return v.x && v.y && v.z && v.w;
    }

    using bool4_t = Bool4;

    [[nodiscard]] NOA_HOST constexpr std::array<bool, 4> toArray(Bool4 v) noexcept {
        return {v.x, v.y, v.z, v.w};
    }

    template<>
    NOA_IH std::string string::typeName<Bool4>() {
        return "bool4";
    }

    NOA_IH std::ostream& operator<<(std::ostream& os, Bool4 v) {
        os << string::format("({},{},{},{})", v.x, v.y, v.z, v.w);
        return os;
    }

    template<>
    struct traits::proclaim_is_boolX<Bool4> : std::true_type {};
}

namespace fmt {
    template<>
    struct formatter<noa::Bool4> : formatter<bool> {
        template<typename FormatContext>
        auto format(const noa::Bool4& vec, FormatContext& ctx) {
            auto out = ctx.out();
            *out = '(';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec.x, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec.y, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec.z, ctx);
            *out = ',';
            ctx.advance_to(out);
            out = formatter<bool>::format(vec.w, ctx);
            *out = ')';
            return out;
        }
    };
}
