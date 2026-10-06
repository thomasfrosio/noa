#pragma once

#include <iostream>

#include "noa/base/Config.hpp"
#include "noa/base/Error.hpp"
#include "noa/base/Strings.hpp"
#include "noa/base/Traits.hpp"
#include "noa/base/Utils.hpp"

namespace noa {
    /// Border mode, i.e. how out-of-bounds indices/coordinates are handled.
    struct Border {
        enum class Method : i32 {
            /// The input is extended by wrapping around to the opposite edge.
            /// (a b c d | a b c d | a b c d)
            PERIODIC = 0,

            /// The input is extended by replicating the last pixel.
            /// (a a a a | a b c d | d d d d)
            CLAMP = 1,

            /// The input is extended by mirroring the input window.
            /// (d c b a | a b c d | d c b a)
            MIRROR = 2,

            /// The input is extended by filling all values beyond the edge with zeros.
            /// (0 0 0 0 | a b c d | 0 0 0 0)
            ZERO = 3,

            /// The input is extended by filling all values beyond the edge with a constant value.
            /// (k k k k | a b c d | k k k k)
            VALUE,

            /// The input is extended by reflection, with the center of the operation on the last pixel.
            /// (d c b | a b c d | c b a)
            REFLECT,

            /// The out-of-bound values are left unchanged.
            NOTHING
        } value;

    public:
        using enum Method;
        constexpr Border() noexcept = default;
        NOA_HD constexpr /*implicit*/ Border(Method value_) noexcept: value(value_) {}
        NOA_HD constexpr /*implicit*/ operator Method() const noexcept { return value; }

    public: // additional methods
        /// Whether the interpolation method is equal to any of the entered values.
        [[nodiscard]] NOA_HD constexpr bool is_any(auto... values) const noexcept {
            return ((value == values) or ...);
        }

        [[nodiscard]] NOA_HD constexpr bool is_finite() const noexcept {
            return is_any(ZERO, VALUE, NOTHING);
        }
    };

    struct BorderSet {
        using reference_type = BitReference<u8>;
        u8 bits{};

        static constexpr auto all() -> BorderSet { return {.bits = 0xFF}; }
        static constexpr auto none() -> BorderSet { return {.bits = 0x00}; }

        static constexpr auto all_but(auto... interps) -> BorderSet {
            auto out = all();
            out.disable(interps...);
            return out;
        }

        static constexpr auto none_but(auto... interps) -> BorderSet {
            auto out = none();
            out.enable(interps...);
            return out;
        }

        constexpr auto enable(auto... interps) noexcept -> BorderSet& {
            (((*this)[interps] = true), ...);
            return *this;
        }

        constexpr auto disable(auto... interps) noexcept -> BorderSet& {
            (((*this)[interps] = false), ...);
            return *this;
        }

        constexpr auto operator[](Border border) const noexcept -> bool {
            switch (border.value) {
                case Border::PERIODIC:  return reference_type::is_set(&bits, 0);
                case Border::CLAMP:     return reference_type::is_set(&bits, 1);
                case Border::MIRROR:    return reference_type::is_set(&bits, 2);
                case Border::ZERO:      return reference_type::is_set(&bits, 3);
                case Border::VALUE:     return reference_type::is_set(&bits, 4);
                case Border::REFLECT:   return reference_type::is_set(&bits, 5);
                case Border::NOTHING:   return reference_type::is_set(&bits, 6);
            }
            unreachable();
        }

        constexpr auto operator[](Border border) noexcept -> reference_type {
            switch (border.value) {
                case Border::PERIODIC:  return reference_type(&bits, 0);
                case Border::CLAMP:     return reference_type(&bits, 1);
                case Border::MIRROR:    return reference_type(&bits, 2);
                case Border::ZERO:      return reference_type(&bits, 3);
                case Border::VALUE:     return reference_type(&bits, 4);
                case Border::REFLECT:   return reference_type(&bits, 5);
                case Border::NOTHING:   return reference_type(&bits, 6);
            }
            unreachable();
        }
    };
}

namespace noa {
    inline auto operator<<(std::ostream& os, Border border) -> std::ostream& {
        switch (border) {
            case Border::NOTHING:
                return os << "Border::NOTHING";
            case Border::ZERO:
                return os << "Border::ZERO";
            case Border::VALUE:
                return os << "Border::VALUE";
            case Border::CLAMP:
                return os << "Border::CLAMP";
            case Border::REFLECT:
                return os << "Border::REFLECT";
            case Border::MIRROR:
                return os << "Border::MIRROR";
            case Border::PERIODIC:
                return os << "Border::PERIODIC";
        }
        return os;
    }

    inline auto operator<<(std::ostream& os, Border::Method interp) -> std::ostream& { return os << Border(interp); }
}

namespace fmt {
    template<> struct formatter<noa::Border> : ostream_formatter {};
    template<> struct formatter<noa::Border::Method> : ostream_formatter {};
}
