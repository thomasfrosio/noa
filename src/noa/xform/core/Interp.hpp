#pragma once

#include "noa/base/Strings.hpp"
#include "noa/base/Utils.hpp"
#include "noa/base/Vec.hpp"

namespace noa::xform {
    /// Enum-class-like object encoding an interpolation method.
    /// \note "_FAST" methods allow the use of lerp fetches (e.g., CUDA textures in linear mode) to speed up the
    ///       interpolation. If textures are not provided to the Interpolator (see below), these methods are equivalent
    ///       to the non-"_FAST" methods. Textures provide multidimensional caching, hardware interpolation
    ///       (nearest or lerp) and addressing (see Border). While it may result in faster computation, textures
    ///       usually encode the floating-point coordinates (usually f32 or f64 values) at which to interpolate using
    ///       low precision representations (e.g. CUDA's textures use 8-bits decimals), thus leading to an overall
    ///       lower precision operation than software interpolation.
    struct Interp {
        enum class Method : i32 {
            /// Nearest neighbor interpolation.
            NEAREST = 0,
            NEAREST_FAST = 100,

            /// Linear interpolation (lerp).
            LINEAR = 1,
            LINEAR_FAST = 101,

            /// Cubic interpolation.
            CUBIC = 2,
            CUBIC_FAST = 102,

            /// Cubic B-spline interpolation.
            CUBIC_BSPLINE = 3,
            CUBIC_BSPLINE_FAST = 103,

            /// Windowed-sinc interpolation, with a Lanczos window of size 4, 6, or 8.
            LANCZOS4 = 4,
            LANCZOS6 = 5,
            LANCZOS8 = 6,
            LANCZOS4_FAST = 104,
            LANCZOS6_FAST = 105,
            LANCZOS8_FAST = 106,
        } value{};

    public: // simplify Interp::Method into Interp
        using enum Method;
        constexpr Interp() noexcept = default;
        NOA_HD constexpr /*implicit*/ Interp(Method value_) noexcept: value(value_) {}
        NOA_HD constexpr /*implicit*/ operator Method() const noexcept { return value; }

    public: // additional methods
        /// Whether the interpolation method is equal to any of the entered values.
        [[nodiscard]] NOA_HD constexpr bool is_any(auto... values) const noexcept {
            return ((value == values) or ...);
        }

        /// Whether the interpolation method, or its (non-)fast alternative, is equal to any of the entered values.
        [[nodiscard]] NOA_HD constexpr bool is_almost_any(auto... values) const noexcept {
            auto underlying = to_underlying(value);
            if (underlying >= 100)
                underlying -= 100;
            auto v = static_cast<Method>(underlying);
            return ((v == values) or ...);
        }

        /// Whether the interpolation method allows fast computation using texture-lerp.
        [[nodiscard]] NOA_HD constexpr bool is_fast() const noexcept {
            return to_underlying(value) >= 100;
        }

        /// Get the size of the interpolation window.
        [[nodiscard]] NOA_HD constexpr auto window_size() const noexcept -> i32 {
            switch (value) {
                case NEAREST:
                case LINEAR:
                case NEAREST_FAST:
                case LINEAR_FAST:
                    return 2;
                case CUBIC:
                case CUBIC_FAST:
                case CUBIC_BSPLINE:
                case CUBIC_BSPLINE_FAST:
                case LANCZOS4:
                case LANCZOS4_FAST:
                    return 4;
                case LANCZOS6:
                case LANCZOS6_FAST:
                    return 6;
                case LANCZOS8:
                case LANCZOS8_FAST:
                    return 8;
            }
            return 0; // unreachable
        }

        [[nodiscard]] NOA_HD constexpr auto erase_fast() const noexcept -> Interp {
            auto underlying = to_underlying(value);
            if (underlying >= 100)
                underlying -= 100;
            return static_cast<Method>(underlying);
        }
    };

    /// Constexpr bitset used to enable and/or disable code generation for interpolation methods.
    struct InterpSet {
        using reference_type = BitReference<u16>;
        u16 bits{};

        static constexpr auto all() -> InterpSet { return {.bits = 0xFFFF}; }
        static constexpr auto none() -> InterpSet { return {.bits = 0x0000}; }

        static constexpr auto all_but(auto... interps) -> InterpSet {
            auto out = all();
            out.disable(interps...);
            return out;
        }

        static constexpr auto none_but(auto... interps) -> InterpSet {
            auto out = none();
            out.enable(interps...);
            return out;
        }

        constexpr auto enable(auto... interps) noexcept -> InterpSet& {
            (((*this)[interps] = true), ...);
            return *this;
        }

        constexpr auto disable(auto... interps) noexcept -> InterpSet& {
            (((*this)[interps] = false), ...);
            return *this;
        }

        constexpr auto enable_fasts() noexcept -> InterpSet& {
            constexpr auto mask = u32{0xFFFF} << 7; // offset to first fast
            auto result = static_cast<u32>(bits);
            result |= mask;
            bits = static_cast<u16>(result);
            return *this;
        }

        constexpr auto disable_fasts() noexcept -> InterpSet& {
            constexpr auto mask = u32{0xFFFF} << 7; // offset to first fast
            auto result = static_cast<u32>(bits);
            result &= ~mask;
            bits = static_cast<u16>(result);
            return *this;
        }

        constexpr auto operator[](Interp interp) const noexcept -> bool {
            switch (interp.value) {
                case Interp::NEAREST:               return reference_type::is_set(&bits, 0);
                case Interp::LINEAR:                return reference_type::is_set(&bits, 1);
                case Interp::CUBIC:                 return reference_type::is_set(&bits, 2);
                case Interp::CUBIC_BSPLINE:         return reference_type::is_set(&bits, 3);
                case Interp::LANCZOS4:              return reference_type::is_set(&bits, 4);
                case Interp::LANCZOS6:              return reference_type::is_set(&bits, 5);
                case Interp::LANCZOS8:              return reference_type::is_set(&bits, 6);
                case Interp::NEAREST_FAST:          return reference_type::is_set(&bits, 7);
                case Interp::LINEAR_FAST:           return reference_type::is_set(&bits, 8);
                case Interp::CUBIC_FAST:            return reference_type::is_set(&bits, 9);
                case Interp::CUBIC_BSPLINE_FAST:    return reference_type::is_set(&bits, 10);
                case Interp::LANCZOS4_FAST:         return reference_type::is_set(&bits, 11);
                case Interp::LANCZOS6_FAST:         return reference_type::is_set(&bits, 12);
                case Interp::LANCZOS8_FAST:         return reference_type::is_set(&bits, 13);
            }
            unreachable();
        }

        constexpr auto operator[](Interp interp) noexcept -> reference_type {
            switch (interp.value) {
                case Interp::NEAREST:               return reference_type(&bits, 0);
                case Interp::LINEAR:                return reference_type(&bits, 1);
                case Interp::CUBIC:                 return reference_type(&bits, 2);
                case Interp::CUBIC_BSPLINE:         return reference_type(&bits, 3);
                case Interp::LANCZOS4:              return reference_type(&bits, 4);
                case Interp::LANCZOS6:              return reference_type(&bits, 5);
                case Interp::LANCZOS8:              return reference_type(&bits, 6);
                case Interp::NEAREST_FAST:          return reference_type(&bits, 7);
                case Interp::LINEAR_FAST:           return reference_type(&bits, 8);
                case Interp::CUBIC_FAST:            return reference_type(&bits, 9);
                case Interp::CUBIC_BSPLINE_FAST:    return reference_type(&bits, 10);
                case Interp::LANCZOS4_FAST:         return reference_type(&bits, 11);
                case Interp::LANCZOS6_FAST:         return reference_type(&bits, 12);
                case Interp::LANCZOS8_FAST:         return reference_type(&bits, 13);
            }
            unreachable();
        }
    };
}

namespace noa::xform {
    inline auto operator<<(std::ostream& os, Interp interp) -> std::ostream& {
        switch (interp) {
            case Interp::NEAREST:
                return os << "Interp::NEAREST";
            case Interp::NEAREST_FAST:
                return os << "Interp::NEAREST_FAST";
            case Interp::LINEAR:
                return os << "Interp::LINEAR";
            case Interp::LINEAR_FAST:
                return os << "Interp::LINEAR_FAST";
            case Interp::CUBIC:
                return os << "Interp::CUBIC";
            case Interp::CUBIC_FAST:
                return os << "Interp::CUBIC_FAST";
            case Interp::CUBIC_BSPLINE:
                return os << "Interp::CUBIC_BSPLINE";
            case Interp::CUBIC_BSPLINE_FAST:
                return os << "Interp::CUBIC_BSPLINE_FAST";
            case Interp::LANCZOS4:
                return os << "Interp::LANCZOS4";
            case Interp::LANCZOS6:
                return os << "Interp::LANCZOS6";
            case Interp::LANCZOS8:
                return os << "Interp::LANCZOS8";
            case Interp::LANCZOS4_FAST:
                return os << "Interp::LANCZOS4_FAST";
            case Interp::LANCZOS6_FAST:
                return os << "Interp::LANCZOS6_FAST";
            case Interp::LANCZOS8_FAST:
                return os << "Interp::LANCZOS8_FAST";
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, Interp::Method interp) { return os << Interp(interp); }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator)
namespace fmt {
    template<> struct formatter<noa::xform::Interp> : ostream_formatter {};
    template<> struct formatter<noa::xform::Interp::Method> : ostream_formatter {};
}
