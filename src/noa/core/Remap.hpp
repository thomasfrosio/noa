#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Enums.hpp"
#include "noa/core/Exception.hpp"

#ifdef NOA_IS_OFFLINE
#include <string_view>
#endif

namespace noa::inline types {
    /// Enum-like type encoding two FFT layouts, referred to as "remap" in the context of input and output layouts.
    /// \details The format is \c {f|h}(c)2{f|h}(c), where f=redundant, h=non-redundant/rfft, c=centered. For example,
    ///          "f2fc" denotes the common fftshift operation and h2hc is the equivalent but for rffts.
    ///          The "non-centered" (or native) layout is used by fft functions, with the origin (the DC) at index 0.
    ///          The "centered" layout is often used in files or for some transformations, with the origin at
    ///          index n//2. The redundancy refers to non-redundant Fourier transforms of real inputs (aka rfft).
    ///          For a {D,H,W} logical/real shape, rffts have a shape of {D,H,W//2+1}. Note that with even dimensions,
    ///          the "highest" frequency (just before Nyquist) is real and the c2r (aka irfft) functions will assume
    ///          its imaginary part is zero.
    ///
    /// \note Position of the Fourier components:
    /// \code
    /// n=8: non-centered, redundant     u=[ 0, 1, 2, 3,-4,-3,-2,-1]
    ///      non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3]
    ///      centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    ///      note: frequency -4 is real, so -4 = 4
    ///
    /// n=9  non-centered, redundant     u=[ 0, 1, 2, 3, 4,-4,-3,-2,-1]
    ///      non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3, 4]
    ///      centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    ///      note: frequency 4 is complex (it's not nyquist), -4 = conj(4)
    /// \endcode
    class Remap {
    public:
        struct Bitset {
            enum : u8 {
                SRC_FULL =     0b00000001, // defaults to half
                DST_FULL =     0b00000010, // defaults to half
                SRC_CENTERED = 0b00000100, // defaults to non-centered
                DST_CENTERED = 0b00001000, // defaults to non-centered
            };
        };

        enum class Enum : u8 {
            H2H = 0,
            H2HC = Bitset::DST_CENTERED,
            H2F = Bitset::DST_FULL,
            H2FC = Bitset::DST_FULL | Bitset::DST_CENTERED,

            HC2H = Bitset::SRC_CENTERED,
            HC2HC = Bitset::SRC_CENTERED | Bitset::DST_CENTERED,
            HC2F = Bitset::SRC_CENTERED | Bitset::DST_FULL,
            HC2FC = Bitset::SRC_CENTERED | Bitset::DST_FULL | Bitset::DST_CENTERED,

            F2H = Bitset::SRC_FULL,
            F2HC = Bitset::SRC_FULL | Bitset::DST_CENTERED,
            F2F = Bitset::SRC_FULL | Bitset::DST_FULL,
            F2FC = Bitset::SRC_FULL | Bitset::DST_FULL | Bitset::DST_CENTERED,

            FC2H = Bitset::SRC_FULL | Bitset::SRC_CENTERED,
            FC2HC = Bitset::SRC_FULL | Bitset::SRC_CENTERED | Bitset::DST_CENTERED,
            FC2F = Bitset::SRC_FULL | Bitset::SRC_CENTERED | Bitset::DST_FULL,
            FC2FC = Bitset::SRC_FULL | Bitset::SRC_CENTERED | Bitset::DST_FULL | Bitset::DST_CENTERED
        } value;

    public: // behave like an enum class
        using enum Enum;
        constexpr /*implicit*/ Remap(Enum value_) noexcept : value(value_) {}
        constexpr /*implicit*/ operator Enum() const noexcept { return value; }

    public: // additional constructors and member functions
#ifdef NOA_IS_OFFLINE
        /// Implicit constructor from string literal.
        template<size_t N>
        constexpr /*implicit*/ Remap(const char (& name)[N]) {
            std::string_view name_(name);
            if (name_ == "h2h" or name_ == "H2H") {
                value = H2H;
            } else if (name_ == "h2hc" or name_ == "H2HC") {
                value = H2HC;
            } else if (name_ == "h2f" or name_ == "H2F") {
                value = H2F;
            } else if (name_ == "h2fc" or name_ == "H2FC") {
                value = H2FC;

            } else if (name_ == "hc2h" or name_ == "HC2H") {
                value = HC2H;
            } else if (name_ == "hc2hc" or name_ == "HC2HC") {
                value = HC2HC;
            } else if (name_ == "hc2f" or name_ == "HC2F") {
                value = HC2F;
            } else if (name_ == "hc2fc" or name_ == "HC2FC") {
                value = HC2FC;

            } else if (name_ == "f2h" or name_ == "F2H") {
                value = F2H;
            } else if (name_ == "f2hc" or name_ == "F2HC") {
                value = F2HC;
            } else if (name_ == "f2f" or name_ == "F2F") {
                value = F2F;
            } else if (name_ == "f2fc" or name_ == "F2FC") {
                value = F2FC;

            } else if (name_ == "fc2h" or name_ == "FC2H") {
                value = FC2H;
            } else if (name_ == "fc2hc" or name_ == "FC2HC") {
                value = FC2HC;
            } else if (name_ == "fc2f" or name_ == "FC2F") {
                value = FC2F;
            } else if (name_ == "fc2fc" or name_ == "FC2FC") {
                value = FC2FC;

            } else {
                // If it is a constant expression, this creates a compile time error because throwing
                // an exception at compile time is not allowed. At runtime, it throws the exception.
                panic("invalid layout");
            }
        }
#endif

        /// Whether the remap is one of these values.
        [[nodiscard]] constexpr bool is_any(auto... values) const noexcept {
            return ((value == values) or ...);
        }

        /// Whether the layout changes between the input and output.
        [[nodiscard]] constexpr bool has_layout_change() const noexcept {
            return not is_any(H2H, HC2HC, F2F, FC2FC);
        }

        [[nodiscard]] constexpr bool is_fx2xx()  const noexcept { return to_u8_() & Bitset::SRC_FULL; }
        [[nodiscard]] constexpr bool is_xx2fx()  const noexcept { return to_u8_() & Bitset::DST_FULL; }
        [[nodiscard]] constexpr bool is_hx2xx()  const noexcept { return not is_fx2xx(); }
        [[nodiscard]] constexpr bool is_xx2hx()  const noexcept { return not is_xx2fx(); }

        [[nodiscard]] constexpr bool is_hx2hx()  const noexcept { return is_hx2xx() and is_xx2hx(); }
        [[nodiscard]] constexpr bool is_fx2fx()  const noexcept { return is_fx2xx() and is_xx2fx(); }
        [[nodiscard]] constexpr bool is_hx2fx()  const noexcept { return is_hx2xx() and is_xx2fx(); }
        [[nodiscard]] constexpr bool is_fx2hx()  const noexcept { return is_fx2xx() and is_xx2hx(); }

        [[nodiscard]] constexpr bool is_xc2xx()  const noexcept { return to_u8_() & Bitset::SRC_CENTERED; }
        [[nodiscard]] constexpr bool is_xx2xc()  const noexcept { return to_u8_() & Bitset::DST_CENTERED; }

        [[nodiscard]] constexpr bool is_hc2xx() const noexcept { return is_hx2xx() and is_xc2xx(); }
        [[nodiscard]] constexpr bool is_h2xx()  const noexcept { return is_hx2xx() and not is_xc2xx(); }
        [[nodiscard]] constexpr bool is_fc2xx() const noexcept { return is_fx2xx() and is_xc2xx(); }
        [[nodiscard]] constexpr bool is_f2xx()  const noexcept { return is_fx2xx() and not is_xc2xx(); }
        [[nodiscard]] constexpr bool is_xx2hc() const noexcept { return is_xx2hx() and is_xx2xc(); }
        [[nodiscard]] constexpr bool is_xx2h()  const noexcept { return is_xx2hx() and not is_xx2xc(); }
        [[nodiscard]] constexpr bool is_xx2fc() const noexcept { return is_xx2fx() and is_xx2xc(); }
        [[nodiscard]] constexpr bool is_xx2f()  const noexcept { return is_xx2fx() and not is_xx2xc(); }

        /// Convert to non-centered output.
        [[nodiscard]] constexpr Remap to_xx2x() const noexcept {
            return static_cast<Enum>(to_u8_() & (~Bitset::DST_CENTERED));
        }

        /// Convert to centered output.
        [[nodiscard]] constexpr Remap to_xx2xc() const noexcept {
            return static_cast<Enum>(to_u8_() | Bitset::DST_CENTERED);
        }

        /// Convert to non-centered input.
        [[nodiscard]] constexpr Remap to_x2xx() const noexcept {
            return static_cast<Enum>(to_u8_() & (~Bitset::SRC_CENTERED));
        }

        /// Convert to centered input.
        [[nodiscard]] constexpr Remap to_xc2xx() const noexcept {
            return static_cast<Enum>(to_u8_() | Bitset::SRC_CENTERED);
        }

        [[nodiscard]] constexpr Remap flip() const noexcept {
            u8 set{};
            if (to_u8_() & Bitset::SRC_FULL)
                set |= Bitset::DST_FULL;
            if (to_u8_() & Bitset::DST_FULL)
                set |= Bitset::SRC_FULL;
            if (to_u8_() & Bitset::SRC_CENTERED)
                set |= Bitset::DST_CENTERED;
            if (to_u8_() & Bitset::DST_CENTERED)
                set |= Bitset::SRC_CENTERED;
            return static_cast<Enum>(set);
        }

        [[nodiscard]] constexpr Remap erase_input() const noexcept {
            u8 set{};
            if (to_u8_() & Bitset::DST_FULL) {
                set |= Bitset::SRC_FULL;
                set |= Bitset::DST_FULL;
            }
            if (to_u8_() & Bitset::DST_CENTERED) {
                set |= Bitset::SRC_CENTERED;
                set |= Bitset::DST_CENTERED;
            }
            return static_cast<Enum>(set);
        }

        [[nodiscard]] constexpr Remap erase_output() const noexcept {
            u8 set{};
            if (to_u8_() & Bitset::SRC_FULL) {
                set |= Bitset::SRC_FULL;
                set |= Bitset::DST_FULL;
            }
            if (to_u8_() & Bitset::SRC_CENTERED) {
                set |= Bitset::SRC_CENTERED;
                set |= Bitset::DST_CENTERED;
            }
            return static_cast<Enum>(set);
        }

    private:
        [[nodiscard]] constexpr u8 to_u8_() const noexcept {
            return static_cast<u8>(value);
        }
    };
}

#ifdef NOA_IS_OFFLINE
namespace noa::inline types {
    inline std::ostream& operator<<(std::ostream& os, Remap::Enum layout) {
        switch (layout) {
            case Remap::H2H:
                return os << "h2h";
            case Remap::H2HC:
                return os << "h2hc";
            case Remap::H2F:
                return os << "h2f";
            case Remap::H2FC:
                return os << "h2fc";
            case Remap::HC2H:
                return os << "hc2h";
            case Remap::HC2HC:
                return os << "hc2hc";
            case Remap::HC2F:
                return os << "hc2f";
            case Remap::HC2FC:
                return os << "hc2fc";
            case Remap::F2H:
                return os << "f2h";
            case Remap::F2HC:
                return os << "f2hc";
            case Remap::F2F:
                return os << "f2f";
            case Remap::F2FC:
                return os << "f2fc";
            case Remap::FC2H:
                return os << "fc2h";
            case Remap::FC2HC:
                return os << "fc2hc";
            case Remap::FC2F:
                return os << "fc2f";
            case Remap::FC2FC:
                return os << "fc2fc";
        }
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, Remap remap) {
        return os << remap.value;
    }
}
namespace fmt {
    template<> struct formatter<noa::Remap::Enum> : ostream_formatter {};
    template<> struct formatter<noa::Remap> : ostream_formatter {};
}
#endif
