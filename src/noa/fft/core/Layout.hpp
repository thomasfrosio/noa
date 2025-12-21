#pragma once

#include "noa/base/Strings.hpp"
#include "noa/base/Traits.hpp"

namespace noa::fft {
    /// Enum-like type encoding the FFT layout of an operation.
    ///
    /// \details
    /// FFT arrays have four possible layouts: f, fc, h, hc, where f=redundant, h=non-redundant, c=centered.
    /// The redundancy refers to non-redundant Fourier transforms of real inputs (aka rFFT). For an array of shape
    /// (B,D,H,W), the rFFT has a shape of (B,D,H,W//2+1). The centering refers to how the frequencies are mapped along
    /// each dimension. The non-centered layout has the origin (the DC) at index 0. The centered layout is often used
    /// for visualization or for some geometric transformations and has the origin at index n//2.
    ///
    /// \details
    /// For functions accepting input and output arrays, and when the FFT layout of the input(s) and output(s)
    /// changes, the full layout is required. It is specified as [input-layout]2[output-layout], so {f|h}(c)2{f|h}(c).
    /// For example, "f2fc" denotes the common fftshift operation for redundant (aka full) FFTs. "h2hc" is the
    /// equivalent but for rFFTs. Note that "h" is equivalent to "h2h", i.e., the input and output are both in the
    /// h-layout (non-redundant, non-centered).
    ///
    /// \note Position of the Fourier components:
    /// \code
    /// n=8: F:  non-centered, redundant     u=[ 0, 1, 2, 3,-4,-3,-2,-1]
    ///      H:  non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      FC: centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3]
    ///      HC: centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    ///      note: frequency -4 is real, so -4 = 4
    ///
    /// n=9  F:  non-centered, redundant     u=[ 0, 1, 2, 3, 4,-4,-3,-2,-1]
    ///      H:  non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      FC: centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3, 4]
    ///      HC: centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    ///      note: frequency 4 is complex (it's not nyquist), -4 = conj(4)
    /// \endcode
    class Layout {
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
        constexpr /*implicit*/ Layout(Enum value_) noexcept : value(value_) {}
        constexpr /*implicit*/ operator Enum() const noexcept { return value; }

    public: // additional constructors and member functions
        static constexpr auto from_string(std::string_view name) -> Layout {
            if (name == "h2h" or name == "H2H" or name == "h" or name == "H") {
                return H2H;
            } else if (name == "h2hc" or name == "H2HC") {
                return H2HC;
            } else if (name == "h2f" or name == "H2F") {
                return H2F;
            } else if (name == "h2fc" or name == "H2FC") {
                return H2FC;

            } else if (name == "hc2h" or name == "HC2H") {
                return HC2H;
            } else if (name == "hc2hc" or name == "HC2HC" or name == "hc" or name == "HC") {
                return HC2HC;
            } else if (name == "hc2f" or name == "HC2F") {
                return HC2F;
            } else if (name == "hc2fc" or name == "HC2FC") {
                return HC2FC;

            } else if (name == "f2h" or name == "F2H") {
                return F2H;
            } else if (name == "f2hc" or name == "F2HC") {
                return F2HC;
            } else if (name == "f2f" or name == "F2F" or name == "f" or name == "F") {
                return F2F;
            } else if (name == "f2fc" or name == "F2FC") {
                return F2FC;

            } else if (name == "fc2h" or name == "FC2H") {
                return FC2H;
            } else if (name == "fc2hc" or name == "FC2HC") {
                return FC2HC;
            } else if (name == "fc2f" or name == "FC2F") {
                return FC2F;
            } else if (name == "fc2fc" or name == "FC2FC" or name == "fc" or name == "FC") {
                return FC2FC;

            } else {
                // If it is a constant expression, this creates a compile time error because throwing
                // an exception at compile time is not allowed. At runtime, it throws the exception.
                panic("invalid layout");
            }
        }

        /// Implicit constructor from string literal.
        template<usize N>
        constexpr /*implicit*/ Layout(const char (& name)[N]) {
            value = from_string(std::string_view(name)).value;
        }

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

        [[nodiscard]] constexpr bool is_f()  const noexcept { return is_f2xx() and is_xx2f(); }
        [[nodiscard]] constexpr bool is_fc()  const noexcept { return is_fc2xx() and is_xx2fc(); }
        [[nodiscard]] constexpr bool is_h()  const noexcept { return is_h2xx() and is_xx2h(); }
        [[nodiscard]] constexpr bool is_hc()  const noexcept { return is_hc2xx() and is_xx2hc(); }


        /// Convert to non-centered output.
        [[nodiscard]] constexpr Layout to_xx2x() const noexcept {
            return static_cast<Enum>(to_u8_() & (~Bitset::DST_CENTERED));
        }

        /// Convert to centered output.
        [[nodiscard]] constexpr Layout to_xx2xc() const noexcept {
            return static_cast<Enum>(to_u8_() | Bitset::DST_CENTERED);
        }

        /// Convert to non-centered input.
        [[nodiscard]] constexpr Layout to_x2xx() const noexcept {
            return static_cast<Enum>(to_u8_() & (~Bitset::SRC_CENTERED));
        }

        /// Convert to centered input.
        [[nodiscard]] constexpr Layout to_xc2xx() const noexcept {
            return static_cast<Enum>(to_u8_() | Bitset::SRC_CENTERED);
        }

        [[nodiscard]] constexpr Layout flip() const noexcept {
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

        [[nodiscard]] constexpr Layout erase_input() const noexcept {
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

        [[nodiscard]] constexpr Layout erase_output() const noexcept {
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

namespace noa::fft {
    inline auto operator<<(std::ostream& os, Layout layout) -> std::ostream& {
        switch (layout) {
            case Layout::H2H:
                return os << "h2h";
            case Layout::H2HC:
                return os << "h2hc";
            case Layout::H2F:
                return os << "h2f";
            case Layout::H2FC:
                return os << "h2fc";
            case Layout::HC2H:
                return os << "hc2h";
            case Layout::HC2HC:
                return os << "hc2hc";
            case Layout::HC2F:
                return os << "hc2f";
            case Layout::HC2FC:
                return os << "hc2fc";
            case Layout::F2H:
                return os << "f2h";
            case Layout::F2HC:
                return os << "f2hc";
            case Layout::F2F:
                return os << "f2f";
            case Layout::F2FC:
                return os << "f2fc";
            case Layout::FC2H:
                return os << "fc2h";
            case Layout::FC2HC:
                return os << "fc2hc";
            case Layout::FC2F:
                return os << "fc2f";
            case Layout::FC2FC:
                return os << "fc2fc";
        }
        return os;
    }

    inline auto operator<<(std::ostream& os, Layout::Enum remap) -> std::ostream& { return os << Layout(remap); }
}

namespace fmt {
    template<> struct formatter<noa::fft::Layout> : ostream_formatter {};
    template<> struct formatter<noa::fft::Layout::Enum> : ostream_formatter {};
}
