#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/Misc.hpp"
#include "noa/core/utils/Strings.hpp"

namespace noa {
    /// Enum-class-like object encoding an interpolation method.
    /// \note "_FAST" methods allow the use of lerp fetches (e.g. CUDA textures in linear mode) to accelerate the
    ///       interpolation. If textures are not provided to the Interpolator (see below), these methods are equivalent
    ///       to the non-"_FAST" methods. Textures provide multidimensional caching, hardware interpolation
    ///       (nearest or lerp) and addressing (see Border). While it may result in faster computation, textures
    ///       usually encode the floating-point coordinates (usually f32 or f64 values) at which to interpolate using
    ///       low precision representations (e.g. CUDA's textures use 8 bits decimals), thus leading to an overall
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

    /// Border mode, i.e. how out of bounds coordinates are handled.
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

    public: // simplify Interp::Method into Interp
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

    enum class Norm {
        /// Set [min,max] range to [0,1].
        MIN_MAX,

        /// Set the mean to 0, and standard deviation to 1.
        MEAN_STD,

        /// Set the l2_norm to 1.
        L2
    };

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

namespace noa::fft {
    /// Sign of the exponent in the formula that defines the c2c Fourier transform.
    /// Either FORWARD (-1) for the forward/direct transform, or BACKWARD (+1) for the backward/inverse transform.
    enum class Sign : i32 {
        FORWARD = -1,
        BACKWARD = 1
    };

    /// Normalization mode. Indicates which direction of the forward/backward pair of transforms is scaled and
    /// with what normalization factor. ORTHO scales each transform with 1/sqrt(N).
    enum class Norm {
        FORWARD,
        ORTHO,
        BACKWARD,
        NONE
    };
}

namespace noa::signal {
    /// Correlation mode to compute the cross-correlation map.
    enum class Correlation {
        /// Conventional cross-correlation. Generates smooth peaks, but often suffer from
        /// roof-top effect if highpass filter is not strong enough.
        CONVENTIONAL,

        /// Phase-only cross-correlation. Generates sharper, but noisier, peaks.
        PHASE,

        /// Double-phase correlation. Generate similar peaks that the conventional approach,
        /// but is more accurate due to the doubling of the peak position.
        DOUBLE_PHASE,

        /// Good general alternative. Sharper than the conventional approach.
        MUTUAL
    };
}

#ifdef NOA_IS_OFFLINE
namespace noa {
    std::ostream& operator<<(std::ostream& os, Interp interp);
    std::ostream& operator<<(std::ostream& os, Border border);
    std::ostream& operator<<(std::ostream& os, Norm norm);
    std::ostream& operator<<(std::ostream& os, Remap layout);

    inline std::ostream& operator<<(std::ostream& os, Interp::Method interp) { return os << Interp(interp); }
    inline std::ostream& operator<<(std::ostream& os, Border::Method interp) { return os << Border(interp); }
    inline std::ostream& operator<<(std::ostream& os, Remap::Enum remap) { return os << Remap(remap); }
}

namespace noa::signal {
    std::ostream& operator<<(std::ostream& os, Correlation correlation);
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator)
namespace fmt {
    template<> struct formatter<noa::Interp> : ostream_formatter {};
    template<> struct formatter<noa::Border> : ostream_formatter {};
    template<> struct formatter<noa::Norm> : ostream_formatter {};
    template<> struct formatter<noa::signal::Correlation> : ostream_formatter {};
    template<> struct formatter<noa::Remap> : ostream_formatter {};

    template<> struct formatter<noa::Interp::Method> : ostream_formatter {};
    template<> struct formatter<noa::Border::Method> : ostream_formatter {};
    template<> struct formatter<noa::Remap::Enum> : ostream_formatter {};
}
#endif
