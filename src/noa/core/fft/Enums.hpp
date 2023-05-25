#pragma once

#include "noa/core/string/Format.hpp"

namespace noa::fft {
    /// Sign of the exponent in the formula that defines the Fourier transform.
    /// Either FORWARD (-1) for the forward/direct transform, or BACKWARD (+1) for the backward/inverse transform.
    enum class Sign : int32_t {
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

    /// Bitmask encoding a FFT layout.
    /// \details
    /// \e Centering:
    ///     The "non-centered" (or native) layout is used by fft routines, with the origin (the DC) at index 0.
    ///     The "centered" layout is often used in files or for geometric transformations, with the origin at index n//2.
    /// \e Redundancy:
    ///     It refers to non-redundant Fourier transforms of real inputs (aka rfft). For a (D,H,W) logical/real shape,
    ///     rffts have a shape of {D,H,W//2+1}. Note that with even dimensions, the Nyquist frequency is real and
    ///     the c2r (aka irfft) functions will assume its imaginary part is zero.
    ///
    /// \example
    /// \code
    /// n=8: non-centered, redundant     u=[ 0, 1, 2, 3,-4,-3,-2,-1]
    ///      non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3]
    ///      centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    ///      note: frequency -4 is real, -4 = 4
    ///
    /// n=9  non-centered, redundant     u=[ 0, 1, 2, 3, 4,-4,-3,-2,-1]
    ///      non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3, 4]
    ///      centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    ///      note: frequency 4 is complex (it's not nyquist), -4 = conj(4)
    /// \endcode
    enum Layout : uint8_t {
        SRC_HALF =          0b00000001,
        SRC_FULL =          0b00000010,
        DST_HALF =          0b00000100,
        DST_FULL =          0b00001000,
        SRC_CENTERED =      0b00010000,
        SRC_NON_CENTERED =  0b00100000,
        DST_CENTERED =      0b01000000,
        DST_NON_CENTERED =  0b10000000,
    };

    /// FFT remapping.
    /// \b F = redundant, non-centered
    /// \b FC = redundant, centered
    /// \b H = non-redundant, non-centered
    /// \b HC = non-redundant, centered
    /// \note With H2H, HC2HC, F2F and FC2FC, there's actually no remapping.
    enum Remap : uint8_t {
        H2H = Layout::SRC_HALF | Layout::SRC_NON_CENTERED | Layout::DST_HALF | Layout::DST_NON_CENTERED,
        H2HC = Layout::SRC_HALF | Layout::SRC_NON_CENTERED | Layout::DST_HALF | Layout::DST_CENTERED,
        H2F = Layout::SRC_HALF | Layout::SRC_NON_CENTERED | Layout::DST_FULL | Layout::DST_NON_CENTERED,
        H2FC = Layout::SRC_HALF | Layout::SRC_NON_CENTERED | Layout::DST_FULL | Layout::DST_CENTERED,

        HC2H = Layout::SRC_HALF | Layout::SRC_CENTERED | Layout::DST_HALF | Layout::DST_NON_CENTERED,
        HC2HC = Layout::SRC_HALF | Layout::SRC_CENTERED | Layout::DST_HALF | Layout::DST_CENTERED,
        HC2F = Layout::SRC_HALF | Layout::SRC_CENTERED | Layout::DST_FULL | Layout::DST_NON_CENTERED,
        HC2FC = Layout::SRC_HALF | Layout::SRC_CENTERED | Layout::DST_FULL | Layout::DST_CENTERED,

        F2H = Layout::SRC_FULL | Layout::SRC_NON_CENTERED | Layout::DST_HALF | Layout::DST_NON_CENTERED,
        F2HC = Layout::SRC_FULL | Layout::SRC_NON_CENTERED | Layout::DST_HALF | Layout::DST_CENTERED,
        F2F = Layout::SRC_FULL | Layout::SRC_NON_CENTERED | Layout::DST_FULL | Layout::DST_NON_CENTERED,
        F2FC = Layout::SRC_FULL | Layout::SRC_NON_CENTERED | Layout::DST_FULL | Layout::DST_CENTERED,

        FC2H = Layout::SRC_FULL | Layout::SRC_CENTERED | Layout::DST_HALF | Layout::DST_NON_CENTERED,
        FC2HC = Layout::SRC_FULL | Layout::SRC_CENTERED | Layout::DST_HALF | Layout::DST_CENTERED,
        FC2F = Layout::SRC_FULL | Layout::SRC_CENTERED | Layout::DST_FULL | Layout::DST_NON_CENTERED,
        FC2FC = Layout::SRC_FULL | Layout::SRC_CENTERED | Layout::DST_FULL | Layout::DST_CENTERED
    };

    inline std::ostream& operator<<(std::ostream& os, Remap remap) {
        switch (remap) {
            case Remap::H2H:
                return os << "Remap::H2H";
            case Remap::H2HC:
                return os << "Remap::H2HC";
            case Remap::H2F:
                return os << "Remap::H2F";
            case Remap::H2FC:
                return os << "Remap::H2FC";
            case Remap::HC2H:
                return os << "Remap::HC2H";
            case Remap::HC2HC:
                return os << "Remap::HC2HC";
            case Remap::HC2F:
                return os << "Remap::HC2F";
            case Remap::HC2FC:
                return os << "Remap::HC2FC";
            case Remap::F2H:
                return os << "Remap::F2H";
            case Remap::F2HC:
                return os << "Remap::F2HC";
            case Remap::F2F:
                return os << "Remap::F2F";
            case Remap::F2FC:
                return os << "Remap::F2FC";
            case Remap::FC2H:
                return os << "Remap::FC2H";
            case Remap::FC2HC:
                return os << "Remap::FC2HC";
            case Remap::FC2F:
                return os << "Remap::FC2F";
            case Remap::FC2FC:
                return os << "Remap::FC2FC";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::fft::Remap> : ostream_formatter {};
}
