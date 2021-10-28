/// \file noa/common/types/Constants.h
/// \brief Some constants and enums that are common to multiple header/namespaces.
/// \author Thomas - ffyr2w
/// \date 11 Jan 2021

#pragma once

#include <ostream>
#include "noa/common/Definitions.h"

namespace noa {
    /// Border mode, i.e. how out of bounds coordinates are handled. It is compatible with cudaTextureAddressMode.
    enum BorderMode {
        /// The input is extended by wrapping around to the opposite edge.
        /// Maps to cudaAddressModeWrap.
        /// (a b c d | a b c d | a b c d)
        BORDER_PERIODIC = 0,

        /// The input is extended by replicating the last pixel.
        /// Maps to cudaAddressModeClamp.
        /// (a a a a | a b c d | d d d d)
        BORDER_CLAMP = 1,

        /// The input is extended by mirroring the input window.
        /// Maps to cudaAddressModeMirror.
        /// (d c b a | a b c d | d c b a)
        BORDER_MIRROR = 2,

        /// The input is extended by filling all values beyond the edge with zeros.
        /// Maps to cudaAddressModeBorder.
        /// (0 0 0 0 | a b c d | 0 0 0 0)
        BORDER_ZERO = 3,

        /// The input is extended by filling all values beyond the edge with a constant value.
        /// (k k k k | a b c d | k k k k)
        BORDER_VALUE,

        /// The input is extended by reflection, with the center of the operation on the last pixel.
        /// (d c b | a b c d | c b a)
        BORDER_REFLECT,

        /// The input is extended but the values are left unchanged.
        BORDER_NOTHING
    };

    NOA_IH std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
        switch (border_mode) {
            case BorderMode::BORDER_NOTHING:
                return os << "BORDER_NOTHING";
            case BorderMode::BORDER_ZERO:
                return os << "BORDER_ZERO";
            case BorderMode::BORDER_VALUE:
                return os << "BORDER_VALUE";
            case BorderMode::BORDER_CLAMP:
                return os << "BORDER_CLAMP";
            case BorderMode::BORDER_REFLECT:
                return os << "BORDER_REFLECT";
            case BorderMode::BORDER_MIRROR:
                return os << "BORDER_MIRROR";
            case BorderMode::BORDER_PERIODIC:
                return os << "BORDER_PERIODIC";
            default:
                return os;
        }
    }

    /// Returns a valid index.
    /// If \p index is out-of-bound, computes a valid index according to \p MODE. \p len should be > 0.
    template<BorderMode MODE>
    NOA_IHD int getBorderIndex(int idx, int len) {
        static_assert(MODE == BORDER_CLAMP || MODE == BORDER_PERIODIC ||
                      MODE == BORDER_MIRROR || MODE == BORDER_REFLECT);
        // a % b == a - b * (a / b) == a + b * (-a / b)
        // Having a < 0 is well defined since C++11.
        if constexpr (MODE == BORDER_CLAMP) {
            if (idx < 0)
                idx = 0;
            else if (idx >= len)
                idx = len - 1;
        } else if constexpr (MODE == BORDER_PERIODIC) {
            // 0 1 2 3 0 1 2 3 0 1 2 3 |  0 1 2 3  | 0 1 2 3 0 1 2 3 0 1 2 3
            int rem = idx % len; // maybe enclose this, at the expense of two jumps?
            idx = rem < 0 ? rem + len : rem;
        } else if constexpr (MODE == BORDER_MIRROR) {
            // 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0 |  0 1 2 3  | 3 2 1 0 0 1 2 3 3 2 1 0
            if (idx < 0)
                idx = -idx - 1;
            if (idx >= len) {
                int period = 2 * len;
                idx %= period;
                if (idx >= len)
                    idx = period - idx - 1;
            }
        } else if constexpr (MODE == BORDER_REFLECT) {
            // 0 1 2 3 2 1 0 1 2 3 2 1 |  0 1 2 3  | 2 1 0 1 2 3 2 1 0
            if (idx < 0)
                idx = -idx;
            if (idx >= len) {
                int period = 2 * len - 2;
                idx %= period;
                if (idx >= len)
                    idx = period - idx;
            }
        }
        return idx;
    }

    /// Interpolation methods. It is compatible with cudaTextureFilterMode.
    enum InterpMode {
        /// Nearest neighbour interpolation.
        /// Corresponds to cudaFilterModePoint.
        INTERP_NEAREST = 0,

        /// (bi|tri)linear interpolation.
        INTERP_LINEAR = 1,

        /// (bi|tri)linear interpolation with cosine smoothing.
        INTERP_COSINE,

        /// (bi|tri)cubic interpolation.
        INTERP_CUBIC,

        /// (bi|tri)cubic B-spline interpolation.
        INTERP_CUBIC_BSPLINE,

        /// (bi|tri)linear interpolation, using CUDA textures in linear mode.
        /// Faster than INTERP_LINEAR, but at the cost of precision.
        /// Corresponds to cudaFilterModeLinear. Only used in the CUDA backend.
        INTERP_LINEAR_FAST,

        /// (bi|tri)linear interpolation, using CUDA textures in linear mode.
        /// Faster than INTERP_COSINE, but at the cost of precision.
        /// Only used in the CUDA backend.
        INTERP_COSINE_FAST,

        /// (bi|tri)linear interpolation, using CUDA textures in linear mode.
        /// Faster than INTER_CUBIC_BSPLINE, but at the cost of precision.
        /// Only used in the CUDA backend.
        INTERP_CUBIC_BSPLINE_FAST
    };

    NOA_IH std::ostream& operator<<(std::ostream& os, InterpMode interp_mode) {
        switch (interp_mode) {
            case InterpMode::INTERP_NEAREST:
                return os << "INTERP_NEAREST";
            case InterpMode::INTERP_LINEAR:
                return os << "INTERP_LINEAR";
            case InterpMode::INTERP_COSINE:
                return os << "INTERP_COSINE";
            case InterpMode::INTERP_CUBIC:
                return os << "INTERP_CUBIC";
            case InterpMode::INTERP_CUBIC_BSPLINE:
                return os << "INTERP_CUBIC_BSPLINE";
            case InterpMode::INTERP_LINEAR_FAST:
                return os << "INTERP_LINEAR_FAST";
            case InterpMode::INTERP_COSINE_FAST:
                return os << "INTERP_COSINE_FAST";
            case InterpMode::INTERP_CUBIC_BSPLINE_FAST:
                return os << "INTERP_CUBIC_BSPLINE_FAST";
            default:
                return os;
        }
    }
}

namespace noa::fft {
    /// Bitmask encoding a FFT layout.
    /// \b F = redundant, non-centered
    /// \b FC = redundant, centered
    /// \b H = non-redundant, non-centered
    /// \b HC = non-redundant, centered
    ///
    /// \details
    /// \e Centering:
    ///     The "non-centered" layout is used by FFT routines, with the origin (the DC) at index 0.
    ///     The "centered" layout is often used in files, with the origin in the "middle right" (N/2).
    /// \e Redundancy:
    ///     It refers to non-redundant Fourier transforms of real inputs, resulting in transforms with a LOGICAL shape
    ///     of {fast, medium, slow} complex elements and PHYSICAL shapes of {fast/2+1, medium, slow} complex elements.
    ///     Note that with even dimensions, the Nyquist frequency is real and the C2R routines will assume its
    ///     imaginary part is zero.
    ///
    /// \example
    /// \code
    /// n=8: non-centered, redundant     u=[ 0, 1, 2, 3,-4,-3,-2,-1]     note: frequency -4 is real, -4 = 4
    ///      non-centered, non-redundant u=[ 0, 1, 2, 3,-4]
    ///      centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3]
    ///      centered,     non-redundant u=[ 0, 1, 2, 3,-4]
    ///
    /// n=9  non-centered, redundant     u=[ 0, 1, 2, 3, 4,-4,-3,-2,-1]  note: frequency 4 is complex, -4 = conj(4)
    ///      non-centered, non-redundant u=[ 0, 1, 2, 3, 4]
    ///      centered,     redundant     u=[-4,-3,-2,-1, 0, 1, 2, 3, 4]
    ///      centered,     non-redundant u=[ 0, 1, 2, 3, 4]
    /// \endcode
    enum Layout : uint8_t {
        SRC_FULL = 0b00000001,
        DST_FULL = 0b00000010,
        SRC_CENTERED = 0b00000100,
        DST_CENTERED = 0b00001000,
        SRC_FULL_CENTERED = Layout::SRC_FULL | Layout::SRC_CENTERED,
        DST_FULL_CENTERED = Layout::DST_FULL | Layout::DST_CENTERED,
    };

    /// FFT remapping.
    enum Remap : uint8_t {
        H2HC = Layout::DST_CENTERED,
        HC2H = Layout::SRC_CENTERED,
        H2F = Layout::DST_FULL,
        F2H = Layout::SRC_FULL,
        F2FC = Layout::SRC_FULL | Layout::DST_FULL_CENTERED,
        FC2F = Layout::SRC_FULL_CENTERED | Layout::DST_FULL,
        HC2F = Layout::SRC_CENTERED | Layout::DST_FULL,
        F2HC = Layout::SRC_FULL | Layout::DST_CENTERED,
        H2FC = Layout::DST_FULL_CENTERED,
        FC2H = Layout::SRC_FULL_CENTERED,
    };

    NOA_IH std::ostream& operator<<(std::ostream& os, Remap remap) {
        switch (remap) {
            case Remap::H2HC:
                return os << "H2HC";
            case Remap::HC2H:
                return os << "HC2H";
            case Remap::H2F:
                return os << "H2F";
            case Remap::F2H:
                return os << "F2H";
            case Remap::F2FC:
                return os << "F2FC";
            case Remap::FC2F:
                return os << "FC2F";
            case Remap::HC2F:
                return os << "HC2F";
            case Remap::F2HC:
                return os << "F2HC";
            case Remap::H2FC:
                return os << "H2FC";
            case Remap::FC2H:
                return os << "FC2H";
            default:
                return os;
        }
    }
}
