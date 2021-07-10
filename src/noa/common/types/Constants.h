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
        std::string buffer;
        switch (border_mode) {
            case BORDER_NOTHING:
                buffer = "BORDER_NOTHING";
                break;
            case BORDER_ZERO:
                buffer = "BORDER_ZERO";
                break;
            case BORDER_VALUE:
                buffer = "BORDER_VALUE";
                break;
            case BORDER_CLAMP:
                buffer = "BORDER_CLAMP";
                break;
            case BORDER_REFLECT:
                buffer = "BORDER_REFLECT";
                break;
            case BORDER_MIRROR:
                buffer = "BORDER_MIRROR";
                break;
            case BORDER_PERIODIC:
                buffer = "BORDER_PERIODIC";
                break;
            default:
                buffer = "UNKNOWN";
        }
        os << buffer;
        return os;
    }

    /// Interpolation methods. It is compatible with cudaTextureFilterMode.
    enum InterpMode {
        /// Nearest neighbour interpolation.
        INTERP_NEAREST = 0,

        /// (bi|tri)linear interpolation.
        INTERP_LINEAR = 1,

        /// (bi|tri)linear interpolation with cosine smoothing.
        INTERP_COSINE = 2,

        /// (bi|tri)cubic interpolation.
        INTERP_CUBIC = 3,

        /// (bi|tri)cubic B-spline interpolation.
        INTERP_CUBIC_BSPLINE = 4
    };

    NOA_IH std::ostream& operator<<(std::ostream& os, InterpMode interp_mode) {
        std::string buffer;
        switch (interp_mode) {
            case INTERP_NEAREST:
                buffer = "INTERP_NEAREST";
                break;
            case INTERP_LINEAR:
                buffer = "INTERP_LINEAR";
                break;
            case INTERP_COSINE:
                buffer = "INTERP_COSINE";
                break;
            case INTERP_CUBIC:
                buffer = "INTERP_CUBIC";
                break;
            case INTERP_CUBIC_BSPLINE:
                buffer = "INTERP_CUBIC_BSPLINE";
                break;
            default:
                buffer = "UNKNOWN";
        }
        os << buffer;
        return os;
    }
}