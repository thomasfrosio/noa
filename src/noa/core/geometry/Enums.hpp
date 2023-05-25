#pragma once

#include "noa/core/string/Format.hpp"

namespace noa {
    /// Border mode, i.e. how out of bounds coordinates are handled. It is compatible with cudaTextureAddressMode.
    enum class BorderMode {
        /// The input is extended by wrapping around to the opposite edge.
        /// Equal to cudaAddressModeWrap.
        /// (a b c d | a b c d | a b c d)
        PERIODIC = 0,

        /// The input is extended by replicating the last pixel.
        /// Equal to cudaAddressModeClamp.
        /// (a a a a | a b c d | d d d d)
        CLAMP = 1,

        /// The input is extended by mirroring the input window.
        /// Equal to cudaAddressModeMirror.
        /// (d c b a | a b c d | d c b a)
        MIRROR = 2,

        /// The input is extended by filling all values beyond the edge with zeros.
        /// Equal to cudaAddressModeBorder.
        /// (0 0 0 0 | a b c d | 0 0 0 0)
        ZERO = 3,

        /// The input is extended by filling all values beyond the edge with a constant value.
        /// (k k k k | a b c d | k k k k)
        VALUE,

        /// The input is extended by reflection, with the center of the operation on the last pixel.
        /// (d c b | a b c d | c b a)
        REFLECT,

        /// The input is extended but the values are left unchanged.
        NOTHING
    };
    inline std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
        switch (border_mode) {
            case BorderMode::NOTHING:
                return os << "BorderMode::NOTHING";
            case BorderMode::ZERO:
                return os << "BorderMode::ZERO";
            case BorderMode::VALUE:
                return os << "BorderMode::VALUE";
            case BorderMode::CLAMP:
                return os << "BorderMode::CLAMP";
            case BorderMode::REFLECT:
                return os << "BorderMode::REFLECT";
            case BorderMode::MIRROR:
                return os << "BorderMode::MIRROR";
            case BorderMode::PERIODIC:
                return os << "BorderMode::PERIODIC";
        }
        return os;
    }

    /// Interpolation methods.
    enum class InterpMode {
        /// Nearest neighbour interpolation.
        /// Corresponds to cudaFilterModePoint.
        NEAREST = 0,

        /// (bi|tri)linear interpolation.
        LINEAR = 1,

        /// (bi|tri)linear interpolation with cosine smoothing.
        COSINE,

        /// (bi|tri)cubic interpolation.
        CUBIC,

        /// (bi|tri)cubic B-spline interpolation.
        CUBIC_BSPLINE,

        /// (bi|tri)linear interpolation, using CUDA textures in linear mode.
        /// Faster than LINEAR, but at the cost of precision.
        /// Corresponds to cudaFilterModeLinear. Only used in the CUDA backend.
        LINEAR_FAST,

        /// (bi|tri)linear interpolation, using CUDA textures in linear mode.
        /// Faster than COSINE, but at the cost of precision.
        /// Only used in the CUDA backend.
        COSINE_FAST,

        /// (bi|tri)linear interpolation, using CUDA textures in linear mode.
        /// Faster than CUBIC_BSPLINE, but at the cost of precision.
        /// Only used in the CUDA backend.
        CUBIC_BSPLINE_FAST
    };
    inline std::ostream& operator<<(std::ostream& os, InterpMode interp_mode) {
        switch (interp_mode) {
            case InterpMode::NEAREST:
                return os << "InterpMode::NEAREST";
            case InterpMode::LINEAR:
                return os << "InterpMode::LINEAR";
            case InterpMode::COSINE:
                return os << "InterpMode::COSINE";
            case InterpMode::CUBIC:
                return os << "InterpMode::CUBIC";
            case InterpMode::CUBIC_BSPLINE:
                return os << "InterpMode::CUBIC_BSPLINE";
            case InterpMode::LINEAR_FAST:
                return os << "InterpMode::LINEAR_FAST";
            case InterpMode::COSINE_FAST:
                return os << "InterpMode::COSINE_FAST";
            case InterpMode::CUBIC_BSPLINE_FAST:
                return os << "InterpMode::CUBIC_BSPLINE_FAST";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::BorderMode> : ostream_formatter {};
    template<> struct formatter<noa::InterpMode> : ostream_formatter {};
}
