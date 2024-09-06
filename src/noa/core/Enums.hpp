#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/utils/Strings.hpp"

namespace noa {
    /// Border mode, i.e. how out of bounds coordinates are handled. It is compatible with cudaTextureAddressMode.
    enum class Border {
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

    enum class Norm {
        /// Set [min,max] range to [0,1].
        MIN_MAX,

        /// Set the mean to 0, and standard deviation to 1.
        MEAN_STD,

        /// Set the l2_norm to 1.
        L2
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
    inline std::ostream& operator<<(std::ostream& os, Border border) {
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

    inline std::ostream& operator<<(std::ostream& os, Norm norm) {
        switch (norm) {
            case Norm::MIN_MAX:
                return os << "Norm::MIN_MAX";
            case Norm::MEAN_STD:
                return os << "Norm::MEAN_STD";
            case Norm::L2:
                return os << "Norm::L2";
        }
        return os;
    }
}

namespace noa::signal {
    inline std::ostream& operator<<(std::ostream& os, Correlation correlation) {
        switch (correlation) {
            case Correlation::CONVENTIONAL:
                return os << "Correlation::CONVENTIONAL";
            case Correlation::PHASE:
                return os << "Correlation::PHASE";
            case Correlation::DOUBLE_PHASE:
                return os << "Correlation::DOUBLE_PHASE";
            case Correlation::MUTUAL:
                return os << "Correlation::MUTUAL";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator)
namespace fmt {
    template<> struct formatter<noa::Border> : ostream_formatter {};
    template<> struct formatter<noa::Norm> : ostream_formatter {};
    template<> struct formatter<noa::signal::Correlation> : ostream_formatter {};
}
#endif
