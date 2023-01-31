#pragma once

#include <ostream>
#include "noa/common/Definitions.h"

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
    std::ostream& operator<<(std::ostream& os, BorderMode border_mode);

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
    std::ostream& operator<<(std::ostream& os, InterpMode interp_mode);
}

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
    ///     The "non-centered" (or native) layout is used by FFT routines, with the origin (the DC) at index 0.
    ///     The "centered" layout is often used in files, with the origin in the "middle", i.e. N // 2.
    /// \e Redundancy:
    ///     It refers to non-redundant Fourier transforms of real inputs, resulting in transforms with a LOGICAL shape
    ///     of {D,H,W} complex elements and PHYSICAL shapes of {D, H, W/2+1} complex elements.
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
    std::ostream& operator<<(std::ostream& os, Remap remap);
}

namespace noa::math {
    struct distribution_t {};
    struct uniform_t : public distribution_t {};
    struct normal_t : public distribution_t {};
    struct log_normal_t : public distribution_t {};
    struct poisson_t : public distribution_t {};

    NOA_IH std::ostream& operator<<(std::ostream& os, uniform_t) { return os << "Distribution::Uniform"; }
    NOA_IH std::ostream& operator<<(std::ostream& os, normal_t) { return os << "Distribution::Normal"; }
    NOA_IH std::ostream& operator<<(std::ostream& os, log_normal_t) { return os << "Distribution::Log-normal"; }
    NOA_IH std::ostream& operator<<(std::ostream& os, poisson_t) { return os << "Distribution::Poisson"; }
}

namespace noa::signal {
    /// Correlation mode to compute the cross-correlation map.
    enum class CorrelationMode {
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
    std::ostream& operator<<(std::ostream& os, CorrelationMode correlation);

    /// Subpixel registration method.
    enum PeakMode {
        /// Updates the pixel location by fitting a 1D parabola along axis.
        /// The peak value is average of the vertex height along each dimension.
        PARABOLA_1D,

        /// Updates the pixel location by adding the center-of-mass (COM).
        /// The peak value is currently not modified and the original max value is returned.
        COM
    };
    std::ostream& operator<<(std::ostream& os, PeakMode peak_mode);
}
