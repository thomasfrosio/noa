#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/string/Format.hpp"

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

    /// Interpolation methods.
    enum class Interp {
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
    ///     The "non-centered" (or native) layout is used by fft functions, with the origin (the DC) at index 0.
    ///     The "centered" layout is often used in files or for geometric transformations, with the origin at index n//2.
    /// \e Redundancy:
    ///     It refers to non-redundant Fourier transforms of real inputs (aka rfft). For a {D,H,W} logical/real shape,
    ///     rffts have a shape of {D,H,W//2+1}. Note that with even dimensions, the Nyquist frequency is real and
    ///     the c2r (aka irfft) functions will assume its imaginary part is zero.
    ///
    /// \example
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
    /// \note F2FC is fftshift and FC2F is ifftshift.
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

    /// Subpixel registration method.
    enum CorrelationRegistration {
        /// Updates the pixel location by fitting a 1D parabola along axis.
        /// The peak value is average of the vertex height along each dimension.
        PARABOLA_1D,

        /// Updates the pixel location by adding the center-of-mass (COM).
        /// The peak value is currently not modified and the original max value is returned.
        COM
    };
}

#if defined(NOA_IS_OFFLINE)
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
    inline std::ostream& operator<<(std::ostream& os, Interp interp) {
        switch (interp) {
            case Interp::NEAREST:
                return os << "Interp::NEAREST";
            case Interp::LINEAR:
                return os << "Interp::LINEAR";
            case Interp::COSINE:
                return os << "Interp::COSINE";
            case Interp::CUBIC:
                return os << "Interp::CUBIC";
            case Interp::CUBIC_BSPLINE:
                return os << "Interp::CUBIC_BSPLINE";
            case Interp::LINEAR_FAST:
                return os << "Interp::LINEAR_FAST";
            case Interp::COSINE_FAST:
                return os << "Interp::COSINE_FAST";
            case Interp::CUBIC_BSPLINE_FAST:
                return os << "Interp::CUBIC_BSPLINE_FAST";
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

namespace noa::fft {
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

    inline std::ostream& operator<<(std::ostream& os, CorrelationRegistration mode) {
        switch (mode) {
            case CorrelationRegistration::PARABOLA_1D:
                return os << "CorrelationRegistration::PARABOLA_1D";
            case CorrelationRegistration::COM:
                return os << "CorrelationRegistration::COM";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator)
namespace fmt {
    template<> struct formatter<noa::Border> : ostream_formatter {};
    template<> struct formatter<noa::Interp> : ostream_formatter {};
    template<> struct formatter<noa::Norm> : ostream_formatter {};
    template<> struct formatter<noa::fft::Remap> : ostream_formatter {};
    template<> struct formatter<noa::signal::Correlation> : ostream_formatter {};
    template<> struct formatter<noa::signal::CorrelationRegistration> : ostream_formatter {};
}
#endif
