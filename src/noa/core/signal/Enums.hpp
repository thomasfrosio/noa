#pragma once

#include "noa/core/string/Format.hpp"

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
    inline std::ostream& operator<<(std::ostream& os, CorrelationMode correlation) {
        switch (correlation) {
            case CorrelationMode::CONVENTIONAL:
                return os << "CorrelationMode::CONVENTIONAL";
            case CorrelationMode::PHASE:
                return os << "CorrelationMode::PHASE";
            case CorrelationMode::DOUBLE_PHASE:
                return os << "CorrelationMode::DOUBLE_PHASE";
            case CorrelationMode::MUTUAL:
                return os << "CorrelationMode::MUTUAL";
        }
        return os;
    }

    /// Subpixel registration method.
    enum PeakMode {
        /// Updates the pixel location by fitting a 1D parabola along axis.
        /// The peak value is average of the vertex height along each dimension.
        PARABOLA_1D,

        /// Updates the pixel location by adding the center-of-mass (COM).
        /// The peak value is currently not modified and the original max value is returned.
        COM
    };
    inline std::ostream& operator<<(std::ostream& os, PeakMode peak_mode) {
        switch (peak_mode) {
            case PeakMode::PARABOLA_1D:
                return os << "PeakMode::PARABOLA_1D";
            case PeakMode::COM:
                return os << "PeakMode::COM";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::signal::CorrelationMode> : ostream_formatter {};
    template<> struct formatter<noa::signal::PeakMode> : ostream_formatter {};
}
