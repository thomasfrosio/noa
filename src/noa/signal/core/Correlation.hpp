#pragma once

#include <iostream>

#include "noa/base/Strings.hpp"

namespace noa::signal {
    /// Correlation mode to compute the cross-correlation map.
    enum class Correlation {
        /// Conventional cross-correlation. Generates smooth peaks but often suffers from
        /// the rooftop effect if the highpass filter is not strong enough.
        CONVENTIONAL,

        /// Phase-only cross-correlation. Generates sharper, but noisier, peaks.
        PHASE,

        /// Double-phase correlation. Generate similar peaks that the conventional approach,
        /// but is more accurate due to the doubling of the peak position.
        DOUBLE_PHASE,

        /// Good general alternative. Sharper than the conventional approach.
        MUTUAL
    };

    inline auto operator<<(std::ostream& os, Correlation correlation) -> std::ostream& {
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

namespace fmt {
    template<> struct formatter<noa::signal::Correlation> : ostream_formatter {};
}
