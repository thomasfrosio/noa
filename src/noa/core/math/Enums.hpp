#pragma once

#include "noa/core/string/Format.hpp"

namespace noa {
    enum class NormalizationMode {
        /// Set [min,max] range to [0,1].
        MIN_MAX,

        /// Set the mean to 0, and standard deviation to 1.
        MEAN_STD,

        /// Set the l2_norm to 1.
        L2_NORM
    };

    inline std::ostream& operator<<(std::ostream& os, NormalizationMode normalization_mode) {
        switch (normalization_mode) {
            case NormalizationMode::MIN_MAX:
                return os << "NormalizationMode::MIN_MAX";
            case NormalizationMode::MEAN_STD:
                return os << "NormalizationMode::MEAN_STD";
            case NormalizationMode::L2_NORM:
                return os << "NormalizationMode::L2_NORM";
        }
        return os;
    }
}

// fmt 9.1.0 fix (Disabled automatic std::ostream insertion operator (operator<<))
namespace fmt {
    template<> struct formatter<noa::NormalizationMode> : ostream_formatter {};
}
