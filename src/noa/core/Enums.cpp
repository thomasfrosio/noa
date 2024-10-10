#include "noa/core/Enums.hpp"

namespace noa {
    std::ostream& operator<<(std::ostream& os, Interp interp) {
        switch (interp) {
            case Interp::NEAREST:
                return os << "Interp::NEAREST";
            case Interp::NEAREST_FAST:
                return os << "Interp::NEAREST_FAST";
            case Interp::LINEAR:
                return os << "Interp::LINEAR";
            case Interp::LINEAR_FAST:
                return os << "Interp::LINEAR_FAST";
            case Interp::CUBIC:
                return os << "Interp::CUBIC";
            case Interp::CUBIC_FAST:
                return os << "Interp::CUBIC_FAST";
            case Interp::CUBIC_BSPLINE:
                return os << "Interp::CUBIC_BSPLINE";
            case Interp::CUBIC_BSPLINE_FAST:
                return os << "Interp::CUBIC_BSPLINE_FAST";
            case Interp::LANCZOS4:
                return os << "Interp::LANCZOS4";
            case Interp::LANCZOS6:
                return os << "Interp::LANCZOS6";
            case Interp::LANCZOS8:
                return os << "Interp::LANCZOS8";
            case Interp::LANCZOS4_FAST:
                return os << "Interp::LANCZOS4_FAST";
            case Interp::LANCZOS6_FAST:
                return os << "Interp::LANCZOS6_FAST";
            case Interp::LANCZOS8_FAST:
                return os << "Interp::LANCZOS8_FAST";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, Border border) {
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

    std::ostream& operator<<(std::ostream& os, Norm norm) {
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

    std::ostream& operator<<(std::ostream& os, Remap layout) {
        switch (layout) {
            case Remap::H2H:
                return os << "h2h";
            case Remap::H2HC:
                return os << "h2hc";
            case Remap::H2F:
                return os << "h2f";
            case Remap::H2FC:
                return os << "h2fc";
            case Remap::HC2H:
                return os << "hc2h";
            case Remap::HC2HC:
                return os << "hc2hc";
            case Remap::HC2F:
                return os << "hc2f";
            case Remap::HC2FC:
                return os << "hc2fc";
            case Remap::F2H:
                return os << "f2h";
            case Remap::F2HC:
                return os << "f2hc";
            case Remap::F2F:
                return os << "f2f";
            case Remap::F2FC:
                return os << "f2fc";
            case Remap::FC2H:
                return os << "fc2h";
            case Remap::FC2HC:
                return os << "fc2hc";
            case Remap::FC2F:
                return os << "fc2f";
            case Remap::FC2FC:
                return os << "fc2fc";
        }
        return os;
    }
}

namespace noa::signal {
    std::ostream& operator<<(std::ostream& os, Correlation correlation) {
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
