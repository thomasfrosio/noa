#include "noa/common/types/Constants.h"

namespace noa {
    std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
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
        }
    }

    std::ostream& operator<<(std::ostream& os, InterpMode interp_mode) {
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
        }
    }
}

namespace noa::fft {
    std::ostream& operator<<(std::ostream& os, Remap remap) {
        switch (remap) {
            case Remap::H2H:
                return os << "H2H";
            case Remap::HC2HC:
                return os << "HC2HC";
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
            case Remap::F2F:
                return os << "F2F";
            case Remap::FC2FC:
                return os << "FC2FC";
        }
    }
}
