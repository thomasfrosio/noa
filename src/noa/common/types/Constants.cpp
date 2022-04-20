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
        return os;
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
        return os;
    }
}

namespace noa::fft {
    std::ostream& operator<<(std::ostream& os, Remap remap) {
        switch (remap) {
            case H2H:
                os << "H2H";
            case H2HC:
                os << "H2HC";
            case H2F:
                os << "H2F";
            case H2FC:
                os << "H2FC";
            case HC2H:
                os << "HC2H";
            case HC2HC:
                os << "HC2HC";
            case HC2F:
                os << "HC2F";
            case HC2FC:
                os << "HC2FC";
            case F2H:
                os << "F2H";
            case F2HC:
                os << "F2HC";
            case F2F:
                os << "F2F";
            case F2FC:
                os << "F2FC";
            case FC2H:
                os << "FC2H";
            case FC2HC:
                os << "FC2HC";
            case FC2F:
                os << "FC2F";
            case FC2FC:
                os << "FC2FC";
        }
        return os;
    }
}
