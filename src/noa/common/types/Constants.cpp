#include "noa/common/types/Constants.h"

namespace noa {
    std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
        switch (border_mode) {
            case BORDER_NOTHING:
                return os << "BORDER_NOTHING";
            case BORDER_ZERO:
                return os << "BORDER_ZERO";
            case BORDER_VALUE:
                return os << "BORDER_VALUE";
            case BORDER_CLAMP:
                return os << "BORDER_CLAMP";
            case BORDER_REFLECT:
                return os << "BORDER_REFLECT";
            case BORDER_MIRROR:
                return os << "BORDER_MIRROR";
            case BORDER_PERIODIC:
                return os << "BORDER_PERIODIC";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, InterpMode interp_mode) {
        switch (interp_mode) {
            case INTERP_NEAREST:
                return os << "INTERP_NEAREST";
            case INTERP_LINEAR:
                return os << "INTERP_LINEAR";
            case INTERP_COSINE:
                return os << "INTERP_COSINE";
            case INTERP_CUBIC:
                return os << "INTERP_CUBIC";
            case INTERP_CUBIC_BSPLINE:
                return os << "INTERP_CUBIC_BSPLINE";
            case INTERP_LINEAR_FAST:
                return os << "INTERP_LINEAR_FAST";
            case INTERP_COSINE_FAST:
                return os << "INTERP_COSINE_FAST";
            case INTERP_CUBIC_BSPLINE_FAST:
                return os << "INTERP_CUBIC_BSPLINE_FAST";
        }
        return os;
    }
}

namespace noa::fft {
    std::ostream& operator<<(std::ostream& os, Remap remap) {
        switch (remap) {
            case H2H:
                return os << "H2H";
            case H2HC:
                return os << "H2HC";
            case H2F:
                return os << "H2F";
            case H2FC:
                return os << "H2FC";
            case HC2H:
                return os << "HC2H";
            case HC2HC:
                return os << "HC2HC";
            case HC2F:
                return os << "HC2F";
            case HC2FC:
                return os << "HC2FC";
            case F2H:
                return os << "F2H";
            case F2HC:
                return os << "F2HC";
            case F2F:
                return os << "F2F";
            case F2FC:
                return os << "F2FC";
            case FC2H:
                return os << "FC2H";
            case FC2HC:
                return os << "FC2HC";
            case FC2F:
                return os << "FC2F";
            case FC2FC:
                return os << "FC2FC";
        }
        return os;
    }
}
