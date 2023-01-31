#include "noa/common/types/Constants.h"

namespace noa {
    std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
        switch (border_mode) {
            case BorderMode::NOTHING:
                return os << "BorderMode::NOTHING";
            case BorderMode::ZERO:
                return os << "BorderMode::ZERO";
            case BorderMode::VALUE:
                return os << "BorderMode::VALUE";
            case BorderMode::CLAMP:
                return os << "BorderMode::CLAMP";
            case BorderMode::REFLECT:
                return os << "BorderMode::REFLECT";
            case BorderMode::MIRROR:
                return os << "BorderMode::MIRROR";
            case BorderMode::PERIODIC:
                return os << "BorderMode::PERIODIC";
        }
        return os;
    }

    std::ostream& operator<<(std::ostream& os, InterpMode interp_mode) {
        switch (interp_mode) {
            case InterpMode::NEAREST:
                return os << "InterpMode::NEAREST";
            case InterpMode::LINEAR:
                return os << "InterpMode::LINEAR";
            case InterpMode::COSINE:
                return os << "InterpMode::COSINE";
            case InterpMode::CUBIC:
                return os << "InterpMode::CUBIC";
            case InterpMode::CUBIC_BSPLINE:
                return os << "InterpMode::CUBIC_BSPLINE";
            case InterpMode::LINEAR_FAST:
                return os << "InterpMode::LINEAR_FAST";
            case InterpMode::COSINE_FAST:
                return os << "InterpMode::COSINE_FAST";
            case InterpMode::CUBIC_BSPLINE_FAST:
                return os << "InterpMode::CUBIC_BSPLINE_FAST";
        }
        return os;
    }
}

namespace noa::fft {
    std::ostream& operator<<(std::ostream& os, Remap remap) {
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
    std::ostream& operator<<(std::ostream& os, CorrelationMode correlation) {
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

    std::ostream& operator<<(std::ostream& os, PeakMode peak_mode) {
        switch (peak_mode) {
            case PeakMode::PARABOLA_1D:
                return os << "PeakMode::PARABOLA_1D";
            case PeakMode::COM:
                return os << "PeakMode::COM";
        }
        return os;
    }
}
