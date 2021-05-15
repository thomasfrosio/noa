/**
 * @file noa/Types.h
 * @brief Some type definitions.
 * @author Thomas - ffyr2w
 * @date 11/01/2021
 */
#pragma once

#include <complex>
#include <cstdint>
#include <filesystem>
#include <ios>

#include "noa/util/Sizes.h"     // defines size2_t, size3_t and size4_t
#include "noa/util/IntX.h"      // defines intX_t, uintX_t, longX_t and ulong_t, where X is 2, 3, or 4.
#include "noa/util/FloatX.h"    // defines floatX_t and doubleX_t, where X is 2, 3, or 4.
#include "noa/util/Complex.h"   // defines cfloat_t and cdouble_t
#include "noa/util/Stats.h"     // defined Stats<T>

namespace Noa {
    namespace fs = std::filesystem;
    using path_t = fs::path;

    enum BorderMode {
        BORDER_NOTHING = 0, BORDER_ZERO, BORDER_VALUE, BORDER_CLAMP, BORDER_MIRROR, BORDER_PERIODIC
    };

    template<typename T>
    std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
        std::string buffer;
        switch (border_mode) {
            case BORDER_NOTHING:
                buffer = "BORDER_NOTHING";
                break;
            case BORDER_ZERO:
                buffer = "BORDER_ZERO";
                break;
            case BORDER_VALUE:
                buffer = "BORDER_VALUE";
                break;
            case BORDER_CLAMP:
                buffer = "BORDER_CLAMP";
                break;
            case BORDER_MIRROR:
                buffer = "BORDER_MIRROR";
                break;
            case BORDER_PERIODIC:
                buffer = "BORDER_PERIODIC";
                break;
        }
        os << buffer;
        return os;
    }

    enum InterpMode {
        INTERP_NEIGHBOUR = 0, INTERP_LINEAR, INTERP_CUBIC
    };
}
