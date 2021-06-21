/// \file noa/util/Constants.h
/// \brief Some constants and enums that are common to multiple header/namespaces.
/// \author Thomas - ffyr2w
/// \date 11/01/2021

#pragma once

#include <ostream>
#include "noa/Definitions.h"

namespace noa {
    enum BorderMode {
        /// The input is extended but the values are left unchanged.
        BORDER_NOTHING = 0,

        /// The input is extended by filling all values beyond the edge with zeros.
        /// (0 0 0 0 | a b c d | 0 0 0 0)
        BORDER_ZERO,

        /// The input is extended by filling all values beyond the edge with a constant value.
        /// (k k k k | a b c d | k k k k)
        BORDER_VALUE,

        /// The input is extended by replicating the last pixel.
        /// (a a a a | a b c d | d d d d)
        BORDER_CLAMP,

        /// The input is extended by reflecting about the edge of the last pixel.
        /// (d c b a | a b c d | d c b a)
        BORDER_REFLECT,

        /// The input is extended by reflecting about the center of the last pixel.
        /// (d c b | a b c d | c b a)
        BORDER_MIRROR,

        /// The input is extended by wrapping around to the opposite edge.
        /// (a b c d | a b c d | a b c d)
        BORDER_PERIODIC
    };

    NOA_IH std::ostream& operator<<(std::ostream& os, BorderMode border_mode) {
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
            case BORDER_REFLECT:
                buffer = "BORDER_REFLECT";
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
