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
#include "noa/util/Stats.h"    // defined Stats<T>

namespace Noa {
    namespace fs = std::filesystem;
    using path_t = fs::path;

    enum PathMode {
        /**Pad with fixed value*/
        PAD_VALUE = 1,
        /**Pad by repeating mirrored data*/
        PAD_MIRROR = 2,
        /**Pad by repeating data*/
        PAD_TILE = 3,
        /**Pad by clamping coordinates**/
        PAD_CLAMP = 4,
        /**Leave memory contents unchanged within pad*/
        PAD_NOTHING = 5
    };

    enum InterpMode {
        INTERP_LINEAR = 1,
        INTERP_CUBIC = 1,
    };
}
