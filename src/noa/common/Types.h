/// \file noa/common/Types.h
/// \brief The basic types used by noa.
/// \author Thomas - ffyr2w
/// \date 11/01/2021

#pragma once

#include <complex>
#include <cstdint>
#include <filesystem>
#include <ios>

#include "noa/common/types/Sizes.h"     // defines size2_t, size3_t and size4_t
#include "noa/common/types/IntX.h"      // defines intX_t, uintX_t, longX_t and ulong_t, where X is 2, 3, or 4.
#include "noa/common/types/FloatX.h"    // defines floatX_t and doubleX_t, where X is 2, 3, or 4.
#include "noa/common/types/MatX.h"      // defines floatXX_t and doubleXX_t, where X is 2, 3, or 4.
#include "noa/common/types/Complex.h"   // defines cfloat_t and cdouble_t
#include "noa/common/types/Stats.h"     // defines Stats<T>
#include "noa/common/types/Constants.h" // defines BorderMode and InterpMode

namespace noa {
    namespace fs = std::filesystem;
    using path_t = fs::path;
}
