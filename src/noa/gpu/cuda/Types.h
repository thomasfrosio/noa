#pragma once

#include "noa/gpu/Backend.h"
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cstdint>

// Half
namespace Noa::CUDA {
    // For now, don't support half/float16, since it is not straightforward to have something working
    // for non-cuda code. Also, not sure it will be used a lot, so not a priority.

    using t_float = float;
    using t_double = double;

    using t_cfloat = cuFloatComplex;
    using t_cdouble = cuDoubleComplex;

    using t_short = short;
    using t_int = int;
    using t_long = long long;

}
