#pragma once

#include "noa/gpu/Base.h"

#ifdef NOA_BUILD_CUDA
    #include "noa/gpu/cuda/Pointer.h"
#elif NOA_BUILD_OPENCL
    #error "OpenCL backend not implemented yet"
#else
    #error "GPU backend not defined"
#endif
