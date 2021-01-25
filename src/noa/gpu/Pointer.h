#pragma once

#include "noa/gpu/Backend.h"

#ifdef NOA_GPU_BACKEND_CUDA
    #include "noa/gpu/cuda/Pointer.h"
#elif NOA_GPU_BACKEND_OPENCL
    #error "OpenCL backend not implemented yet"
#else
    #error "GPU backend not defined"
#endif
