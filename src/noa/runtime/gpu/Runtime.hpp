#pragma once

#if defined(NOA_ENABLE_CUDA)
#   include <cuda_runtime.h>
#   include <cuComplex.h>
#   include <cuda_fp16.h>
#elif defined(NOA_ENABLE_SYCL)
#   include <sycl/sycl.hpp>
#endif
