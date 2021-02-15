#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "noa/Base.h"
#include "noa/gpu/cuda/Exception.h"

// Add description support for cfloat_t, which has the same layout as float2.
// Mostly used for CUDA arrays and textures.
template<>
__inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<Noa::Complex<float>>() {
    return cudaCreateChannelDesc<float2>();
}
