/// \file noa/gpu/cuda/Types.h
/// \brief Expansion of Types.h for noa::cuda.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

#include "noa/Types.h"
#include "noa/gpu/cuda/util/Sizes.h"

// Add specialization for cfloat_t, which has the same layout as float2.
// Mostly used for CUDA arrays (i.e. PtrArray) and textures (i.e. PtrTexture).
template<>
__inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<noa::cfloat_t>() {
    return cudaCreateChannelDesc<float2>();
}
