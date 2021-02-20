/**
 * @file noa/gpu/cuda/CudaRuntime.h
 * @brief To include the cuda runtime.
 * @author Thomas - ffyr2w
 * @date 05 Jan 2021
 */
#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>

// Add specialization for cfloat_t, which has the same layout as float2.
// Mostly used for CUDA arrays (i.e. PtrArray) and textures (i.e. PtrTexture).
template<>
__inline__ __host__ cudaChannelFormatDesc cudaCreateChannelDesc<Noa::Complex<float>>() {
    return cudaCreateChannelDesc<float2>();
}
