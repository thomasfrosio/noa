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

namespace noa::cuda {
    /// CUDA stream mode.
    /// @a STREAM_SERIAL        Work running in the created stream is implicitly synchronized with the NULL stream.
    /// @a STREAM_CONCURRENT    Work running in the created stream may run concurrently with work in stream 0 (the
    ///                         NULL stream) and there is no implicit synchronization performed between it and stream 0.
    enum StreamMode : uint {
        STREAM_CONCURRENT = cudaStreamNonBlocking,
        STREAM_SERIAL = cudaStreamDefault
    };

    /// Bitmask for CUDA events.
    /// @a EVENT_BUSY_TIMER             Default behavior, i.e. record time and busy-wait on synchronization.
    /// @a EVENT_BLOCK_WHILE_WAITING    When synchronizing on this event, shall a thread block?
    /// @a EVENT_DISABLE_TIMING         Can this event be used to record time values (e.g. duration between events)?
    /// @a EVENT_INTERPROCESS           Can multiple processes work with the constructed event?
    enum EventMode : uint {
        EVENT_BUSY_TIMER = 0U,
        EVENT_BLOCK_WHILE_WAITING = cudaEventBlockingSync,
        EVENT_DISABLE_TIMING = cudaEventDisableTiming,
        EVENT_INTERPROCESS = cudaEventInterprocess
    };
}
