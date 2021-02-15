#pragma once

#include <cuda_runtime.h>

#ifdef NOA_DEBUG
#include <atomic>
#endif

#include "noa/Define.h"

namespace Noa::CUDA {
    /** Centralizes CUDA allocations for debugging purposes. */
    class Allocator {
#ifdef NOA_DEBUG
    public:
        static std::atomic<size_t> debug_count_pinned;
        static std::atomic<size_t> debug_count_device;
        static std::atomic<size_t> debug_count_array;
        static std::atomic<size_t> debug_count_texture;

        NOA_HOST static size_t countDevice() { return debug_count_device; }
        NOA_HOST static size_t countPinned() { return debug_count_pinned; }
        NOA_HOST static size_t countArray() { return debug_count_array; }
        NOA_HOST static size_t countTexture() { return debug_count_texture; }
#else
        NOA_HOST static size_t countDevice() { return 0; }
        NOA_HOST static size_t countPinned() { return 0; }
        NOA_HOST static size_t countArray() { return 0; }
        NOA_HOST static size_t countTexture() { return 0; }
#endif

    public:
        /** Allocates pinned "linear" memory with cudaMallocHost. */
        NOA_HOST static cudaError_t mallocHost(void** ptr, size_t bytes);

        /** Allocates "linear" memory on the current device with cudaMalloc. */
        NOA_HOST static cudaError_t malloc(void** ptr, size_t bytes);

        /** Allocates "padded" memory on the current device with cudaMalloc3D. */
        NOA_HOST static cudaError_t mallocPadded(cudaPitchedPtr* pitched_ptr, cudaExtent extent);

        /** Allocates a CUDA array on the current device with cudaMalloc3DArray. */
        NOA_HOST static cudaError_t mallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                                                cudaExtent extent, uint flags);

        /** Creates a texture object with cudaCreateTextureObject. */
        NOA_HOST static cudaError_t mallocTexture(cudaTextureObject_t* texture,
                                                  const cudaResourceDesc* resource_desc,
                                                  const cudaTextureDesc* texture_desc,
                                                  const cudaResourceViewDesc* resource_view_desc = nullptr);

        /** Frees "linear" memory allocated with cudaMalloc(Pitch/3D). */
        NOA_FH static cudaError_t free(void* ptr);

        /** Frees a CUDA array allocated with cudaMalloc3DArray. */
        NOA_FH static cudaError_t freeArray(cudaArray_t ptr);

        /** Frees "linear" memory allocated with cudaMallocHost. */
        NOA_FH static cudaError_t freeHost(void* ptr);

        /** Destroys a texture object created by cudaCreateTextureObject. */
        NOA_FH static cudaError_t freeTexture(cudaTextureObject_t ptr);
    };

#ifdef NOA_DEBUG
    std::atomic<size_t> Allocator::debug_count_pinned{0};
    std::atomic<size_t> Allocator::debug_count_device{0};
    std::atomic<size_t> Allocator::debug_count_array{0};
    std::atomic<size_t> Allocator::debug_count_texture{0};
#endif

    NOA_HOST cudaError_t Allocator::mallocHost(void** ptr, size_t bytes) {
        #ifdef NOA_DEBUG
        debug_count_pinned++;
        #endif
        return cudaMallocHost(ptr, bytes);
    }

    NOA_HOST cudaError_t Allocator::malloc(void** ptr, size_t bytes) {
        #ifdef NOA_DEBUG
        debug_count_device++;
        #endif
        return cudaMalloc(ptr, bytes);
    }

    NOA_HOST cudaError_t Allocator::mallocPadded(cudaPitchedPtr* pitched_ptr, cudaExtent extent) {
        #ifdef NOA_DEBUG
        debug_count_device++;
        #endif
        return cudaMalloc3D(pitched_ptr, extent);
    }

    cudaError_t Allocator::mallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                                       cudaExtent extent, uint flags = 0) {
        #ifdef NOA_DEBUG
        debug_count_array++;
        #endif
        return cudaMalloc3DArray(array, desc, extent, flags);
    }

    NOA_HOST cudaError_t Allocator::free(void* ptr) {
        #ifdef NOA_DEBUG
        debug_count_device -= ptr ? 1 : 0;
        #endif
        return cudaFree(ptr); // if nullptr, does nothing.
    }

    cudaError_t Allocator::freeArray(cudaArray_t ptr) {
        #ifdef NOA_DEBUG
        debug_count_array -= ptr ? 1 : 0;
        #endif
        return cudaFreeArray(ptr); // if nullptr, does nothing.
    }

    cudaError_t Allocator::freeHost(void* ptr) {
        #ifdef NOA_DEBUG
        debug_count_pinned -= ptr ? 1 : 0; // it should _not_ be nullptr, isn't it?
        #endif
        return cudaFreeHost(ptr);
    }

    cudaError_t Allocator::mallocTexture(cudaTextureObject_t* texture,
                                         const cudaResourceDesc* resource_desc,
                                         const cudaTextureDesc* texture_desc,
                                         const cudaResourceViewDesc* resource_view_desc) {
        #ifdef NOA_DEBUG
        debug_count_texture++;
        #endif
        return cudaCreateTextureObject(texture, resource_desc, texture_desc, resource_view_desc);
    }

    cudaError_t Allocator::freeTexture(cudaTextureObject_t texture) {
        #ifdef NOA_DEBUG
        debug_count_texture--;
        #endif
        return cudaDestroyTextureObject(texture);
    }
}
