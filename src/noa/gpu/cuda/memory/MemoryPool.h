#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Device.h"

// Memory pools: (since CUDA 11.2)
//  - When called without an explicit pool argument, each call to cudaMallocAsync infers the device from the
//    specified stream and attempts to allocate memory from that device’s current pool.
//  - By default, unused memory accumulated in the pool is returned to the OS during the next synchronization
//    operation on an event, stream, or device, as the following code example shows. The application can configure
//    a release threshold to enable unused memory to persist beyond the synchronization operation.
//  - The pool threshold is just a hint. Memory in the pool can also be released implicitly by the CUDA driver to
//    enable an unrelated memory allocation request in the same process to succeed.
//    See cudaMemPoolTrimTo(pool, bytesToKeep).
//  - Memory pools are not attached to a particular stream, they can reuse memory device-wide.

#if CUDART_VERSION >= 11020

namespace noa::cuda::memory {
    // Memory pool.
    class Pool {
    public:
        // Gets the default memory pool of device.
        static cudaMemPool_t current(Device device) {
            cudaMemPool_t pool{};
            NOA_THROW_IF(cudaDeviceGetDefaultMemPool(&pool, device.id()));
            return pool;
        }

        // Sets the default memory pool of device.
        static void set_current(Device device, cudaMemPool_t pool) {
            NOA_THROW_IF(cudaDeviceSetMemPool(device.id(), pool));
        }

    public:
        // Gets the default memory pool of the current device.
        Pool() : Pool(Device::current()) {}

        // Gets the default memory pool of device.
        explicit Pool(Device device) : m_pool(Pool::current(device)) {}

        // Sets this pool as default memory pool of device.
        void attach(Device device) const {
            set_current(device, m_pool);
        }

        // Sets the amount of reserved memory in bytes to hold onto before trying to release memory back to the OS.
        // When more than the release threshold bytes of memory are held by the memory pool, the allocator will
        // try to release memory back to the OS on the next call to stream, event or context synchronize. The
        // default value is 0 bytes (i.e. stream synchronization frees the cached memory).
        void set_threshold(size_t threshold_bytes) const {
            NOA_THROW_IF(cudaMemPoolSetAttribute(m_pool, cudaMemPoolAttrReleaseThreshold, &threshold_bytes));
        }

        // Releases memory back to the OS until the pool contains fewer than "bytes_to_keep" reserved bytes,
        // or there is no more memory that the allocator can safely release. The allocator cannot release OS
        // allocations that back outstanding asynchronous allocations. The OS allocations may happen at different
        // granularity from the user allocations. If the pool has less than this amount reserved, do nothing.
        // Otherwise, the pool will be guaranteed to have at least that amount of bytes reserved.
        void trim(size_t bytes_to_keep) const {
            cudaMemPoolTrimTo(m_pool, bytes_to_keep);
        }

        [[nodiscard]] cudaMemPool_t get() const noexcept { return m_pool; }
        [[nodiscard]] cudaMemPool_t id() const noexcept { return m_pool; }

    private:
        cudaMemPool_t m_pool{nullptr};
    };
}

#endif
