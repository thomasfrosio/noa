/// \file noa/gpu/cuda/memory/PtrManaged.h
/// \brief Unified memory.
/// \author Thomas - ffyr2w
/// \date 19 Oct 2021

#pragma once

#include <utility> // std::exchange

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"

// Unified memory:
//  - Managed memory is interoperable and interchangeable with device-specific allocations, such as those created
//    using the cudaMalloc() routine. All CUDA operations that are valid on device memory are also valid on managed
//    memory; the primary difference is that the host portion of a program is able to reference and access the
//    memory as well.
//
//  - Data movement still happens, of course. On compute capabilities >= 6.X, page faulting means that the CUDA
//    system software doesn't need to synchronize all managed memory allocations to the GPU before each kernel
//    launch. If a kernel running on the GPU accesses a page that is not resident in its memory, it faults, allowing
//    the page to be automatically migrated to the GPU memory on-demand. The same thing occurs with CPU page faults.
//
//  - GPU memory over-subscription: On compute capabilities >= 6.X, applications can allocate and access more
//    managed memory than the physical size of GPU memory.

namespace noa::cuda::memory {
    template<typename Type>
    class PtrManaged {
    private:
        size_t m_elements{0};
        std::enable_if_t<noa::traits::is_valid_ptr_type_v<Type>, Type*> m_ptr{nullptr};

    public: // static functions
        /// Allocates device memory using cudaMalloc.
        /// \param elements     Number of elements to allocate on the current device.
        /// \return             Pointer pointing to device memory.
        /// \throw This function can throw if cudaMalloc fails.
        static NOA_HOST Type* alloc(size_t elements) {
            void* tmp{nullptr}; // X** to void** is not allowed
            NOA_THROW_IF(cudaMallocManaged(&tmp, elements * sizeof(Type)));
            return static_cast<Type*>(tmp);
        }

        /// Deallocates device memory allocated by the cudaMalloc* functions.
        /// \param[out] ptr     Pointer pointing to device memory, or nullptr.
        /// \throw This function can throw if cudaFree fails (e.g. double free).
        static NOA_HOST void dealloc(Type* ptr) {
            NOA_THROW_IF(cudaFree(ptr)); // if nullptr, it does nothing
        }
    };
}
