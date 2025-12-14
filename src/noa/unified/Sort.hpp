#pragma once

#include "noa/core/Config.hpp"
#include "noa/cpu/Sort.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Sort.cuh"
#endif

#include "noa/unified/Stream.hpp"
#include "noa/unified/Traits.hpp"

namespace noa {
    struct SortOptions {
        /// Whether to sort in ascending or descending order.
        bool ascending{true};

        /// Axis along which to sort. The default is -1, which sorts along the first non-empty
        /// dimension in the rightmost order. Otherwise, it should be from 0 to 3, included.
        i32 axis{-1};
    };

    /// Sorts an array, in-place.
    /// \param[in,out] array    Array to sort, in-place.
    /// \param options          Sorting options.
    /// \note The sort algorithms make temporary copies of the data when sorting along any but the last axis.
    ///       Consequently, sorting along the last axis is faster and uses less memory than sorting along any
    ///       other axis.
    template<nt::writable_varray_decay_of_scalar VArray>
    void sort(VArray&& array, SortOptions options = {}) {
        check(not array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.enqueue([=, a = std::forward<VArray>(array)](){
                noa::cpu::sort(a.get(), a.strides(), a.shape(), options.ascending, options.axis);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::sort(array.get(), array.strides(), array.shape(), options.ascending, options.axis, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<VArray>(array));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    // TODO Add sort by keys.
}
