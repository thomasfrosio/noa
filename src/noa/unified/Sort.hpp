#pragma once

#include "noa/cpu/Sort.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Sort.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa {
    /// Sorts an array, in-place.
    /// \tparam T               Any restricted scalar.
    /// \param[in,out] array    Array to sort, in-place.
    /// \param ascending        Whether to sort in ascending or descending order.
    /// \param axis             Axis along which to sort. The default is -1, which sorts along the first non-empty
    ///                         dimension in the rightmost order. Otherwise, it should be from 0 to 3, included.
    /// \note All the sort algorithms make temporary copies of the data when sorting along any but the last axis.
    ///       Consequently, sorting along the last axis is faster and uses less space than sorting along any other axis.
    template<typename VArray,
             typename = std::enable_if_t<nt::is_varray_of_restricted_scalar_v<VArray>>>
    void sort(const VArray& array, bool ascending = true, i32 axis = -1) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.enqueue([=](){
                cpu::sort(array.get(), array.strides(), array.shape(), ascending, axis);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::sort(array.get(), array.strides(), array.shape(), ascending, axis, cuda_stream);
            cuda_stream.enqueue_attach(array);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    // TODO Add sort by keys.
}
