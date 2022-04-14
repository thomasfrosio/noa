#pragma once

#include "noa/cpu/memory/Copy.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Copy.h"
#endif

#include "noa/unified/Array.h"

namespace noa::memory {
    /// (Deep-)Copies of arrays.
    /// \details Contiguous arrays have no copy restrictions and can be copied to any device. This is also true for
    ///          pitched arrays. However, other non-contiguous memory layouts can only be copied if the source and
    ///          destination are both on the same GPU or both on the CPU.
    /// \tparam T           Any data type.
    /// \param[in] input    Source.
    /// \param[out] output  Destination.
    template<typename T>
    void copy(const Array<T>& input, const Array<T>& output) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape:{} into an array of shape:{}",
                      input.shape(), output.shape());
        }

        const Device input_device = input.device();
        const Device output_device = output.device();

        if (input_device.cpu() && output_device.cpu()) {
            cpu::memory::copy(input.share(), input_stride,
                              output.share(), output.stride(),
                              output.shape(), Stream::current(input_device).cpu());
        } else if (output_device.cpu()) { // gpu->cpu
            #ifdef NOA_ENABLE_CUDA
            Stream::current(output_device).synchronize();
            cuda::Stream& cuda_stream = Stream::current(input_device).cuda();
            cuda::memory::copy(input.share(), input_stride,
                               output.share(), output.stride(),
                               output.shape(), cuda_stream);
            cuda_stream.synchronize();
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        } else { // gpu->gpu or cpu->gpu
            #ifdef NOA_ENABLE_CUDA
            if (input_device != output_device)
                Stream::current(input_device).synchronize(); // wait for the input
            cuda::memory::copy(input.share(), input_stride,
                               output.share(), output.stride(),
                               output.shape(), Stream::current(output_device).cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
