#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/utils/Indexing.hpp"
#include "noa/unified/Stream.h"
#include "noa/unified/Handle.hpp"

#include "noa/cpu/memory/Copy.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Copy.h"
#endif

namespace noa::memory {
    /// (Deep-)Copies arrays.
    /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
    ///          also true for pitched layouts and colum or row vectors. However, other non-contiguous memory
    ///          layouts can only be copied if the source and destination are both on the same GPU or on the CPU.
    /// \param[in] input    Source.
    /// \param[out] output  Destination. It should not overlap with \p input.
    template<typename ArrayOrView, typename = std::enable_if_t<noa::traits::is_array_or_view_v<ArrayOrView>>>
    void copy(const ArrayOrView& input, const ArrayOrView& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device input_device = input.device();
        const Device output_device = output.device();
        const auto& input_handle = noa::details::get_handle(input);
        const auto& output_handle = noa::details::get_handle(output);

        if (input_device.is_cpu() && output_device.is_cpu()) {
            auto& cpu_stream = Stream::current(input_device).cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=](){
                cpu::memory::copy(input_handle.get(), input_strides,
                                  output_handle.get(), output.strides(),
                                  output.shape(), threads);
            });
        } else if (output_device.is_cpu()) { // gpu -> cpu
            #ifdef NOA_ENABLE_CUDA
            Stream::current(output_device).synchronize();
            auto& cuda_stream = Stream::current(input_device).cuda();
            cuda::memory::copy(input_handle.get(), input_strides,
                               output_handle.get(), output.strides(),
                               output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input_handle, output_handle);
            cuda_stream.synchronize();
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        } else { // gpu -> gpu or cpu -> gpu
            #ifdef NOA_ENABLE_CUDA
            if (input_device != output_device)
                Stream::current(input_device).synchronize(); // wait for the input
            auto& cuda_stream = Stream::current(output_device).cuda();
            cuda::memory::copy(input_handle.get(), input_strides,
                               output_handle.get(), output.strides(),
                               output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input_handle, output_handle);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// (Deep-)Copies arrays.
    template<typename Value, typename Vector>
    void copy(const Array<Value>& input, const Array<Value>& output,
              const Vector& batch_indexes) {
        // If the batches to copy into output are next to each other,
        // this becomes a slice operation.

        // If the arrays are on the same device, use extract for
        // better performance (only one ewise call).

        // Worst case, copy batches across devices.

    }
}
