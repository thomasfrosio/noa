#pragma once

#include "noa/cpu/memory/Cast.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Cast.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::memory {
    /// Casts an array.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the casted values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    template<typename Input, typename Output, typename = std::enable_if_t<
             nt::are_varray_of_restricted_numeric_v<Input, Output>>>
    void cast(const Input& input, const Output& output, bool clamp = false) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, input.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::memory::cast(input.get(), input_strides, output.get(), output.strides(),
                                  output.shape(), clamp, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            using input_t = nt::value_type_t<Input>;
            using output_t = nt::value_type_t<Output>;
            if constexpr (cuda::memory::details::is_valid_cast_v<input_t, output_t>) {
                auto& cuda_stream = stream.cuda();
                cuda::memory::cast(input.get(), input_strides, output.get(), output.strides(),
                                   output.shape(), clamp, cuda_stream);
                cuda_stream.enqueue_attach(input, output);
            } else {
                // TODO Add nvrtc to support all types.
                NOA_THROW("This cast ({} -> {}) is not supported by the CUDA backend",
                          noa::string::human<input_t>(), noa::string::human<output_t>());
            }
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
