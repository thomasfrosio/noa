#pragma once

#include "noa/signal/cpu/MedianFilter.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/signal/cuda/MedianFilter.cuh"
#endif

#include "noa/runtime/Array.hpp"

namespace noa::signal {
    struct MedianFilterOptions {
        /// Number of elements to consider for the computation of the median. Only odd numbers are supported.
        /// 1d: This corresponds to the width dimension. On the GPU, this is limited to 21.
        /// 2d: This corresponds to the height and width dimension. On the GPU, this is limited to 11.
        /// 3d: This corresponds to the depth, height and width dimensions.  On the GPU, this is limited to 5.
        i32 window_size;

        /// Border mode used for the "implicit padding".
        /// Either Border::ZERO or Border::REFLECT.
        /// With Border::REFLECT, the filtered dimensions should be larger or equal than `window_size/2+1`.
        Border border_mode{Border::REFLECT};
    };

    /// Computes the median filter using a 1D window.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param options      Filter options.
    template<nt::readable_varray_decay_of_scalar Input,
             nt::writable_varray_decay_of_scalar Output>
    void median_filter_1d(Input&& input, Output&& output, const MedianFilterOptions& options) {
        if (options.window_size <= 1) {
            std::forward<Input>(input).to(output);
            return;
        }

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        check(is_odd(options.window_size), "Only odd windows are currently supported");
        check(options.border_mode == Border::ZERO or output.shape()[3] >= options.window_size / 2 + 1,
              "With Border::REFLECT and a window of {}, the width should be larger or equal than {}, but got {}",
              options.window_size, options.window_size / 2 + 1, output.shape()[3]);

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, i = std::forward<Input>(input), o = std::forward<Output>(output)] {
                noa::signal::cpu::median_filter_1d(
                    i.get(), input_strides,
                    o.get(), o.strides(), o.shape(),
                    options.border_mode, options.window_size, n_threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            check(nd::is_accessor_access_safe<i32>(input, output.shape()) and
                  nd::is_accessor_access_safe<i32>(output, output.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            auto& cuda_stream = stream.cuda();
            noa::signal::cuda::median_filter_1d(
                input.get(), input_strides.template as<i32>(),
                output.get(), output.strides().template as<i32>(), output.shape(),
                options.border_mode, options.window_size, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Computes the median filter using a 2d square window.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param options      Filter options.
    template<nt::readable_varray_decay_of_scalar Input,
             nt::writable_varray_decay_of_scalar Output>
    void median_filter_2d(Input&& input, Output&& output, const MedianFilterOptions& options) {
        if (options.window_size <= 1) {
            std::forward<Input>(input).to(output);
            return;
        }

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        check(is_odd(options.window_size), "Only odd windows are currently supported");
        check(options.border_mode == Border::ZERO or
              (output.shape()[3] >= options.window_size / 2 + 1 and output.shape()[2] >= options.window_size / 2 + 1),
              "With Border::REFLECT and a window of {}, the height and width should be larger or equal than {}, but got {}",
              options.window_size, options.window_size / 2 + 1, output.shape().filter(2, 3));

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, i = std::forward<Input>(input), o = std::forward<Output>(output)] {
                noa::signal::cpu::median_filter_2d(
                    i.get(), input_strides,
                    o.get(), o.strides(), o.shape(),
                    options.border_mode, options.window_size, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            check(nd::is_accessor_access_safe<i32>(input, output.shape()) and
                  nd::is_accessor_access_safe<i32>(output, output.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            auto& cuda_stream = stream.cuda();
            noa::signal::cuda::median_filter_2d(
                input.get(), input_strides.template as<i32>(),
                output.get(), output.strides().template as<i32>(), output.shape(),
                options.border_mode, options.window_size, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Computes the median filter using a 3d cubic window.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param options      Filter options.
    template<nt::readable_varray_decay_of_scalar Input,
             nt::writable_varray_decay_of_scalar Output>
    void median_filter_3d(Input&& input, Output&& output, const MedianFilterOptions& options) {
        if (options.window_size <= 1) {
            std::forward<Input>(input).to(output);
            return;
        }

        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        check(is_odd(options.window_size), "Only odd windows are currently supported");
        check(options.border_mode == Border::ZERO or output.shape().pop_front() >= options.window_size / 2 + 1,
              "With Border::REFLECT and a window of {}, the depth, height and width should be >= than {}, but got {}",
              options.window_size, options.window_size / 2 + 1, output.shape().pop_front());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, i = std::forward<Input>(input), o = std::forward<Output>(output)] {
                noa::signal::cpu::median_filter_3d(
                    i.get(), input_strides,
                    o.get(), o.strides(), o.shape(),
                    options.border_mode, options.window_size, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            check(nd::is_accessor_access_safe<i32>(input, output.shape()) and
                  nd::is_accessor_access_safe<i32>(output, output.shape()),
                  "GPU backend only instantiate i32-based accessor indexing, "
                  "which is unsafe for the given input and output arrays. "
                  "Please report this.");
            auto& cuda_stream = stream.cuda();
            noa::signal::cuda::median_filter_3d(
                input.get(), input_strides.template as<i32>(),
                output.get(), output.strides().template as<i32>(), output.shape(),
                options.border_mode, options.window_size, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), std::forward<Output>(output));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }
}
