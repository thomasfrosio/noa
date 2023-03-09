#pragma once

#include "noa/cpu/signal/Median.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Median.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::signal {
    /// Computes the median filter using a 1D window.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median.
    ///                     This corresponds to the width dimension.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 21.
    /// \param border_mode  Border mode used for the "implicit padding".
    ///                     Either BorderMode::ZERO or BorderMode::REFLECT.
    ///                     With BorderMode::REFLECT, the width should be >= than ``window_size/2 + 1``.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f16, f32, f64, i32, i64, u32, u64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f16, f32, f64, i32, i64, u32, u64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void median_filter_1d(const Input& input, const Output& output,
                          i64 window_size, BorderMode border_mode = BorderMode::REFLECT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(border_mode == BorderMode::ZERO || output.shape()[3] >= window_size / 2 + 1,
                  "With BorderMode::REFLECT and a window of {}, the width should be >= than {}, but got {}",
                  window_size, window_size / 2 + 1, output.shape()[3]);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::signal::median_filter_1d(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        border_mode, window_size, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::median_filter_1d(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    border_mode, window_size, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the median filter using a 2D square window.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median, for each dimension.
    ///                     This corresponds to the height and width dimension.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 11.
    /// \param border_mode  Border mode used for the "implicit padding". Either BorderMode::ZERO or BorderMode::REFLECT.
    ///                     With BorderMode::REFLECT, the height and width should be >= than ``window_size/2 + 1``.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f16, f32, f64, i32, i64, u32, u64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f16, f32, f64, i32, i64, u32, u64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void median_filter_2d(const Input& input, const Output& output,
                          i64 window_size, BorderMode border_mode = BorderMode::REFLECT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(border_mode == BorderMode::ZERO ||
                  (output.shape()[3] >= window_size / 2 + 1 && output.shape()[2] >= window_size / 2 + 1),
                  "With BorderMode::REFLECT and a window of {}, the height and width should be >= than {}, but got ({}, {})",
                  window_size, window_size / 2 + 1, output.shape()[2], output.shape()[3]);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::signal::median_filter_2d(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        border_mode, window_size, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::median_filter_2d(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    border_mode, window_size, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the median filter using a 3D cubic window.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median, for each dimension.
    ///                     This corresponds to the depth, height and width dimensions.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 5.
    /// \param border_mode  Border mode used for the "implicit padding". Either BorderMode::ZERO or BorderMode::REFLECT.
    ///                     With BorderMode::REFLECT, the depth, height and width should be >= than ``window_size/2 + 1``.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f16, f32, f64, i32, i64, u32, u64> &&
             noa::traits::is_array_or_view_of_any_v<Output, f16, f32, f64, i32, i64, u32, u64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void median_filter_3d(const Input& input, const Output& output,
                          i64 window_size, BorderMode border_mode = BorderMode::REFLECT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(border_mode == BorderMode::ZERO || all(dim3_t(output.shape().get(1)) >= window_size / 2 + 1),
                  "With BorderMode::REFLECT and a window of {}, the depth, height and width should be >= than {}, but got {}",
                  window_size, window_size / 2 + 1, dim3_t(output.shape().get(1)));

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::signal::median_filter_3d(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        border_mode, window_size, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::median_filter_3d(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    border_mode, window_size, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
