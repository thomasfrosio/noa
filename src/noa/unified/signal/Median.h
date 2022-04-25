#pragma once

#include "noa/cpu/signal/Median.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Median.h"
#endif

#include "noa/unified/Array.h"

namespace noa::filter {
    /// Computes the median filter using a 1D window.
    /// \tparam T           (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median.
    ///                     This corresponds to the innermost dimension.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 21.
    /// \param border_mode  Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    ///                     With BORDER_REFLECT, the innermost dimension should be >= than ``window_size/2 + 1``.
    template<typename T>
    void median1(const Array<T>& input, const Array<T>& output,
                 size_t window_size, BorderMode border_mode = BORDER_REFLECT) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::median1(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median1(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the median filter using a 2D square window.
    /// \tparam T           (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median, for each dimension.
    ///                     This corresponds to the second and innermost dimension.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 11.
    /// \param border_mode  Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    ///                     With BORDER_REFLECT, the second and innermost dimensions should be >= than ``window_size/2 + 1``.
    template<typename T>
    void median2(const Array<T>& input, const Array<T>& output,
                 size_t window_size, BorderMode border_mode = BORDER_REFLECT) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::median2(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median2(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the median filter using a 3D cubic window.
    /// \tparam T           (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Array to filter.
    /// \param[out] output  Filtered array. Should not overlap with \p input.
    /// \param window_size  Number of elements to consider for the computation of the median, for each dimension.
    ///                     This corresponds to the 3 innermost dimensions.
    ///                     Only odd numbers are supported. On the GPU, this is limited to 5.
    /// \param border_mode  Border mode used for the "implicit padding". Either BORDER_ZERO or BORDER_REFLECT.
    ///                     With BORDER_REFLECT, each of the 3 innermost dimension should be >= than ``window_size/2 + 1``.
    template<typename T>
    void median3(const Array<T>& input, const Array<T>& output,
                 size_t window_size, BorderMode border_mode = BORDER_REFLECT) {
        size4_t input_stride = input.stride();
        if (!indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::median3(input.share(), input_stride,
                                 output.share(), output.stride(), output.shape(),
                                 border_mode, window_size, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::median3(input.share(), input_stride,
                                  output.share(), output.stride(), output.shape(),
                                  border_mode, window_size, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
