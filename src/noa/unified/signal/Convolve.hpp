#pragma once

#include "noa/cpu/signal/Convolve.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Convolve.hpp"
namespace noa::signal::details {
    constexpr i64 CUDA_FILTER_MAX_BYTES = 1032;
}
#endif

#include "noa/unified/Array.hpp"

namespace noa::signal::details {
    template<typename Filter>
    void check_separable_filter(const Filter& filter, const Device& compute_device) {
        if (filter.is_empty())
            return;

        NOA_CHECK(noa::indexing::is_contiguous_vector(filter) && filter.elements() % 2 == 1,
                  "The input filters should be contiguous vectors with an odd number of elements, "
                  "but got a filter with a shape {} and strides {}", filter.shape(), filter.strides());
        NOA_CHECK(filter.device() == compute_device,
                  "The input filters must be on the same device as the compute device, but got device:{}, filter:{}",
                  compute_device, filter.device());
    }
}

namespace noa::signal {
    /// ND convolution.
    /// \param[in] input    Array to convolve.
    /// \param[out] output  Convolved array. Should not overlap with \p input.
    /// \param[in] filter   1D, 2D or 3D C-contiguous filter. The same filter is applied to every output batch.
    ///                     Dimensions should have an odd number of elements. Dimensions don't have to have the same size.
    ///                     On the GPU, each dimension is limited to 129 (1d), 17 (2d) and 5 (3d) elements.
    ///
    /// \bug This function modifies the GPU state via the usage of constant memory. As such, there should be no
    ///      concurrent calls from different streams associated to the same GPU.
    template<typename Input, typename Output, typename Filter, typename = std::enable_if_t<
             nt::is_varray_of_almost_any_v<Input, f16, f32, f64> &&
             nt::is_varray_of_any_v<Output, f16, f32, f64> &&
             nt::is_varray_of_almost_any_v<Filter, f16, f32, f64> &&
             nt::are_almost_same_value_type_v<Input, Output, Filter>>>
    void convolve(const Input& input, const Output& output, const Filter& filter) {
        NOA_CHECK(!input.is_empty() && !output.is_empty() && !filter.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        NOA_CHECK(!filter.shape().is_batched() && filter.are_contiguous(),
                  "The input filter shouldn't be batched and must be contiguous, but got shape {} and strides {}",
                  filter.shape(), filter.strides());

        const auto filter_shape = filter.shape().pop_front();
        auto ndim = output.shape().ndim();
        NOA_CHECK(filter_shape.ndim() <= ndim && noa::all(filter_shape % 2 == 1),
                  "Given a {}d (batched) output, the input filter should be {}d at most and each dimension "
                  "should have an odd number of elements, but got filter shape {}", ndim, ndim, filter_shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device() && device == filter.device(),
                  "The input and output arrays must be on the same device, but got input:{}, filter:{}, output:{}",
                  input.device(), filter.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::convolve(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        filter.get(), filter_shape, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            using value_t = typename Filter::value_type;
            NOA_CHECK(filter_shape.elements() * static_cast<i64>(sizeof(value_t)) <= details::CUDA_FILTER_MAX_BYTES,
                      "In the CUDA backend, the filter size is limited to {} bytes, "
                      "but got filter shape {} and type {}",
                      details::CUDA_FILTER_MAX_BYTES, filter_shape, string::human<value_t>());

            auto& cuda_stream = stream.cuda();
            cuda::signal::convolve(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter.get(), filter_shape, cuda_stream);
            cuda_stream.enqueue_attach(input, output, filter);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Separable convolutions. \p input is convolved with \p filter_depth, then \p filter_height, then \p filter_width.
    /// \param[in] input            Input array to convolve.
    /// \param[out] output          Output convolved array. Should not overlap with \p input.
    /// \param[in] filter_depth     1d filter with an odd number of elements applied along the depth dimension.
    /// \param[in] filter_height    1d filter with an odd number of elements applied along the height dimension.
    /// \param[in] filter_width     1d filter with an odd number of elements applied along the width dimension.
    /// \param[out] buffer          Temporary array. If only one dimension is filtered, this is ignored. Otherwise,
    ///                             it should be an array of the same shape as \p output, or be an empty array,
    ///                             in which case a temporary array will be allocated internally.
    ///
    /// \note Filters can be empty. In these cases, the convolution in the corresponding dimension is not applied
    ///       and it goes directly to the next filter, if any. Filters can be equal to each other.
    /// \note If the output is on the GPU, filters are limited to a maximum size of 1032 bytes.
    ///
    /// \bug This function modifies the GPU state via the usage of constant memory. As such, there should be no
    ///      concurrent calls from different streams associated to the same GPU.
    template<typename Input, typename Output,
             typename FilterDepth = View<const nt::value_type_t<Output>>,
             typename FilterHeight = View<const nt::value_type_t<Output>>,
             typename FilterWidth = View<const nt::value_type_t<Output>>,
             typename Buffer = View<nt::value_type_t<Output>>, typename = std::enable_if_t<
                    nt::is_varray_of_almost_any_v<Input, f16, f32, f64> &&
                    nt::is_varray_of_any_v<Output, f16, f32, f64> &&
                    nt::is_varray_of_almost_any_v<FilterDepth, f16, f32, f64> &&
                    nt::is_varray_of_almost_any_v<FilterHeight, f16, f32, f64> &&
                    nt::is_varray_of_almost_any_v<FilterWidth, f16, f32, f64> &&
                    nt::is_varray_of_any_v<Buffer, f16, f32, f64> &&
                    nt::are_almost_same_value_type_v<Input, Output, FilterDepth, FilterHeight, FilterWidth, Buffer>>>
    void convolve_separable(const Input& input, const Output& output,
                            const FilterDepth& filter_depth,
                            const FilterHeight& filter_height,
                            const FilterWidth& filter_width,
                            const Buffer& buffer = Buffer{}) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        details::check_separable_filter(filter_depth, device);
        details::check_separable_filter(filter_height, device);
        details::check_separable_filter(filter_width, device);

        if (!buffer.is_empty()) {
            NOA_CHECK(noa::all(buffer.shape() == output.shape()) && !noa::any(buffer.strides() == 0),
                      "The temporary array should be able to hold an array of shape {}, but got shape {} and strides {}",
                      output.shape(), buffer.shape(), buffer.strides());
            NOA_CHECK(device == buffer.device(),
                      "The temporary array must be on the same device as the output, but got buffer:{}, output:{}",
                      buffer.device(), device);
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::convolve_separable(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        filter_depth.get(), filter_depth.elements(),
                        filter_height.get(), filter_height.elements(),
                        filter_width.get(), filter_width.elements(),
                        buffer.get(), buffer.strides(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            using value_t = typename FilterDepth::value_type;
            constexpr i64 BYTES_PER_ELEMENTS = sizeof(value_t);
            NOA_CHECK(filter_depth.elements() * BYTES_PER_ELEMENTS <= details::CUDA_FILTER_MAX_BYTES &&
                      filter_height.elements() * BYTES_PER_ELEMENTS <= details::CUDA_FILTER_MAX_BYTES &&
                      filter_width.elements() * BYTES_PER_ELEMENTS <= details::CUDA_FILTER_MAX_BYTES,
                      "In the CUDA backend, separable filters have a size limited to {} bytes",
                      details::CUDA_FILTER_MAX_BYTES);

            auto& cuda_stream = stream.cuda();
            cuda::signal::convolve_separable(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter_depth.get(), filter_depth.elements(),
                    filter_height.get(), filter_height.elements(),
                    filter_width.get(), filter_width.elements(),
                    buffer.get(), buffer.strides(), cuda_stream);
            cuda_stream.enqueue_attach(
                    input, output, filter_depth, filter_height, filter_width);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
