#pragma once

#include "noa/cpu/signal/Convolve.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/Convolve.hpp"
#endif
#include "noa/unified/Array.hpp"

namespace noa::signal::guts {
    template<typename Filter>
    void check_separable_filter(const Filter& filter, const Device& compute_device) {
        if (filter.is_empty())
            return;
        check(ni::is_contiguous_vector(filter) and is_odd(filter.elements()),
              "The input filters should be contiguous vectors with an odd number of elements, "
              "but got a filter with a shape {} and strides {}", filter.shape(), filter.strides());
        check(filter.device() == compute_device,
              "The input filters must be on the same device as the compute device, but got device:{}, filter:{}",
              compute_device, filter.device());
    }
}

namespace noa::signal {
    /// 1d, 2d or 3d convolution.
    /// \param[in] input    Array to convolve.
    /// \param[out] output  Convolved array. Should not overlap with \p input.
    /// \param[in] filter   1d, 2d or 3d C-contiguous filter. The same filter is applied to every output batch.
    ///                     Dimensions should have an odd number of elements. Dimensions don't have to have the same size.
    ///                     On the GPU, each dimension is currently limited to 129 (1d), 17 (2d) and 5 (3d) elements.
    /// \note The precision of the convolution is the floating-point precision of the \p filter value type.
    template<typename Input, typename Output, typename Filter>
    requires (nt::are_varray_of_real_v<Input, Output, Filter> and nt::is_varray_of_mutable_v<Output>)
    void convolve(const Input& input, const Output& output, const Filter& filter) {
        check(not input.is_empty() and not output.is_empty() and not filter.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(ni::broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        check(not filter.shape().is_batched() and filter.are_contiguous(),
              "The input filter shouldn't be batched and should be contiguous, but got shape={} and strides={}",
              filter.shape(), filter.strides());

        const auto filter_shape = filter.shape().pop_front();
        auto ndim = output.shape().ndim();
        check(filter_shape.ndim() <= ndim and all(filter_shape % 2 == 1),
              "Given a {}d (batched) output, the input filter should be {}d at most and each dimension "
              "should have an odd number of elements, but got filter shape {}", ndim, ndim, filter_shape);

        const Device device = output.device();
        check(device == input.device() and device == filter.device(),
              "The input and output arrays must be on the same device, but got input={}, filter={}, output={}",
              input.device(), filter.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::signal::convolve(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        filter.get(), filter_shape, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            using value_t = Filter::value_type;
            NOA_CHECK(filter_shape.elements() * static_cast<i64>(sizeof(value_t)) <= 1032,
                      "In the CUDA backend, the filter size is limited to 1032 bytes, but got filter shape {} and type {}",
                      filter_shape, string::human<value_t>());

            auto& cuda_stream = stream.cuda();
            noa::cuda::signal::convolve(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter.get(), filter_shape, cuda_stream);
            cuda_stream.enqueue_attach(input, output, filter);
            #else
            panic("No GPU backend detected");
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
    /// \note The precision of the convolution is the floating-point precision of the filters value type.
    /// \note Filters can be empty. In these cases, the convolution in the corresponding dimension is not applied
    ///       and it goes directly to the next filter, if any. Filters can be equal to each other.
    /// \note If the output is on the GPU, filters are limited to a maximum size of 1032 bytes.
    template<typename Input, typename Output,
             typename FilterDepth = View<const nt::value_type_t<Input>>,
             typename FilterHeight = View<const nt::value_type_t<Input>>,
             typename FilterWidth = View<const nt::value_type_t<Input>>,
             typename Buffer = View<nt::value_type_t<Output>>>
    requires (nt::are_varray_of_real_v<Input, Output, FilterDepth, FilterHeight, FilterWidth, Buffer> and
              nt::are_varray_of_mutable_v<Output, Buffer> &&
              nt::are_almost_same_value_type_v<FilterDepth, FilterHeight, FilterWidth, Buffer>)
    void convolve_separable(
            const Input& input,
            const Output& output,
            const FilterDepth& filter_depth,
            const FilterHeight& filter_height,
            const FilterWidth& filter_width,
            const Buffer& buffer = Buffer{}
    ) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(ni::broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input={}, output={}",
              input.device(), device);

        guts::check_separable_filter(filter_depth, device);
        guts::check_separable_filter(filter_height, device);
        guts::check_separable_filter(filter_width, device);

        if (not buffer.is_empty()) {
            check(all(buffer.shape() == output.shape()) and all(buffer.strides() > 1),
                  "The temporary array should be able to hold an array of shape {}, but got shape {} and strides {}",
                  output.shape(), buffer.shape(), buffer.strides());
            check(device == buffer.device(),
                  "The temporary array must be on the same device as the output, but got buffer={}, output={}",
                  buffer.device(), device);
        }

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::signal::convolve_separable(
                        input.get(), input_strides,
                        output.get(), output.strides(), output.shape(),
                        filter_depth.get(), filter_depth.elements(),
                        filter_height.get(), filter_height.elements(),
                        filter_width.get(), filter_width.elements(),
                        buffer.get(), buffer.strides(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            using value_t = FilterDepth::value_type;
            check(filter_depth.elements() * static_cast<i64>(sizeof(value_t)) <= 1032 and
                  filter_height.elements() * static_cast<i64>(sizeof(value_t)) <= 1032 and
                  filter_width.elements() * static_cast<i64>(sizeof(value_t)) <= 1032,
                  "In the CUDA backend, separable filters have a size limited to 1032 bytes");

            auto& cuda_stream = stream.cuda();
            noa::cuda::signal::convolve_separable(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter_depth.get(), filter_depth.elements(),
                    filter_height.get(), filter_height.elements(),
                    filter_width.get(), filter_width.elements(),
                    buffer.get(), buffer.strides(), cuda_stream);
            cuda_stream.enqueue_attach(
                    input, output, filter_depth, filter_height, filter_width);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }
}
