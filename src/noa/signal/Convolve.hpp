#pragma once

#include "noa/signal/cpu/Convolve.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/signal/cuda/Convolve.cuh"
#endif

#include "noa/runtime/Array.hpp"

namespace noa::signal {
    struct ConvolveOptions {
        /// Border mode used for the convolution.
        /// Only Border::ZERO and Border::REFLECT are currently supported.
        Border border{Border::ZERO};
    };

    /// 1d, 2d or 3d convolution.
    /// \param[in] input    Array to convolve.
    /// \param[out] output  Convolved array. Should not overlap with \p input.
    /// \param[in] filter   1d, 2d or 3d C-contiguous filter. The same filter is applied to every output batch.
    ///                     Dimensions should have an odd number of elements. Dimensions don't have to have the same size.
    /// \note The precision of the convolution is the floating-point precision of the \p filter value type.
    template<nt::readable_varray_decay_of_real Input,
             nt::writable_varray_decay_of_real Output,
             nt::readable_varray_decay_of_real Filter>
    void convolve(Input&& input, Output&& output, Filter&& filter, const ConvolveOptions& options = {}) {
        check(nd::are_arrays_valid(input, output, filter), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(ni::broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        check(not filter.shape().is_batched() and filter.are_contiguous(),
              "The input filter shouldn't be batched and should be contiguous, "
              "but got filter:shape={} and filter:strides={}",
              filter.shape(), filter.strides());

        const auto filter_shape = filter.shape().pop_front();
        auto ndim = output.shape().ndim();
        check(filter_shape.ndim() <= ndim and (filter_shape % 2) == 1,
              "Given a {}d (batched) output, the input filter should be {}d at most and each dimension "
              "should have an odd number of elements, but got filter shape {}", ndim, ndim, filter_shape);

        const Device device = output.device();
        check(device == input.device() and device == filter.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={}, filter:device={}, output:device={}",
              input.device(), filter.device(), device);

        check(options.border.is_any(Border::ZERO, Border::REFLECT), "The provided border mode is not supported");

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=,
                i = std::forward<Input>(input),
                o = std::forward<Output>(output),
                f = std::forward<Filter>(filter)] {
                if (options.border == Border::ZERO) {
                    noa::signal::cpu::convolve<Border::ZERO>(
                        i.get(), input_strides,
                        o.get(), o.strides(), o.shape(),
                        f.get(), filter_shape, threads);
                } else {
                    noa::signal::cpu::convolve<Border::REFLECT>(
                        i.get(), input_strides,
                        o.get(), o.strides(), o.shape(),
                        f.get(), filter_shape, threads);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if (options.border == Border::ZERO) {
                noa::signal::cuda::convolve<Border::ZERO>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter.get(), filter_shape, cuda_stream);
            } else {
                noa::signal::cuda::convolve<Border::REFLECT>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter.get(), filter_shape, cuda_stream);
            }
            cuda_stream.enqueue_attach(
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<Filter>(filter));
            #else
            panic_no_gpu_backend();
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
    template<nt::readable_varray_decay_of_real Input,
             nt::writable_varray_decay_of_real Output,
             nt::readable_varray_decay_of_real FilterDepth = View<nt::const_value_type_t<Input>>,
             nt::readable_varray_decay_of_real FilterHeight = View<nt::const_value_type_t<Input>>,
             nt::readable_varray_decay_of_real FilterWidth = View<nt::const_value_type_t<Input>>,
             nt::writable_varray_decay_of_real Buffer = View<nt::value_type_t<Output>>>
    requires nt::varray_decay_of_almost_same_type<FilterDepth, FilterHeight, FilterWidth, Buffer>
    void convolve_separable(
        Input&& input,
        Output&& output,
        FilterDepth&& filter_depth,
        FilterHeight&& filter_height,
        FilterWidth&& filter_width,
        Buffer&& buffer = Buffer{},
        const ConvolveOptions& options = {}
    ) {
        check(nd::are_arrays_valid(input, output), "Empty array detected");
        check(not ni::are_overlapped(input, output), "The input and output array should not overlap");

        auto input_strides = input.strides();
        check(ni::broadcast(input.shape(), input_strides, output.shape()),
              "Cannot broadcast an array of shape {} into an array of shape {}",
              input.shape(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input={}, output={}",
              input.device(), device);

        auto check_separable_filter = [&]<typename T>(
            const T& filter,
            std::source_location location
        ) {
            if (filter.is_empty())
                return;
            check_at_location(
                location, ni::is_contiguous_vector(filter) and is_odd(filter.n_elements()),
                "The input filters should be contiguous vectors with an odd number of elements, "
                "but got a filter with a shape {} and strides {}", filter.shape(), filter.strides());
            check_at_location(
                location, filter.device() == device,
                "The input filters must be on the same device as the compute device, "
                "but got output:device={}, filter:device={}",
                device, filter.device());
        };
        check_separable_filter(filter_depth, std::source_location::current());
        check_separable_filter(filter_height, std::source_location::current());
        check_separable_filter(filter_width, std::source_location::current());

        if (not buffer.is_empty()) {
            check(buffer.shape() == output.shape() and ni::are_elements_unique(buffer.strides(), output.shape()),
                  "The temporary array should be able to hold an array of shape {}, but got shape {} and strides {}",
                  output.shape(), buffer.shape(), buffer.strides());
            check(device == buffer.device(),
                  "The temporary array must be on the same device as the output, "
                  "but got buffer:device={}, output:device={}",
                  buffer.device(), device);
        }

        check(options.border.is_any(Border::ZERO, Border::REFLECT), "The provided border mode is not supported");

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=,
                i = std::forward<Input>(input),
                o = std::forward<Output>(output),
                fd = std::forward<FilterDepth>(filter_depth),
                fh = std::forward<FilterHeight>(filter_height),
                fw = std::forward<FilterWidth>(filter_width),
                b = std::forward<Buffer>(buffer)] {
                if (options.border == Border::ZERO) {
                    noa::signal::cpu::convolve_separable<Border::ZERO>(
                        i.get(), input_strides,
                        o.get(), o.strides(), o.shape(),
                        fd.get(), fd.n_elements(),
                        fh.get(), fh.n_elements(),
                        fw.get(), fw.n_elements(),
                        b.get(), b.strides(), threads);
                } else {
                    noa::signal::cpu::convolve_separable<Border::REFLECT>(
                        i.get(), input_strides,
                        o.get(), o.strides(), o.shape(),
                        fd.get(), fd.n_elements(),
                        fh.get(), fh.n_elements(),
                        fw.get(), fw.n_elements(),
                        b.get(), b.strides(), threads);
                }
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            if (options.border == Border::ZERO) {
                noa::signal::cuda::convolve_separable<Border::ZERO>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter_depth.get(), filter_depth.n_elements(),
                    filter_height.get(), filter_height.n_elements(),
                    filter_width.get(), filter_width.n_elements(),
                    buffer.get(), buffer.strides(), cuda_stream);
            } else {
                noa::signal::cuda::convolve_separable<Border::REFLECT>(
                    input.get(), input_strides,
                    output.get(), output.strides(), output.shape(),
                    filter_depth.get(), filter_depth.n_elements(),
                    filter_height.get(), filter_height.n_elements(),
                    filter_width.get(), filter_width.n_elements(),
                    buffer.get(), buffer.strides(), cuda_stream);
            }
            cuda_stream.enqueue_attach(
                std::forward<Input>(input),
                std::forward<Output>(output),
                std::forward<FilterDepth>(filter_depth),
                std::forward<FilterHeight>(filter_height),
                std::forward<FilterWidth>(filter_width),
                std::forward<Buffer>(buffer));
            #else
            panic_no_gpu_backend();
            #endif
        }
    }
}
