#pragma once

#include "noa/algorithms/memory/Resize.hpp"
#include "noa/cpu/memory/Resize.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Resize.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as \c shape/2) aligned.
    /// \param input_shape  Current shape.
    /// \param output_shape Desired shape.
    /// \return             1: The elements to add/remove from the left side of the dimensions.
    ///                     2: The elements to add/remove from the right side of the dimension.
    ///                     Positive values correspond to padding, while negative values correspond to cropping.
    [[nodiscard]] inline auto shape2borders(
            const Shape4<i64>& input_shape,
            const Shape4<i64>& output_shape
    ) -> std::pair<Vec4<i64>, Vec4<i64>> {
        return noa::algorithm::memory::borders(input_shape, output_shape);
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array. The output shape should be the sum of the input shape and the borders.
    /// \param border_left  Elements to add/remove from the left side of the axes.
    /// \param border_right Elements to add/remove from the right side of the axes.
    /// \param border_mode  Border mode to use. See BorderMode for more details.
    /// \param border_value Border value. Only used for padding if \p mode is BorderMode::VALUE.
    /// \note \p output == \p input is not valid.
    template<typename Input, typename Output,
             typename Value = nt::value_type_t<Output>, typename = std::enable_if_t<
             nt::are_varray_of_restricted_numeric_v<Input, Output> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             nt::is_almost_same_v<nt::value_type_t<Output>, Value>>>
    void resize(const Input& input, const Output& output,
                const Vec4<i64>& border_left, const Vec4<i64>& border_right,
                BorderMode border_mode = BorderMode::ZERO, Value border_value = Value{0}) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  input.device(), device);
        NOA_CHECK(noa::all(output.shape().vec() == input.shape().vec() + border_left + border_right),
                  "The output shape {} does not match the expected shape (input:{}, left:{}, right:{})",
                  output.shape(), input.shape(), border_left, border_right);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=](){
                cpu::memory::resize(input.get(), input.strides(), input.shape(),
                                    border_left, border_right,
                                    output.get(), output.strides(),
                                    border_mode, border_value, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::memory::resize(input.get(), input.strides(), input.shape(),
                                 border_left, border_right,
                                 output.get(), output.strides(),
                                 border_mode, border_value, cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \param[in] input    Input array.
    /// \param border_left  Elements to add/remove from the left side of the axes.
    /// \param border_right Elements to add/remove from the right side of the axes.
    /// \param border_mode  Border mode to use. See BorderMode for more details.
    /// \param border_value Border value. Only used for padding if \p mode is BorderMode::VALUE.
    template<typename Input, typename Value = nt::mutable_value_type_t<Input>, typename = std::enable_if_t<
             nt::is_varray_of_numeric_v<Input> &&
             nt::is_almost_same_v<nt::value_type_t<Input>, Value>>>
    [[nodiscard]] auto resize(const Input& input,
                              const Vec4<i64>& border_left, const Vec4<i64>& border_right,
                              BorderMode border_mode = BorderMode::ZERO, Value border_value = Value{0}) {
        const auto output_shape = Shape4<i64>(input.shape().vec() + border_left + border_right);
        NOA_CHECK(noa::all(output_shape > 0),
                  "Cannot resize [left:{}, right:{}] an array of shape {} into an array of shape {}",
                  border_left, border_right, input.shape(), output_shape);
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(output_shape, input.options());
        resize(input, output, border_left, border_right, border_mode, border_value);
        return output;
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array.
    /// \param border_mode  Border mode to use. See BorderMode for more details.
    /// \param border_value Border value. Only used if \p mode is BorderMode::VALUE.
    /// \note \p output == \p input is not valid.
    template<typename Input, typename Output,
             typename Value = nt::value_type_t<Output>, typename = std::enable_if_t<
             nt::are_varray_of_restricted_numeric_v<Input, Output> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             nt::is_almost_same_v<nt::value_type_t<Input>, Value>>>
    void resize(const Input& input, const Output& output,
                BorderMode border_mode = BorderMode::ZERO, Value border_value = Value{0}) {
        const auto [border_left, border_right] = shape2borders(input.shape(), output.shape());
        resize(input, output, border_left, border_right, border_mode, border_value);
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \param[in] input            Input array.
    /// \param[out] output_shape    Output shape.
    /// \param border_mode          Border mode to use. See BorderMode for more details.
    /// \param border_value         Border value. Only used if \p mode is BorderMode::VALUE.
    template<typename Input, typename Value = nt::mutable_value_type_t<Input>, typename = std::enable_if_t<
             nt::is_varray_of_numeric_v<Input> &&
             nt::is_almost_same_v<nt::value_type_t<Input>, Value>>>
    [[nodiscard]] auto resize(const Input& input, const Shape4<i64>& output_shape,
                              BorderMode border_mode = BorderMode::ZERO, Value border_value = Value{0}) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(output_shape, input.options());
        resize(input, output, border_mode, border_value);
        return output;
    }
}
