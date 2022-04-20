#pragma once

#include "noa/cpu/memory/Resize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/memory/Resize.h"
#endif

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as ``shape / 2``) aligned.
    /// \param input_shape  Current rightmost shape.
    /// \param output_shape Desired rightmost shape.
    /// \return             1: The rightmost elements to add/remove from the left side of the dimension.
    ///                     2: The rightmost elements to add/remove from the right side of the dimension.
    ///                     Positive values correspond to padding, while negative values correspond to cropping.
    NOA_IH std::pair<int4_t, int4_t> borders(size4_t input_shape, size4_t output_shape) {
        int4_t o_shape(output_shape);
        int4_t i_shape(input_shape);
        int4_t diff(o_shape - i_shape);

        int4_t border_left = o_shape / 2 - i_shape / 2;
        int4_t border_right = diff - border_left;
        return {border_left, border_right}; // TODO If noa::Pair<> is added, use it instead.
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \tparam T               Any data type.
    /// \param[in] input        Input array.
    /// \param[out] output      Output array. The output shape should be the sum of the input shape and the borders.
    /// \param border_left      Rightmost elements to add/remove from the left side of the axes.
    /// \param border_right     Rightmost elements to add/remove from the right side of the axes.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used for padding if \p mode is BORDER_VALUE.
    /// \note \p output == \p input is not valid.
    template<typename T>
    void resize(const Array<T>& input, const Array<T>& output,
                int4_t border_left, int4_t border_right,
                BorderMode border_mode = BORDER_ZERO, T border_value = T(0)) {
        const Device device(output.device());
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, input.device());
        NOA_CHECK(all(int4_t{output.shape()} == int4_t{input.shape()} + border_left + border_right),
                  "The output shape ({}) does not math the expected shape (input:{}, left:{}, right:{})",
                  output.shape(), input.shape(), border_left, border_right);
        NOA_CHECK(input.get() != output.get(), "In-place resizing is not allowed");

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::memory::resize<T>(input.share(), input.stride(), input.shape(),
                                   border_left, border_right,
                                   output.share(), output.stride(),
                                   border_mode, border_value, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::memory::resize<T>(input.share(), input.stride(), input.shape(),
                                    border_left, border_right,
                                    output.share(), output.stride(),
                                 border_mode, border_value, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T               Any data type.
    /// \param[in] input        On the \b device. Input array.
    /// \param[out] output      On the \b device. Output array.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used if \p mode is BORDER_VALUE.
    /// \note \p output == \p input is not valid.
    template<typename T>
    NOA_IH void resize(const Array<T>& input, const Array<T>& output,
                       BorderMode border_mode = BORDER_ZERO, T border_value = T(0)) {
        auto[border_left, border_right] = borders(input.shape(), output.shape());
        resize(input, output, border_left, border_right, border_mode, border_value);
    }
}
