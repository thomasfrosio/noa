/// \file noa/cpu/memory/Resize.h
/// \brief Resize memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as ``shape / 2``) aligned.
    ///
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
    /// \param[in] input        On the \b host. Input array.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape of \p input.
    /// \param border_left      Rightmost elements to add/remove from the left side of the axes.
    /// \param border_right     Rightmost elements to add/remove from the right side of the axes.
    /// \param[out] output      On the \b host. Output array.
    ///                         The output shape is \p input_shape + \p border_left + \p border_right.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used for padding if \p mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output == \p input is not valid.
    /// \note The resulting output shape should be valid, i.e. no dimensions should be <= 0.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    void resize(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                int4_t border_left, int4_t border_right,
                const shared_t<T[]>& output, size4_t output_stride,
                BorderMode border_mode, T border_value, Stream& stream);

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T               Any data type.
    /// \param[in] input        On the \b host. Input array.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Output array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape of \p output.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used if \p mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output == \p input is not valid.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void resize(const shared_t<const T[]>& input, size4_t input_stride, size4_t input_shape,
                       const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                       BorderMode border_mode, T border_value, Stream& stream) {
        auto[border_left, border_right] = borders(input_shape, output_shape);
        resize(input, input_stride, input_shape,
               border_left, border_right,
               output, output_stride,
               border_mode, border_value, stream);
    }
}
