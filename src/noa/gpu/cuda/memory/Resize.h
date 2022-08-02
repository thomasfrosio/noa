/// \file noa/gpu/cuda/memory/Resize.h
/// \brief Resize memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as ``shape / 2``) aligned.
    ///
    /// \param input_shape  Current shape.
    /// \param output_shape Desired shape.
    /// \return             1: The elements to add/remove from the left side of the dimensions.
    ///                     2: The elements to add/remove from the right side of the dimension.
    ///                     Positive values correspond to padding, while negative values correspond to cropping.
    NOA_IH std::pair<int4_t, int4_t> borders(size4_t input_shape, size4_t output_shape) {
        const int4_t o_shape(output_shape);
        const int4_t i_shape(input_shape);
        const int4_t diff(o_shape - i_shape);

        const int4_t border_left = o_shape / 2 - i_shape / 2;
        const int4_t border_right = diff - border_left;
        return {border_left, border_right};
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \tparam T               Any data type.
    /// \param[in] input        On the \b device. Input array.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param input_shape      Shape of \p input.
    /// \param border_left      Elements to add/remove from the left side of the axes.
    /// \param border_right     Elements to add/remove from the right side of the axes.
    /// \param[out] output      On the \b device. Output array.
    ///                         The output shape is \p input_shape + \p border_left + \p border_right.
    /// \param output_strides   Strides, in elements, of \p output.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used for padding if \p mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output == \p input is not valid.
    /// \note The resulting output shape should be valid, i.e. no dimensions should be <= 0.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void resize(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                int4_t border_left, int4_t border_right,
                const shared_t<T[]>& output, size4_t output_strides,
                BorderMode border_mode, T border_value, Stream& stream);

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T               Any data type.
    /// \param[in] input        On the \b device. Input array.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param input_shape      Shape of \p input.
    /// \param[out] output      On the \b device. Output array.
    /// \param output_strides   Strides, in elements, of \p output.
    /// \param output_shape     Shape of \p output.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used if \p mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output == \p input is not valid.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    NOA_IH void resize(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                       const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                       BorderMode border_mode, T border_value, Stream& stream) {
        auto[border_left, border_right] = borders(input_shape, output_shape);
        resize(input, input_strides, input_shape,
               border_left, border_right,
               output, output_strides,
               border_mode, border_value, stream);
    }
}
