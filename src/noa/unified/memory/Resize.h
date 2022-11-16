#pragma once

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as ``shape / 2``) aligned.
    /// \param input_shape  Current shape.
    /// \param output_shape Desired shape.
    /// \return             1: The elements to add/remove from the left side of the dimensions.
    ///                     2: The elements to add/remove from the right side of the dimension.
    ///                     Positive values correspond to padding, while negative values correspond to cropping.
    [[nodiscard]] inline std::pair<int4_t, int4_t> borders(dim4_t input_shape, dim4_t output_shape);

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \tparam T           Any data type.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array. The output shape should be the sum of the input shape and the borders.
    /// \param border_left  Elements to add/remove from the left side of the axes.
    /// \param border_right Elements to add/remove from the right side of the axes.
    /// \param border_mode  Border mode to use. See BorderMode for more details.
    /// \param border_value Border value. Only used for padding if \p mode is BORDER_VALUE.
    /// \note \p output == \p input is not valid.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void resize(const Array<T>& input, const Array<T>& output,
                int4_t border_left, int4_t border_right,
                BorderMode border_mode = BORDER_ZERO, T border_value = T(0));

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \tparam T           Any data type.
    /// \param[in] input    Input array.
    /// \param border_left  Elements to add/remove from the left side of the axes.
    /// \param border_right Elements to add/remove from the right side of the axes.
    /// \param border_mode  Border mode to use. See BorderMode for more details.
    /// \param border_value Border value. Only used for padding if \p mode is BORDER_VALUE.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    [[nodiscard]] Array<T> resize(const Array<T>& input,
                                  int4_t border_left, int4_t border_right,
                                  BorderMode border_mode = BORDER_ZERO, T border_value = T(0));

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T           Any data type.
    /// \param[in] input    Input array.
    /// \param[out] output  Output array.
    /// \param border_mode  Border mode to use. See BorderMode for more details.
    /// \param border_value Border value. Only used if \p mode is BORDER_VALUE.
    /// \note \p output == \p input is not valid.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    void resize(const Array<T>& input, const Array<T>& output,
                BorderMode border_mode = BORDER_ZERO, T border_value = T(0));

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T                   Any data type.
    /// \param[in] input            Input array.
    /// \param[out] output_shape    Output shape.
    /// \param border_mode          Border mode to use. See BorderMode for more details.
    /// \param border_value         Border value. Only used if \p mode is BORDER_VALUE.
    template<typename T, typename = std::enable_if_t<traits::is_restricted_data_v<T>>>
    [[nodiscard]] Array<T> resize(const Array<T>& input, dim4_t output_shape,
                                  BorderMode border_mode = BORDER_ZERO, T border_value = T(0));
}

#define NOA_UNIFIED_RESIZE_
#include "noa/unified/memory/Resize.inl"
#undef NOA_UNIFIED_RESIZE_
