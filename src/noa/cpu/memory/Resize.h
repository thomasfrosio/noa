/// \file noa/cpu/memory/Resize.h
/// \brief Resize memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

// 2D Borders:
//                             (2)
//  shape(x=6,y=5): [[ 0,  1,  2,  3,  4,  5],
//                   [ 6,  7,  8,  9, 10, 11],
//               (1) [12, 13, 14, 15, 16, 17], (3)
//                   [18, 19, 20, 21, 22, 23],
//                   [24, 25, 26, 27, 28, 29]]
//                             (4)
// border_left = [1,2]
// border_right = [3,4]

namespace noa::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \a input_shape to
    /// \a output_shape, while keeping the centers of the input and output array (defined as `shape / 2`) aligned.
    ///
    /// \param input_shape      Current shape
    /// \param output_shape     Desired shape
    /// \return                 1: The {x, y, z} elements to add/remove from the left side of the dimension.
    /// \param[out]             2: The {x, y, z} elements to add/remove from the right side of the dimension.
    ///                         Positive values correspond to padding, while negative values correspond to cropping.
    NOA_IH std::pair<int3_t, int3_t> setBorders(size3_t input_shape, size3_t output_shape) {
        int3_t o_shape(output_shape);
        int3_t i_shape(input_shape);
        int3_t diff(o_shape - i_shape);

        int3_t border_left = o_shape / 2 - i_shape / 2;
        int3_t border_right = diff - border_left;
        return {border_left, border_right};
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \tparam T               float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs       Input array(s). One per batch.
    /// \param input_shape      Physical {fast, medium, slow} shape of \a inputs, ignoring the batch size.
    /// \param border_left      The {x, y, z} elements to add/remove from the left side of the dimension.
    /// \param border_right     The {x, y, z} elements to add/remove from the right side of the dimension.
    /// \param[out] outputs     Output array(s). One per batch.
    ///                         The output shape is \a input_shape + \a border_left + \a border_right.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used if \a mode == BORDER_VALUE.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    ///
    /// \note \a outputs == \a inputs is not valid.
    /// \note The implicit output shape should be valid, i.e. no dimensions should be <= 0.
    template<typename T>
    NOA_HOST void resize(const T* inputs, size3_t input_shape, int3_t border_left, int3_t border_right,
                         T* outputs, BorderMode border_mode, T border_value, uint batches);

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T               float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs       Input array(s). One per batch.
    /// \param input_shape      Physical {fast, medium, slow} shape of \a inputs, ignoring the batch size.
    /// \param[out] outputs     Output array(s). One per batch.
    /// \param output_shape     Physical {fast, medium, slow} shape of \a inputs, ignoring the batch size.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used if \a mode == BORDER_VALUE.
    /// \param batches          Number of batches in \a inputs and \a outputs.
    /// \note \a outputs == \a inputs is not valid.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                       BorderMode border_mode, T border_value, uint batches) {
        auto[border_left, border_right] = setBorders(input_shape, output_shape);
        resize(inputs, input_shape, border_left, border_right, outputs, border_mode, border_value, batches);
    }
}
