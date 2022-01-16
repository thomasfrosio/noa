/// \file noa/cpu/memory/Resize.h
/// \brief Resize memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

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

namespace noa::cpu::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as `shape / 2`) aligned.
    ///
    /// \param input_shape      Current {fast, medium, slow} shape.
    /// \param output_shape     Desired {fast, medium, slow} shape.
    /// \return                 1: The {fast, medium, slow} elements to add/remove from the left side of the dimension.
    ///                         2: The {fast, medium, slow} elements to add/remove from the right side of the dimension.
    ///                         Positive values correspond to padding, while negative values correspond to cropping.
    NOA_IH std::pair<int3_t, int3_t> setBorders(size3_t input_shape, size3_t output_shape) {
        int3_t o_shape(output_shape);
        int3_t i_shape(input_shape);
        int3_t diff(o_shape - i_shape);

        int3_t border_left = o_shape / 2 - i_shape / 2;
        int3_t border_right = diff - border_left;
        return {border_left, border_right}; // TODO If noa::Pair<> is added, use it instead.
    }

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \tparam T               bool, (u)char, (u)short, (u)int, (u)long, (u)long long, or (complex) floating-point.
    /// \param[in] inputs       On the \b host. Input array(s). One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param input_shape      Logical {fast,medium,slow} shape of \p inputs.
    /// \param border_left      The {x, y, z} elements to add/remove from the left side of the dimension.
    /// \param border_right     The {x, y, z} elements to add/remove from the right side of the dimension.
    /// \param[out] outputs     On the \b host. Output array(s). One per batch.
    ///                         The output shape is \p input_shape + \p border_left + \p border_right.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param batches          Number of batches to compute in \p inputs and \p outputs.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used for padding if \p mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs == \p inputs is not valid.
    /// \note The resulting output shape should be valid, i.e. no dimensions should be <= 0.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void resize(const T* inputs, size3_t input_pitch, size3_t input_shape,
                         int3_t border_left, int3_t border_right,
                         T* outputs, size3_t output_pitch, size_t batches,
                         BorderMode border_mode, T border_value, Stream& stream);

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T               bool, (u)char, (u)short, (u)int, (u)long, (u)long long, or (complex) floating-point.
    /// \param[in] inputs       On the \b host. Input array(s). One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param input_shape      Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Output array(s). One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param output_shape     Logical {fast,medium,slow} shape of \p outputs.
    /// \param batches          Number of batches to compute in \p inputs and \p outputs.
    /// \param border_mode      Border mode to use. See BorderMode for more details.
    /// \param border_value     Border value. Only used if \p mode is BORDER_VALUE.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs == \p inputs is not valid.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_pitch, size3_t input_shape,
                       T* outputs, size3_t output_pitch, size3_t output_shape, size_t batches,
                       BorderMode border_mode, T border_value, Stream& stream) {
        auto[border_left, border_right] = setBorders(input_shape, output_shape);
        resize(inputs, input_pitch, input_shape, border_left, border_right, outputs, output_pitch, batches,
               border_mode, border_value, stream);
    }
}
