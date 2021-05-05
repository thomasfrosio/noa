#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

// 2D Borders:
//                             (1)
//  shape(x=6,y=5): [[ 0,  1,  2,  3,  4,  5],
//                   [ 6,  7,  8,  9, 10, 11],
//               (2) [12, 13, 14, 15, 16, 17], (4)
//                   [18, 19, 20, 21, 22, 23],
//                   [24, 25, 26, 27, 28, 29]]
//                             (3)
// border_left = [1,2]
// border_right = [3,4]

namespace Noa::Memory {
    /**
     * Sets the number of element(s) to pad/crop for each border of each dimension to get from @a input_shape to
     * @a output_shape, while keeping the centers of the input and output array (defined as shape / 2) aligned.
     *
     * @param input_shape       Current shape
     * @param output_shape      Desired shape
     * @param[out] border_left  The {x, y, z} elements to add/remove from the left side of the dimension.
     * @param[out] border_right The {x, y, z} elements to add/remove from the right side of the dimension.
     * @note Positive values correspond to padding, while negative values correspond to cropping.
     */
    NOA_IH void setBorders(size3_t input_shape, size3_t output_shape, int3_t* border_left, int3_t* border_right) {
        int3_t o_shape(output_shape);
        int3_t i_shape(input_shape);
        int3_t diff(o_shape - i_shape);

        *border_left = o_shape / 2 - i_shape / 2;
        *border_right = diff - *border_left;
    }

    /**
     * Resizes the input array(s) by padding and/or cropping the edges of the array.
     * @tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] inputs    Input array(s). One per batch.
     * @param input_shape   Physical {fast, medium, slow} shape of @a inputs, ignoring the batch size.
     * @param[out] outputs  Output array(s). One per batch.
     * @param output_shape  Physical {fast, medium, slow} shape of @a outputs, ignoring the batch size.
     * @param border_left   The {x, y, z} elements to add/remove from the left side of the dimension.
     * @param border_right  The {x, y, z} elements to add/remove from the right side of the dimension.
     * @param mode          Border mode to use. See BorderMode for more details.
     * @param border_value  Border value. Only used if @a mode == BORDER_VALUE.
     * @param batches       Number of batches in @a inputs and @a outputs.
     *
     * @throw   If (@a input_shape + @a border_left + @a border_right) != @a output_shape
     *          If @a inputs == @a outputs, i.e. in-place resizing is not allowed.
     *
     * @warning Edge case: if @a mode == BORDER_MIRROR and any of the (left/right) border is padded by more that
     *          one time the original shape, the padding in this region will probably not be what one would expect.
     *          A warning will be logged if this situation ever arise.
     *          Example: shape = 5, i.e. input = [0,1,2,3,4].
     *                   resize with border_left = 6, gives: [0,4,3,2,1,0,0,1,2,3,4], where one might expect the first
     *                   element to be 4 as opposed to 0.
     */
    template<typename T>
    NOA_HOST void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                         int3_t border_left, int3_t border_right, BorderMode mode, T border_value, uint batches);

    /**
     * Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
     * @tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] inputs    Input array(s). One per batch.
     * @param input_shape   Physical {fast, medium, slow} shape of @a inputs, ignoring the batch size.
     * @param[out] outputs  Output array(s). One per batch.
     * @param output_shape  Physical {fast, medium, slow} shape of @a inputs, ignoring the batch size.
     * @param mode          Border mode to use. See BorderMode for more details.
     * @param border_value  Border value. Only used if @a mode == BORDER_VALUE.
     * @param batches       Number of batches in @a inputs and @a outputs.
     *
     * @throw If @a inputs == @a outputs, i.e. in-place resizing is not allowed.
     */
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                       BorderMode mode, T border_value, uint batches) {
        int3_t border_left, border_right;
        setBorders(input_shape, output_shape, &border_left, &border_right);
        resize(inputs, input_shape, outputs, output_shape, border_left, border_right, mode, border_value, batches);
    }
}
