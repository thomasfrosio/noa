#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Mask {
    /**
     * Applies a rectangular mask to the input array(s).
     * @tparam INVERT       Whether the mask should be inverted. If true, everything within the rectangle is removed.
     * @tparam T            float or double.
     * @param[in] inputs    Input arrays. One per batch.
     * @param[out] outputs  Output arrays. One per batch.
     * @param shape         Logical {fast, medium, slow} shape of @a inputs and @a outputs (does not include the batch).
     * @param shifts        Shifts relative to the center of the @a shape (the center is at shape / 2).
     *                      Positive shifts translate the rectangle to the right.
     * @param radius        Radii of the rectangle.
     * @param taper_size    Width, in elements, of the raised-cosine, including the first zero.
     *                      For instance: taper_size = 6 gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
     * @param batches       Number of batches.
     */
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                            float taper_size, uint batches);

    /// Computes a rectangular mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(T* output_mask, size3_t shape, float3_t shifts, float3_t radius, float taper_size);
}
