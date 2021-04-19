#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Mask::Details {
    template<bool INVERT, bool ON_THE_FLY, typename T>
    void rectangle3D(T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                     float taper_size, uint batches);

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void rectangle2D(T* inputs, T* outputs, size2_t shape, float2_t shifts, float2_t radius,
                     float taper_size, uint batches);

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void rectangle(T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius, uint batches);
}

namespace Noa::Mask {
    /**
     * Applies a rectangular mask to the input array(s).
     * @tparam INVERT       Whether the mask should be inverted. In this case, everything within the rectangle is removed.
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
    NOA_IH void rectangle(T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                          float taper_size, uint batches) {
        if (taper_size < 1e-5f)
            Details::rectangle<INVERT, true, T>(inputs, outputs, shape, shifts, radius, batches);
        else if (getNDim(shape) == 2)
            Details::rectangle2D<INVERT, true, T>(inputs, outputs, {shape.x, shape.y}, {shifts.x, shifts.y},
                                                  {radius.x, radius.y}, taper_size, batches);
        else
            Details::rectangle3D<INVERT, true, T>(inputs, outputs, shape, shifts, radius, taper_size, batches);
    }

    /// Computes a rectangular mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_IH void rectangle(T* output_mask, size3_t shape, float3_t shifts, float3_t radius, float taper_size) {
        if (taper_size < 1e-5f)
            Details::rectangle<INVERT, false, T>(nullptr, output_mask, shape, shifts, radius, 0);
        else if (getNDim(shape) == 2)
            Details::rectangle2D<INVERT, false, T>(nullptr, output_mask, {shape.x, shape.y}, {shifts.x, shifts.y},
                                                  {radius.x, radius.y}, taper_size, 0);
        else
            Details::rectangle3D<INVERT, false, T>(nullptr, output_mask, shape, shifts, radius, taper_size, 0);
    }
}
