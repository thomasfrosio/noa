#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Mask::Details {
    template<bool INVERT, bool ON_THE_FLY, typename T>
    void cylinder(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                  float taper_size, uint batches);

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void cylinder(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius_xy, float radius_z, uint batches);
}

namespace Noa::Mask {
    /**
     * Applies a cylindrical mask to the input array(s).
     * @tparam INVERT       Whether the mask should be inverted. In this case, everything within the cylinder is removed.
     * @tparam T            float or double.
     * @param[in] inputs    Input arrays. One per batch.
     * @param[out] outputs  Output arrays. One per batch.
     * @param shape         Logical {fast, medium, slow} shape of @a inputs and @a outputs (does not include the batch).
     * @param shifts        Shifts relative to the center of the @a shape (the center is at shape / 2).
     *                      Positive shifts translate the cylinder to the right.
     * @param radius_xy     Radius of the cylinder.
     * @param radius_z      Length of the cylinder.
     * @param taper_size    Width, in elements, of the raised-cosine, including the first zero.
     *                      For instance: taper_size = 6 gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
     * @param batches       Number of batches.
     */
    template<bool INVERT = false, typename T>
    NOA_IH void cylinder(T* inputs, T* outputs, size3_t shape,
                       float3_t shifts, float radius_xy, float radius_z, float taper_size, uint batches) {
        if (taper_size > 1e-5f)
            Details::cylinder<INVERT, true, T>(inputs, outputs, shape, shifts, radius_xy, radius_z, taper_size, batches);
        else
            Details::cylinder<INVERT, true, T>(inputs, outputs, shape, shifts, radius_xy, radius_z, batches);
    }

    /// Computes a cylindrical mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_IH void cylinder(T* output_mask, size3_t shape, float3_t shifts, float radius_xy, float radius_z, float taper_size) {
        if (taper_size > 1e-5f)
            Details::cylinder<INVERT, false, T>(nullptr, output_mask, shape, shifts, radius_xy, radius_z, taper_size, 0);
        else
            Details::cylinder<INVERT, false, T>(nullptr, output_mask, shape, shifts, radius_xy, radius_z, 0);
    }
}
