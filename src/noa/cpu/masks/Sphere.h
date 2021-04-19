#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

namespace Noa::Mask::Details {
    template<bool INVERT, bool ON_THE_FLY, typename T>
    void sphere(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius, float taper_size, uint batches);

    template<bool INVERT, bool ON_THE_FLY, typename T>
    void sphere(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius, uint batches);
}

namespace Noa::Mask {
    /**
     * Applies a spherical mask to the input array(s).
     * @tparam INVERT       Whether the mask should be inverted. In this case, everything within the sphere is removed.
     * @tparam T            float or double.
     * @param[in] inputs    Input arrays. One per batch.
     * @param[out] outputs  Output arrays. One per batch.
     * @param shape         Logical {fast, medium, slow} shape of @a inputs and @a outputs (does not include the batch).
     * @param shifts        Shifts relative to the center of the @a shape (the center is at shape / 2).
     *                      Positive shifts translate the sphere to the right.
     * @param radius        Radius of the sphere.
     * @param taper_size    Width, in elements, of the raised-cosine, including the first zero.
     *                      For instance: taper_size = 6 gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
     * @param batches       Number of batches.
     */
    template<bool INVERT = false, typename T>
    NOA_IH void sphere(T* inputs, T* outputs, size3_t shape,
                       float3_t shifts, float radius, float taper_size, uint batches) {
        if (taper_size > 1e-5f)
            Details::sphere<INVERT, true, T>(inputs, outputs, shape, shifts, radius, taper_size, batches);
        else
            Details::sphere<INVERT, true, T>(inputs, outputs, shape, shifts, radius, batches);
    }

    /// Computes a spherical mask. This is otherwise identical to the overload above.
    template<bool INVERT = false, typename T>
    NOA_IH void sphere(T* output_mask, size3_t shape, float3_t shifts, float radius, float taper_size) {
        if (taper_size > 1e-5f)
            Details::sphere<INVERT, false, T>(nullptr, output_mask, shape, shifts, radius, taper_size, 0);
        else
            Details::sphere<INVERT, false, T>(nullptr, output_mask, shape, shifts, radius, 0);
    }
}
