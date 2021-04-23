#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Mask {
    /**
     * Applies a cylindrical mask to the input array(s).
     * @tparam INVERT           Whether the mask should be inverted.
     * @tparam T                float or double.
     * @param[in] inputs        Input arrays. One per batch.
     * @param pitch_inputs      Pitch of @a inputs, in elements.
     * @param[out] outputs      Output arrays. One per batch.
     * @param pitch_outputs     Pitch of @a inputs, in elements.
     * @param shape             Logical {fast, medium, slow} shape of @a inputs and @a outputs/
     * @param shifts            Shifts relative to the center of the @a shape (the center is at shape / 2).
     * @param radius_xy         Radius of the cylinder.
     * @param radius_z          Length of the cylinder.
     * @param taper_size        Width, in elements, of the raised-cosine, including the first zero.
     * @param batches           Number of batches.
     * @param[in,out] stream    Stream on which to enqueue this function.
     *
     * @see This is the CUDA version of Noa::Mask::cylinder. See the CPU version for more details.
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs, size3_t shape,
                           float3_t shifts, float radius_xy, float radius_z,
                           float taper_size, uint batches, Stream& stream);

    /// Computes a cylindrical mask. @see The version for padded arrays for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(T* output_mask, size_t pitch_output_mask, size3_t shape,
                           float3_t shifts, float radius_xy, float radius_z,
                           float taper_size, Stream& stream);

    /// Applies a cylindrical mask to the input array(s). @see The version for padded arrays for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void cylinder(T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                         float taper_size, uint batches, Stream& stream) {
        cylinder<INVERT>(inputs, shape.x, outputs, shape.x, shape, shifts, radius_xy, radius_z,
                         taper_size, batches, stream);
    }

    /// Computes a cylindrical mask. @see The version for padded arrays for more details.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void cylinder(T* output_mask, size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                         float taper_size, Stream& stream) {
        cylinder<INVERT>(output_mask, shape.x, shape, shifts, radius_xy, radius_z, taper_size, stream);
    }
}
