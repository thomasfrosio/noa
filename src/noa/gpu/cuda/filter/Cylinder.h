/// \file noa/gpu/cuda/mask/Cylinder.h
/// \brief Cylinder masks.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::filter {
    /// Applies a cylindrical mask to the input array(s).
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the cylinder is removed.
    /// \tparam T               float or double.
    /// \param[in] inputs       On the \b host. Contiguous input arrays. One per batch.
    /// \param input_pitch      Pitch of \a inputs, in elements.
    /// \param[out] outputs     On the \b host. Contiguous output arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch of \a inputs, in elements.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param shifts           Shifts, in elements, corresponding to \p shape.
    ///                         Positive shifts translate the cylinder to the right.
    ///                         The center of the cylinder without shifts is at shape / 2.
    /// \param radius_xy        Radius of the cylinder.
    /// \param radius_z         Length of the cylinder.
    /// \param taper_size       Width, in elements, of the raised-cosine, including the first zero.
    ///                         For instance: taper_size = 6, gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                           float3_t shifts, float radius_xy, float radius_z,
                           float taper_size, uint batches, Stream& stream);

    /// Computes a cylindrical mask. This is otherwise identical to the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void cylinder(T* output_mask, size_t output_mask_pitch, size3_t shape,
                           float3_t shifts, float radius_xy, float radius_z,
                           float taper_size, Stream& stream);

    /// Applies a cylindrical mask to the input array(s). Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void cylinder(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                         float taper_size, uint batches, Stream& stream) {
        cylinder<INVERT>(inputs, shape.x, outputs, shape.x, shape, shifts, radius_xy, radius_z,
                         taper_size, batches, stream);
    }

    /// Computes a cylindrical mask. Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void cylinder(T* output_mask, size3_t shape, float3_t shifts, float radius_xy, float radius_z,
                         float taper_size, Stream& stream) {
        cylinder<INVERT>(output_mask, shape.x, shape, shifts, radius_xy, radius_z, taper_size, stream);
    }
}
