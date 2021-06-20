/// \file noa/gpu/cuda/mask/Sphere.h
/// \brief Sphere masks.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::mask {
    /// Applies a spherical mask to the input array(s).
    /// \tparam INVERT          Whether the mask should be inverted.
    /// \tparam T               float or double.
    /// \param[in] inputs       Input arrays. One per batch.
    /// \param inputs_pitch     Pitch of the input arrays, in elements.
    /// \param[out] outputs     Output arrays. One per batch. \a inputs == \a outputs is valid.
    /// \param outputs_pitch    Pitch of the output arrays, in elements.
    /// \param shape            Logical {fast, medium, slow} shape.
    /// \param shifts           Shifts relative to the center of the \a shape.
    /// \param radius           Radius of the sphere.
    /// \param taper_size       Width, in elements, of the raised-cosine.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \see This is the CUDA version of noa::mask::sphere. See the CPU version for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch, size3_t shape,
                         float3_t shifts, float radius, float taper_size, uint batches, Stream& stream);

    /// Computes a spherical mask. This is otherwise identical to the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void sphere(T* output_mask, size_t output_mask_pitch, size3_t shape, float3_t shifts,
                         float radius, float taper_size, Stream& stream);

    /// Applies a spherical mask to the input array(s). This is the version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void sphere(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float radius, float taper_size,
                       uint batches, Stream& stream) {
        sphere<INVERT>(inputs, shape.x, outputs, shape.x, shape, shifts, radius, taper_size, batches, stream);
    }

    /// Computes a spherical mask. This is the version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void sphere(T* output_mask, size3_t shape, float3_t shifts, float radius, float taper_size, Stream& stream) {
        sphere<INVERT>(output_mask, shape.x, shape, shifts, radius, taper_size, stream);
    }
}
