/// \file noa/gpu/cuda/mask/Rectangle.h
/// \brief Rectangle masks.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::filter {
    /// Applies a rectangular mask to the input array(s).
    /// \tparam INVERT          Whether the mask should be inverted. If true, everything within the rectangle is removed.
    /// \tparam T               float or double.
    /// \param[in] inputs       On the \b device. Contiguous input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Contiguous output arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs, ignoring the batches.
    /// \param shifts           Shifts, in elements, corresponding to \p shape.
    ///                         Positive shifts translate the rectangle to the right.
    ///                         The center of the rectangle without shifts is at shape / 2.
    ///                         If \p shape describes a 2D array, \p shift.z is ignored.
    /// \param radius           Radii, in elements, of the rectangle.
    /// \param taper_size       Width, in elements, of the raised-cosine, including the first zero.
    ///                         For instance: taper_size = 6, gives [..., 0. , 0.067, 0.25 , 0.5 , 0.75 , 0.933, 1., ...].
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                            size3_t shape, float3_t shifts, float3_t radius,
                            float taper_size, uint batches, Stream& stream);

    /// Computes a rectangular mask. This is otherwise identical to the overload above.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(T* output_mask, size_t output_mask_pitch,
                            size3_t shape, float3_t shifts, float3_t radius, float taper_size, Stream& stream);

    /// Applies a rectangular mask to the input array(s). Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void rectangle(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                          float taper_size, uint batches, Stream& stream) {
        rectangle<INVERT>(inputs, shape.x, outputs, shape.x, shape, shifts, radius, taper_size, batches, stream);
    }

    /// Computes a rectangular mask. Version for contiguous layouts.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void rectangle(T* output_mask, size3_t shape, float3_t shifts, float3_t radius,
                          float taper_size, Stream& stream) {
        rectangle<INVERT>(output_mask, shape.x, shape, shifts, radius, taper_size, stream);
    }
}
