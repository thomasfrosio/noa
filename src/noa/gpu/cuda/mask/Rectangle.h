/// \file noa/gpu/cuda/mask/Rectangle.h
/// \brief Rectangle masks.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::mask {
    /// Applies a rectangular mask to the input array(s).
    /// \tparam INVERT           Whether the mask should be inverted.
    /// \tparam T                float or double.
    /// \param[in] inputs        Input arrays. One per batch.
    /// \param inputs_pitch      Pitch of \a inputs, in elements.
    /// \param[out] outputs      Output arrays. One per batch. \a inputs == \a outputs is valid.
    /// \param outputs_pitch     Pitch of \a outputs, in elements.
    /// \param shape             Logical {fast, medium, slow} shape of \a inputs and \a outputs.
    /// \param shifts            Shifts relative to the center of the \a shape (the center is at shape / 2).
    /// \param radius            Radii of the rectangle, corresponding to \a shape.
    /// \param taper_size        Width, in elements, of the raised-cosine.
    /// \param batches           Number of batches.
    /// \param[in,out] stream    Stream on which to enqueue this function.
    ///
    /// \see This is the CUDA version of noa::mask::rectangle. See the CPU version for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(const T* inputs, size_t inputs_pitch, T* outputs, size_t outputs_pitch,
                            size3_t shape, float3_t shifts, float3_t radius,
                            float taper_size, uint batches, Stream& stream);

    /// Computes a rectangular mask. \see The version for padded arrays for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_HOST void rectangle(T* output_mask, size_t pitch_output_mask,
                            size3_t shape, float3_t shifts, float3_t radius, float taper_size, Stream& stream);

    /// Applies a rectangular mask to the input array(s). \see The version for padded arrays for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void rectangle(const T* inputs, T* outputs, size3_t shape, float3_t shifts, float3_t radius,
                          float taper_size, uint batches, Stream& stream) {
        rectangle<INVERT>(inputs, shape.x, outputs, shape.x, shape, shifts, radius, taper_size, batches, stream);
    }

    /// Computes a rectangular mask. \see The version for padded arrays for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<bool INVERT = false, typename T>
    NOA_IH void rectangle(T* output_mask, size3_t shape, float3_t shifts, float3_t radius,
                          float taper_size, Stream& stream) {
        rectangle<INVERT>(output_mask, shape.x, shape, shifts, radius, taper_size, stream);
    }
}
