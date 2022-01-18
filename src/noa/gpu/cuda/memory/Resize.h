/// \file noa/gpu/cuda/memory/Resize.h
/// \brief Resize memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::memory {
    /// Sets the number of element(s) to pad/crop for each border of each dimension to get from \p input_shape to
    /// \p output_shape, while keeping the centers of the input and output array (defined as `shape / 2`) aligned.
    /// This is identical to the CPU version.
    ///
    /// \param input_shape      Current {fast, medium, slow} shape.
    /// \param output_shape     Desired {fast, medium, slow} shape.
    /// \return                 1: The {fast, medium, slow} elements to add/remove from the left side of the dimension.
    ///                         2: The {fast, medium, slow} elements to add/remove from the right side of the dimension.
    ///                         Positive values correspond to padding, while negative values correspond to cropping.
    NOA_IH std::pair<int3_t, int3_t> setBorders(size3_t input_shape, size3_t output_shape) {
        int3_t o_shape(output_shape);
        int3_t i_shape(input_shape);
        int3_t diff(o_shape - i_shape);

        int3_t border_left = o_shape / 2 - i_shape / 2;
        int3_t border_right = diff - border_left;
        return {border_left, border_right}; // TODO If noa::Pair<> is added, use it instead.
    }

     /// Resizes the input array(s) by padding and/or cropping the edges of the array.
     /// \tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
     /// \param[in] inputs    On the \b device. Input array(s). One per batch.
     /// \param input_pitch   Pitch of \p inputs, in elements.
     /// \param input_shape   Physical {fast, medium, slow} shape of \p inputs, ignoring the batch size.
     /// \param border_left   The {x, y, z} elements to add/remove from the left side of the dimension.
     /// \param border_right  The {x, y, z} elements to add/remove from the right side of the dimension.
     /// \param[out] outputs  On the \b device. Output array(s). One per batch.
     ///                      The output shape is \p input_shape + \p border_left + \p border_right.
     /// \param output_pitch  Pitch of \p outputs, in elements.
     /// \param border_mode   Border mode to use. See BorderMode for more details.
     /// \param border_value  Border value. Only used if \p mode == BORDER_VALUE.
     /// \param batches       Number of batches in \p inputs and \p outputs.
     /// \param stream        Stream on which to enqueue this function.
     ///
     /// \see See the corresponding resize function on the CPU backend. This function has the same limitations.
     /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                         int3_t border_left, int3_t border_right, T* outputs, size_t output_pitch,
                         BorderMode border_mode, T border_value, size_t batches, Stream& stream);

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, int3_t border_left, int3_t border_right, T* outputs,
                       BorderMode border_mode, T border_value, size_t batches, Stream& stream) {
        int o_pitch = static_cast<int>(input_shape.x) + border_left.x + border_right.x;
        resize(inputs, input_shape.x, input_shape, border_left, border_right, outputs, static_cast<size_t>(o_pitch),
               border_mode, border_value, batches, stream);
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs    On the \b device. Input array(s). One per batch.
    /// \param input_pitch   Pitch of \p inputs, in elements.
    /// \param input_shape   Physical {fast, medium, slow} shape of \p inputs, ignoring the batch size.
    /// \param[out] outputs  On the \b device. Output array(s). One per batch.
    /// \param output_pitch  Pitch of \p outputs, in elements.
    /// \param output_shape  Physical {fast, medium, slow} shape of \p inputs, ignoring the batch size.
    /// \param border_mode   Border mode to use. See BorderMode for more details.
    /// \param border_value  Border value. Only used if \p mode == BORDER_VALUE.
    /// \param batches       Number of batches in \p inputs and \p outputs.
    /// \param stream        Stream on which to enqueue this function.
    ///
    /// \see See the corresponding resize function on the CPU backend. This function has the same limitations.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                       T* outputs, size_t output_pitch, size3_t output_shape,
                       BorderMode border_mode, T border_value, size_t batches, Stream& stream) {
        auto[border_left, border_right] = setBorders(input_shape, output_shape);
        resize(inputs, input_pitch, input_shape, border_left, border_right, outputs, output_pitch,
               border_mode, border_value, batches, stream);
    }
}
