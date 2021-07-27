/// \file noa/gpu/cuda/memory/Resize.h
/// \brief Resize memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/cpu/memory/Resize.h" // noa::memory::setBorders()
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::memory {
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
                         BorderMode border_mode, T border_value, uint batches, Stream& stream);

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, int3_t border_left, int3_t border_right, T* outputs,
                       BorderMode border_mode, T border_value, uint batches, Stream& stream) {
        int o_pitch = static_cast<int>(input_shape.x) + border_left.x + border_right.x; // assumed to be > 0
        resize(inputs, input_shape.x, input_shape, border_left, border_right, outputs, static_cast<size_t>(o_pitch),
               border_mode, border_value, batches, stream);
    }

    /**
     * Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
     * \tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
     * \param[in] inputs    On the \b device. Input array(s). One per batch.
     * \param input_pitch   Pitch of \p inputs, in elements.
     * \param input_shape   Physical {fast, medium, slow} shape of \p inputs, ignoring the batch size.
     * \param[out] outputs  On the \b device. Output array(s). One per batch.
     * \param output_pitch  Pitch of \p outputs, in elements.
     * \param output_shape  Physical {fast, medium, slow} shape of \p inputs, ignoring the batch size.
     * \param border_mode   Border mode to use. See BorderMode for more details.
     * \param border_value  Border value. Only used if \p mode == BORDER_VALUE.
     * \param batches       Number of batches in \p inputs and \p outputs.
     * \param stream        Stream on which to enqueue this function.
     *
     * \see See the corresponding resize function on the CPU backend. This function has the same limitations.
     * \note This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_IH void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                       T* outputs, size_t output_pitch, size3_t output_shape,
                       BorderMode border_mode, T border_value, uint batches, Stream& stream) {
        auto[border_left, border_right] = noa::memory::setBorders(input_shape, output_shape);
        resize(inputs, input_pitch, input_shape, border_left, border_right, outputs, output_pitch,
               border_mode, border_value, batches, stream);
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                       BorderMode border_mode, T border_value, uint batches, Stream& stream) {
        resize(inputs, input_shape.x, input_shape, outputs, output_shape.x, output_shape,
               border_mode, border_value, batches, stream);
    }
}
