#pragma once

#include "noa/Definitions.h"
#include "noa/cpu/memory/Resize.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Memory {
    /**
     * Resizes the input array(s) by padding and/or cropping the edges of the array.
     * @tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] inputs    Input array(s). One per batch.
     * @param input_pitch   Pitch of @a inputs, in elements.
     * @param input_shape   Physical {fast, medium, slow} shape of @a inputs, ignoring the batch size.
     * @param border_left   The {x, y, z} elements to add/remove from the left side of the dimension.
     * @param border_right  The {x, y, z} elements to add/remove from the right side of the dimension.
     * @param[out] outputs  Output array(s). One per batch.
     *                      The output shape is @a input_shape + @a border_left + @a border_right.
     * @param output_pitch  Pitch of @a outputs, in elements.
     * @param mode          Border mode to use. See BorderMode for more details.
     * @param border_value  Border value. Only used if @a mode == BORDER_VALUE.
     * @param batches       Number of batches in @a inputs and @a outputs.
     * @param stream        Stream on which to enqueue this function.
     *
     * @see See the corresponding resize function on the CPU backend. This function has the same limitations.
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                         int3_t border_left, int3_t border_right, T* outputs, size_t output_pitch,
                         BorderMode border_mode, T border_value, uint batches, Stream& stream);

    /// Resizes the input array(s) by padding and/or cropping the edges of the array.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, int3_t border_left, int3_t border_right, T* outputs,
                       BorderMode mode, T border_value, uint batches, Stream& stream) {
        int o_pitch = static_cast<int>(input_shape.x) + border_left.x + border_right.x; // assumed to be > 0
        resize(inputs, input_shape.x, input_shape, border_left, border_right, outputs, static_cast<size_t>(o_pitch),
               mode, border_value, batches, stream);
    }

    /**
     * Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
     * @tparam T            float, double, bool, (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] inputs    Input array(s). One per batch.
     * @param input_pitch   Pitch of @a inputs, in elements.
     * @param input_shape   Physical {fast, medium, slow} shape of @a inputs, ignoring the batch size.
     * @param[out] outputs  Output array(s). One per batch.
     * @param output_pitch  Pitch of @a outputs, in elements.
     * @param output_shape  Physical {fast, medium, slow} shape of @a inputs, ignoring the batch size.
     * @param mode          Border mode to use. See BorderMode for more details.
     * @param border_value  Border value. Only used if @a mode == BORDER_VALUE.
     * @param batches       Number of batches in @a inputs and @a outputs.
     * @param stream        Stream on which to enqueue this function.
     *
     * @see See the corresponding resize function on the CPU backend. This function has the same limitations.
     * @warning This function is asynchronous relative to the host and may return before completion.
     */
    template<typename T>
    NOA_IH void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                       T* outputs, size_t output_pitch, size3_t output_shape,
                       BorderMode mode, T border_value, uint batches, Stream& stream) {
        auto[border_left, border_right] = Noa::Memory::setBorders(input_shape, output_shape);
        resize(inputs, input_pitch, input_shape, border_left, border_right, outputs, output_pitch,
               mode, border_value, batches, stream);
    }

    /// Resizes the input array(s) to the desired shape while keeping the center (defined as shape / 2) aligned.
    /// @warning This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                       BorderMode border_mode, T border_value, uint batches, Stream& stream) {
        resize(inputs, input_shape.x, input_shape, outputs, output_shape.x, output_shape,
               border_mode, border_value, batches, stream);
    }
}
