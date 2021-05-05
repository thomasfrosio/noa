#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"
#include "noa/cpu/memory/Resize.h"

namespace Noa::CUDA::Memory {
    NOA_IH void setBorders(size3_t input_shape, size3_t output_shape, int3_t* border_left, int3_t* border_right) {
        ::Noa::Memory::setBorders(input_shape, output_shape, border_left, border_right);
    }

    template<typename T>
    NOA_HOST void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                         T* outputs, size_t output_pitch, size3_t output_shape,
                         int3_t border_left, int3_t border_right, BorderMode border_mode, T border_value,
                         uint batches, Stream& stream);

    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                       int3_t border_left, int3_t border_right, BorderMode mode, T border_value,
                       uint batches, Stream& stream) {
        resize(inputs, input_shape.x, input_shape, outputs, output_shape.x, output_shape,
               border_left, border_right, mode, border_value, batches, stream);
    }

    template<typename T>
    NOA_IH void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                       T* outputs, size_t output_pitch, size3_t output_shape,
                       BorderMode mode, T border_value, uint batches, Stream& stream) {
        int3_t border_left, border_right;
        setBorders(input_shape, output_shape, &border_left, &border_right);
        resize(inputs, input_pitch, input_shape, outputs, output_pitch, output_shape,
               border_left, border_right, mode, border_value, batches, stream);
    }

    template<typename T>
    NOA_IH void resize(const T* inputs, size3_t input_shape, T* outputs, size3_t output_shape,
                       BorderMode border_mode, T border_value, uint batches, Stream& stream) {
        resize(inputs, input_shape.x, input_shape, outputs, output_shape.x, output_shape,
               border_mode, border_value, batches, stream);
    }
}
