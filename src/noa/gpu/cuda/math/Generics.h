#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Math {

    template<typename T>
    NOA_HOST void oneMinus(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void oneMinus(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void inverse(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void inverse(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void square(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void square(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void sqrt(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void sqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void rsqrt(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void rsqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void pow(T* input, T exponent, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void pow(T* input, size_t pitch_input, T exponent, T* output, size_t pitch_output,
                      size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void exp(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void exp(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void log(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void log(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void abs(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void abs(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void cos(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void cos(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void sin(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void sin(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void normalize(T* input, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void normalize(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void min(T* lhs, T* rhs, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void min(T* lhs, size_t pitch_lhs,
                      T* rhs, size_t pitch_rhs,
                      T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void min(T* input, T threshold, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void min(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output,
                      size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void max(T* lhs, T* rhs, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void max(T* lhs, size_t pitch_lhs,
                      T* rhs, size_t pitch_rhs,
                      T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void max(T* input, T threshold, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void max(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output,
                      size3_t shape, Stream& stream);

    template<typename T>
    NOA_HOST void clamp(T* input, T low, T high, T* output, size_t elements, Stream& stream);

    template<typename T>
    NOA_HOST void clamp(T* input, size_t pitch_input, T low, T high, T* output, size_t pitch_output,
                        size3_t shape, Stream& stream);
}
