#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

// Implementation details:
namespace Noa::CUDA::Math::Details {
    enum : int {
        GEN_ONE_MINUS, GEN_INVERSE, GEN_SQUARE, GEN_SQRT, GEN_RSQRT, GEN_EXP, GEN_LOG, GEN_ABS, GEN_COS, GEN_SIN,
        GEN_NORMALIZE, GEN_POW, GEN_MIN, GEN_MAX
    };

    template<int GEN, typename T>
    NOA_HOST void generic(T* input, T* output, size_t elements, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithValue(T* input, T value, T* output, size_t elements, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithArray(T* input, T* array, T* output, size_t elements, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void generic(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithValue(T* input, size_t pitch_input, T value, T* output, size_t pitch_output,
                                   size3_t shape, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithArray(T* input, size_t pitch_input, T* array, size_t pitch_array,
                                   T* output, size_t pitch_output, size3_t shape, Stream& stream);
}

namespace Noa::CUDA::Math {
    /// CUDA version of Noa::Math::oneMinus(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void oneMinus(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_ONE_MINUS>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::oneMinus(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void oneMinus(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_ONE_MINUS>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::inverse(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void inverse(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_INVERSE>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::inverse(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void inverse(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_INVERSE>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::square(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void square(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_SQUARE>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::square(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void square(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_SQUARE>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::sqrt(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sqrt(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_SQRT>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::sqrt(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_SQRT>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::rsqrt(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void rsqrt(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_RSQRT>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::rsqrt(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void rsqrt(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_RSQRT>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::pow(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void pow(T* input, T exponent, T* output, size_t elements, Stream& stream) {
        Details::genericWithValue<Details::GEN_POW>(input, exponent, output, elements, stream);
    }

    /// CUDA version of Noa::Math::pow(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void pow(T* input, size_t pitch_input, T exponent, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        Details::genericWithValue<Details::GEN_POW>(input, pitch_input, exponent, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::exp(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void exp(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_EXP>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::exp(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void exp(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_EXP>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::log(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void log(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_LOG>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::log(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void log(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_LOG>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::abs(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void abs(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_ABS>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::abs(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void abs(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_ABS>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::cos(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void cos(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_COS>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::cos(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void cos(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_COS>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::sin(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sin(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_SIN>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::sin(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sin(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_SIN>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::normalize(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void normalize(T* input, T* output, size_t elements, Stream& stream) {
        Details::generic<Details::GEN_NORMALIZE>(input, output, elements, stream);
    }

    /// CUDA version of Noa::Math::normalize(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void normalize(T* input, size_t pitch_input, T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::generic<Details::GEN_NORMALIZE>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::min(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(T* lhs, T* rhs, T* output, size_t elements, Stream& stream) {
        Details::genericWithArray<Details::GEN_MIN>(lhs, rhs, output, elements, stream);
    }

    /// CUDA version of Noa::Math::min(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(T* lhs, size_t pitch_lhs, T* rhs, size_t pitch_rhs, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        Details::genericWithArray<Details::GEN_MIN>(lhs, pitch_lhs, rhs, pitch_rhs, output, pitch_output,
                                                    shape, stream);
    }

    /// CUDA version of Noa::Math::min(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(T* input, T threshold, T* output, size_t elements, Stream& stream) {
        Details::genericWithValue<Details::GEN_MIN>(input, threshold, output, elements, stream);
    }

    /// CUDA version of Noa::Math::min(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        Details::genericWithValue<Details::GEN_MIN>(input, pitch_input, threshold, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::max(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(T* lhs, T* rhs, T* output, size_t elements, Stream& stream) {
        Details::genericWithArray<Details::GEN_MAX>(lhs, rhs, output, elements, stream);
    }

    /// CUDA version of Noa::Math::max(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(T* lhs, size_t pitch_lhs, T* rhs, size_t pitch_rhs,
                    T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        Details::genericWithArray<Details::GEN_MAX>(lhs, pitch_lhs, rhs, pitch_rhs, output, pitch_output,
                                                    shape, stream);
    }

    /// CUDA version of Noa::Math::max(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(T* input, T threshold, T* output, size_t elements, Stream& stream) {
        Details::genericWithValue<Details::GEN_MAX>(input, threshold, output, elements, stream);
    }

    /// CUDA version of Noa::Math::max(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        Details::genericWithValue<Details::GEN_MAX>(input, pitch_input, threshold, output, pitch_output, shape, stream);
    }

    /// CUDA version of Noa::Math::clamp(). The same features and restrictions apply to this function.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_HOST void clamp(T* input, T low, T high, T* output, size_t elements, Stream& stream);

    /// CUDA version of Noa::Math::clamp(), for padded layouts (pitches are in number of elements.
    /// @warning This function is enqueued to @a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_HOST void clamp(T* input, size_t pitch_input, T low, T high, T* output, size_t pitch_output,
                        size3_t shape, Stream& stream);
}
