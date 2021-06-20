/// \file noa/gpu/cuda/math/Generics.h
/// \brief Generic math functions for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::math::details {
    enum : int {
        GEN_ONE_MINUS, GEN_INVERSE, GEN_SQUARE, GEN_SQRT, GEN_RSQRT, GEN_EXP, GEN_LOG, GEN_ABS, GEN_COS, GEN_SIN,
        GEN_NORMALIZE, GEN_POW, GEN_MIN, GEN_MAX
    };

    template<int GEN, typename T>
    NOA_HOST void generic(const T* input, T* output, size_t elements, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithValue(const T* input, T value, T* output, size_t elements, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithArray(const T* input, const T* array, T* output, size_t elements, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void generic(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                          size3_t shape, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithValue(const T* input, size_t pitch_input, T value, T* output, size_t pitch_output,
                                   size3_t shape, Stream& stream);

    template<int GEN, typename T>
    NOA_HOST void genericWithArray(const T* input, size_t pitch_input, const T* array, size_t pitch_array,
                                   T* output, size_t pitch_output, size3_t shape, Stream& stream);
}

namespace noa::cuda::math {
    /// CUDA version of noa::math::oneMinus(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void oneMinus(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_ONE_MINUS>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::oneMinus(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void oneMinus(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                         size3_t shape, Stream& stream) {
        details::generic<details::GEN_ONE_MINUS>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::inverse(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void inverse(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_INVERSE>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::inverse(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void inverse(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                        size3_t shape, Stream& stream) {
        details::generic<details::GEN_INVERSE>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::square(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void square(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_SQUARE>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::square(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void square(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                       size3_t shape, Stream& stream) {
        details::generic<details::GEN_SQUARE>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::sqrt(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sqrt(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_SQRT>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::sqrt(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sqrt(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                     size3_t shape, Stream& stream) {
        details::generic<details::GEN_SQRT>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::rsqrt(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void rsqrt(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_RSQRT>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::rsqrt(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void rsqrt(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                      size3_t shape, Stream& stream) {
        details::generic<details::GEN_RSQRT>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::pow(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void pow(const T* input, T exponent, T* output, size_t elements, Stream& stream) {
        details::genericWithValue<details::GEN_POW>(input, exponent, output, elements, stream);
    }

    /// CUDA version of noa::math::pow(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void pow(const T* input, size_t pitch_input, T exponent, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::genericWithValue<details::GEN_POW>(input, pitch_input, exponent, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::exp(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void exp(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_EXP>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::exp(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void exp(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::generic<details::GEN_EXP>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::log(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void log(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_LOG>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::log(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void log(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::generic<details::GEN_LOG>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::abs(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void abs(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_ABS>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::abs(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void abs(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::generic<details::GEN_ABS>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::cos(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void cos(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_COS>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::cos(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void cos(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::generic<details::GEN_COS>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::sin(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sin(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_SIN>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::sin(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void sin(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::generic<details::GEN_SIN>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::normalize(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void normalize(const T* input, T* output, size_t elements, Stream& stream) {
        details::generic<details::GEN_NORMALIZE>(input, output, elements, stream);
    }

    /// CUDA version of noa::math::normalize(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void normalize(const T* input, size_t pitch_input, T* output, size_t pitch_output,
                          size3_t shape, Stream& stream) {
        details::generic<details::GEN_NORMALIZE>(input, pitch_input, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::min(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(const T* lhs, const T* rhs, T* output, size_t elements, Stream& stream) {
        details::genericWithArray<details::GEN_MIN>(lhs, rhs, output, elements, stream);
    }

    /// CUDA version of noa::math::min(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(const T* lhs, size_t pitch_lhs, const T* rhs, size_t pitch_rhs, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::genericWithArray<details::GEN_MIN>(lhs, pitch_lhs, rhs, pitch_rhs, output, pitch_output,
                                                    shape, stream);
    }

    /// CUDA version of noa::math::min(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(const T* input, T threshold, T* output, size_t elements, Stream& stream) {
        details::genericWithValue<details::GEN_MIN>(input, threshold, output, elements, stream);
    }

    /// CUDA version of noa::math::min(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void min(const T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::genericWithValue<details::GEN_MIN>(input, pitch_input, threshold, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::max(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(const T* lhs, const T* rhs, T* output, size_t elements, Stream& stream) {
        details::genericWithArray<details::GEN_MAX>(lhs, rhs, output, elements, stream);
    }

    /// CUDA version of noa::math::max(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(const T* lhs, size_t pitch_lhs, const T* rhs, size_t pitch_rhs,
                    T* output, size_t pitch_output, size3_t shape, Stream& stream) {
        details::genericWithArray<details::GEN_MAX>(lhs, pitch_lhs, rhs, pitch_rhs, output, pitch_output,
                                                    shape, stream);
    }

    /// CUDA version of noa::math::max(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(const T* input, T threshold, T* output, size_t elements, Stream& stream) {
        details::genericWithValue<details::GEN_MAX>(input, threshold, output, elements, stream);
    }

    /// CUDA version of noa::math::max(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_IH void max(const T* input, size_t pitch_input, T threshold, T* output, size_t pitch_output,
                    size3_t shape, Stream& stream) {
        details::genericWithValue<details::GEN_MAX>(input, pitch_input, threshold, output, pitch_output, shape, stream);
    }

    /// CUDA version of noa::math::clamp(). The same features and restrictions apply to this function.
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_HOST void clamp(const T* input, T low, T high, T* output, size_t elements, Stream& stream);

    /// CUDA version of noa::math::clamp(), for padded layouts (pitches are in number of elements).
    /// \note This function is enqueued to \a stream and is asynchronous with respect to the host.
    template<typename T>
    NOA_HOST void clamp(const T* input, size_t pitch_input, T low, T high, T* output, size_t pitch_output,
                        size3_t shape, Stream& stream);
}
