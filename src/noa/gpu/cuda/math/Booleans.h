/// \file noa/gpu/cuda/math/Booleans.h
/// \brief Boolean "operators" for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::math {
    /// Computes output[x] = threshold < input[x], for every x from 0 to \a elements.
    /// \tparam T               float, double or any integral type.
    /// \tparam U               Any type allowed for \a T or bool.
    /// \param[in] input        Input array.
    /// \param threshold        Value to use as threshold.
    /// \param[out] output      Output array. Can be equal to \a input.
    /// \param elements         Number of elements to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isLess(const T* input, T threshold, U* output, size_t elements, Stream& stream);

    /// Computes output[x] = threshold < input[x], for every x in \a shape.
    /// \tparam T               float, double or any integral type.
    /// \tparam U               Any type allowed for \a T or bool.
    /// \param[in] input        Input array.
    /// \param input_pitch      Pitch, in elements, of \a input.
    /// \param threshold        Value to use as threshold.
    /// \param[out] output      Output array. Can be equal to \a input.
    /// \param output_pitch     Pitch, in elements, of \a output.
    /// \param elements         Number of elements to compute.
    /// \param shape            Logical {fast, medium, slow} shape of \a input and \a output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isLess(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                         size3_t shape, Stream& stream);

    /// Computes output[x] = input[x] < threshold, for every x from 0 to \a elements.
    /// \see This is similar to noa::cuda::math::isLess(). The same features and restrictions apply to this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isGreater(const T* input, T threshold, U* output, size_t elements, Stream& stream);

    /// \see This version is for padded layouts. See the overload for contiguous layouts for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isGreater(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                            size3_t shape, Stream& stream);

    /// Computes output[x] = threshold < input[x] && input[x] < threshold, for every x from 0 to \a elements.
    /// \see This is similar to noa::cuda::math::isLess(). The same features and restrictions apply to this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isWithin(const T* input, T low, T high, U* output, size_t elements, Stream& stream);

    /// \see This version is for padded layouts. See the overload for contiguous layouts for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isWithin(const T* input, size_t input_pitch, T low, T high, U* output, size_t output_pitch,
                           size3_t shape, Stream& stream);

    /// Computes the logical NOT, i.e. output[x] = !input[x], for every x from 0 to \a elements.
    /// \tparam T               Any integral type.
    /// \tparam U               Any type allowed for \a T or bool.
    /// \param[in] input        Input array.
    /// \param[out] output      Output array. Can be equal to \a input.
    /// \param elements         Number of elements to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void logicNOT(const T* input, U* output, size_t elements, Stream& stream);

    /// \see This version is for padded layouts. See the overload for contiguous layouts for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void logicNOT(const T* input, size_t input_pitch, U* output, size_t output_pitch,
                           size3_t shape, Stream& stream);
}
