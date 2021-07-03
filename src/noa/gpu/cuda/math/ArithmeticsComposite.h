/// \file noa/gpu/cuda/math/ArithmeticsComposite.h
/// \brief Basic composite arithmetics for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::math {
    /// For each batch, computes the element-wise fused multiply-add.
    /// \see    This is the CUDA version of noa::math::multiplyAddArray.
    ///         The full documentation is described on the CPU version.
    ///         The same features and restrictions apply to this function.
    /// \note   This functions is enqueued to \a stream, thus is asynchronous with respect to the host and may
    ///         return before completion.
    template<typename T>
    NOA_HOST void multiplyAddArray(const T* inputs, const T* multipliers, const T* addends, T* outputs,
                                   size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the element-wise fused multiply-add.
    /// \see This version is for padded layouts. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void multiplyAddArray(const T* inputs, size_t inputs_pitch,
                                   const T* multipliers, size_t multipliers_pitch,
                                   const T* addends, size_t addends_pitch,
                                   T* outputs, size_t outputs_pitch,
                                   size3_t shape, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value.
    /// \see    This is the CUDA version of noa::math::squaredDistanceFromValue.
    ///         The full documentation is described on the CPU version.
    ///         The same features and restrictions apply to this function.
    /// \note   This functions is enqueued to \a stream, thus is asynchronous with respect to the host and may
    ///         return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, const T* values, T* outputs,
                                           size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value.
    /// \see This version is for padded layouts. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, size_t inputs_pitch, const T* values,
                                           T* outputs, size_t outputs_pitch,
                                           size3_t shape, uint batches, Stream& stream);

    /// For each batch, computes the element-wise squared distance from an array.
    /// \see    This is the CUDA version of noa::math::squaredDistanceFromValue.
    ///         The full documentation is described on the CPU version.
    ///         The same features and restrictions apply to this function.
    /// \note   This functions is enqueued to \a stream, thus is asynchronous with respect to the host and may
    ///         return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(const T* inputs, const T* array, T* outputs,
                                           size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the element-wise squared distance from an array.
    /// \see This version is for padded layouts. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(const T* inputs, size_t inputs_pitch,
                                           const T* array, size_t array_pitch,
                                           T* outputs, size_t outputs_pitch,
                                           size3_t shape, uint batches, Stream& stream);
}
