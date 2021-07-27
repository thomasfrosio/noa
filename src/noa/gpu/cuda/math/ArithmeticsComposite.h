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
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b device. Input arrays. One array per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[in] multipliers  On the \b device. One array of multipliers. The same array is used for every batch.
    /// \param multiplier_pitch Pitch, in elements, of \p multipliers.
    /// \param[in] addends      On the \b device. One array of addends. The same array is used for every batch.
    /// \param addend_pitch     Pitch, in elements, of \p addends.
    /// \param[out] outputs     On the \b device. Output arrays. One array per batch.
    ///                         Can be equal to \p inputs, \p multipliers or \p addends.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void multiplyAddArray(const T* inputs, size_t inputs_pitch,
                                   const T* multipliers, size_t multipliers_pitch,
                                   const T* addends, size_t addends_pitch,
                                   T* outputs, size_t outputs_pitch,
                                   size3_t shape, uint batches, Stream& stream);

    /// For each batch, computes the element-wise fused multiply-add. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void multiplyAddArray(const T* inputs, const T* multipliers, const T* addends, T* outputs,
                                   size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b device. Input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param values           On the \b device. Values to subtract. One value per batch.
    /// \param[out] outputs     On the \b device. Output arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, size_t inputs_pitch, const T* values,
                                           T* outputs, size_t outputs_pitch,
                                           size3_t shape, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, const T* values, T* outputs,
                                           size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b device. Input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param value            Value to subtract. The same value is used for every batch.
    /// \param[out] outputs     On the \b device. Output arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, size_t inputs_pitch, T value,
                                           T* outputs, size_t outputs_pitch,
                                           size3_t shape, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, T value, T* outputs,
                                           size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the element-wise squared distance from an array.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b device. Input arrays. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[in] array        On the \b device. Array to subtract. The same array is subtracted to every batch.
    /// \param array_pitch      Pitch, in elements, of \p array.
    /// \param[out] outputs     On the \b device. Output arrays. One per batch. Can be equal to \p inputs or \p array.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(const T* inputs, size_t inputs_pitch,
                                           const T* array, size_t array_pitch,
                                           T* outputs, size_t outputs_pitch,
                                           size3_t shape, uint batches, Stream& stream);

    /// For each batch, computes the element-wise squared distance from an array. Version for contiguous layouts.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(const T* inputs, const T* array, T* outputs,
                                           size_t elements, uint batches, Stream& stream);
}
