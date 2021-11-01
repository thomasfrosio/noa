/// \file noa/cpu/math/ArithmeticsComposite.h
/// \brief Basic composite arithmetics for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::math {
    /// For each batch, computes the element-wise fused multiply-add.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b host. Input arrays. One array per batch.
    /// \param[in] multipliers  On the \b host. One array of multipliers. The same array is used for every batch.
    /// \param[in] addends      On the \b host. One array of addends. The same array is used for every batch.
    /// \param[out] outputs     On the \b host. Output arrays. One array per batch.
    ///                         Can be equal to \p inputs, \p multipliers or \p addends.
    /// \param elements         Number of elements to compute per batch.
    /// \param batches          Number of batches to compute.
    template<typename T>
    NOA_HOST void multiplyAddArray(const T* inputs, const T* multipliers, const T* addends, T* outputs,
                                   size_t elements, size_t batches);

    /// For each batch, computes the squared distance from a single value.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs   On the \b host. Input arrays. One per batch.
    /// \param values       On the \b host. Values to subtract. One value per batch.
    /// \param[out] outputs On the \b host. Output arrays. One per batch. Can be equal to \p inputs.
    /// \param elements     Number of elements to compute per batch.
    /// \param batches      Number of batches to compute.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, const T* values, T* outputs,
                                           size_t elements, size_t batches);

    /// Computes the squared distance from a single value. Un-batched version.
    template<typename T>
    NOA_IH void squaredDistanceFromValue(const T* input, T value, T* output, size_t elements) {
        squaredDistanceFromValue(input, &value, output, elements, 1);
    }

    /// For each batch, computes the element-wise squared distance from an array.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs   On the \b host. Input arrays. One per batch.
    /// \param[in] array    On the \b host. Array to subtract. The same array is subtracted to every batch.
    /// \param[out] outputs On the \b host. Output arrays. One per batch. Can be equal to \p inputs or \p array.
    /// \param elements     Number of elements to compute per batch.
    /// \param batches      Number of batches to compute.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(const T* inputs, const T* array, T* outputs,
                                           size_t elements, size_t batches);
}
