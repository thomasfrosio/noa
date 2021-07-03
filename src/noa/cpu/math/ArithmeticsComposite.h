/// \file noa/cpu/math/ArithmeticsComposite.h
/// \brief Basic composite arithmetics for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::math {
    /// For each batch, computes the element-wise fused multiply-add:
    /// outputs[b][x] = inputs[b][x] * multipliers[x] + addends[x], for every x from 0 to \a elements and b from 0 to \a batches.
    ///
    /// \tparam T               float, double, int32_t, uint32_t.
    /// \param[in] inputs       Input arrays. One array of \a elements elements per batch.
    /// \param[in] multipliers  One array of multipliers. Should be at least \a elements elements.
    /// \param[in] addends      One array of addends. Should be at least \a elements elements.
    /// \param[out] outputs     Output arrays. One array per batch. Can be equal to \a inputs/multipliers/addends.
    /// \param elements         Number of elements per batch.
    /// \param batches          Number of batches.
    template<typename T>
    NOA_HOST void multiplyAddArray(const T* inputs, const T* multipliers, const T* addends, T* outputs,
                                   size_t elements, uint batches);

    /// For each batch, computes the squared distance from a single value:
    /// outputs[b][x] = (inputs[b][x] - values[b])^2, for every x from 0 to \a elements and for every b from 0 to \a batches.
    ///
    /// \tparam T           float, double, int32_t, uint32_t.
    /// \param[in] inputs   Input arrays. One per batch.
    /// \param values       Values to subtract. One per batch.
    /// \param[out] outputs Output arrays. One per batch. Can be equal to \a inputs.
    /// \param elements     Number of elements per batch.
    /// \param batches      Number of batches to compute.
    ///
    /// \note \a inputs and \a outputs should be at least \a elements * \a batches elements.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(const T* inputs, const T* values, T* outputs, size_t elements, uint batches);

    /// Computes the squared distance from a single value.
    /// \see This is a version without batches, but it is otherwise identical to the overload above.
    template<typename T>
    NOA_IH void squaredDistanceFromValue(const T* input, T value, T* output, size_t elements) {
        squaredDistanceFromValue(input, &value, output, elements, 1);
    }

    /// For each batch, computes the element-wise squared distance from an array:
    /// outputs[b][x] = (inputs[b][x] - array[x])^2, for every x from 0 to \a elements and for every b from 0 to \a batches.
    ///
    /// \tparam T           float, double, int32_t, uint32_t.
    /// \param[in] inputs   Input arrays. One per batch.
    /// \param[in] array    Array to subtract. The same array is subtracted to every batch.
    /// \param[out] outputs Output arrays. One per batch. Can be equal to \a inputs or \a arrays.
    /// \param elements     Number of elements per batch.
    /// \param batches      Number of batches to compute.
    ///
    /// \note \a inputs and \a outputs should be at least `\a elements * \a batches` elements,
    ///       whereas \a array should be at least \a elements elements.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(const T* inputs, const T* array, T* outputs, size_t elements, uint batches);
}
