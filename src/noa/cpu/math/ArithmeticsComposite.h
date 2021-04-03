#pragma once

#include <algorithm>
#include <execution>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

namespace Noa::Math {
    /**
     * For each batch, computes the element-wise fused multiply-add:
     * outputs[b][x] = inputs[b][x] * multipliers[x] + addends[x], for every x from 0 to @a elements and b from 0 to @a batches.
     *
     * @tparam T                Any type with `T operator*(T, T)` and `T operator+(T, T)` defined.
     * @param[in] inputs        Input arrays. One array of @a elements elements per batch.
     * @param[in] multipliers   One array of multipliers. Should be at least @a elements elements.
     * @param[in] addends       One array of addends. Should be at least @a elements elements.
     * @param[out] outputs      Output arrays. One array per batch. Can be equal to @a inputs/multipliers/addends.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches.
     */
    template<typename T>
    NOA_IH void multiplyAddArray(T* inputs, T* multipliers, T* addends, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch{0}; batch < batches; ++batch)
            for (size_t idx{0}; idx < elements; ++idx)
                outputs[batch * elements + idx] = inputs[batch * elements + idx] * multipliers[idx] + addends[idx];
    }

    /**
     * For each batch, computes the squared distance from a single value:
     * outputs[b][x] = (inputs[b][x] - values[b])^2, for every x from 0 to @a elements and for every b from 0 to @a batches.
     *
     * @tparam T            Any type with `T operator-(T, T)` and `T operator*(T, T)` defined.
     * @param[in] inputs    Input arrays. One per batch.
     * @param values        Values to subtract. One per batch.
     * @param[out] outputs  Output arrays. One per batch. Can be equal to @a inputs.
     * @param elements      Number of elements per batch.
     * @param batches       Number of batches to compute.
     *
     * @note @a inputs and @a outputs should be at least @a elements * @a batches elements.
     */
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(T* inputs, T* values, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T& value = values[batch];
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(std::execution::par_unseq,
                           inputs + batch_offset, inputs + batch_offset + elements, outputs + batch_offset,
                           [value](const T& a) -> T {
                               T distance = a - value;
                               return distance * distance;
                           });
        }
    }

    /// Computes the squared distance from a single value.
    /// @see This is a version without batches, but it is otherwise identical to the overload above.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(T* input, T value, T* output, size_t elements) {
        squaredDistanceFromValue(input, &value, output, elements, 1);
    }

    /**
     * For each batch, computes the element-wise squared distance from an array:
     * outputs[b][x] = (inputs[b][x] - array[x])^2, for every x from 0 to @a elements and for every b from 0 to @a batches.
     *
     * @tparam T            Any type with `T operator-(T, T)` and `T operator*(T, T)` defined.
     * @param[in] inputs    Input arrays. One per batch.
     * @param[in] array     Array to subtract. The same array is subtracted to every batch.
     * @param[out] outputs  Output arrays. One per batch. Can be equal to @a inputs or @a arrays.
     * @param elements      Number of elements per batch.
     * @param batches       Number of batches to compute.
     *
     * @note @a inputs and @a outputs should be at least `@a elements * @a batches` elements,
     *       whereas @a array should be at least @a elements elements.
     */
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(T* inputs, T* array, T* outputs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * static_cast<size_t>(batch);
            std::transform(std::execution::par_unseq,
                           inputs + batch_offset, inputs + batch_offset + elements, array, outputs + batch_offset,
                           [](const T& a, const T& b) {
                               T distance = a - b;
                               return distance * distance;
                           });
        }
    }
}
