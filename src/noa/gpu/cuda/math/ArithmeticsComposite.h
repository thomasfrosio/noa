#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Math {
    /**
     * For each batch, computes the element-wise fused multiply-add.
     * @see This is the CUDA version of Noa::Math::multiplyAddArray.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void multiplyAddArray(T* inputs, T* multipliers, T* addends, T* outputs,
                                   size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the element-wise fused multiply-add.
    /// @see This version is for padded memory. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void multiplyAddArray(T* inputs, size_t pitch_inputs,
                                   T* multipliers, size_t pitch_multipliers,
                                   T* addends, size_t pitch_addends,
                                   T* outputs, size_t pitch_outputs,
                                   size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, computes the squared distance from a single value.
     * @see This is the CUDA version of Noa::Math::squaredDistanceFromValue.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(T* inputs, T* values, T* outputs,
                                           size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the squared distance from a single value.
    /// @see This version is for padded memory. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromValue(T* inputs, size_t pitch_inputs, T* values,
                                           T* outputs, size_t pitch_outputs,
                                           size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, computes the element-wise squared distance from an array.
     * @see This is the CUDA version of Noa::Math::squaredDistanceFromValue.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(T* inputs, T* array, T* outputs,
                                           size_t elements, uint batches, Stream& stream);

    /// For each batch, computes the element-wise squared distance from an array.
    /// @see This version is for padded memory. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void squaredDistanceFromArray(T* inputs, size_t pitch_inputs,
                                           T* array, size_t pitch_array,
                                           T* outputs, size_t pitch_outputs,
                                           size3_t shape, uint batches, Stream& stream);
}
