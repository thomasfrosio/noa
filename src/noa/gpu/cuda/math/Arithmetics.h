#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

// Implementation details:
namespace Noa::CUDA::Math::Details {
    enum : int { ARITH_ADD, ARITH_SUBTRACT, ARITH_MULTIPLY, ARITH_DIVIDE, ARITH_DIVIDE_SAFE };

    template<int ARITH, typename T, typename U>
    void arithByValue(T* input, U value, T* output, size_t elements, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByValue(T* input, size_t pitch_input, U value, T* output, size_t pitch_output,
                      size3_t shape, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByValue(T* inputs, size_t pitch_inputs, U* values, T* outputs, size_t pitch_outputs,
                      size3_t shape, uint batches, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByArray(T* inputs, size_t pitch_inputs, U* array, size_t pitch_array, T* outputs, size_t pitch_outputs,
                      size3_t shape, uint batches, Stream& stream);
}

namespace Noa::CUDA::Math {
    /* ----------------------------- */
    /* --- Contiguous - Multiply --- */
    /* ----------------------------- */

    /**
     * Multiplies the input array by a single value.
     * @see This is the CUDA version of Noa::Math::multiplyByValue.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @tparam U            Equal to @a T except for complex types where @a U can be the corresponding real type.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        Details::arithByValue<Details::ARITH_MULTIPLY>(input, value, output, elements, stream);
    }

    /**
     * For each batch, multiplies one input array by a single value.
     * @see This is the CUDA version of Noa::Math::multiplyByValue.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @tparam U            Equal to @a T except for complex types where @a U can be the corresponding real type.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_MULTIPLY>(inputs, values, outputs, elements, batches, stream);
    }

    /**
     * For each batch, multiplies one input array by another array.
     * @see This is the CUDA version of Noa::Math::multiplyByArray.
     *      The full documentation is described on the CPU version.
     *      The same features and restrictions apply to this function.
     *
     * @tparam T            float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @tparam U            Equal to @a T except for complex types where @a U can be the corresponding real type.
     * @param[out] stream   Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_MULTIPLY>(inputs, array, outputs, elements, batches, stream);
    }

    /* --------------------------- */
    /* --- Contiguous - Divide --- */
    /* --------------------------- */

    /// Divides the input array by a single value.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByValue.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        Details::arithByValue<Details::ARITH_DIVIDE>(input, value, output, elements, stream);
    }

    /// For each batch, divides one input array by a single value.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByValue.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_DIVIDE>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise division between one of the input array and the second array.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByArray.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_DIVIDE>(inputs, array, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise division between one of the input array and the second array.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByArray.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideSafeByArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_DIVIDE_SAFE>(inputs, array, outputs, elements, batches, stream);
    }

    /* ------------------------ */
    /* --- Contiguous - Add --- */
    /* ------------------------ */

    /// For each batch, adds one input array by a single value.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByValue.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        Details::arithByValue<Details::ARITH_ADD>(input, value, output, elements, stream);
    }

    /// For each batch, adds one input array by a single value.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByValue.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_ADD>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise addition between one array and the weights.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByArray.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_ADD>(inputs, array, outputs, elements, batches, stream);
    }

    /* ----------------------------- */
    /* --- Contiguous - Subtract --- */
    /* ----------------------------- */

    /// For each batch, subtracts one input array by a single value.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByValue.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(T* input, U value, T* output, size_t elements, Stream& stream) {
        Details::arithByValue<Details::ARITH_SUBTRACT>(input, value, output, elements, stream);
    }

    /// For each batch, subtracts one input array by a single value.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByValue.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(T* inputs, U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_SUBTRACT>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise subtraction between one array and the weights.
    /// @see This function supports same features and restrictions than Noa::CUDA::Math::multiplyByArray.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractArray(T* inputs, U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_SUBTRACT>(inputs, array, outputs, elements, batches, stream);
    }
}

namespace Noa::CUDA::Math {
    /* ------------------------- */
    /* --- Padded - Multiply --- */
    /* ------------------------- */

    /**
     * Multiplies the input array by a single value.
     * @see This version is for padded layout. See the overload for contiguous memory for more details.
     *
     * @tparam T                float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @tparam U                Equal to @a T except for complex types where @a U can be the corresponding real type.
     * @param[in] input         Input array.
     * @param pitch_input       Pitch, in elements, of the input array.
     * @param[in] array         Multiplier.
     * @param[out] output       Output array. Can be equal to @a input.
     * @param pitch_output      Pitch, in elements, of the output array.
     * @param shape             Physical {fast, medium, slow} shape of @a input and @a output.
     * @param[out] stream       Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByValue(T* input, size_t pitch_input, U value, T* output, size_t pitch_output,
                                size3_t shape, Stream& stream) {
        Details::arithByValue<Details::ARITH_MULTIPLY>(input, pitch_input, value, output, pitch_output, shape, stream);
    }

    /**
     * For each batch, multiplies one input array by a single value.
     * @see This version is for padded layout. See the overload for contiguous memory for more details.
     *
     * @tparam T                float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @tparam U                Equal to @a T except for complex types where @a U can be the corresponding real type.
     * @param[in] inputs        Input arrays. One array per batch.
     * @param pitch_inputs      Pitch, in elements, of the input arrays.
     * @param values            Multipliers. One per batch.
     * @param[out] outputs      Output arrays. One array per batch. Can be equal to @a inputs.
     * @param pitch_outputs     Pitch, in elements, of the output arrays.
     * @param shape             Physical {fast, medium, slow} shape of @a inputs and @a outputs.
     * @param[out] stream       Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByValue(T* inputs, size_t pitch_inputs, U* values, T* outputs, size_t pitch_outputs,
                                size3_t shape, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_MULTIPLY>(inputs, pitch_inputs, values, outputs, pitch_outputs,
                                                       shape, batches, stream);
    }

    /**
     * For each batch, computes the element-wise multiplication between one input array and the second array.
     * @see This version is for padded layout. See the overload for contiguous memory for more details.
     *
     * @tparam T                float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
     * @tparam U                Equal to @a T except for complex types where @a U can be the corresponding real type.
     * @param[in] inputs        Input arrays. One array per batch.
     * @param pitch_inputs      Pitch, in elements, of the input arrays.
     * @param[in] array         Multipliers. The same array is applied to every batch.
     * @param pitch_array       Pitch, in elements, of @a array.
     * @param[out] outputs      Output arrays. One array per batch. Can be equal to @a inputs.
     * @param pitch_outputs     Pitch, in elements, of the output arrays.
     * @param shape             Physical {fast, medium, slow} shape of @a inputs and @a outputs.
     * @param[out] stream       Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_IH void multiplyByArray(T* inputs, size_t pitch_inputs,
                                U* array, size_t pitch_array,
                                T* outputs, size_t pitch_outputs,
                                size3_t shape, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_MULTIPLY>(inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs,
                                                       shape, batches, stream);
    }

    /* ----------------------- */
    /* --- Padded - Divide --- */
    /* ----------------------- */

    /// Divides the input array by a single value.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(T* input, size_t pitch_input, U value, T* output, size_t pitch_output,
                              size3_t shape, Stream& stream) {
        Details::arithByValue<Details::ARITH_DIVIDE>(input, pitch_input, value, output, pitch_output, shape, stream);
    }

    /// For each batch, divides one input array by a single value.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(T* inputs, size_t pitch_inputs, U* values, T* outputs, size_t pitch_outputs,
                              size3_t shape, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_DIVIDE>(inputs, pitch_inputs, values, outputs, pitch_outputs,
                                                     shape, batches, stream);
    }

    /// For each batch, computes the element-wise division between one of the input array and the second array.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByArray(T* inputs, size_t pitch_inputs,
                              U* array, size_t pitch_array,
                              T* outputs, size_t pitch_outputs,
                              size3_t shape, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_DIVIDE>(inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs,
                                                     shape, batches, stream);
    }

    /// For each batch, computes the element-wise safe division between one of the input array and the second array.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideSafeByArray(T* inputs, size_t pitch_inputs,
                                  U* array, size_t pitch_array,
                                  T* outputs, size_t pitch_outputs,
                                  size3_t shape, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_DIVIDE_SAFE>(inputs, pitch_inputs, array, pitch_array,
                                                          outputs, pitch_outputs, shape, batches, stream);
    }

    /* -------------------- */
    /* --- Padded - Add --- */
    /* -------------------- */

    /// For each batch, adds one input array by a single value.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(T* input, size_t pitch_input, U value, T* output, size_t pitch_output,
                         size3_t shape, Stream& stream) {
        Details::arithByValue<Details::ARITH_ADD>(input, pitch_input, value, output, pitch_output, shape, stream);
    }

    /// For each batch, adds one input array by a single value.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(T* inputs, size_t pitch_inputs, U* values, T* outputs, size_t pitch_outputs,
                         size3_t shape, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_ADD>(inputs, pitch_inputs, values, outputs, pitch_outputs,
                                                  shape, batches, stream);
    }

    /// For each batch, computes the element-wise addition between one array and the weights.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addArray(T* inputs, size_t pitch_inputs,
                         U* array, size_t pitch_array,
                         T* outputs, size_t pitch_outputs,
                         size3_t shape, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_ADD>(inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs,
                                                  shape, batches, stream);
    }

    /* ------------------------- */
    /* --- Padded - Subtract --- */
    /* ------------------------- */

    /// For each batch, subtracts one input array by a single value.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(T* input, size_t pitch_input, U value, T* output, size_t pitch_output,
                              size3_t shape, Stream& stream) {
        Details::arithByValue<Details::ARITH_SUBTRACT>(input, pitch_input, value, output, pitch_output, shape, stream);
    }

    /// For each batch, subtracts one input array by a single value.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(T* inputs, size_t pitch_inputs, U* values, T* outputs, size_t pitch_outputs,
                              size3_t shape, uint batches, Stream& stream) {
        Details::arithByValue<Details::ARITH_SUBTRACT>(inputs, pitch_inputs, values, outputs, pitch_outputs,
                                                       shape, batches, stream);
    }

    /// For each batch, computes the element-wise subtraction between one array and the weights.
    /// @see This version is for padded layout. See the overload above for more details.
    /// @warning This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractArray(T* inputs, size_t pitch_inputs,
                              U* array, size_t pitch_array,
                              T* outputs, size_t pitch_outputs,
                              size3_t shape, uint batches, Stream& stream) {
        Details::arithByArray<Details::ARITH_SUBTRACT>(inputs, pitch_inputs, array, pitch_array, outputs, pitch_outputs,
                                                       shape, batches, stream);
    }
}
