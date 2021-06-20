/// \file noa/gpu/cuda/math/Arithmetics.h
/// \brief Basic arithmetic operators for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::math::details {
    enum : int { ARITH_ADD, ARITH_SUBTRACT, ARITH_MULTIPLY, ARITH_DIVIDE, ARITH_DIVIDE_SAFE };

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* input, U value, T* output, size_t elements, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* inputs, const U* values, T* outputs, size_t elements, uint batches, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByArray(const T* inputs, const U* array, T* outputs, size_t elements, uint batches, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                      size3_t shape, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                      size3_t shape, uint batches, Stream& stream);

    template<int ARITH, typename T, typename U>
    void arithByArray(const T* inputs, size_t inputs_pitch, const U* array, size_t array_pitch,
                      T* outputs, size_t outputs_pitch, size3_t shape, uint batches, Stream& stream);
}

namespace noa::cuda::math {
    // -- Contiguous - Multiply -- //

    /// Multiplies the input array by a single value.
    /// \see This is the CUDA version of noa::math::multiplyByValue.
    ///      The full documentation is described on the CPU version.
    ///      The same features and restrictions apply to this function.
    ///
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               Equal to \a T except for complex types where \a U can be the corresponding real type.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(input, value, output, elements, stream);
    }

    /// For each batch, multiplies one input array by a single value.
    /// \see This is the CUDA version of noa::math::multiplyByValue.
    ///      The full documentation is described on the CPU version.
    ///      The same features and restrictions apply to this function.
    ///
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               Equal to \a T except for complex types where \a U can be the corresponding real type.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* inputs, const U* values, T* outputs, size_t elements,
                                uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, multiplies one input array by another array.
    /// \see This is the CUDA version of noa::math::multiplyByArray.
    ///      The full documentation is described on the CPU version.
    ///      The same features and restrictions apply to this function.
    ///
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               Equal to \a T except for complex types where \a U can be the corresponding real type.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByArray(const T* inputs, const U* array, T* outputs, size_t elements,
                                uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_MULTIPLY>(inputs, array, outputs, elements, batches, stream);
    }

    // -- Contiguous - Divide -- //

    /// Divides the input array by a single value.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByValue.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(input, value, output, elements, stream);
    }

    /// For each batch, divides one input array by a single value.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByValue.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* inputs, const U* values, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise division between one of the input array and the second array.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByArray.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByArray(const T* inputs, const U* array, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE>(inputs, array, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise division between one of the input array and the second array.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByArray.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideSafeByArray(const T* inputs, const U* array, T* outputs, size_t elements,
                                  uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE_SAFE>(inputs, array, outputs, elements, batches, stream);
    }

    // -- Contiguous - Add -- //

    /// For each batch, adds one input array by a single value.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByValue.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(input, value, output, elements, stream);
    }

    /// For each batch, adds one input array by a single value.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByValue.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* inputs, const U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise addition between one array and the weights.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByArray.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addArray(const T* inputs, const U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_ADD>(inputs, array, outputs, elements, batches, stream);
    }

    // -- Contiguous - Subtract -- //

    /// For each batch, subtracts one input array by a single value.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByValue.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(input, value, output, elements, stream);
    }

    /// For each batch, subtracts one input array by a single value.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByValue.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* inputs, const U* values, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise subtraction between one array and the weights.
    /// \see This function supports same features and restrictions than noa::cuda::math::multiplyByArray.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractArray(const T* inputs, const U* array, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_SUBTRACT>(inputs, array, outputs, elements, batches, stream);
    }

    // -- Padded - Multiply -- //

    /// Multiplies the input array by a single value.
    /// \see This version is for padded layout. See the overload for contiguous memory for more details.
    ///
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               Equal to \a T except for complex types where \a U can be the corresponding real type.
    /// \param[in] input        Input array.
    /// \param input_pitch      Pitch, in elements, of the input array.
    /// \param[in] array        Multiplier.
    /// \param[out] output      Output array. Can be equal to \a input.
    /// \param output_pitch     Pitch, in elements, of the output array.
    /// \param shape            Physical {fast, medium, slow} shape of \a input and \a output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                                size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// For each batch, multiplies one input array by a single value.
    /// \see This version is for padded layout. See the overload for contiguous memory for more details.
    ///
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               Equal to \a T except for complex types where \a U can be the corresponding real type.
    /// \param[in] inputs       Input arrays. One array per batch.
    /// \param inputs_pitch     Pitch, in elements, of the input arrays.
    /// \param values           Multipliers. One per batch.
    /// \param[out] outputs     Output arrays. One array per batch. Can be equal to \a inputs.
    /// \param outputs_pitch    Pitch, in elements, of the output arrays.
    /// \param shape            Physical {fast, medium, slow} shape of \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                                size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(inputs, inputs_pitch, values, outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    /// For each batch, computes the element-wise multiplication between one input array and the second array.
    /// \see This version is for padded layout. See the overload for contiguous memory for more details.
    ///
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               Equal to \a T except for complex types where \a U can be the corresponding real type.
    /// \param[in] inputs       Input arrays. One array per batch.
    /// \param inputs_pitch     Pitch, in elements, of the input arrays.
    /// \param[in] array        Multipliers. The same array is applied to every batch.
    /// \param array_pitch      Pitch, in elements, of \a array.
    /// \param[out] outputs     Output arrays. One array per batch. Can be equal to \a inputs.
    /// \param outputs_pitch    Pitch, in elements, of the output arrays.
    /// \param shape            Physical {fast, medium, slow} shape of \a inputs and \a outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByArray(const T* inputs, size_t inputs_pitch,
                                const U* array, size_t array_pitch,
                                T* outputs, size_t outputs_pitch,
                                size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_MULTIPLY>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    // -- Padded - Divide -- //

    /// Divides the input array by a single value.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                              size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// For each batch, divides one input array by a single value.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(inputs, inputs_pitch, values, outputs, outputs_pitch,
                                                     shape, batches, stream);
    }

    /// For each batch, computes the element-wise division between one of the input array and the second array.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByArray(const T* inputs, size_t inputs_pitch,
                              const U* array, size_t array_pitch,
                              T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                     shape, batches, stream);
    }

    /// For each batch, computes the element-wise safe division between one of the input array and the second array.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideSafeByArray(const T* inputs, size_t inputs_pitch,
                                  const U* array, size_t array_pitch,
                                  T* outputs, size_t outputs_pitch,
                                  size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE_SAFE>(inputs, inputs_pitch, array, array_pitch,
                                                          outputs, outputs_pitch, shape, batches, stream);
    }

    // -- Padded - Add -- //

    /// For each batch, adds one input array by a single value.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                         size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// For each batch, adds one input array by a single value.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                         size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(inputs, inputs_pitch, values, outputs, outputs_pitch,
                                                  shape, batches, stream);
    }

    /// For each batch, computes the element-wise addition between one array and the weights.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addArray(const T* inputs, size_t inputs_pitch,
                         const U* array, size_t array_pitch,
                         T* outputs, size_t outputs_pitch,
                         size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_ADD>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                  shape, batches, stream);
    }

    // -- Padded - Subtract -- //

    /// For each batch, subtracts one input array by a single value.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                              size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// For each batch, subtracts one input array by a single value.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(inputs, inputs_pitch, values, outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    /// For each batch, computes the element-wise subtraction between one array and the weights.
    /// \see This version is for padded layout. See the overload above for more details.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractArray(const T* inputs, size_t inputs_pitch,
                              const U* array, size_t array_pitch,
                              T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_SUBTRACT>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                       shape, batches, stream);
    }
}
