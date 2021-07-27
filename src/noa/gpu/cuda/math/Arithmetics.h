/// \file noa/gpu/cuda/math/Arithmetics.h
/// \brief Basic arithmetic operators for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
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
    /// Multiplies \p input by \p value.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U can be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] input        On the \b device. Input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[in] value        Multiplier.
    /// \param[out] output      On the \b device. Output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* input, size_t input_pitch, U value,
                                T* output, size_t output_pitch,
                                size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// Multiplies \p input by \p value. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(input, value, output, elements, stream);
    }

    /// For each batch, multiplies \p inputs by a single value.
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U can be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] inputs       On the \b device. Input arrays. One array per batch.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param values           On the \b device. Multipliers. One value per batch.
    /// \param[out] outputs     On the \b device. Output arrays. One array per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* inputs, size_t inputs_pitch, const U* values,
                                T* outputs, size_t outputs_pitch,
                                size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(inputs, inputs_pitch, values, outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    /// For each batch, multiplies \p inputs by a single value. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByValue(const T* inputs, const U* values, T* outputs,
                                size_t elements, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_MULTIPLY>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise multiplication between \p input and \p array.
    /// \tparam T               float, double, int32_t, uint32_t, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U can be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] inputs       On the \b device. Input arrays. One array per batch.
    /// \param inputs_pitch     Pitch, in elements, of \p inputs.
    /// \param[in] array        On the \b device. Multipliers. The same array is applied to every batch.
    /// \param array_pitch      Pitch, in elements, of \p array.
    /// \param[out] outputs     On the \b device.  Output arrays. One array per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByArray(const T* inputs, size_t inputs_pitch,
                                const U* array, size_t array_pitch,
                                T* outputs, size_t outputs_pitch,
                                size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_MULTIPLY>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    /// For each batch, computes the element-wise multiplication between \p input and \p array. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void multiplyByArray(const T* inputs, const U* array, T* outputs,
                                size_t elements, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_MULTIPLY>(inputs, array, outputs, elements, batches, stream);
    }
}

namespace noa::cuda::math {
    /// Divides \p input by \p value.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                              size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// Divides \p input by \p value. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(input, value, output, elements, stream);
    }

    /// For each batch, divides \p input by a single value.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* inputs, size_t inputs_pitch, const U* values,
                              T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(inputs, inputs_pitch, values,
                                                     outputs, outputs_pitch,
                                                     shape, batches, stream);
    }

    /// For each batch, divides \p input by a single value. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByValue(const T* inputs, const U* values, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_DIVIDE>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, computes the element-wise division between \p input and \p array.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByArray(const T* inputs, size_t inputs_pitch,
                              const U* array, size_t array_pitch,
                              T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                     shape, batches, stream);
    }

    /// For each batch, computes the element-wise division between \p input and \p array. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideByArray(const T* inputs, const U* array, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE>(inputs, array, outputs, elements, batches, stream);
    }


    /// For each batch, computes the element-wise safe division (division by 0 returns 0) between \p inputs and \p array.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByArray(),
    ///      with the additional restriction that \p U cannot be complex (cfloat_t or cdouble_t).
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideSafeByArray(const T* inputs, size_t inputs_pitch,
                                  const U* array, size_t array_pitch,
                                  T* outputs, size_t outputs_pitch,
                                  size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE_SAFE>(inputs, inputs_pitch, array, array_pitch,
                                                          outputs, outputs_pitch, shape, batches, stream);
    }

    /// For each batch, computes the element-wise safe division (division by 0 returns 0) between \p inputs and \p array.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void divideSafeByArray(const T* inputs, const U* array, T* outputs, size_t elements,
                                  uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_DIVIDE_SAFE>(inputs, array, outputs, elements, batches, stream);
    }
}

namespace noa::cuda::math {
    /// Adds \p value to \p input.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                         size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// Adds \p value to \p input. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(input, value, output, elements, stream);
    }

    /// For each batch, adds a single value to \p input.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                         size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(inputs, inputs_pitch, values, outputs, outputs_pitch,
                                                  shape, batches, stream);
    }

    /// For each batch, adds a single value to \p input. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addValue(const T* inputs, const U* values, T* outputs, size_t elements, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_ADD>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, adds \p array to \p inputs.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByArray().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addArray(const T* inputs, size_t inputs_pitch,
                         const U* array, size_t array_pitch,
                         T* outputs, size_t outputs_pitch,
                         size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_ADD>(inputs, inputs_pitch, array, array_pitch, outputs, outputs_pitch,
                                                  shape, batches, stream);
    }

    /// For each batch, adds \p array to \p inputs. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void addArray(const T* inputs, const U* array, T* outputs, size_t elements, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_ADD>(inputs, array, outputs, elements, batches, stream);
    }
}

namespace noa::cuda::math {
    /// Subtracts \p value to \p input.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* input, size_t input_pitch, U value, T* output, size_t output_pitch,
                              size3_t shape, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(input, input_pitch, value, output, output_pitch, shape, stream);
    }

    /// Subtracts \p value to \p input. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* input, U value, T* output, size_t elements, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(input, value, output, elements, stream);
    }

    /// For each batch, subtracts a single value to \p input.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByValue().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* inputs, size_t inputs_pitch, const U* values, T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(inputs, inputs_pitch, values,
                                                       outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    /// For each batch, subtracts a single value to \p input. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractValue(const T* inputs, const U* values, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByValue<details::ARITH_SUBTRACT>(inputs, values, outputs, elements, batches, stream);
    }

    /// For each batch, subtracts \p array to \p input.
    /// \see This function has the same features and restrictions than noa::cuda::math::multiplyByArray().
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractArray(const T* inputs, size_t inputs_pitch,
                              const U* array, size_t array_pitch,
                              T* outputs, size_t outputs_pitch,
                              size3_t shape, uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_SUBTRACT>(inputs, inputs_pitch, array, array_pitch,
                                                       outputs, outputs_pitch,
                                                       shape, batches, stream);
    }

    /// For each batch, subtracts \p array to \p input. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_IH void subtractArray(const T* inputs, const U* array, T* outputs, size_t elements,
                              uint batches, Stream& stream) {
        details::arithByArray<details::ARITH_SUBTRACT>(inputs, array, outputs, elements, batches, stream);
    }
}
