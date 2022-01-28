/// \file noa/cpu/math/Ewise.h
/// \brief Element-wise transformations.
/// \author Thomas - ffyr2w
/// \date 11 Jan 2022

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"

#include "noa/cpu/Stream.h"

namespace noa::cpu::math {
    /// Element-wise transformation using an unary operator()(\p T) -> \p U
    /// \param[in] input        On the \b host. Inputs to transform.
    /// \param input_stride     Rightmost stride, in elements of \p input.
    /// \param[out] output      On the \b host. Transformed arrays.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param unary_op         Unary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename UnaryOp>
    NOA_HOST void ewise(const T* input, size4_t input_stride, U* output, size4_t output_stride,
                        size4_t shape, UnaryOp unary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] input        On the \b host. Inputs to transform.
    /// \param input_stride     Rightmost stride, in elements of \p input.
    /// \param values           Value(s) to use as right-hand side argument.
    ///                         If \p U is a pointer, there should be one value per batch.
    ///                         Otherwise, the same value is applied to all batches.
    /// \param[out] output      On the \b host. Transformed arrays.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp>
    NOA_HOST void ewise(const T* input, size4_t input_stride, U values,
                        V* output, size4_t output_stride,
                        size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] input        On the \b host. Inputs to transform.
    /// \param input_stride     Rightmost stride, in elements of \p input.
    /// \param array            Array(s) to use as right-hand side argument.
    /// \param[out] output      On the \b host. Transformed arrays.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input, \p arrays and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp>
    NOA_HOST void ewise(const T* input, size4_t input_stride,
                        const U* array, size4_t array_stride,
                        V* output, size4_t output_stride,
                        size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U) -> \p V
    /// \tparam U               If \p U is a pointer, there should be one \p value1 and \p value2 value per batch.
    ///                         Otherwise, the same values are applied to all batches.
    /// \param[in] input        On the \b host. Inputs to transform.
    /// \param input_stride     Rightmost stride, in elements of \p input.
    /// \param value1           First value(s) to use as right-hand side argument.
    /// \param value2           Second value(s) to use as right-hand side argument.
    /// \param[out] output      On the \b host. Transformed arrays.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename TrinaryOp>
    NOA_HOST void ewise(const T* input, size4_t input_stride, U value1, U value2,
                        V* output, size4_t output_stride,
                        size4_t shape, TrinaryOp trinary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U) -> \p V
    /// \param[in] input        On the \b host. Inputs to transform.
    /// \param input_stride     Rightmost stride, in elements of \p input.
    /// \param array1           First array(s) to use as right-hand side argument.
    /// \param array1_stride    Rightmost stride, in elements, of \p array1.
    /// \param array2           Second array(s) to use as right-hand side argument.
    /// \param array2_stride    Rightmost stride, in elements, of \p array2.
    /// \param[out] output      On the \b host. Transformed arrays.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input, \p array1, \p array2 and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    NOA_HOST void ewise(const T* input, size4_t input_stride,
                        const U* array1, size4_t array1_stride,
                        const V* array2, size4_t array2_stride,
                        W* output, size4_t output_stride,
                        size4_t shape, TrinaryOp trinary_op, Stream& stream);
}

#define NOA_EWISE_INL_
#include "noa/cpu/math/Ewise.inl"
#undef NOA_EWISE_INL_
