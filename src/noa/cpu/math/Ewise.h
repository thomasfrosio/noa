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
    /// \param[in] inputs       On the \b host. Inputs to transform. One per batch.
    /// \param input_pitch      Pitch, in elements of \p inputs.
    /// \param[out] outputs     On the \b host. Transformed arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param unary_op         Unary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename UnaryOp>
    NOA_HOST void ewise(const T* inputs, size3_t input_pitch, U* outputs, size3_t output_pitch,
                        size3_t shape, size_t batches, UnaryOp unary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] inputs       On the \b host. Inputs to transform. One or one per batch.
    /// \param input_pitch      Pitch, in elements of \p inputs.
    /// \param values           Value(s) to use as right-hand side argument.
    ///                         If \p U is a pointer, there should be one per batch.
    ///                         Otherwise, the same value is applied to all batches.
    /// \param[out] outputs     On the \b host. Transformed arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp>
    NOA_HOST void ewise(const T* inputs, size3_t input_pitch, U values,
                        V* outputs, size3_t output_pitch,
                        size3_t shape, size_t batches, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] inputs       On the \b host. Inputs to transform. One or one per batch.
    /// \param input_pitch      Pitch, in elements of \p inputs.
    /// \param arrays           Array(s) to use as right-hand side argument. One or one per batch.
    /// \param[out] outputs     On the \b host. Transformed arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs, \p arrays and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp>
    NOA_HOST void ewise(const T* inputs, size3_t input_pitch, const U* arrays, size3_t array_pitch,
                        V* output, size3_t output_pitch,
                        size3_t shape, size_t batches, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U) -> \p V
    /// \tparam U               If \p U is a pointer, there should be one \p v1 and \p v2 value per batch.
    ///                         Otherwise, the same values are applied to all batches.
    /// \param[in] inputs       On the \b host. Inputs to transform. One or one per batch.
    /// \param input_pitch      Pitch, in elements of \p inputs.
    /// \param v1               First value(s) to use as right-hand side argument.
    /// \param v2               Second value(s) to use as right-hand side argument.
    /// \param[out] outputs     On the \b host. Transformed arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename TrinaryOp>
    NOA_HOST void ewise(const T* inputs, size3_t input_pitch, U v1, U v2,
                        V* outputs, size3_t output_pitch,
                        size3_t shape, size_t batches, TrinaryOp trinary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U) -> \p V
    /// \param[in] inputs       On the \b host. Inputs to transform. One or one per batch.
    /// \param input_pitch      Pitch, in elements of \p inputs.
    /// \param a1               First array(s) to use as right-hand side argument. One or one per batch.
    /// \param output_pitch     Pitch, in elements, of \p a1.
    /// \param a2               Second array(s) to use as right-hand side argument. One or one per batch.
    /// \param a2_pitch         Pitch, in elements, of \p a2.
    /// \param[out] outputs     On the \b host. Transformed arrays. One per batch.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs, \p a1, \p a2 and \p outputs.
    /// \param batches          Number of batches to compute.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    NOA_HOST void ewise(const T* inputs, size3_t input_pitch,
                        const U* a1, size3_t a1_pitch, const V* a2, size3_t a2_pitch,
                        W* output, size3_t output_pitch,
                        size3_t shape, size_t batches, TrinaryOp trinary_op, Stream& stream);
}

#define NOA_EWISE_INL_
#include "noa/cpu/math/Ewise.inl"
#undef NOA_EWISE_INL_
