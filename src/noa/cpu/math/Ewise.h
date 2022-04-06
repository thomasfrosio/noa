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
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param unary_op         Unary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename UnaryOp>
    void ewise(const shared_t<const T[]>& input, size4_t input_stride,
               const shared_t<U[]>& output, size4_t output_stride,
               size4_t shape, UnaryOp unary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs          On the \b host. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param rhs              Right-hand side argument.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const shared_t<const T[]>& lhs, size4_t lhs_stride, U rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs          On the \b host. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] rhs          On the \b host. Right-hand side argument.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<const T[]>& lhs, size4_t lhs_stride,
               const shared_t<const U[]>& rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs          On the \b host. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] rhs          On the \b host. Right-hand side argument.
    /// \param rhs_stride       Rightmost stride, in elements of \p rhs.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs, \p rhs and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<const T[]>& lhs, size4_t lhs_stride,
               const shared_t<const U[]>& rhs, size4_t rhs_stride,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    /// \param[in] lhs          On the \b host. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param mhs              Middle-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const shared_t<const T[]>& lhs, size4_t lhs_stride, U mhs, U rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    /// \param[in] lhs          On the \b host. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] mhs          On the \b host. Middle-hand side argument. One value per batch.
    /// \param[in] rhs          On the \b host. Right-hand side argument. One value per batch.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename TrinaryOp>
    void ewise(const shared_t<const T[]>& lhs, size4_t lhs_stride,
               const shared_t<const U[]>& mhs, const shared_t<const U[]>& rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    /// \param[in] lhs          On the \b host. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] mhs          On the \b host. Middle-hand side argument.
    /// \param mhs_stride       Rightmost stride, in elements, of \p mhs.
    /// \param[in] rhs          On the \b host. Right-hand side argument.
    /// \param rhs_stride       Rightmost stride, in elements, of \p rhs.
    /// \param[out] output      On the \b host. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs, \p mhs, \p rhs and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const shared_t<const T[]>& lhs, size4_t lhs_stride,
               const shared_t<const U[]>& mhs, size4_t mhs_stride,
               const shared_t<const V[]>& rhs, size4_t rhs_stride,
               const shared_t<W[]>& output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);
}

#define NOA_EWISE_INL_
#include "noa/cpu/math/Ewise.inl"
#undef NOA_EWISE_INL_
