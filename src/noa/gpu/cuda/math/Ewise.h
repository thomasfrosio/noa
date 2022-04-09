/// \file noa/gpu/cuda/math/Ewise.h
/// \brief Element-wise transformations.
/// \author Thomas - ffyr2w
/// \date 3 Feb 2022
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math {
    /// Element-wise transformation using an unary operator()(\p T) -> \p U
    /// \param[in] input        On the \b device. Inputs to transform.
    /// \param input_stride     Rightmost stride, in elements of \p input.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param unary_op         Unary operation function object that will be applied. Either:
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note Supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (copy|square|abs|negate|one_minus|nonzero|logical_not)_t(A) -> A
    ///       - (copy|square|nonzero|logical_not)_t(B) -> B
    ///       - (nonzero|logical_not)_t(A|B) -> bool
    ///     Floating-points:
    ///       - (copy|square|abs|negate|one_minus|inverse|sqrt|rsqrt|exp|log|cos|sin)_t(C) -> C
    ///       - (one_minus|inverse|normalize)_t(D) -> D
    ///     Complex:
    ///       - (abs|real|imag)_t(D) -> C
    ///     Where:
    ///         A = int16_t, int32_t, or int64_t
    ///         B = uint16_t, uint32_t, or uint64_t
    ///         C = half_t, float, or double
    ///         D = chalf_t, cfloat_t, or cdouble_t
    template<typename T, typename U, typename UnaryOp>
    void ewise(const shared_t<T[]>& input, size4_t input_stride,
               const shared_t<U[]>& output, size4_t output_stride,
               size4_t shape, UnaryOp unary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param rhs              Right-hand side argument.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note Supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(A,A) -> A
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|modulo_t|logical_and|logical_not)_t(A,A) -> A
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|logical_and|logical_not)_t(A,A) -> bool
    ///     Floating-points:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2|min|max)_t(B,B) -> B
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal|pow)_t(B,B) -> B
    ///       - (equal|not_equal|less|less_equal|greater|greater_equal)_t(B,B) -> bool
    ///     Complex:
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(C,C) -> C
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(C,B) -> C
    ///       - (plus|minus|multiply|divide|divide_safe|dist2)_t(B,C) -> C
    ///     Where:
    ///         A = int16_t, int32_t, int64_t, uint16_t, uint32_t, or uint64_t
    ///         B = half_t, float, or double
    ///         C = chalf_t, cfloat_t, or cdouble_t
    template<typename T, typename U, typename V, typename BinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride, U rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] rhs          On the \b host or device. Right-hand side argument. One value per batch.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note The same operators and types are supported as the overload above.
    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride, const shared_t<U[]>& rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a binary operator()(\p T, \p U) -> \p V
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] rhs          On the \b device. Right-hand side argument.
    /// \param rhs_stride       Rightmost stride, in elements of \p rhs.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs, \p rhs and \p output.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note The same operators and types are supported as the overload above.
    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride,
               const shared_t<U[]>& rhs, size4_t rhs_stride,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param mhs              Middle-hand side argument.
    /// \param rhs              Right-hand side argument.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note Supported operators and types are limited to the following list:
    ///     Integers:
    ///       - (within|within_equal|clamp)_t(A,A,A) -> A or bool
    ///     Floating-points:
    ///       - (within|within_equal|clamp)_t(B,B,B) -> B or bool
    ///       - fma_t(B,B,B) -> B
    ///     Complex:
    ///       - fma_t(C,C,C) -> C
    ///       - fma_t(C,B,B) -> C
    ///     Where:
    ///         A = (u)int16_t, (u)int32_t, (u)int64_t
    ///         B = half_t, float, double
    ///         C = chalf_t, cfloat_t, cdouble_t
    template<typename T, typename U, typename V, typename TrinaryOp,
             typename = std::enable_if_t<noa::traits::is_data_v<U>>>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride, U mhs, U rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p U) -> \p V
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] mhs          On the \b host or \b device. Middle-hand side argument. One value per batch.
    /// \param[in] rhs          On the \b host or \b device. Right-hand side argument. One value per batch.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note The same operators and types are supported as the overload above.
    template<typename T, typename U, typename V, typename TrinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride,
               const shared_t<U[]>& mhs, const shared_t<U[]>& rhs,
               const shared_t<V[]>& output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);

    /// Element-wise transformation using a trinary operator()(\p T, \p U, \p V) -> \p W
    /// \param[in] lhs          On the \b device. Left-hand side argument.
    /// \param lhs_stride       Rightmost stride, in elements of \p lhs.
    /// \param[in] mhs          On the \b device. Middle-hand side argument.
    /// \param mhs_stride       Rightmost stride, in elements, of \p mhs.
    /// \param[in] rhs          On the \b device. Right-hand side argument.
    /// \param rhs_stride       Rightmost stride, in elements, of \p rhs.
    /// \param[out] output      On the \b device. Transformed array.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p lhs, \p mhs, \p rhs and \p output.
    /// \param trinary_op       Trinary operation function object that will be applied.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note The same operators and types are supported as the overload above.
    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const shared_t<T[]>& lhs, size4_t lhs_stride,
               const shared_t<U[]>& mhs, size4_t mhs_stride,
               const shared_t<V[]>& rhs, size4_t rhs_stride,
               const shared_t<W[]>& output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream);
}
