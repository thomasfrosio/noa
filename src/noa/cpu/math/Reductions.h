/// \file noa/cpu/math/Reductions.h
/// \brief Reduction operations for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/common/Functors.h"
#include "noa/cpu/Stream.h"

// -- Reduce each batch to one value -- //
namespace noa::cpu::math {
    /// Reduces the input array along its 3 innermost dimensions using a binary operator()(\p T, \p T) -> \p T.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Reduced value(s). One per batch.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param init             Initial value for the reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename BinaryOp>
    NOA_HOST void reduce(const T* input, size4_t stride, size4_t shape, T* outputs,
                         BinaryOp binary_op, T init, Stream& stream);

    /// Returns the minimum value of the input array(s).
    /// \tparam T               Any type with a noa::math::min(T,T) overload defined.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Minimum value(s). One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void min(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream);

    /// Returns the maximum value of the input array(s).
    /// \tparam T               Any type with a noa::math::max(T,T) overload defined.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Maximum value(s). One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void max(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream);

    /// Returns the sum of the input array(s).
    /// \tparam T               Any type with `T operator+(T,T)` defined.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Sum(s). One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T>
    NOA_HOST void sum(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream);

    /// Returns the mean of the input array(s).
    /// \tparam T               Any type with `T operator+(T,T)` defined.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Mean(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T>
    NOA_IH void mean(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream);

    /// Returns the variance of the input array(s).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Variance(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void var(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream);

    /// Returns the standard-deviation of the input array(s).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param[out] outputs     On the \b host. Standard-deviation(s). One per batch.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void std(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream);

    /// Returns some statistics of the input array(s).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Input array(s) to reduce. One per batch.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] out_mins    On the \b host. Output minimum values.   One value per batch. If nullptr, ignore it.
    /// \param[out] out_maxs    On the \b host. Output maximum values.   One value per batch. If nullptr, ignore it.
    /// \param[out] out_sums    On the \b host. Output sum values.       One value per batch. If nullptr, ignore it.
    /// \param[out] out_means   On the \b host. Output mean values.      One value per batch. If nullptr, ignore it.
    /// \param[out] out_vars    On the \b host. Output variance values.  One value per batch. If nullptr, ignore it.
    /// \param[out] out_stds    On the \b host. Output stddev values.    One value per batch. If nullptr, ignore it.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void statistics(const T* input, size4_t stride, size4_t shape,
                             T* output_mins, T* output_maxs,
                             T* output_sums, T* output_means,
                             T* output_vars, T* output_stds,
                             Stream& stream);
}

// -- Reduce along particular axes -- //
namespace noa::cpu::math {
    /// Reduces an array along some dimensions using a binary operator()(\p T, \p T) -> \p T.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one elements.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param init             Initial value for the reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \example
    /// Reduce a stack of 2D arrays into one single 2D array.
    /// \code
    /// const size4_t input_shape{1,41,4096,4096};
    /// const size4_t output_shape{1,1,4096,4096};
    /// memory::PtrHost<T> stack(input_shape.elements());
    /// memory::PtrHost<T> sum(output_shape.elements());
    /// // do something with stack...
    /// reduce(stack.get(), input_shape.strides(), input_shape,
    ///        sum.get(), output_shape.strides(), output_shape,
    ///        noa::math::plus_t{}, T(0), stream);
    /// \endcode
    template<typename T, typename BinaryOp>
    NOA_HOST void reduce(const T* input, size4_t input_stride, size4_t input_shape,
                         T* output, size4_t output_stride, size4_t output_shape,
                         BinaryOp binary_op, T init, Stream& stream);
}

#define NOA_REDUCTIONS_INL_
#include "noa/cpu/math/Reductions.inl"
#undef NOA_REDUCTIONS_INL_
