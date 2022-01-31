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
    /// Reduces the input array to one value using a binary operator()(\p T, \p T) -> \p T.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Reduced value.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param init             Initial value for the reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename BinaryOp>
    NOA_HOST void reduce(const T* input, size4_t stride, size4_t shape, T* output,
                         BinaryOp binary_op, T init, Stream& stream);

    /// Returns the minimum value of the input array.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Minimum value.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void min(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the maximum value of the input array.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Maximum value.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void max(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the sum of the input array(s).
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Sum.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T>
    NOA_HOST void sum(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the mean of the input array.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Mean.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T>
    NOA_IH void mean(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the variance of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Variance.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void var(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream);

    /// Returns the standard-deviation of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host. Standard-deviation.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<int DDOF = 0, typename T, typename U>
    NOA_IH void std(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream);

    /// Returns some statistics of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance and standard deviation.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] out_min     On the \b host. Output minimum value.  If nullptr, ignore it.
    /// \param[out] out_max     On the \b host. Output maximum value.  If nullptr, ignore it.
    /// \param[out] out_sum     On the \b host. Output sum value.      If nullptr, ignore it.
    /// \param[out] out_mean    On the \b host. Output mean value.     If nullptr, ignore it.
    /// \param[out] out_var     On the \b host. Output variance value. If nullptr, ignore it.
    /// \param[out] out_std     On the \b host. Output stddev value.   If nullptr, ignore it.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void statistics(const T* input, size4_t stride, size4_t shape,
                             T* output_min, T* output_max,
                             T* output_sum, T* output_mean,
                             U* output_var, U* output_std,
                             Stream& stream);
}

// -- Reduce along particular axes -- //
namespace noa::cpu::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array of minimum values.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one elements.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void min(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array of maximum values.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one elements.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void max(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the sum.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array of sums.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one element.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded double-precision Kahan summation (with Neumaier variation) algorithm is used.
    /// \note Reducing more than one axis at a time is only supported if the reduction results to having one
    ///       value per batch, i.e. the three innermost dimensions are of size 1 after reduction.
    ///
    /// \example
    /// Reduce a stack of 2D arrays into one single 2D array.
    /// \code
    /// const size4_t input_shape{1,41,4096,4096};
    /// const size4_t output_shape{1,1,4096,4096};
    /// memory::PtrHost<T> stack(input_shape.elements());
    /// memory::PtrHost<T> sum(output_shape.elements());
    /// // do something with stack...
    /// sum(stack.get(), input_shape.strides(), input_shape,
    ///     sum.get(), output_shape.strides(), output_shape, stream);
    /// \endcode
    template<typename T>
    NOA_HOST void sum(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the mean.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array of means.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one elements.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded double-precision Kahan summation (with Neumaier variation) algorithm is used.
    /// \note Reducing more than one axis at a time is only supported if the reduction results to having one
    ///       value per batch, i.e. the three innermost dimensions are of size 1 after reduction.
    template<typename T>
    NOA_HOST void mean(const T* input, size4_t input_stride, size4_t input_shape,
                       T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the variance.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array of variances.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one elements.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note Reducing more than one axis at a time is only supported if the reduction results to having one
    ///       value per batch, i.e. the three innermost dimensions are of size 1 after reduction.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void var(const T* input, size4_t input_stride, size4_t input_shape,
                      U* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      Reduced array of standard deviations.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced to one element. If all dimensions are 1, \p input is reduced to one elements.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note Reducing more than one axis at a time is only supported if the reduction results to having one
    ///       value per batch, i.e. the three innermost dimensions are of size 1 after reduction.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U>
    NOA_IH void std(const T* input, size4_t input_stride, size4_t input_shape,
                    U* output, size4_t output_stride, size4_t output_shape, Stream& stream);
}

#define NOA_REDUCTIONS_INL_
#include "noa/cpu/math/Reductions.inl"
#undef NOA_REDUCTIONS_INL_
