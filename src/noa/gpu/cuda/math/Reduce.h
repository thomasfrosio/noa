/// \file noa/gpu/cuda/math/Reduce.h
/// \brief Reduction operations for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math {
    /// Returns the minimum value of the input array.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Minimum value.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void min(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the maximum value of the input array.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Maximum value.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void max(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the sum of the input array(s).
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Sum.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void sum(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the mean of the input array.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Mean.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void mean(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream);

    /// Returns the variance of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Variance.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void var(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream);

    /// Returns the standard-deviation of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Standard-deviation.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void std(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream);

    /// Returns some statistics of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance and standard deviation.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] out_sum     On the \b host or \b device. Output sum value.
    /// \param[out] out_mean    On the \b host or \b device. Output mean value.
    /// \param[out] out_var     On the \b host or \b device. Output variance value.
    /// \param[out] out_std     On the \b host or \b device. Output stddev value.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void statistics(const T* input, size4_t stride, size4_t shape,
                             T* output_sum, T* output_mean,
                             U* output_var, U* output_std,
                             Stream& stream);
}

// -- Reduce along particular axes -- //
namespace noa::cuda::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Reduced array of minimum values.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void min(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Reduced array of maximum values.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void max(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the sum.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Reduced array of sums.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void sum(const T* input, size4_t input_stride, size4_t input_shape,
                      T* output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the mean.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Reduced array of means.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function may be asynchronous relative to the host and may return before completion.
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
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Reduced array of variances.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
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
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b device. Reduced array of standard deviations.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note This function may be asynchronous relative to the host and may return before completion.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U>
    NOA_HOST void std(const T* input, size4_t input_stride, size4_t input_shape,
                    U* output, size4_t output_stride, size4_t output_shape, Stream& stream);
}
