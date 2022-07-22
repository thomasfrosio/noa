/// \file noa/cpu/math/Reduce.h
/// \brief Reduction operations for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/common/Functors.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::math::details {
    template<typename T>
    constexpr bool is_valid_min_max_v = traits::is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double>;

    template<typename T>
    constexpr bool is_valid_sum_mean_v = traits::is_any_v<T, int32_t, int64_t, uint32_t, uint64_t, float, double, cfloat_t, cdouble_t>;

    template<int DDOF, typename T, typename U>
    constexpr bool is_valid_var_std_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> && std::is_same_v<U, traits::value_type_t<T>> &&
            (DDOF == 0 || DDOF == 1);
}

// -- Reduce each batch to one value -- //
namespace noa::cpu::math {
    /// Reduces the input array to one value using a binary operator()(\p T, \p T) -> \p T.
    /// \param[in] input        On the \b host. Input array(s) to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param init             Initial value for the reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename T, typename BinaryOp>
    [[nodiscard]] T reduce(const shared_t<T[]>& input, size4_t stride, size4_t shape,
                           BinaryOp binary_op, T init, Stream& stream);

    /// Returns the minimum value of the input array.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    [[nodiscard]] T min(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream);

    /// Returns the maximum value of the input array.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    [[nodiscard]] T max(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream);

    /// Returns the sum of the input array(s).
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T sum(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream);

    /// Returns the mean of the input array.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] NOA_IH T mean(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream);

    /// Returns the variance of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    ///                         If \p T is complex, return the corresponding real type.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    [[nodiscard]] U var(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream);

    /// Returns the standard-deviation of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    ///                         If \p T is complex, return the corresponding real type.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    [[nodiscard]] NOA_IH U std(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream);

    /// Returns the sum, mean, variance and stddev of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance and standard deviation.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param stride           Rightmost strides, in elements, of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    [[nodiscard]] std::tuple<T, T, U, U> statistics(const shared_t<T[]>& input,
                                                    size4_t stride, size4_t shape, Stream& stream);
}

// -- Reduce along particular axes -- //
namespace noa::cpu::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Reduced array of minimum values.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    void min(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \tparam T               (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Reduced array of maximum values.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    void max(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the sum.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Reduced array of sums.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded double-precision Kahan summation (with Neumaier variation) algorithm is used.
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
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void sum(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the mean.
    /// \tparam T               (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Reduced array of means.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For floating-point and complex types, this function is not equivalent to reduce().
    ///       Instead, a multi-threaded double-precision Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void mean(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
              const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the variance.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Reduced array of variances.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    void var(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<U[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream);

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b host. Input array to reduce.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param input_shape      Rightmost shape, in elements, of \p input.
    /// \param[out] output      On the \b host. Reduced array of standard deviations.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param output_shape     Rightmost shape, in elements, of \p output.
    ///                         Dimensions should match \p input_shape, or be 1, indicating the dimension should be
    ///                         reduced. Reducing more than one axis at a time is only supported if the reduction
    ///                         results to having one value or one value per batch, i.e. the three or four innermost
    ///                         dimensions are of size 1 after reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note For complex types, the absolute value is taken before squaring, so the result is always real and positive.
    template<int DDOF = 0, typename T, typename U,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    NOA_IH void std(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                    const shared_t<U[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream);
}

#define NOA_REDUCTIONS_INL_
#include "noa/cpu/math/Reduce.inl"
#undef NOA_REDUCTIONS_INL_
