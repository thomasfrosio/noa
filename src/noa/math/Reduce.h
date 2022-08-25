#pragma once

#include "noa/Array.h"

namespace noa::math::details {
    template<typename T>
    constexpr bool is_valid_min_max_v =
            traits::is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double>;

    template<typename T>
    constexpr bool is_valid_sum_mean_v =
            traits::is_any_v<T, int32_t, int64_t, uint32_t, uint64_t, float, double, cfloat_t, cdouble_t>;

    template<typename T, typename U>
    constexpr bool is_valid_var_std_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> && std::is_same_v<U, traits::value_type_t<T>>;
}

namespace noa::math {
    /// Returns the minimum value of the input array.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] array    Array to reduce.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    [[nodiscard]] T min(const Array<T>& array);

    /// Returns the maximum value of the input array.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] array    Array to reduce.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    [[nodiscard]] T max(const Array<T>& array);

    /// Returns the sum of the input array.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] array    Array to reduce.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T sum(const Array<T>& array);

    /// Returns the mean of the input array.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] array    Array to reduce.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T mean(const Array<T>& array);

    /// Returns the variance of the input array.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the result is always real and positive.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    [[nodiscard]] auto var(const Array<T>& array, int ddof = 0);

    /// Returns the standard-deviation of the input array.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    [[nodiscard]] auto std(const Array<T>& array, int ddof = 0);

    /// Returns a tuple with the sum, mean, variance and stddev of the input array.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<traits::is_any_v<T, float, double, cfloat_t, cdouble_t>>>
    [[nodiscard]] auto statistics(const Array<T>& array, int ddof = 0);
}

// -- Reduce along particular axes -- //
namespace noa::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of minimum values.
    /// \note On the GPU, if the depth, height and width dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    void min(const Array<T>& input, const Array<T>& output);

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int16_t, (u)int32_t, (u)int64_t, half_t, float, double.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    /// \note On the GPU, if the depth, height and width dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_v<T>>>
    void max(const Array<T>& input, const Array<T>& output);

    /// Reduces an array along some dimensions by taking the sum.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced sums.
    /// \note On the GPU, if the depth, height and width dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void sum(const Array<T>& input, const Array<T>& output);

    /// Reduces an array along some dimensions by taking the mean.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced means.
    /// \note On the GPU, if the depth, height and width dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void mean(const Array<T>& input, const Array<T>& output);

    /// Reduces an array along some dimensions by taking the variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \tparam U           If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \note On the GPU, if the depth, height and width dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    void var(const Array<T>& input, const Array<U>& output, int ddof = 0);

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \tparam U           If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \note On the GPU, if the depth, height and width dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    /// \note For floating-point and complex types, the CPU backend uses a multi-threaded
    ///       Kahan summation (with Neumaier variation) algorithm is used.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    void std(const Array<T>& input, const Array<U>& output, int ddof = 0);
}

#define NOA_UNIFIED_REDUCE_
#include "noa/math/details/Reduce.inl"
#undef NOA_UNIFIED_REDUCE_
