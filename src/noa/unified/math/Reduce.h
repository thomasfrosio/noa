#pragma once

#include "noa/unified/Array.h"

namespace noa::math::details {
    template<typename T>
    constexpr bool is_valid_min_max_v =
            traits::is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double>;

    template<typename T>
    constexpr bool is_valid_sum_mean_v =
            traits::is_any_v<T, int32_t, int64_t, uint32_t, uint64_t, float, double, cfloat_t, cdouble_t>;

    template<int DDOF, typename T, typename U>
    constexpr bool is_valid_var_std_v =
            traits::is_any_v<T, float, double, cfloat_t, cdouble_t> &&
            std::is_same_v<U, traits::value_type_t<T>> &&
            (DDOF == 0 || DDOF == 1);
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
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T sum(const Array<T>& array);

    /// Returns the mean of the input array.
    /// \tparam T           (u)int32_t, (u)int64_t, float, double, cfloat_t, cdouble_t.
    /// \param[in] array    Array to reduce.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T mean(const Array<T>& array);

    /// Returns the variance of the input array.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    [[nodiscard]] U var(const Array<T>& array);

    /// Returns the standard-deviation of the input array.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    [[nodiscard]] U std(const Array<T>& array);

    /// Returns the sum, mean, variance and stddev of the input array.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance and standard deviation.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t. For complex types, the absolute value is taken
    ///                     before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    template<int DDOF = 0, typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    [[nodiscard]] std::tuple<T, T, U, U> statistics(const Array<T>& array);
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
    /// \note On the GPU, if the three innermost dimensions are reduced,
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
    /// \note On the GPU, if the three innermost dimensions are reduced,
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
    /// \note On the GPU, if the three innermost dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
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
    /// \note On the GPU, if the three innermost dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void mean(const Array<T>& input, const Array<T>& output);

    /// Reduces an array along some dimensions by taking the variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \note On the GPU, if the three innermost dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    template<int DDOF = 0, typename T, typename U,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    void var(const Array<T>& input, const Array<T>& output);

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the three or four innermost dimensions
    ///          are of size 1 after reduction.
    /// \tparam DDOF        Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \tparam T           float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \note On the GPU, if the three innermost dimensions are reduced,
    ///       \p output can be on any device, including the CPU.
    template<int DDOF = 0, typename T, typename U,
             typename = std::enable_if_t<details::is_valid_var_std_v<DDOF, T, U>>>
    void std(const Array<T>& input, const Array<T>& output);
}

#define NOA_UNIFIED_REDUCE_
#include "noa/unified/math/Reduce.inl"
#undef NOA_UNIFIED_REDUCE_
