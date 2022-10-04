#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    template<typename T>
    constexpr bool is_valid_min_max_median_v = traits::is_any_v<T, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t, half_t, float, double>;

    template<typename T>
    constexpr bool is_valid_sum_mean_v = traits::is_any_v<T, int32_t, int64_t, uint32_t, uint64_t, float, double, cfloat_t, cdouble_t>;

    template<typename T, typename U>
    constexpr bool is_valid_var_std_v = traits::is_any_v<T, float, double, cfloat_t, cdouble_t> && std::is_same_v<U, traits::value_type_t<T>>;
}

namespace noa::cuda::math {
    // Returns the minimum value of the input array.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_median_v<T>>>
    [[nodiscard]] T min(const shared_t<T[]>& input, dim4_t strides, dim4_t shape, Stream& stream);

    // Returns the maximum value of the input array.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_median_v<T>>>
    [[nodiscard]] T max(const shared_t<T[]>& input, dim4_t strides, dim4_t shape, Stream& stream);

    // Returns the median of the input array.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_median_v<T>>>
    [[nodiscard]] T median(const shared_t<T[]>& input, dim4_t strides, dim4_t shape,
                           bool overwrite, Stream& stream);

    // Returns the sum of the input array(s).
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T sum(const shared_t<T[]>& input, dim4_t strides, dim4_t shape, Stream& stream);

    // Returns the mean of the input array.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    [[nodiscard]] T mean(const shared_t<T[]>& input, dim4_t strides, dim4_t shape, Stream& stream);

    // Returns the variance of the input array.
    template<typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    [[nodiscard]] U var(const shared_t<T[]>& input, dim4_t strides, dim4_t shape, int32_t ddof, Stream& stream);

    // Returns the standard-deviation of the input array.
    template<typename T, typename U = noa::traits::value_type_t<T>,
             typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    [[nodiscard]] U std(const shared_t<T[]>& input, dim4_t strides, dim4_t shape, int32_t ddof, Stream& stream);

    // Returns the sum, mean, variance and stddev of the input array.
    template<typename T, typename U = noa::traits::value_type_t<T>,
            typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    [[nodiscard]] std::tuple<T, T, U, U> statistics(const shared_t<T[]>& input,
                                                    dim4_t strides, dim4_t shape, int32_t ddof, Stream& stream);
}

// -- Reduce along particular axes -- //
namespace noa::cuda::math {
    // Reduces an array along some dimensions by taking the minimum value.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_median_v<T>>>
    void min(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);

    // Reduces an array along some dimensions by taking the maximum value.
    template<typename T, typename = std::enable_if_t<details::is_valid_min_max_median_v<T>>>
    void max(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);

    // Reduces an array along some dimensions by taking the sum.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void sum(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);

    // Reduces an array along some dimensions by taking the mean.
    template<typename T, typename = std::enable_if_t<details::is_valid_sum_mean_v<T>>>
    void mean(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);

    // Reduces an array along some dimensions by taking the variance.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    void var(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<U[]>& output, dim4_t output_strides, dim4_t output_shape, int32_t ddof, Stream& stream);

    // Reduces an array along some dimensions by taking the standard-deviation.
    template<typename T, typename U, typename = std::enable_if_t<details::is_valid_var_std_v<T, U>>>
    void std(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<U[]>& output, dim4_t output_strides, dim4_t output_shape, int32_t ddof, Stream& stream);
}
