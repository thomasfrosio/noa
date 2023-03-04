#pragma once

#include "noa/cpu/math/Reduce.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Reduce.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"

namespace noa::math::details {
    template<typename Input>
    constexpr bool is_valid_min_max_median_v =
            noa::traits::is_any_v<std::remove_const_t<noa::traits::value_type_t<Input>>,
                                  i16, i32, i64, u16, u32, u64, f16, f32, f64>;

    template<typename Input>
    constexpr bool is_valid_sum_mean_v =
            noa::traits::is_any_v<std::remove_const_t<noa::traits::value_type_t<Input>>,
                                  i32, i64, u32, u64, f32, f64, c32, c64>;

    template<typename Input, typename Output>
    constexpr bool is_valid_var_std_v =
            noa::traits::is_any_v<std::remove_const_t<noa::traits::value_type_t<Input>>, f32, f64, c32, c64> &&
            std::is_same_v<noa::traits::value_type_t<Output>,
                           noa::traits::value_type_t<noa::traits::value_type_t<Input>>>;
}

namespace noa::math {
    /// Returns the minimum value of the input array.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_min_max_median_v<Input>>>
    [[nodiscard]] auto min(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::min(array.get(), array.strides(), array.shape(), cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::min(array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the maximum value of the input array.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_min_max_median_v<Input>>>
    [[nodiscard]] auto max(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::max(array.get(), array.strides(), array.shape(), cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::max(array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the median of the input array.
    /// \param[in,out] array    Input array.
    /// \param overwrite        Whether the function is allowed to overwrite \p array.
    ///                         If true and if the array is contiguous, the content of \p array is left
    ///                         in an undefined state. Otherwise, array is unchanged and a temporary
    ///                         buffer is allocated.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_min_max_median_v<Input>>>
    [[nodiscard]] auto median(const Input& array, bool overwrite = false) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            return cpu::math::median(array.get(), array.strides(), array.shape(), overwrite);
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::median(array.get(), array.strides(), array.shape(), overwrite, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the sum of the input array.
    /// \note For (complex)-floating-point types, the CPU backend uses
    ///       a multi-threaded Kahan summation (with Neumaier variation).
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_sum_mean_v<Input>>>
    [[nodiscard]] auto sum(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::sum(array.get(), array.strides(), array.shape(), cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::sum(array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the mean of the input array.
    /// \note For (complex)-floating-point types, the CPU backend uses
    ///       a multi-threaded Kahan summation (with Neumaier variation).
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_sum_mean_v<Input>>>
    [[nodiscard]] auto mean(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::mean(array.get(), array.strides(), array.shape(), cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::mean(array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the variance of the input array.
    /// \tparam Input       Array or View of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_var_std_v<Input, noa::traits::value_type_t<Input>>>>
    [[nodiscard]] auto var(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::var(array.get(), array.strides(), array.shape(), ddof, cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::var(array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the mean and variance of the input array.
    /// \tparam Input       Array or View of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_var_std_v<Input, noa::traits::value_type_t<Input>>>>
    [[nodiscard]] auto mean_var(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::mean_var(array.get(), array.strides(), array.shape(), ddof, cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::mean_var(array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the standard-deviation of the input array.
    /// \tparam Input       Array or View of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_var_std_v<Input, noa::traits::value_type_t<Input>>>>
    [[nodiscard]] auto std(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::std(array.get(), array.strides(), array.shape(), ddof, cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::std(array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Normalizes (and standardizes) the input array in-place, by setting its mean to 0 and variance to 1.
    /// It also returns the mean and variance before normalization.
    template<typename Input, typename = std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_var_std_v<Input, Input>>>
    auto normalize(const Input& array, i64 ddof = 0) {
        const auto [mean, var] = mean_var(array, ddof);
        ewise_trinary(array, mean, var, array, minus_divide_t{});
        return std::pair{mean, var};
    }
}

namespace noa::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of minimum values.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::are_array_or_view_v<Input, Output> &&
            details::is_valid_min_max_median_v<Input> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void min(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::min(input.get(), input.strides(), input.shape(),
                               output.get(), output.strides(), output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::min(input.get(), input.strides(), input.shape(),
                            output.get(), output.strides(), output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::are_array_or_view_v<Input, Output> &&
            details::is_valid_min_max_median_v<Input> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void max(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::max(input.get(), input.strides(), input.shape(),
                               output.get(), output.strides(), output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::max(input.get(), input.strides(), input.shape(),
                            output.get(), output.strides(), output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    // TODO Add median()

    /// Reduces an array along some dimensions by taking the sum.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced sums.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::are_array_or_view_v<Input, Output> &&
            details::is_valid_sum_mean_v<Input> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void sum(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::sum(input.get(), input.strides(), input.shape(),
                               output.get(), output.strides(), output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::sum(input.get(), input.strides(), input.shape(),
                            output.get(), output.strides(), output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the mean.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced means.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::are_array_or_view_v<Input, Output> &&
            details::is_valid_sum_mean_v<Input> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void mean(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::mean(input.get(), input.strides(), input.shape(),
                                output.get(), output.strides(), output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::mean(input.get(), input.strides(), input.shape(),
                             output.get(), output.strides(), output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input       Array or View of f32, f64, c32, f64. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \tparam Output      If the input value type is complex, the output value type should be the corresponding
    ///                     real type. Otherwise, should be the same as the input value type.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::are_array_or_view_v<Input, Output> &&
            details::is_valid_var_std_v<Input, Output> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void var(const Input& input, const Output& output, i64 ddof = 0) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::var(input.get(), input.strides(), input.shape(),
                               output.get(), output.strides(), output.shape(),
                               ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::var(input.get(), input.strides(), input.shape(),
                            output.get(), output.strides(), output.shape(),
                            ddof, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input       Array or View of f32, f64, c32, f64. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \tparam Output      If the input value type is complex, the output value type should be the corresponding
    ///                     real type. Otherwise, should be the same as the input value type.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::are_array_or_view_v<Input, Output> &&
            details::is_valid_var_std_v<Input, Output> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void std(const Input& input, const Output& output, i64 ddof = 0) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::std(input.get(), input.strides(), input.shape(),
                               output.get(), output.strides(), output.shape(),
                               ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::std(input.get(), input.strides(), input.shape(),
                            output.get(), output.strides(), output.shape(),
                            ddof, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
