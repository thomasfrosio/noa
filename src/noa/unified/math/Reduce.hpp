#pragma once

#include "noa/cpu/math/Reduce.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Reduce.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"

namespace noa::math::details {
    namespace nt = noa::traits;

    template<typename Input>
    constexpr bool is_valid_min_max_median_v =
            nt::is_any_v<nt::mutable_value_type_t<Input>,
                         i16, i32, i64, u16, u32, u64, f16, f32, f64>;

    template<typename Input, typename PreProcessOp>
    using sum_mean_return_t = std::conditional_t<
            nt::is_any_v<PreProcessOp, noa::copy_t, noa::square_t>,
            nt::value_type_t<Input>, nt::value_type_twice_t<Input>>;

    template<typename Input,
             typename PreProcessOp = noa::copy_t,
             typename Output = View<sum_mean_return_t<Input, PreProcessOp>>>
    constexpr bool is_valid_sum_mean_v =
            nt::is_any_v<nt::mutable_value_type_t<Input>,
                         i32, i64, u32, u64, f32, f64, c32, c64> &&
            nt::is_any_v<nt::value_type_t<Output>, sum_mean_return_t<Input, PreProcessOp>> &&
            nt::is_any_v<PreProcessOp,
                         noa::copy_t, noa::nonzero_t, noa::square_t,
                         noa::abs_t, noa::abs_squared_t>;

    template<typename Input, typename Output>
    constexpr bool is_valid_var_std_v =
            nt::is_any_v<nt::mutable_value_type_t<Input>, f32, f64, c32, c64> &&
            std::is_same_v<nt::value_type_t<Output>,
                           nt::value_type_twice_t<Input>>;
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
    /// \param[in] array        Array to reduce.
    /// \param pre_process_op   Preprocessing operator. Used to apply a unary
    ///                         element-wise operator before the reduction.
    ///                         Must be copy_t, nonzero_t, square_t, abs_t or abs_squared_t.
    ///                         For complex input types, abs_t or abs_squared_t return a real value.
    /// \note For (complex)-floating-point types, the CPU backend uses
    ///       a multi-threaded Kahan summation (with Neumaier variation).
    template<typename Input, typename PreProcessOp = noa::copy_t, typename std::enable_if_t<
             noa::traits::is_array_or_view_v<Input> &&
             details::is_valid_sum_mean_v<Input, PreProcessOp>, bool> = true>
    [[nodiscard]] auto sum(const Input& array, PreProcessOp pre_process_op = {}) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::sum(array.get(), array.strides(), array.shape(),
                                  pre_process_op, cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::sum(array.get(), array.strides(), array.shape(),
                                   pre_process_op, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the mean of the input array.
    /// \param[in] array        Array to reduce.
    /// \param pre_process_op   Preprocessing operator. Used to apply a unary
    ///                         element-wise operator before the reduction.
    ///                         Must be copy_t, nonzero_t, square_t, abs_t or abs_squared_t.
    ///                         For complex input types, abs_t or abs_squared_t return a real value.
    /// \note For (complex)-floating-point types, the CPU backend uses
    ///       a multi-threaded Kahan summation (with Neumaier variation).
    template<typename Input, typename PreProcessOp = noa::copy_t, std::enable_if_t<
             noa::traits::is_array_or_view_v<Input> &&
             details::is_valid_sum_mean_v<Input, PreProcessOp>, bool> = true>
    [[nodiscard]] auto mean(const Input& array, PreProcessOp pre_process_op = {}) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::mean(array.get(), array.strides(), array.shape(),
                                   pre_process_op, cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::mean(array.get(), array.strides(), array.shape(),
                                    pre_process_op, stream.cuda());
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

    /// Returns the root-mean-square deviation.
    template<typename Lhs, typename Rhs, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Lhs, Rhs> &&
             noa::traits::are_almost_same_value_type_v<Lhs, Rhs> &&
             noa::traits::is_array_or_view_of_almost_any_v<Lhs, f32, f64>>>
    [[nodiscard]] auto rmsd(const Lhs& lhs, const Rhs& rhs) {
        NOA_CHECK(!lhs.is_empty() && !rhs.is_empty(), "Empty array detected");
        const Device device = rhs.device();

        NOA_CHECK(noa::all(lhs.shape() == rhs.shape()),
                  "The lhs and rhs arrays should have the same shape, but got lhs={} and rhs={}",
                  lhs.shape(), rhs.shape());
        NOA_CHECK(device == lhs.device(),
                  "The arrays should be on the same device, but got lhs={} and rhs={}",
                  lhs.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return cpu::math::rmsd(
                    lhs.get(), lhs.strides(), rhs.get(), rhs.strides(),
                    rhs.shape(), cpu_stream.threads());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return cuda::math::rmsd(
                    lhs.get(), lhs.strides(), rhs.get(), rhs.strides(),
                    rhs.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

namespace noa::math {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of minimum values.
    template<typename Input, typename Output, std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             details::is_valid_min_max_median_v<Input> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>, bool> = true>
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

    /// Reduces an array along some dimensions by taking the minimum value.
    template<typename Input, std::enable_if_t<
             noa::traits::is_array_or_view_v<Input> &&
             details::is_valid_min_max_median_v<Input>, bool> = true>
    [[nodiscard]] auto min(const Input& input, const Shape4<i64>& output_shape) {
        using output_value_t = noa::traits::value_type_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        min(input, output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    template<typename Input, typename Output, std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             details::is_valid_min_max_median_v<Input> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>, bool> = true>
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

    /// Reduces an array along some dimensions by taking the maximum value.
    template<typename Input, std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_min_max_median_v<Input>, bool> = true>
    [[nodiscard]] auto max(const Input& input, const Shape4<i64>& output_shape) {
        using output_value_t = noa::traits::value_type_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        max(input, output);
        return output;
    }

    // TODO Add median()

    /// Reduces an array along some dimensions by taking the sum.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced sums.
    /// \param pre_process_op   Preprocessing operator. Used to apply a unary element-wise operator before the reduction.
    ///                         Must be copy_t, nonzero_t, square_t, abs_t or abs_squared_t.
    ///                         For complex input types, abs_t or abs_squared_t return a real value.
    template<typename Input, typename PreProcessOp = noa::copy_t, typename Output, std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             details::is_valid_sum_mean_v<Input, PreProcessOp, Output>, bool> = true>
    void sum(const Input& input, const Output& output, PreProcessOp pre_process_op = {}) {
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
                               output.get(), output.strides(), output.shape(),
                               pre_process_op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::sum(input.get(), input.strides(), input.shape(),
                            output.get(), output.strides(), output.shape(),
                            pre_process_op, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the sum.
    template<typename Input, typename PreProcessOp = noa::copy_t, std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_sum_mean_v<Input, PreProcessOp>, bool> = true>
    [[nodiscard]] auto sum(const Input& input, const Shape4<i64>& output_shape, PreProcessOp pre_process_op = {}) {
        using output_value_t = details::sum_mean_return_t<Input, PreProcessOp>;
        Array<output_value_t> output(output_shape, input.options());
        sum(input, output, pre_process_op);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced means.
    /// \param pre_process_op   Preprocessing operator. Used to apply a unary element-wise operator before the reduction.
    ///                         Must be copy_t, nonzero_t, square_t, abs_t or abs_squared_t.
    ///                         For complex input types, abs_t or abs_squared_t return a real value.
    template<typename Input, typename PreProcessOp = noa::copy_t, typename Output, std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             details::is_valid_sum_mean_v<Input, PreProcessOp, Output>, bool> = true>
    void mean(const Input& input, const Output& output, PreProcessOp pre_process_op = {}) {
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
                                output.get(), output.strides(), output.shape(),
                                pre_process_op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::mean(input.get(), input.strides(), input.shape(),
                             output.get(), output.strides(), output.shape(),
                             pre_process_op, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the average.
    template<typename Input, typename PreProcessOp = noa::copy_t, std::enable_if_t<
            noa::traits::is_array_or_view_v<Input> &&
            details::is_valid_sum_mean_v<Input, PreProcessOp>, bool> = true>
    [[nodiscard]] auto mean(const Input& input, const Shape4<i64>& output_shape, PreProcessOp pre_process_op = {}) {
        using output_value_t = details::sum_mean_return_t<Input, PreProcessOp>;
        Array<output_value_t> output(output_shape, input.options());
        mean(input, output, pre_process_op);
        return output;
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
             details::is_valid_var_std_v<Input, Output>>>
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

    /// Reduces an array along some dimensions by taking the variance.
    template<typename Input, std::enable_if_t<
             noa::traits::is_array_or_view_v<Input> &&
             details::is_valid_var_std_v<Input, View<noa::traits::value_type_twice_t<Input>>>, bool> = true>
    [[nodiscard]] auto var(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_value_t = noa::traits::value_type_twice_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        var(input, output, ddof);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input           Array or View of f32, f64, c32, f64. To compute the variance of complex types, the absolute
    ///                         value is taken before squaring, so the variance is always real and positive.
    /// \tparam Variance        If the input value type is complex, the variance value type should be the corresponding
    ///                         real type. Otherwise, should be the same as the input value type.
    /// \param[in] input        Input array to reduce.
    /// \param[in] mean         Reduced means.
    /// \param[out] variance    Reduced variances.
    /// \param ddof             Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    template<typename Input, typename Mean, typename Variance, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Mean, Variance> &&
             noa::traits::are_almost_same_value_type_v<Input, Mean> &&
             details::is_valid_var_std_v<Input, Variance>>>
    void mean_var(const Input& input, const Mean& mean, const Variance& variance, i64 ddof = 0) {
        NOA_CHECK(!input.is_empty() && !mean.is_empty() && !variance.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, mean) &&
                  !noa::indexing::are_overlapped(input, variance),
                  "The input and output arrays should not overlap");
        NOA_CHECK(noa::all(mean.shape() == variance.shape()),
                  "The mean and variance arrays should have the same shape, but got mean={} and variance={}",
                  mean.shape(), variance.shape());

        const Device device = input.device();
        NOA_CHECK(device == mean.device() && device == variance.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input={} mean={} and variance={}",
                  device, mean.device(), variance.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::mean_var(
                        input.get(), input.strides(), input.shape(),
                        mean.get(), mean.strides(),
                        variance.get(), variance.strides(),
                        variance.shape(), ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::mean_var(
                    input.get(), input.strides(), input.shape(),
                    mean.get(), mean.strides(),
                    variance.get(), variance.strides(),
                    variance.shape(), ddof, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), mean.share(), variance.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    template<typename Input, std::enable_if_t<
             noa::traits::is_array_or_view_v<Input> &&
             details::is_valid_var_std_v<Input, View<noa::traits::value_type_twice_t<Input>>>, bool> = true>
    [[nodiscard]] auto mean_var(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_mean_value_t = noa::traits::value_type_t<Input>;
        using output_variance_value_t = noa::traits::value_type_twice_t<Input>;
        Array<output_mean_value_t> mean(output_shape, input.options());
        Array<output_variance_value_t> variance(output_shape, input.options());
        mean_var(input, mean, variance, ddof);
        return std::pair{mean, variance};
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
              details::is_valid_var_std_v<Input, Output>>>
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

    /// Reduces an array along some dimensions by taking the standard-deviation.
    template<typename Input, std::enable_if_t<
             noa::traits::is_array_or_view_v<Input> &&
             details::is_valid_var_std_v<Input, View<noa::traits::value_type_twice_t<Input>>>, bool> = true>
    [[nodiscard]] auto std(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_value_t = noa::traits::value_type_twice_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        std(input, output, ddof);
        return output;
    }

    /// Normalizes (and standardizes) an array, by setting its mean to 0 and variance to 1.
    /// Can be in-place or out-of-place. It also returns the mean and variance before normalization.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             noa::traits::is_almost_any_v<noa::traits::value_type_t<Output>, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    auto normalize(const Input& input, const Output& output, i64 ddof = 0) {
        NOA_CHECK(noa::all(input.shape() == output.shape()),
                  "The input and output arrays should have the same shape, but got input={} and output={}",
                  input.shape(), output.shape());
        const auto [mean, var] = mean_var(input, ddof);
        ewise_trinary(input, mean, var, output, minus_divide_t{});
        return std::pair{mean, var};
    }

    /// Normalizes (and standardizes) each batch of an array, by setting mean=0 and variance=1 of each batch.
    /// Can be in-place or out-of-place. It also returns the mean and variance before normalization.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_v<Input, Output> &&
             noa::traits::is_almost_any_v<noa::traits::value_type_t<Output>, f32, f64, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    auto normalize_per_batch(const Input& input, const Output& output, i64 ddof = 0) {
        NOA_CHECK(noa::all(input.shape() == output.shape()),
                  "The input and output arrays should have the same shape, but got input={} and output={}",
                  input.shape(), output.shape());
        const auto batches = Shape4<i64>{output.shape()[0], 1, 1, 1};
        const auto [means, variances] = mean_var(input, batches, ddof);
        ewise_trinary(input, means, variances, output, minus_divide_t{});
        return std::pair{means, variances};
    }
}
