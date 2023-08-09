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

    template<typename Input>
    constexpr bool is_valid_var_std_input_v =
            nt::is_varray_of_almost_any_v<Input, f32, f64, c32, c64>;

    template<typename Input, typename Output>
    constexpr bool is_valid_var_std_v =
            is_valid_var_std_input_v<Input> &&
            std::is_same_v<nt::value_type_t<Output>, nt::value_type_twice_t<Input>>;

    template<typename VArray>
    auto axes_to_output_shape(const VArray& varray, Vec4<bool> axes) -> Shape4<i64> {
        auto output_shape = varray.shape();
        for (size_t i = 0; i < 4; ++i)
            if (axes[i])
                output_shape[i] = 1;
        return output_shape;
    }
}

namespace noa::math {
    /// Returns the minimum value of the input array.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto min(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::min(array.get(), array.strides(), array.shape(), cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::min(array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the maximum value of the input array.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto max(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::max(array.get(), array.strides(), array.shape(), cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::max(array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns a pair with the minimum and maximum value of the input array.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto min_max(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::min_max(array.get(), array.strides(), array.shape(), cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::min_max(array.get(), array.strides(), array.shape(), stream.cuda());
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
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto median(const Input& array, bool overwrite = false) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            return noa::cpu::math::median(array.get(), array.strides(), array.shape(), overwrite);
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::median(array.get(), array.strides(), array.shape(), overwrite, stream.cuda());
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
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_sum_mean_v<Input, PreProcessOp>> = true>
    [[nodiscard]] auto sum(const Input& array, PreProcessOp pre_process_op = {}) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::sum(
                    array.get(), array.strides(), array.shape(),
                    pre_process_op, cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::sum(
                    array.get(), array.strides(), array.shape(),
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
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_sum_mean_v<Input, PreProcessOp>> = true>
    [[nodiscard]] auto mean(const Input& array, PreProcessOp pre_process_op = {}) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::mean(
                    array.get(), array.strides(), array.shape(),
                    pre_process_op, cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::mean(
                    array.get(), array.strides(), array.shape(),
                    pre_process_op, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the L2-norm (i.e. sqrt(sum(abs(array)^2))) of the input array.
    /// \tparam Input       Array of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto norm(const Input& array) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::norm(
                    array.get(), array.strides(), array.shape(), cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::norm(
                    array.get(), array.strides(), array.shape(), stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the variance of the input array.
    /// \tparam Input       Array of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto var(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::var(
                    array.get(), array.strides(), array.shape(), ddof, cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::var(
                    array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the standard-deviation of the input array.
    /// \tparam Input       Array of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto std(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::std(
                    array.get(), array.strides(), array.shape(), ddof, cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::std(
                    array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the mean and variance of the input array.
    /// \tparam Input       Array of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto mean_var(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::mean_var(
                    array.get(), array.strides(), array.shape(), ddof, cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::mean_var(
                    array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the mean and standard deviation of the input array.
    /// \tparam Input       Array of f32, f64, c32 or c64. For complex value types, the absolute
    ///                     value is taken before squaring, so the result is always real and positive.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto mean_std(const Input& array, i64 ddof = 0) {
        NOA_CHECK(!array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            cpu_stream.synchronize();
            return noa::cpu::math::mean_std(
                    array.get(), array.strides(), array.shape(), ddof, cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::mean_std(
                    array.get(), array.strides(), array.shape(), ddof, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns the root-mean-square deviation.
    template<typename Lhs, typename Rhs, nt::enable_if_bool_t<
             nt::are_varray_v<Lhs, Rhs> &&
             nt::are_almost_same_value_type_v<Lhs, Rhs> &&
             nt::is_varray_of_almost_any_v<Lhs, f32, f64>> = true>
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
            return noa::cpu::math::rmsd(
                    lhs.get(), lhs.strides(), rhs.get(), rhs.strides(),
                    rhs.shape(), cpu_stream.thread_limit());
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::math::rmsd(
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
    template<typename Input, typename Output, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Output> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_min_max_median_v<Input>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::min(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::min(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the minimum value.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto min(const Input& input, const Shape4<i64>& output_shape) {
        using output_value_t = nt::value_type_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        min(input, output);
        return output;
    }

    /// Reduces an array along some dimension by taking the minimum value.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto min(const Input& input, Vec4<bool> axes) {
        return min(input, details::axes_to_output_shape(input, axes));
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    template<typename Input, typename Output, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Output> &&
             nt::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_min_max_median_v<Input>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::max(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::max(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto max(const Input& input, const Shape4<i64>& output_shape) {
        using output_value_t = nt::value_type_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        max(input, output);
        return output;
    }

    /// Reduces an array along some dimension by taking the maximum value.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_min_max_median_v<Input>> = true>
    [[nodiscard]] auto max(const Input& input, Vec4<bool> axes) {
        return max(input, details::axes_to_output_shape(input, axes));
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
    template<typename Input, typename PreProcessOp = noa::copy_t, typename Output, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Output> && details::is_valid_sum_mean_v<Input, PreProcessOp, Output>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::sum(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        pre_process_op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::sum(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    pre_process_op, cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the sum.
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_sum_mean_v<Input, PreProcessOp>> = true>
    [[nodiscard]] auto sum(const Input& input, const Shape4<i64>& output_shape, PreProcessOp pre_process_op = {}) {
        using output_value_t = details::sum_mean_return_t<Input, PreProcessOp>;
        Array<output_value_t> output(output_shape, input.options());
        sum(input, output, pre_process_op);
        return output;
    }

    /// Reduces an array along some dimensions by taking the sum.
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
            nt::is_varray_v<Input> && details::is_valid_sum_mean_v<Input, PreProcessOp>> = true>
    [[nodiscard]] auto sum(const Input& input, Vec4<bool> axes, PreProcessOp pre_process_op = {}) {
        return sum(input, details::axes_to_output_shape(input, axes), pre_process_op);
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
    template<typename Input, typename PreProcessOp = noa::copy_t, typename Output, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Output> &&
             details::is_valid_sum_mean_v<Input, PreProcessOp, Output>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::mean(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        pre_process_op, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::mean(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    pre_process_op, cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the average.
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
            nt::is_varray_v<Input> && details::is_valid_sum_mean_v<Input, PreProcessOp>> = true>
    [[nodiscard]] auto mean(const Input& input, const Shape4<i64>& output_shape, PreProcessOp pre_process_op = {}) {
        using output_value_t = details::sum_mean_return_t<Input, PreProcessOp>;
        Array<output_value_t> output(output_shape, input.options());
        mean(input, output, pre_process_op);
        return output;
    }

    /// Reduces an array along some dimension by taking the average.
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_sum_mean_v<Input, PreProcessOp>> = true>
    [[nodiscard]] auto mean(const Input& input, Vec4<bool> axes, PreProcessOp pre_process_op = {}) {
        return mean(input, details::axes_to_output_shape(input, axes), pre_process_op);
    }

    /// Reduces an array along some dimensions by taking the L2-norm.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced norms.
    template<typename Input, typename Output, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Output> &&
             details::is_valid_var_std_v<Input, Output>> = true>
    void norm(const Input& input, const Output& output) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "The input and output arrays should not overlap");

        const Device device = input.device();
        NOA_CHECK(device == output.device(),
                  "The input and output arrays must be on the same device, but got input:{} and output:{}",
                  device, output.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::norm(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::norm(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the norm.
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto norm(const Input& input, const Shape4<i64>& output_shape) {
        using output_value_t = nt::value_type_twice_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        norm(input, output);
        return output;
    }

    /// Reduces an array along some dimension by taking the norm.
    template<typename Input, typename PreProcessOp = noa::copy_t, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto norm(const Input& input, Vec4<bool> axes) {
        return norm(input, details::axes_to_output_shape(input, axes));
    }

    /// Reduces an array along some dimensions by taking the variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input       VArray of f32, f64, c32, f64. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \tparam Output      VArray with the same (but mutable) value-type as \p Input, except for if \p Input has
    ///                     a complex value-type, in which case it should be the corresponding real type.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<typename Input, typename Output, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Output> &&
             details::is_valid_var_std_v<Input, Output>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::var(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::var(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    ddof, cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the variance.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto var(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_value_t = nt::value_type_twice_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        var(input, output, ddof);
        return output;
    }

    /// Reduces an array along some dimension by taking the variance.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto var(const Input& input, Vec4<bool> axes, i64 ddof = true) {
        return var(input, details::axes_to_output_shape(input, axes), ddof);
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input       VArray of f32, f64, c32, f64. For complex types, the absolute value is taken
    ///                     before squaring, so the variance and stddev are always real and positive.
    /// \tparam Output      VArray with the same (but mutable) value-type as \p Input, except for if \p Input has
    ///                     a complex value-type, in which case it should be the corresponding real type.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced variances.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<typename Input, typename Output, nt::enable_if_bool_t<
              nt::are_varray_v<Input, Output> &&
              details::is_valid_var_std_v<Input, Output>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::std(
                        input.get(), input.strides(), input.shape(),
                        output.get(), output.strides(), output.shape(),
                        ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::std(
                    input.get(), input.strides(), input.shape(),
                    output.get(), output.strides(), output.shape(),
                    ddof, cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto std(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_value_t = nt::value_type_twice_t<Input>;
        Array<output_value_t> output(output_shape, input.options());
        std(input, output, ddof);
        return output;
    }

    /// Reduces an array along some dimension by taking the standard-deviation.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto std(const Input& input, Vec4<bool> axes, i64 ddof = 0) {
        return std(input, details::axes_to_output_shape(input, axes), ddof);
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input           VArray of f32, f64, c32, f64. To compute the variance of complex types,
    ///                         the absolute value is taken before squaring, so the variance is always real and positive.
    /// \tparam Mean            VArray with the same (but mutable) value-type as \p Input.
    /// \tparam Stddev          VArray with the same (but mutable) value-type as \p Input, except for if \p Input has
    ///                         a complex value-type, in which case it should be the corresponding real type.
    /// \param[in] input        Input array to reduce.
    /// \param[out] mean        Reduced means.
    /// \param[out] variance    Reduced variances.
    /// \param ddof             Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    template<typename Input, typename Mean, typename Variance, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Mean, Variance> &&
             nt::are_almost_same_value_type_v<Input, Mean> &&
             details::is_valid_var_std_v<Input, Variance>> = true>
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
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::mean_var(
                        input.get(), input.strides(), input.shape(),
                        mean.get(), mean.strides(),
                        variance.get(), variance.strides(),
                        variance.shape(), ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::mean_var(
                    input.get(), input.strides(), input.shape(),
                    mean.get(), mean.strides(),
                    variance.get(), variance.strides(),
                    variance.shape(), ddof, cuda_stream);
            cuda_stream.enqueue_attach(input, mean, variance);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto mean_var(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_mean_value_t = nt::value_type_t<Input>;
        using output_variance_value_t = nt::value_type_twice_t<Input>;
        Array<output_mean_value_t> mean(output_shape, input.options());
        Array<output_variance_value_t> variance(output_shape, input.options());
        mean_var(input, mean, variance, ddof);
        return std::pair{mean, variance};
    }

    /// Reduces an array along some dimension by taking the mean and variance.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto mean_var(const Input& input, Vec4<bool> axes, i64 ddof = 0) {
        return mean_var(input, details::axes_to_output_shape(input, axes), ddof);
    }

    /// Reduces an array along some dimensions by taking the mean and standard deviation.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input           VArray of f32, f64, c32, f64. To compute the variance of complex types,
    ///                         the absolute value is taken before squaring, so the variance is always real and positive.
    /// \tparam Mean            VArray with the same (but mutable) value-type as \p Input.
    /// \tparam Stddev          VArray with the same (but mutable) value-type as \p Input, except for if \p Input has
    ///                         a complex value-type, in which case it should be the corresponding real type.
    /// \param[in] input        Input array to reduce.
    /// \param[out] mean        Reduced means.
    /// \param[out] stddev      Reduced standard deviations.
    /// \param ddof             Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    template<typename Input, typename Mean, typename Stddev, nt::enable_if_bool_t<
             nt::are_varray_v<Input, Mean, Stddev> &&
             nt::are_almost_same_value_type_v<Input, Mean> &&
             details::is_valid_var_std_v<Input, Stddev>> = true>
    void mean_std(const Input& input, const Mean& mean, const Stddev& stddev, i64 ddof = 0) {
        NOA_CHECK(!input.is_empty() && !mean.is_empty() && !stddev.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, mean) &&
                  !noa::indexing::are_overlapped(input, stddev),
                  "The input and output arrays should not overlap");
        NOA_CHECK(noa::all(mean.shape() == stddev.shape()),
                  "The mean and stddev arrays should have the same shape, but got mean={} and stddev={}",
                  mean.shape(), stddev.shape());

        const Device device = input.device();
        NOA_CHECK(device == mean.device() && device == stddev.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input={} mean={} and stddev={}",
                  device, mean.device(), stddev.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                noa::cpu::math::mean_std(
                        input.get(), input.strides(), input.shape(),
                        mean.get(), mean.strides(),
                        stddev.get(), stddev.strides(),
                        stddev.shape(), ddof, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::math::mean_std(
                    input.get(), input.strides(), input.shape(),
                    mean.get(), mean.strides(),
                    stddev.get(), stddev.strides(),
                    stddev.shape(), ddof, cuda_stream);
            cuda_stream.enqueue_attach(input, mean, stddev);
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Reduces an array along some dimensions by taking the mean and standard deviation.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto mean_std(const Input& input, const Shape4<i64>& output_shape, i64 ddof = 0) {
        using output_mean_value_t = nt::value_type_t<Input>;
        using output_stddev_value_t = nt::value_type_twice_t<Input>;
        Array<output_mean_value_t> mean(output_shape, input.options());
        Array<output_stddev_value_t> stddev(output_shape, input.options());
        mean_std(input, mean, stddev, ddof);
        return std::pair{mean, stddev};
    }

    /// Reduces an array along some dimension by taking the mean and standard deviation.
    template<typename Input, nt::enable_if_bool_t<
             nt::is_varray_v<Input> && details::is_valid_var_std_input_v<Input>> = true>
    [[nodiscard]] auto mean_std(const Input& input, Vec4<bool> axes, i64 ddof = 0) {
        return mean_std(input, details::axes_to_output_shape(input, axes), ddof);
    }
}

namespace noa::math {
    /// Normalizes an array, according to a normalization mode.
    /// Can be in-place or out-of-place.
    template<typename Input, typename Output, typename = std::enable_if_t<
             nt::are_varray_v<Input, Output> &&
             nt::is_almost_any_v<nt::value_type_t<Output>, f32, f64, c32, c64> &&
             nt::are_almost_same_value_type_v<Input, Output>>>
    void normalize(
            const Input& input,
            const Output& output,
            NormalizationMode normalization_mode = NormalizationMode::MEAN_STD,
            i64 ddof = 0
    ) {
        NOA_CHECK(noa::all(input.shape() == output.shape()),
                  "The input and output arrays should have the same shape, but got input={} and output={}",
                  input.shape(), output.shape());

        switch (normalization_mode) {
            case NormalizationMode::MIN_MAX: {
                const auto [min, max] = min_max(input);
                ewise_trinary(input, min, max - min, output, minus_divide_t{});
                break;
            }
            case NormalizationMode::MEAN_STD: {
                const auto [mean, stddev] = mean_std(input, ddof);
                ewise_trinary(input, mean, stddev, output, minus_divide_t{});
                break;
            }
            case NormalizationMode::L2_NORM: {
                const auto l2_norm = norm(input);
                ewise_binary(input, l2_norm, output, divide_t{});
                break;
            }
        }
    }

    /// Normalizes each batch of an array, according to a normalization mode.
    /// Can be in-place or out-of-place.
    template<typename Input, typename Output, typename = std::enable_if_t<
             nt::are_varray_v<Input, Output> &&
             nt::is_almost_any_v<nt::value_type_t<Output>, f32, f64, c32, c64> &&
             nt::are_almost_same_value_type_v<Input, Output>>>
    void normalize_per_batch(
            const Input& input,
            const Output& output,
            NormalizationMode normalization_mode = NormalizationMode::MEAN_STD,
            i64 ddof = 0
    ) {
        NOA_CHECK(noa::all(input.shape() == output.shape()),
                  "The input and output arrays should have the same shape, but got input={} and output={}",
                  input.shape(), output.shape());
        const auto batches = Shape4<i64>{output.shape()[0], 1, 1, 1};

        switch (normalization_mode) {
            case NormalizationMode::MIN_MAX: {
                const auto min = noa::math::min(input, batches);
                const auto max = noa::math::max(input, batches);
                ewise_binary(max, min, max, noa::minus_t{}); // not ideal...
                ewise_trinary(input, min, max, output, minus_divide_t{});
                break;
            }
            case NormalizationMode::MEAN_STD: {
                const auto [means, stddevs] = mean_std(input, batches, ddof);
                ewise_trinary(input, means, stddevs, output, minus_divide_t{});
                break;
            }
            case NormalizationMode::L2_NORM: {
                const auto l2_norms = norm(input, batches);
                ewise_binary(input, l2_norms, output, divide_t{});
                break;
            }
        }
    }
}
