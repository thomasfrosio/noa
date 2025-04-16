#pragma once

#include "noa/core/Reduce.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Ewise.hpp"
#include "noa/unified/ReduceEwise.hpp"
#include "noa/unified/ReduceAxesEwise.hpp"
#include "noa/unified/ReduceIwise.hpp"
#include "noa/unified/ReduceAxesIwise.hpp"

#include "noa/cpu/Median.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/Median.cuh"
#endif

// TODO We used to have a backend specific kernel to compute the variance along an axis, add it back?
//      The current API requires to first compute the means and save them in an array, then relaunch
//      to compute the variances. This can be fused into a single backend call and doesn't require to
//      allocate the temporary array.
//      Or add a core interface to compute "nested" reductions??

// TODO Add support for variance/stddev/l2_norm of integers, numpy casts to f64.
//      This will ultimately add support for the normalize(_per_batch) function.

namespace noa::guts {
    template<typename VArray>
    auto axes_to_output_shape(const VArray& varray, ReduceAxes axes) -> Shape4<i64> {
        auto output_shape = varray.shape();
        if (axes.batch)
            output_shape[0] = 1;
        if (axes.depth)
            output_shape[1] = 1;
        if (axes.height)
            output_shape[2] = 1;
        if (axes.width)
            output_shape[3] = 1;
        return output_shape;
    }

    inline auto n_elements_to_reduce(const Shape4<i64>& input_shape, const Shape4<i64>& output_shape) -> i64 {
        i64 n_elements_to_reduce{1};
        for (size_t i{}; i < 4; ++i) {
            if (input_shape[i] > 1 and output_shape[i] == 1)
                n_elements_to_reduce *= input_shape[i];
        }
        return n_elements_to_reduce;
    }

    template<typename Input, typename ReducedValue, typename ReducedOffset, typename ArgValue, typename ArgOffset,
         typename InputValue = nt::value_type_t<Input>,
         typename ArgValue_ = std::conditional_t<nt::empty<ArgValue>, InputValue, nt::value_type_t<ArgValue>>,
         typename ArgOffset_ = std::conditional_t<nt::empty<ArgOffset>, i64, nt::value_type_t<ArgOffset>>,
         typename ReducedValue_ = std::conditional_t<std::is_void_v<ReducedValue>, InputValue, ReducedValue>,
         typename ReducedOffset_ = std::conditional_t<std::is_void_v<ReducedOffset>, i64, ReducedOffset>>
    concept arg_reduceable =
        nt::readable_varray_decay<Input> and
        (nt::writable_varray_decay<ArgValue> or nt::empty<ArgValue>) and
        (nt::writable_varray_decay<ArgOffset> or nt::empty<ArgOffset>) and
        (nt::numeric<InputValue, ReducedValue_> or (nt::complex<InputValue> and nt::real<ReducedValue_>)) and
        nt::numeric<ArgValue_> and nt::integer<ArgOffset_, ReducedOffset_>;
}

namespace noa {
    /// Returns the minimum value of the input array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto min(const Input& array) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::max();
        value_type output;
        reduce_ewise(array, init, output, ReduceMin{});
        return output;
    }

    /// Returns the maximum value of the input array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto max(const Input& array) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::lowest();
        value_type output;
        reduce_ewise(array, init, output, ReduceMax{});
        return output;
    }

    /// Returns a pair with the minimum and maximum value of the input array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto min_max(const Input& array) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init_min = std::numeric_limits<value_type>::max();
        auto init_max = std::numeric_limits<value_type>::lowest();
        Pair<value_type, value_type> output;
        reduce_ewise(array, wrap(init_min, init_max), wrap(output.first, output.second), ReduceMinMax{});
        return output;
    }

    /// Returns the median of the input array.
    /// \param[in,out] array    Input array.
    /// \param overwrite        Whether the function is allowed to overwrite \p array.
    ///                         If true and if the array is contiguous, the content of \p array is left
    ///                         in an undefined state. Otherwise, array is unchanged and a temporary
    ///                         buffer is allocated.
    template<nt::readable_varray_of_scalar Input>
    [[nodiscard]] auto median(const Input& array, bool overwrite = false) {
        check(not array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            return noa::cpu::median(array.get(), array.strides(), array.shape(), overwrite);
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::median(array.get(), array.strides(), array.shape(), overwrite, stream.cuda());
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Returns the sum of the input array.
    /// \note For (complex)-floating-point types, the CPU backend uses a Kahan summation (with Neumaier variation).
    template<nt::readable_varray Input>
    [[nodiscard]] auto sum(const Input& array) {
        using value_t = nt::mutable_value_type_t<Input>;
        value_t output;

        if constexpr (nt::real_or_complex<value_t>) {
            if (array.device().is_cpu()) {
                using op_t = ReduceAccurateSum<value_t>;
                using pair_t = op_t::pair_type;
                reduce_ewise<ReduceEwiseOptions{.generate_gpu = false}>(array, pair_t{}, output, op_t{});
                return output;
            }
        }
        reduce_ewise(array, value_t{}, output, ReduceSum{});
        return output;
    }

    /// Returns the mean of the input array.
    /// \note For (complex)-floating-point types, the CPU backend uses a Kahan summation (with Neumaier variation).
    template<nt::readable_varray Input>
    [[nodiscard]] auto mean(const Input& array) {
        using value_t = nt::mutable_value_type_t<Input>;
        value_t output;

        if constexpr (nt::real_or_complex<value_t>) {
            if (array.device().is_cpu()) {
                using op_t = ReduceAccurateMean<value_t>;
                using pair_t = op_t::pair_type;
                auto op = op_t{.size=static_cast<op_t::mean_type>(array.ssize())};
                reduce_ewise<ReduceEwiseOptions{.generate_gpu = false}>(array, pair_t{}, output, op);
                return output;
            }
        }
        using mean_t = nt::value_type_t<value_t>;
        reduce_ewise(array, value_t{}, output, ReduceMean{.size=static_cast<mean_t>(array.ssize())});
        return output;
    }

    /// Returns the L2-norm of the input array.
    template<nt::readable_varray_of_real_or_complex Input>
    [[nodiscard]] auto l2_norm(const Input& array) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        real_t output;

        if (array.device().is_cpu()) {
            reduce_ewise<ReduceEwiseOptions{.generate_gpu = false}>(
                array, Pair<f64, f64>{}, output, ReduceAccurateL2Norm{});
            return output;
        }
        reduce_ewise(array, real_t{}, output, ReduceL2Norm{});
        return output;
    }

    /// Returns the mean and variance of the input array.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<nt::readable_varray_of_real_or_complex Input>
    [[nodiscard]] auto mean_variance(const Input& array, i64 ddof = 0) {
        auto mean = noa::mean(array);
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        real_t variance;

        if (array.device().is_cpu()) {
            using double_t = std::conditional_t<nt::real<value_t>, f64, c64>;
            auto mean_double = static_cast<double_t>(mean);
            auto size = static_cast<f64>(array.ssize() - ddof);
            reduce_ewise<ReduceEwiseOptions{.generate_gpu = false}>(
                wrap(array, mean_double), f64{}, variance, ReduceVariance{size});
            return Pair{mean, variance};
        }
        auto size = static_cast<real_t>(array.ssize() - ddof);
        reduce_ewise(wrap(array, mean), real_t{}, variance, ReduceVariance{size});
        return Pair{mean, variance};
    }

    /// Returns the mean and standard deviation of the input array.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<nt::readable_varray_of_real_or_complex Input>
    [[nodiscard]] auto mean_stddev(const Input& array, i64 ddof = 0) {
        auto pair = mean_variance(array, ddof);
        pair.second = sqrt(pair.second);
        return pair;
    }

    /// Returns the variance of the input array.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<nt::readable_varray_of_real_or_complex Input>
    [[nodiscard]] auto variance(const Input& array, i64 ddof = 0) {
        const auto& [mean, variance] = mean_variance(array, ddof);
        return variance;
    }

    /// Returns the standard-deviation of the input array.
    /// \param[in] array    Array to reduce.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    template<nt::readable_varray_of_real_or_complex Input>
    [[nodiscard]] auto stddev(const Input& array, i64 ddof = 0) {
        return sqrt(variance(array, ddof));
    }

    /// Returns the root-mean-square deviation.
    template<nt::readable_varray_of_real Lhs,
             nt::readable_varray_of_real Rhs>
    [[nodiscard]] auto rmsd(const Lhs& lhs, const Rhs& rhs) {
        using lhs_value_t = nt::mutable_value_type_t<Lhs>;
        using rhs_value_t = nt::mutable_value_type_t<Rhs>;
        using real_t = std::conditional_t<nt::same_as<f64, lhs_value_t>, lhs_value_t, rhs_value_t>;
        real_t output;

        if (lhs.device().is_cpu()) {
            reduce_ewise<ReduceEwiseOptions{.generate_gpu = false}>(
                wrap(lhs, rhs), f64{}, output, ReduceRMSD{static_cast<f64>(lhs.ssize())});
        } else {
            reduce_ewise<ReduceEwiseOptions{.generate_cpu = false}>(
                wrap(lhs, rhs), real_t{}, output, ReduceRMSD{static_cast<real_t>(lhs.ssize())});
        }
        return output;
    }

    /// Returns {offset, maximum} of the input array.
    /// \note If the maximum value appears more than once, this function makes no guarantee to which one is selected.
    /// \note To get the corresponding 4d indices, noa::indexing::offset2index(offset, input) can be used.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto argmax(const Input& input) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduced_t = Pair<value_t, i64>;
        using op_t = ReduceFirstMax<AccessorI64<const value_t, 4>, reduced_t>;
        reduced_t reduced{std::numeric_limits<value_t>::lowest(), i64{}};
        reduced_t output{};
        reduce_iwise(input.shape(), input.device(), reduced,
                     wrap(output.first, output.second),
                     op_t{{ng::to_accessor(input)}});
        return output;
    }

    /// Returns {offset, minimum} of the input array.
    /// \note If the minimum value appears more than once, this function makes no guarantee to which one is selected.
    /// \note To get the corresponding 4d indices, noa::indexing::offset2index(offset, input) can be used.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto argmin(const Input& input) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduced_t = Pair<value_t, i64>;
        using op_t = ReduceFirstMin<AccessorI64<const value_t, 4>, reduced_t>;
        reduced_t reduced{std::numeric_limits<value_t>::max(), i64{}};
        reduced_t output{};
        reduce_iwise(input.shape(), input.device(), reduced,
                     wrap(output.first, output.second),
                     op_t{{ng::to_accessor(input)}});
        return output;
    }
}

namespace noa {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of minimum values.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_numeric Output>
    void min(Input&& input, Output&& output) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::max();
        reduce_axes_ewise(std::forward<Input>(input), init, std::forward<Output>(output), ReduceMin{});
    }

    /// Reduces an array along some dimension by taking the minimum value.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto min(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        min(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_numeric Output>
    void max(Input&& input, Output&& output) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::lowest();
        reduce_axes_ewise(std::forward<Input>(input), init, std::forward<Output>(output), ReduceMax{});
    }

    /// Reduces an array along some dimension by taking the maximum value.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto max(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        max(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimension(s) by taking the minimum and maximum values.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_numeric Min,
             nt::writable_varray_decay_of_numeric Max>
    void min_max(Input&& input, Min&& min, Max&& max) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init_min = std::numeric_limits<value_type>::max();
        auto init_max = std::numeric_limits<value_type>::lowest();
        reduce_axes_ewise(
            std::forward<Input>(input),
            wrap(init_min, init_max),
            wrap(std::forward<Min>(min), std::forward<Max>(max)),
            ReduceMinMax{});
    }

    /// Reduces an array along some dimension(s) by taking the minimum and maximum values.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto min_max(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
            Array<value_t>(output_shape, input.options()),
            Array<value_t>(output_shape, input.options()),
        };
        min_max(std::forward<Input>(input), output.first, output.second);
        return output;
    }

    // TODO Add median()

    /// Reduces an array along some dimensions by taking the sum.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced sums.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    void sum(Input&& input, Output&& output) {
        using input_value_t = nt::mutable_value_type_t<Input>;

        if constexpr (nt::real_or_complex<input_value_t>) {
            if (input.device().is_cpu()) {
                using op_t = ReduceAccurateSum<input_value_t>;
                using pair_t = op_t::pair_type;
                return reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_gpu = false}>(
                    std::forward<Input>(input), pair_t{}, std::forward<Output>(output), op_t{});
            }
        }
        reduce_axes_ewise(std::forward<Input>(input), input_value_t{}, std::forward<Output>(output), ReduceSum{});
    }

    /// Reduces an array along some dimensions by taking the sum.
    template<nt::readable_varray_decay Input>
    [[nodiscard]] auto sum(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        sum(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced means.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    void mean(Input&& input, Output&& output) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using scalar_t = nt::value_type_t<input_value_t>;
        const auto n_elements_to_reduce = guts::n_elements_to_reduce(input.shape(), output.shape());

        if constexpr (nt::real_or_complex<input_value_t>) {
            if (input.device().is_cpu()) {
                using op_t = ReduceAccurateMean<input_value_t>;
                using pair_t = op_t::pair_type;
                return reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_gpu = false}>(
                    std::forward<Input>(input), pair_t{}, std::forward<Output>(output),
                    op_t{.size=static_cast<op_t::mean_type>(n_elements_to_reduce)});
            }
        }
        reduce_axes_ewise(
            std::forward<Input>(input),
            input_value_t{},
            std::forward<Output>(output),
            ReduceMean{.size = static_cast<scalar_t>(n_elements_to_reduce)});
    }

    /// Reduces an array along some dimensions by taking the average.
    template<nt::readable_varray_decay Input>
    [[nodiscard]] auto mean(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        mean(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the L2-norm.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced norms.
    template<nt::readable_varray_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_real Output>
    void l2_norm(Input&& input, Output&& output) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;

        if constexpr (nt::real_or_complex<value_t>) {
            if (input.device().is_cpu()) {
                return reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_gpu = false}>(
                    std::forward<Input>(input), Pair<f64, f64>{}, std::forward<Output>(output),
                    ReduceAccurateL2Norm{});
            }
        }
        reduce_axes_ewise(std::forward<Input>(input), real_t{}, std::forward<Output>(output), ReduceL2Norm{});
    }

    /// Reduces an array along some dimensions by taking the norm.
    template<nt::readable_varray_decay_of_real_or_complex Input>
    [[nodiscard]] auto l2_norm(Input&& input, ReduceAxes axes) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<real_t> output(guts::axes_to_output_shape(input, axes), input.options());
        l2_norm(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \tparam Input           VArray of f32, f64, c32, f64. To compute the variance of complex types,
    ///                         the absolute value is taken before squaring, so the variance is always real and positive.
    /// \tparam Mean            VArray with the same (but mutable) value-type as \p Input.
    /// \tparam Variance        VArray with the same (but mutable) value-type as \p Input, except for if \p Input has
    ///                         a complex value-type, in which case it should be the corresponding real type.
    /// \param[in] input        Input array to reduce.
    /// \param[out] means       Reduced means.
    /// \param[out] variances   Reduced variances.
    /// \param ddof             Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    template<nt::readable_varray_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_real_or_complex Mean,
             nt::writable_varray_decay_of_real Variance>
    void mean_variance(Input&& input, Mean&& means, Variance&& variances, i64 ddof = 0) {
        check(vall(Equal{}, means.shape(), variances.shape()),
              "The means and variances should have the same shape, but got means={} and variances={}",
              means.shape(), variances.shape());

        mean(input, means);
        auto n_reduced = guts::n_elements_to_reduce(input.shape(), means.shape()) - ddof;
        auto shape = input.shape();

        if (input.device().is_cpu()) {
            reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_gpu = false}>(
                wrap(std::forward<Input>(input), ni::broadcast(std::forward<Mean>(means), shape)),
                f64{}, std::forward<Variance>(variances), ReduceVariance{static_cast<f64>(n_reduced)});
        } else {
            using real_t = nt::value_type_t<nt::mutable_value_type_t<Input>>;
            reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_cpu = false}>(
                wrap(std::forward<Input>(input), ni::broadcast(std::forward<Mean>(means), shape)),
                real_t{}, std::forward<Variance>(variances), ReduceVariance{static_cast<real_t>(n_reduced)});
        }
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    template<nt::readable_varray_decay_of_real_or_complex Input>
    [[nodiscard]] auto mean_variance(Input&& input, ReduceAxes axes, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::mutable_value_type_twice_t<Input>;

        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
            Array<mean_t>(output_shape, input.options()),
            Array<variance_t>(output_shape, input.options()),
        };
        mean_variance(std::forward<Input>(input), output.first, output.second, ddof);
        return output;
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
    /// \param[out] means      Reduced means.
    /// \param[out] stddevs     Reduced standard deviations.
    /// \param ddof             Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    template<nt::readable_varray_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_real_or_complex Mean,
             nt::writable_varray_decay_of_real Stddev>
    void mean_stddev(Input&& input, Mean&& means, Stddev&& stddevs, i64 ddof = 0) {
        check(vall(Equal{}, means.shape(), stddevs.shape()),
              "The means and stddevs should have the same shape, but got means={} and stddevs={}",
              means.shape(), stddevs.shape());

        mean(input, means);
        auto n_reduced = guts::n_elements_to_reduce(input.shape(), means.shape()) - ddof;
        auto shape = input.shape();

        if (input.device().is_cpu()) {
            reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_gpu = false}>(
                wrap(std::forward<Input>(input), ni::broadcast(std::forward<Mean>(means), shape)),
                f64{}, std::forward<Stddev>(stddevs), ReduceStddev{static_cast<f64>(n_reduced)});
        } else {
            using real_t = nt::value_type_t<nt::mutable_value_type_t<Input>>;
            reduce_axes_ewise<ReduceAxesEwiseOptions{.generate_cpu = false}>(
                wrap(std::forward<Input>(input), ni::broadcast(std::forward<Mean>(means), shape)),
                real_t{}, std::forward<Stddev>(stddevs), ReduceStddev{static_cast<real_t>(n_reduced)});
        }
    }

    /// Reduces an array along some dimensions by taking the mean and standard deviation.
    template<nt::readable_varray_decay_of_real_or_complex Input>
    [[nodiscard]] auto mean_stddev(Input&& input, ReduceAxes axes, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::value_type_t<mean_t>;

        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
            Array<mean_t>(output_shape, input.options()),
            Array<variance_t>(output_shape, input.options()),
        };
        mean_stddev(std::forward<Input>(input), output.first, output.second, ddof);
        return output;
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
    template<nt::readable_varray_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_real Output>
    void variance(Input&& input, Output&& output, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        Array<mean_t> means(output.shape(), output.options());
        mean_variance(std::forward<Input>(input), means, std::forward<Output>(output), ddof);
    }

    /// Reduces an array along some dimensions by taking the variance.
    template<nt::readable_varray_decay_of_real_or_complex Input>
    [[nodiscard]] auto variance(Input&& input, ReduceAxes axes, i64 ddof = 0) {
        using value_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::value_type_t<value_t>;
        Array<variance_t> variances(guts::axes_to_output_shape(input, axes), input.options());
        variance(std::forward<Input>(input), variances, ddof);
        return variances;
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
    template<nt::readable_varray_decay_of_real_or_complex Input,
             nt::writable_varray_decay_of_real Output>
    void stddev(Input&& input, Output&& output, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        Array<mean_t> means(output.shape(), output.options());
        mean_stddev(std::forward<Input>(input), means, std::forward<Output>(output), ddof);
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    template<nt::readable_varray_decay_of_real_or_complex Input>
    [[nodiscard]] auto stddev(Input&& input, ReduceAxes axes, i64 ddof = 0) {
        using value_t = nt::mutable_value_type_t<Input>;
        using stddev_t = nt::value_type_t<value_t>;
        Array<stddev_t> stddevs(guts::axes_to_output_shape(input, axes), input.options());
        stddev(std::forward<Input>(input), stddevs, ddof);
        return stddevs;
    }

    /// Reduces an array along some dimensions by taking the maximum value along the reduced axis/axes.
    /// \details Dimensions of the output arrays should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    ///
    /// \tparam ReducedValue        Value type used for the reduction. Any numeric or void.
    ///                             If void, this defaults to the input value type.
    ///                             The input is explicitly converted to this type.
    ///                             If real and the input is complex, the power spectrum of the input is first computed.
    /// \tparam ReducedOffset       Offset type used for the reduction. Any integer or void.
    ///                             If void, this defaults to the input index type (i64).
    /// \param[in] input            Array to reduce.
    /// \param[out] output_values   Array where to save the maximum values, or empty.
    ///                             If real and the reduced value is complex, the power is computed.
    /// \param[out] output_offsets  Array where to save the offsets of the maximum values, or empty.
    /// \note If the maximum value appears more than once, this function makes no guarantee to which one is selected.
    template<typename ReducedValue = void,
             typename ReducedOffset = void,
             typename Input,
             typename Values = Empty,
             typename Offsets = Empty>
    requires guts::arg_reduceable<Input, ReducedValue, ReducedOffset, Values, Offsets>
    void argmax(
        Input&& input,
        Values&& output_values,
        Offsets&& output_offsets
    ) {
        constexpr bool has_values = not nt::empty<Values>;
        constexpr bool has_offsets = not nt::empty<Offsets>;
        if constexpr (has_offsets or has_values) {
            check(not input.is_empty(), "Empty array detected");
            auto shape = input.shape();
            auto accessor = ng::to_accessor(input);

            using input_value_t = nt::mutable_value_type_t<Input>;
            using accessor_t = AccessorI64<const input_value_t, 4>;
            using reduce_value_t = std::conditional_t<std::is_void_v<ReducedValue>, input_value_t, ReducedValue>;
            using reduce_offset_t = std::conditional_t<std::is_void_v<ReducedOffset>, i64, ReducedOffset>;
            using pair_t = Pair<reduce_value_t, reduce_offset_t>;
            auto reduced = pair_t{std::numeric_limits<reduce_value_t>::lowest(), reduce_offset_t{}};

            auto device = input.device();
            if constexpr (has_offsets and has_values) {
                reduce_axes_iwise(
                    shape, device, reduced, wrap(output_values.view(), output_offsets.view()),
                    ReduceFirstMax<accessor_t, pair_t, true>{{accessor}},
                    std::forward<Input>(input),
                    std::forward<Values>(output_values),
                    std::forward<Offsets>(output_offsets));

            } else if constexpr (has_offsets) {
                reduce_axes_iwise(
                    shape, device, reduced, output_offsets.view(),
                    ReduceFirstMax<accessor_t, pair_t, false>{{accessor}},
                    std::forward<Input>(input),
                    std::forward<Offsets>(output_offsets));

            } else {
                // Reorder DHW to rightmost if offsets are not computed.
                auto arg_values = output_values.view();
                const auto order_3d = ni::order(input.strides().pop_front(), shape.pop_front());
                if (vany(NotEqual{}, order_3d, Vec{0, 1, 2})) {
                    auto order_4d = (order_3d + 1).push_front(0);
                    shape = shape.reorder(order_4d);
                    accessor.reorder(order_4d);
                    arg_values = arg_values.permute(order_4d);
                }

                reduce_axes_iwise(
                    shape, device, reduced, arg_values,
                    ReduceFirstMax<accessor_t, pair_t, true>{{accessor}},
                    std::forward<Input>(input),
                    std::forward<Values>(output_values));
            }
        }
    }

    /// Reduces an array along some dimensions by taking the minimum value along the reduced axis/axes.
    /// \details Dimensions of the output arrays should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    ///
    /// \tparam ReducedValue        Value type used for the reduction. Any numeric or void.
    ///                             If void, this defaults to the input value type.
    ///                             The input is explicitly converted to this type.
    ///                             If real and the input is complex, the power spectrum of the input is first computed.
    /// \tparam ReducedOffset       Offset type used for the reduction. Any integer or void.
    ///                             If void, this defaults to the input index type (i64).
    /// \param[in] input            Array to reduce.
    /// \param[out] output_values   Array where to save the minimum values, or empty.
    ///                             If real and the reduced value is complex, the power is computed.
    /// \param[out] output_offsets  Array where to save the offsets of the minimum values, or empty.
    /// \note If the minimum value appears more than once, this function makes no guarantee to which one is selected.
    template<typename ReducedValue = void,
             typename ReducedOffset = void,
             typename Input,
             typename Values = Empty,
             typename Offsets = Empty>
    requires guts::arg_reduceable<Input, ReducedValue, ReducedOffset, Values, Offsets>
    void argmin(
        Input&& input,
        Values&& output_values,
        Offsets&& output_offsets
    ) {
        // noa::argmin(f32, f64, {});
        constexpr bool has_values = not nt::empty<Values>;
        constexpr bool has_offsets = not nt::empty<Offsets>;
        if constexpr (has_offsets or has_values) {
            check(not input.is_empty(), "Empty array detected");
            auto shape = input.shape();
            auto accessor = ng::to_accessor(input);

            using input_value_t = nt::mutable_value_type_t<Input>;
            using accessor_t = AccessorI64<const input_value_t, 4>;
            using reduce_value_t = std::conditional_t<std::is_void_v<ReducedValue>, input_value_t, ReducedValue>;
            using reduce_offset_t = std::conditional_t<std::is_void_v<ReducedOffset>, i64, ReducedOffset>;
            using pair_t = Pair<reduce_value_t, reduce_offset_t>;
            auto reduced = pair_t{std::numeric_limits<reduce_value_t>::max(), reduce_offset_t{}};

            auto device = input.device();
            if constexpr (has_offsets and has_values) {
                reduce_axes_iwise(
                    shape, device, reduced, wrap(output_values.view(), output_offsets.view()),
                    ReduceFirstMin<accessor_t, pair_t, true>{{accessor}},
                    std::forward<Input>(input),
                    std::forward<Values>(output_values),
                    std::forward<Offsets>(output_offsets));

            } else if constexpr (has_offsets) {
                reduce_axes_iwise(
                    shape, device, reduced, output_offsets.view(),
                    ReduceFirstMin<accessor_t, pair_t, false>{{accessor}},
                    std::forward<Input>(input),
                    std::forward<Offsets>(output_offsets));

            } else {
                // Reorder DHW to rightmost if offsets are not computed.
                auto arg_values = output_values.view();
                const auto order_3d = ni::order(input.strides().pop_front(), shape.pop_front());
                if (vany(NotEqual{}, order_3d, Vec{0, 1, 2})) {
                    auto order_4d = (order_3d + 1).push_front(0);
                    shape = shape.reorder(order_4d);
                    accessor.reorder(order_4d);
                    arg_values = arg_values.permute(order_4d);
                }

                reduce_axes_iwise(
                    shape, device, reduced, arg_values,
                    ReduceFirstMin<accessor_t, pair_t, true>{{accessor}},
                    std::forward<Input>(input),
                    std::forward<Values>(output_values));
            }
        }
    }
}

namespace noa {
    struct NormalizeOptions {
        Norm mode = Norm::MEAN_STD;
        i64 ddof = 0;
    };

    /// Normalizes an array, according to a normalization mode.
    /// Can be in-place or out-of-place.
    template<nt::readable_varray_decay Input, nt::writable_varray_decay Output>
    requires (nt::varray_decay_of_complex<Input, Output> or
              nt::varray_decay_of_real<Input, Output>)
    void normalize(
        Input&& input,
        Output&& output,
        const NormalizeOptions& options = {}
    ) {
        switch (options.mode) {
            case Norm::MIN_MAX: {
                const auto [min, max] = min_max(input);
                return ewise(wrap(std::forward<Input>(input), min, max),
                             std::forward<Output>(output), NormalizeMinMax{});
            }
            case Norm::MEAN_STD: {
                const auto [mean, stddev] = mean_stddev(input, options.ddof);
                return ewise(wrap(std::forward<Input>(input), mean, stddev),
                             std::forward<Output>(output), NormalizeMeanStddev{});
            }
            case Norm::L2: {
                const auto norm = l2_norm(input);
                return ewise(wrap(std::forward<Input>(input), norm),
                             std::forward<Output>(output), NormalizeNorm{});
            }
        }
    }

    /// Normalizes each batch of an array, according to a normalization mode.
    /// Can be in-place or out-of-place.
    template<nt::readable_varray_decay Input, nt::writable_varray_decay Output>
    requires (nt::varray_decay_of_complex<Input, Output> or
              nt::varray_decay_of_real<Input, Output>)
    void normalize_per_batch(
        Input&& input,
        Output&& output,
        const NormalizeOptions& options = {}
    ) {
        check(vall(Equal{}, input.shape(), output.shape()),
              "The input and output arrays should have the same shape, but got input={} and output={}",
              input.shape(), output.shape());

        constexpr auto axes_to_reduced = ReduceAxes::all_but(0);
        switch (options.mode) {
            case Norm::MIN_MAX: {
                const auto mins_maxs = min_max(input, axes_to_reduced);
                ewise(wrap(std::forward<Input>(input), mins_maxs.first, mins_maxs.second),
                      std::forward<Output>(output), NormalizeMinMax{});
                break;
            }
            case Norm::MEAN_STD: {
                const auto [means, stddevs] = mean_stddev(input, axes_to_reduced, options.ddof);
                ewise(wrap(std::forward<Input>(input), means, stddevs),
                      std::forward<Output>(output), NormalizeMeanStddev{});
                break;
            }
            case Norm::L2: {
                const auto l2_norms = l2_norm(input, axes_to_reduced);
                ewise(wrap(std::forward<Input>(input), l2_norms),
                      std::forward<Output>(output), NormalizeNorm{});
                break;
            }
        }
    }
}
