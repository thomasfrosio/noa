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
    struct SumOptions {
        /// Reduce (complex) floating-points using double-precision Kahan summation, with the Neumaier variation.
        /// Otherwise, (complex) floating-points are reduced using a double-precision linear or partial sum.
        ///
        /// \note Integers:
        ///     For integers, this parameter has no effect as integer are summed using 64-bits integers (signed or
        ///     unsigned depending on whether the operator needs to support negative values). When computing values
        ///     that can have a decimal part (mean, variance, L2-norm), this integral sum is then static_cast to f64.
        ///
        /// \note Accuracy:
        ///     The Kahan summation is numerically stable and can deal with very large arrays and a very wide range
        ///     of values. However, by default (accurate=false), (complex) floating-points are first static_cast to
        ///     double-precision. In the vast majority of cases, this is enough to guarantee good stability.
        ///     Furthermore, both the GPU and the parallel CPU implementations use a partial sum algorithm,
        ///     making the need for the Kahan sum even less likely.
        ///
        /// \note Runtime performance:
        ///     On the GPU, the Kahan sum has a similar runtime cost than the default double-precision partial sum.
        ///     For maximum performance, single-precision is still the best option. On the CPU, the single- and
        ///     double-precision sums perform identically, but the Kahan summation is much slower.
        bool accurate = false;
    };

    struct VarianceOptions {
        /// Same as SumOptions::accurate.
        bool accurate = false;

        /// Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
        /// In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
        /// of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
        /// of the variance for normally distributed variables.
        i64 ddof = 0;
    };

    struct MedianOptions {
        /// Whether the input array can be overwritten.
        /// If true and if the array is contiguous, the content of the array is left in an undefined state.
        /// Otherwise, the array is unchanged and a temporary buffer is allocated.
        /// This is ignored if a View of const data is passed.
        bool overwrite = false;
    };
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

    /// Returns the median of an array.
    template<nt::readable_varray_of_scalar Input>
    [[nodiscard]] auto median(const Input& array, const MedianOptions& options = {}) {
        check(not array.is_empty(), "Empty array detected");
        const Device device = array.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            stream.synchronize();
            return noa::cpu::median(array.get(), array.strides(), array.shape(), options.overwrite);
        } else {
            #ifdef NOA_ENABLE_CUDA
            return noa::cuda::median(array.get(), array.strides(), array.shape(), options.overwrite, stream.cuda());
            #else
            panic_no_gpu_backend();
            #endif
        }
    }

    /// Returns the sum of an array.
    template<nt::readable_varray Input>
    [[nodiscard]] auto sum(const Input& array, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        value_t output;
        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                reduce_ewise(array, Vec<reduce_t, 2>{}, output, ReduceSumKahan{});
                return output;
            }
        }
        reduce_ewise(array, reduce_t{}, output, ReduceSum{});
        return output;
    }

    /// Returns the mean of an array.
    template<nt::readable_varray Input>
    [[nodiscard]] auto mean(const Input& array, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        using size_t = nt::value_type_t<reduce_t>;
        const auto size = static_cast<size_t>(array.ssize());

        value_t output;
        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                reduce_ewise(array, Vec<reduce_t, 2>{}, output, ReduceMeanKahan{size});
                return output;
            }
        }
        reduce_ewise(array, reduce_t{}, output, ReduceMean{size});
        return output;
    }

    /// Returns the L2-norm of the input array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto l2_norm(const Input& array, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        using reduce_t = std::conditional_t<nt::integer<value_t>, u64, real_t>;
        using output_t = std::conditional_t<nt::integer<value_t>, f64, real_t>;

        output_t output;
        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                reduce_ewise(array, Vec<reduce_t, 2>{}, output, ReduceL2NormKahan{});
                return output;
            };
        }
        reduce_ewise(array, reduce_t{}, output, ReduceL2Norm{});
        return output;
    }

    namespace guts {
        template<bool STDDEV, typename Input>
        [[nodiscard]] auto mean_variance_or_stddev(const Input& array, const VarianceOptions& options) {
            using value_t = nt::mutable_value_type_t<Input>;
            using real_t = nt::value_type_t<value_t>;
            using sum_t = std::conditional_t<nt::integer<value_t>, i64, nt::double_precision_t<value_t>>;
            using sum_sqd_t = nt::value_type_t<sum_t>;

            using output_mean_t = std::conditional_t<nt::integer<value_t>, f64, value_t>;
            using output_variance_t = std::conditional_t<nt::integer<value_t>, f64, real_t>;
            const auto size = static_cast<f64>(array.ssize() - options.ddof);

            output_mean_t mean;
            output_variance_t variance;
            if constexpr (nt::real_or_complex<value_t>) {
                if (options.accurate) {
                    using sum_error_t = Vec<sum_t, 2>;
                    using sum_sqd_error_t = Vec<sum_sqd_t, 2>;
                    reduce_ewise(
                        array, wrap(sum_error_t{}, sum_sqd_error_t{}),
                        wrap(mean, variance), ReduceMeanVarianceKahan<f64, STDDEV>{size}
                    );
                    return Pair{mean, variance};
                }
            }
            reduce_ewise(array, wrap(sum_t{}, sum_sqd_t{}), wrap(mean, variance), ReduceMeanVariance<f64, STDDEV>{size});
            return Pair{mean, variance};
        }
    }

    /// Returns the mean and variance of an array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto mean_variance(const Input& array, const VarianceOptions& options = {}) {
        return guts::mean_variance_or_stddev<false>(array, options);
    }

    /// Returns the mean and standard deviation of an array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto mean_stddev(const Input& array, const VarianceOptions& options = {}) {
        return guts::mean_variance_or_stddev<true>(array, options);
    }

    /// Returns the variance of an array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto variance(const Input& array, const VarianceOptions& options = {}) {
        const auto& [mean, variance] = mean_variance(array, options);
        return variance;
    }

    /// Returns the standard-deviation of an array.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto stddev(const Input& array, const VarianceOptions& options = {}) {
        const auto& [mean, stddev] = mean_stddev(array, options);
        return stddev;
    }

    /// Returns the root-mean-square deviation.
    template<nt::readable_varray_of_real Lhs,
             nt::readable_varray_of_real Rhs>
    [[nodiscard]] auto rmsd(const Lhs& lhs, const Rhs& rhs) {
        using lhs_value_t = nt::mutable_value_type_t<Lhs>;
        using rhs_value_t = nt::mutable_value_type_t<Rhs>;
        using real_t = nt::largest_type_t<lhs_value_t, rhs_value_t>;
        real_t output;
        reduce_ewise(wrap(lhs, rhs), f64{}, output, ReduceRMSD{static_cast<f64>(lhs.ssize())});
        return output;
    }

    /// Returns {maximum, offset: i64} of an array.
    /// \note If the maximum value appears more than once, this function makes no guarantee to which one is selected.
    /// \note To get the corresponding 4d indices, noa::indexing::offset2index(offset, input) can be used.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto argmax(const Input& input) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduced_t = Pair<value_t, i64>;
        using op_t = ReduceFirstMax<AccessorI64<const value_t, 4>, reduced_t>;
        reduced_t reduced{std::numeric_limits<value_t>::lowest(), i64{}};
        reduced_t output{};
        reduce_iwise(
            input.shape(), input.device(), reduced,
            wrap(output.first, output.second),
            op_t{{ng::to_accessor(input)}}
        );
        return output;
    }

    /// Returns {minimum, offset: i64} of an array.
    /// \note If the minimum value appears more than once, this function makes no guarantee to which one is selected.
    /// \note To get the corresponding 4d indices, noa::indexing::offset2index(offset, input) can be used.
    template<nt::readable_varray_of_numeric Input>
    [[nodiscard]] auto argmin(const Input& input) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduced_t = Pair<value_t, i64>;
        using op_t = ReduceFirstMin<AccessorI64<const value_t, 4>, reduced_t>;
        reduced_t reduced{std::numeric_limits<value_t>::max(), i64{}};
        reduced_t output{};
        reduce_iwise(
            input.shape(), input.device(), reduced,
            wrap(output.first, output.second),
            op_t{{ng::to_accessor(input)}}
        );
        return output;
    }
}

namespace noa {
    /// Reduces an array along some dimensions by taking the minimum value.
    //// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_numeric Output>
    void min(Input&& input, Output&& output) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::max();
        reduce_axes_ewise(std::forward<Input>(input), init, std::forward<Output>(output), ReduceMin{});
    }

    /// Reduces an array along some dimension by taking the minimum value.
    /// //// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto min(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t>(guts::axes_to_output_shape(input, axes), input.options());
        min(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    //// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_numeric Output>
    void max(Input&& input, Output&& output) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::lowest();
        reduce_axes_ewise(std::forward<Input>(input), init, std::forward<Output>(output), ReduceMax{});
    }

    /// Reduces an array along some dimension by taking the maximum value.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto max(Input&& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t>(guts::axes_to_output_shape(input, axes), input.options());
        max(std::forward<Input>(input), output);
        return output;
    }

    /// Reduces an array along some dimension(s) by taking the minimum and maximum values.
    /// \see reduce_axes_ewise for more details about the reduction.
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
            ReduceMinMax{}
        );
    }

    /// Reduces an array along some dimension(s) by taking the minimum and maximum values.
    /// \see reduce_axes_ewise for more details about the reduction.
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
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    void sum(Input&& input, Output&& output, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                return reduce_axes_ewise(
                    std::forward<Input>(input), Vec<reduce_t, 2>{},
                    std::forward<Output>(output), ReduceSumKahan{}
                );
            }
        }
        reduce_axes_ewise(
            std::forward<Input>(input), reduce_t{},
            std::forward<Output>(output), ReduceSum{}
        );
    }

    /// Reduces an array along some dimensions by taking the sum.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay Input>
    [[nodiscard]] auto sum(Input&& input, ReduceAxes axes, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t>(guts::axes_to_output_shape(input, axes), input.options());
        sum(std::forward<Input>(input), output, options);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Output>
    void mean(Input&& input, Output&& output, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using reduce_t = std::conditional_t<nt::real<value_t>, f64,
                         std::conditional_t<nt::complex<value_t>, c64,
                         value_t>>;

        using real_t = nt::value_type_t<reduce_t>;
        const auto size = static_cast<real_t>(guts::n_elements_to_reduce(input.shape(), output.shape()));
        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                return reduce_axes_ewise(
                    std::forward<Input>(input), Vec<reduce_t, 2>{},
                    std::forward<Output>(output), ReduceMeanKahan{size}
                );
            }
        }
        reduce_axes_ewise(
            std::forward<Input>(input), reduce_t{},
            std::forward<Output>(output), ReduceMean{size}
        );
    }

    /// Reduces an array along some dimensions by taking the average.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay Input>
    [[nodiscard]] auto mean(Input&& input, ReduceAxes axes, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t>(guts::axes_to_output_shape(input, axes), input.options());
        mean(std::forward<Input>(input), output, options);
        return output;
    }

    /// Reduces an array along some dimensions by taking the L2-norm.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_numeric Output>
    void l2_norm(Input&& input, Output&& output, const SumOptions& options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        using reduce_t = std::conditional_t<nt::integer<value_t>, u64, real_t>;

        if constexpr (nt::real_or_complex<value_t>) {
            if (options.accurate) {
                return reduce_axes_ewise(
                    std::forward<Input>(input), Vec<reduce_t, 2>{},
                    std::forward<Output>(output), ReduceL2NormKahan{}
                );
            }
        }
        reduce_axes_ewise(
            std::forward<Input>(input), reduce_t{},
            std::forward<Output>(output), ReduceL2Norm{}
        );
    }

    /// Reduces an array along some dimensions by taking the norm.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::writable_varray_decay_of_numeric Input>
    [[nodiscard]] auto l2_norm(Input&& input, ReduceAxes axes, const SumOptions& options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        auto output = Array<real_t>(guts::axes_to_output_shape(input, axes), input.options());
        l2_norm(std::forward<Input>(input), output, options);
        return output;
    }

    namespace guts {
        template<bool STDDEV, typename Input, typename Mean, typename Variance>
        void mean_variance_or_stddev(Input&& input, Mean&& means, Variance&& variances, const VarianceOptions options) {
            constexpr bool HAS_MEAN = not nt::empty<std::remove_reference_t<Mean>>;
            if constexpr (HAS_MEAN) {
                check(vall(Equal{}, means.shape(), variances.shape()),
                      "The means and variances should have the same shape, but got means={} and variances={}",
                      means.shape(), variances.shape());
            }

            using value_t = nt::mutable_value_type_t<Input>;
            using sum_t = std::conditional_t<nt::integer<value_t>, i64, nt::double_precision_t<value_t>>;
            using sum_sqd_t = nt::value_type_t<sum_t>;
            const auto size = static_cast<f64>(guts::n_elements_to_reduce(input.shape(), variances.shape()) - options.ddof);

            if constexpr (nt::real_or_complex<value_t>) {
                if (options.accurate) {
                    using sum_error_t = Vec<sum_t, 2>;
                    using sum_sqd_error_t = Vec<sum_sqd_t, 2>;
                    if constexpr (HAS_MEAN) {
                        reduce_axes_ewise(
                            std::forward<Input>(input),
                            wrap(sum_error_t{}, sum_sqd_error_t{}),
                            wrap(std::forward<Mean>(means), std::forward<Variance>(variances)),
                            ReduceMeanVarianceKahan<f64, STDDEV>{size}
                        );
                    } else {
                        reduce_axes_ewise(
                            std::forward<Input>(input),
                            wrap(sum_error_t{}, sum_sqd_error_t{}),
                            std::forward<Variance>(variances),
                            ReduceMeanVarianceKahan<f64, STDDEV>{size}
                        );
                    }
                }
            }
            if constexpr (HAS_MEAN) {
                reduce_axes_ewise(
                    std::forward<Input>(input),
                    wrap(sum_t{}, sum_sqd_t{}),
                    wrap(std::forward<Mean>(means), std::forward<Variance>(variances)),
                    ReduceMeanVariance<f64, STDDEV>{size}
                );
            } else {
                reduce_axes_ewise(
                    std::forward<Input>(input),
                    wrap(sum_t{}, sum_sqd_t{}),
                    std::forward<Variance>(variances),
                    ReduceMeanVariance<f64, STDDEV>{size}
                );
            }
        }
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Mean,
             nt::writable_varray_decay Variance>
    requires ((nt::varray_decay_of_complex<Input, Mean> and nt::varray_decay_of_real<Variance>) or
               nt::varray_decay_of_scalar<Input, Mean, Variance>)
    void mean_variance(Input&& input, Mean&& means, Variance&& variances, const VarianceOptions options = {}) {
        guts::mean_variance_or_stddev<false>(
            std::forward<Input>(input),
            std::forward<Mean>(means),
            std::forward<Variance>(variances),
            options
        );
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto mean_variance(Input&& input, ReduceAxes axes, const VarianceOptions options = {}) {
        using mean_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::mutable_value_type_twice_t<Input>;
        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
            Array<mean_t>(output_shape, input.options()),
            Array<variance_t>(output_shape, input.options()),
        };
        mean_variance(std::forward<Input>(input), output.first, output.second, options);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean and standard deviation.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Mean,
             nt::writable_varray_decay Stddev>
    requires ((nt::varray_decay_of_complex<Input, Mean> and nt::varray_decay_of_real<Stddev>) or
               nt::varray_decay_of_scalar<Input, Mean, Stddev>)
    void mean_stddev(Input&& input, Mean&& means, Stddev&& stddevs, const VarianceOptions options = {}) {
        guts::mean_variance_or_stddev<true>(
           std::forward<Input>(input),
           std::forward<Mean>(means),
           std::forward<Stddev>(stddevs),
           options
       );
    }

    /// Reduces an array along some dimensions by taking the mean and standard deviation.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto mean_stddev(Input&& input, ReduceAxes axes, const VarianceOptions options = {}) {
        using mean_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::value_type_t<mean_t>;
        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
            Array<mean_t>(output_shape, input.options()),
            Array<variance_t>(output_shape, input.options()),
        };
        mean_stddev(std::forward<Input>(input), output.first, output.second, options);
        return output;
    }

    /// Reduces an array along some dimensions by taking the variance.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_real Output>
    void variance(Input&& input, Output&& output, const VarianceOptions options = {}) {
        guts::mean_variance_or_stddev<false>(
            std::forward<Input>(input),
            Empty{},
            std::forward<Output>(output),
            options
        );
    }

    /// Reduces an array along some dimensions by taking the variance.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto variance(Input&& input, ReduceAxes axes, const VarianceOptions options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::value_type_t<value_t>;
        auto variances = Array<variance_t>(guts::axes_to_output_shape(input, axes), input.options());
        variance(std::forward<Input>(input), variances, options);
        return variances;
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    ///\see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input,
             nt::writable_varray_decay_of_real Output>
    void stddev(Input&& input, Output&& output, const VarianceOptions options = {}) {
        guts::mean_variance_or_stddev<true>(
            std::forward<Input>(input),
            Empty{},
            std::forward<Output>(output),
            options
        );
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    /// \see reduce_axes_ewise for more details about the reduction.
    template<nt::readable_varray_decay_of_numeric Input>
    [[nodiscard]] auto stddev(Input&& input, ReduceAxes axes, const VarianceOptions options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        using stddev_t = nt::value_type_t<value_t>;
        auto stddevs = Array<stddev_t>(guts::axes_to_output_shape(input, axes), input.options());
        stddev(std::forward<Input>(input), stddevs, options);
        return stddevs;
    }

    /// Reduces an array along some dimensions by taking the maximum value along the reduced axis/axes.
    /// \see reduce_axes_ewise for more details about the reduction.
    ///
    /// \tparam ReducedValue        Value type used for the reduction. Any numeric or void.
    ///                             If void, it defaults to the input value type.
    ///                             The input is explicitly converted to this type.
    ///                             If real and the input is complex, the abs_squared of the input is first computed.
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
    /// \see reduce_axes_ewise for more details about the reduction.
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

        /// Same as SumOptions::accurate.
        ///  Only used if mode == Norm::MEAN_STD or Norm::L2.
        bool accurate = false;

        /// Same as VarianceOptions::ddof.
        /// Only used if mode == Norm::MEAN_STD.
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
                const auto [mean, stddev] = mean_stddev(input, {.accurate = options.accurate, .ddof = options.ddof});
                return ewise(wrap(std::forward<Input>(input), mean, stddev),
                             std::forward<Output>(output), NormalizeMeanStddev{});
            }
            case Norm::L2: {
                const auto norm = l2_norm(input, {.accurate = options.accurate});
                return ewise(wrap(std::forward<Input>(input), norm),
                             std::forward<Output>(output), NormalizeNorm{});
            }
        }
    }

    /// Normalizes an array, according to a normalization mode.
    /// The normalization bounds are computed by reducing the provided axes.
    /// Can be in-place or out-of-place.
    template<nt::readable_varray_decay Input, nt::writable_varray_decay Output>
    requires (nt::varray_decay_of_complex<Input, Output> or
              nt::varray_decay_of_real<Input, Output>)
    void normalize(
        Input&& input,
        Output&& output,
        ReduceAxes axes,
        const NormalizeOptions& options = {}
    ) {
        check(vall(Equal{}, input.shape(), output.shape()),
              "The input and output arrays should have the same shape, but got input={} and output={}",
              input.shape(), output.shape());

        switch (options.mode) {
            case Norm::MIN_MAX: {
                const auto mins_maxs = min_max(input, axes);
                ewise(wrap(std::forward<Input>(input), mins_maxs.first, mins_maxs.second),
                      std::forward<Output>(output), NormalizeMinMax{});
                break;
            }
            case Norm::MEAN_STD: {
                const auto [means, stddevs] = mean_stddev(input, axes, {.accurate = options.accurate, .ddof = options.ddof});
                ewise(wrap(std::forward<Input>(input), means, stddevs),
                      std::forward<Output>(output), NormalizeMeanStddev{});
                break;
            }
            case Norm::L2: {
                const auto l2_norms = l2_norm(input, axes, {.accurate = options.accurate});
                ewise(wrap(std::forward<Input>(input), l2_norms),
                      std::forward<Output>(output), NormalizeNorm{});
                break;
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
        normalize(std::forward<Input>(input), std::forward<Output>(output), ReduceAxes::all_but(0), options);
    }
}
