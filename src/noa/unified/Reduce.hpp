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
//      to compute the variances. This can be fused into a single backend call.
//      Or add a core interface to compute "nested" reductions??

namespace noa {
    struct ReduceAxes {
        bool batch{};
        bool depth{};
        bool height{};
        bool width{};

        static constexpr ReduceAxes from_shape(const Shape4<i64>& output_shape) {
            ReduceAxes axes{};
            if (output_shape[0] == 1)
                axes.batch = true;
            if (output_shape[1] == 1)
                axes.depth = true;
            if (output_shape[2] == 1)
                axes.height = true;
            if (output_shape[3] == 1)
                axes.width = true;
            return axes;
        }
    };
}

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
        for (size_t i = 0; i < 4; ++i) {
            if (input_shape[i] > 1 and output_shape[i] == 1)
                n_elements_to_reduce *= input_shape[i];
        }
        return n_elements_to_reduce;
    }
}

namespace noa {
    /// Returns the minimum value of the input array.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto min(const Input& array) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::max();
        value_type output;
        reduce_ewise(array, init, output, ReduceMin{});
        return output;
    }

    /// Returns the maximum value of the input array.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto max(const Input& array) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::lowest();
        value_type output;
        reduce_ewise(array, init, output, ReduceMax{});
        return output;
    }

    /// Returns a pair with the minimum and maximum value of the input array.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto min_max(const Input& array) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init_min = std::numeric_limits<value_type>::max();
        auto init_max = std::numeric_limits<value_type>::lowest();
        value_type output_min, output_max;
        reduce_ewise(array, wrap(init_min, init_max), wrap(output_min, output_max), ReduceMinMax{});
        return Pair{output_min, output_max};
    }

    /// Returns the median of the input array.
    /// \param[in,out] array    Input array.
    /// \param overwrite        Whether the function is allowed to overwrite \p array.
    ///                         If true and if the array is contiguous, the content of \p array is left
    ///                         in an undefined state. Otherwise, array is unchanged and a temporary
    ///                         buffer is allocated.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
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
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Returns the sum of the input array.
    /// \note For (complex)-floating-point types, the CPU backend uses a Kahan summation (with Neumaier variation).
    template<typename Input> requires nt::is_varray_v<Input>
    [[nodiscard]] auto sum(const Input& array) {
        using value_t = nt::mutable_value_type_t<Input>;
        value_t output;

        if constexpr (nt::is_real_or_complex_v<value_t>) {
            if (array.device().is_cpu()) {
                using op_t = ReduceAccurateSum<value_t>;
                using pair_t = op_t::pair_type;
                reduce_ewise(array, pair_t{}, output, op_t{});
                return output;
            }
        }
        reduce_ewise(array, value_t{}, output, ReduceSum{});
        return output;
    }

    /// Returns the mean of the input array.
    /// \note For (complex)-floating-point types, the CPU backend uses a Kahan summation (with Neumaier variation).
    template<typename Input> requires nt::is_varray_v<Input>
    [[nodiscard]] auto mean(const Input& array) {
        using value_t = nt::mutable_value_type_t<Input>;
        value_t output;

        if constexpr (nt::is_real_or_complex_v<value_t>) {
            if (array.device().is_cpu()) {
                using op_t = ReduceAccurateMean<value_t>;
                using pair_t = op_t::pair_type;
                using mean_t = op_t::mean_type;
                reduce_ewise(array, pair_t{}, output, op_t{.size=static_cast<mean_t>(array.ssize())});
                return output;
            }
        }
        using mean_t = nt::value_type_t<value_t>;
        reduce_ewise(array, value_t{}, output, ReduceMean{.size=static_cast<mean_t>(array.ssize())});
        return output;
    }

    /// Returns the L2-norm of the input array.
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto l2_norm(const Input& array) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        real_t output;

        if (array.device().is_cpu()) {
            reduce_ewise(array, Pair<f64, f64>{}, output, ReduceAccurateL2Norm{});
            return output;
        }
        reduce_ewise(array, real_t{}, output, ReduceL2Norm{});
        return output;
    }

    /// Returns the mean and variance of the input array.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto mean_variance(const Input& array, i64 ddof = 0) {
        auto mean = noa::mean(array);
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        real_t variance;

        if (array.device().is_cpu()) {
            using double_t = std::conditional_t<nt::is_real_v<value_t>, f64, c64>;
            auto mean_double = static_cast<double_t>(mean);
            auto size = static_cast<f64>(array.ssize() - ddof);
            reduce_ewise(wrap(array, mean_double), f64{}, variance, ReduceVariance{size});
            return Pair{mean, variance};
        }
        auto size = static_cast<real_t>(array.ssize() - ddof);
        reduce_ewise(wrap(array, mean), real_t{}, variance, ReduceVariance{size});
        return Pair{mean, variance};
    }

    /// Returns the mean and standard deviation of the input array.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto mean_stddev(const Input& array, i64 ddof = 0) {
        auto pair = mean_variance(array, ddof);
        pair.second = sqrt(pair.second);
        return pair;
    }

    /// Returns the variance of the input array.
    /// \param ddof         Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                     In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                     of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                     of the variance for normally distributed variables.
    /// \param[in] array    Array to reduce.
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
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
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto stddev(const Input& array, i64 ddof = 0) {
        return sqrt(variance(array, ddof));
    }

    /// Returns the root-mean-square deviation.
    template<typename Lhs, typename Rhs> requires nt::are_varray_of_real_v<Lhs, Rhs>
    [[nodiscard]] auto rmsd(const Lhs& lhs, const Rhs& rhs) {
        using lhs_value_t = nt::mutable_value_type_t<Lhs>;
        using rhs_value_t = nt::mutable_value_type_t<Rhs>;
        using real_t = std::conditional_t<std::is_same_v<f64, lhs_value_t>, lhs_value_t, rhs_value_t>;
        real_t output;

        if (lhs.device().is_cpu())
            reduce_ewise(wrap(lhs, rhs), f64{}, output, ReduceRMSD{static_cast<f64>(lhs.ssize())});
        else
            reduce_ewise(wrap(lhs, rhs), real_t{}, output, ReduceRMSD{static_cast<real_t>(lhs.ssize())});
        return output;
    }

    /// Returns {offset, maximum} of the input array.
    /// \note If the maximum value appears more than once, this function makes no guarantee to which one is selected.
    /// \note To get the corresponding 4d indices, noa::indexing::offset2index(offset, input) can be used.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto argmax(const Input& input) {
        using value_t = nt::mutable_value_type_t<Input>;
        using pair_t = Pair<value_t, i64>;
        using op_t = ReduceFirstMax<AccessorI64<const value_t, 4>, i64, false>;
        pair_t reduced{std::numeric_limits<value_t>::lowest(), i64{}};
        value_t output_value{};
        i64 output_offset{};
        reduce_iwise(input.shape(), input.device(), reduced,
                     wrap(output_value, output_offset),
                     op_t{{input.accessor()}});
        return Pair{output_value, output_offset};
    }

    /// Returns {offset, minimum} of the input array.
    /// \note If the minimum value appears more than once, this function makes no guarantee to which one is selected.
    /// \note To get the corresponding 4d indices, noa::indexing::offset2index(offset, input) can be used.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto argmin(const Input& input) {
        using value_t = nt::mutable_value_type_t<Input>;
        using pair_t = Pair<value_t, i64>;
        using op_t = ReduceFirstMin<AccessorI64<const value_t, 4>, i64, false>;
        pair_t reduced{std::numeric_limits<value_t>::max(), i64{}};
        value_t output_value{};
        i64 output_offset{};
        reduce_iwise(input.shape(), input.device(), reduced,
                     wrap(output_value, output_offset),
                     op_t{{input.accessor()}});
        return Pair{output_value, output_offset};
    }
}

namespace noa {
    /// Reduces an array along some dimensions by taking the minimum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of minimum values.
    template<typename Input, typename Output> requires nt::are_varray_of_scalar_v<Input, Output>
    void min(const Input& input, const Output& output) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::max();
        reduce_axes_ewise(input, init, output, ReduceMin{});
    }

    /// Reduces an array along some dimension by taking the minimum value.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto min(const Input& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        min(input, output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the maximum value.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input    Input array to reduce.
    /// \param[out] output  Reduced array of maximum values.
    template<typename Input, typename Output> requires nt::are_varray_of_scalar_v<Input, Output>
    void max(const Input& input, const Output& output) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init = std::numeric_limits<value_type>::lowest();
        reduce_axes_ewise(input, init, output, ReduceMax{});
    }

    /// Reduces an array along some dimension by taking the maximum value.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto max(const Input& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        max(input, output);
        return output;
    }

    /// Reduces an array along some dimension(s) by taking the minimum and maximum values.
    template<typename Input, typename Min, typename Max> requires nt::are_varray_of_scalar_v<Input, Min, Max>
    void min_max(const Input& array, const Min& min, const Max& max) {
        using value_type = nt::mutable_value_type_t<Input>;
        auto init_min = std::numeric_limits<value_type>::max();
        auto init_max = std::numeric_limits<value_type>::lowest();
        reduce_axes_ewise(array, wrap(init_min, init_max), wrap(min, max), ReduceMinMax{});
    }

    /// Reduces an array along some dimension(s) by taking the minimum and maximum values.
    template<typename Input> requires nt::is_varray_of_scalar_v<Input>
    [[nodiscard]] auto min_max(const Input& array, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output_shape = guts::axes_to_output_shape(array, axes);
        Pair output{
                Array<value_t>(output_shape, array.options()),
                Array<value_t>(output_shape, array.options()),
        };
        min_max(array, output.first, output.second);
        return output;
    }

    // TODO Add median()

    /// Reduces an array along some dimensions by taking the sum.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced sums.
    template<typename Input, typename Output>
    requires (nt::are_varray_v<Input, Output> and nt::is_varray_of_mutable_v<Output>)
    void sum(const Input& input, const Output& output) {
        using input_value_t = nt::mutable_value_type_t<Input>;

        if constexpr (nt::is_real_or_complex_v<input_value_t>) {
            if (input.device().is_cpu()) {
                using op_t = ReduceAccurateSum<input_value_t>;
                using pair_t = op_t::pair_type;
                return reduce_axes_ewise(input, pair_t{}, output, op_t{});
            }
        }
        reduce_axes_ewise(input, input_value_t{}, output, ReduceSum{});
    }

    /// Reduces an array along some dimensions by taking the sum.
    template<typename Input> requires nt::is_varray_v<Input>
    [[nodiscard]] auto sum(const Input& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        sum(input, output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the mean.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced means.
    template<typename Input, typename Output>
    requires (nt::are_varray_v<Input, Output> and nt::is_varray_of_mutable_v<Output>)
    void mean(const Input& input, const Output& output) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using scalar_t = nt::value_type_t<input_value_t>;

        const auto n_elements_to_reduce = guts::n_elements_to_reduce(input.shape(), output.shape());
        if constexpr (nt::is_real_or_complex_v<input_value_t>) {
            if (input.device().is_cpu()) {
                using op_t = ReduceAccurateMean<input_value_t>;
                using pair_t = op_t::pair_type;
                using mean_t = op_t::mean_type;
                auto size = static_cast<mean_t>(n_elements_to_reduce);
                return reduce_axes_ewise(input, pair_t{}, output, op_t{.size=size});
            }
        }
        auto mean = static_cast<scalar_t>(n_elements_to_reduce);
        reduce_axes_ewise(input, input_value_t{}, output, ReduceMean{.size=mean});
    }

    /// Reduces an array along some dimensions by taking the average.
    template<typename Input> requires nt::is_varray_v<Input>
    [[nodiscard]] auto mean(const Input& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        Array<value_t> output(guts::axes_to_output_shape(input, axes), input.options());
        mean(input, output);
        return output;
    }

    /// Reduces an array along some dimensions by taking the L2-norm.
    /// \details Dimensions of the output array should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input        Input array to reduce.
    /// \param[out] output      Reduced norms.
    template<typename Input, typename Output>
    requires (nt::is_varray_of_real_or_complex_v<Input> and
              nt::is_varray_of_real_v<Output> and
              nt::is_varray_of_mutable_v<Output>)
    void l2_norm(const Input& input, const Output& output) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;

        if constexpr (nt::is_real_or_complex_v<value_t>) {
            if (input.device().is_cpu())
                return reduce_axes_ewise(input, Pair<f64, f64>{}, output, ReduceAccurateL2Norm{});
        }
        reduce_axes_ewise(input, real_t{}, output, ReduceL2Norm{});
    }

    /// Reduces an array along some dimensions by taking the norm.
    template<typename Input> requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto l2_norm(const Input& input, ReduceAxes axes) {
        using value_t = nt::mutable_value_type_t<Input>;
        using real_t = nt::value_type_t<value_t>;
        Array<real_t> output(guts::axes_to_output_shape(input, axes), input.options());
        l2_norm(input, output);
        return output;
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
    template<typename Input, typename Mean, typename Variance>
    requires (nt::are_varray_of_real_or_complex_v<Input, Mean> and
              nt::is_varray_of_real_v<Variance> and
              nt::are_varray_of_mutable_v<Mean, Variance>)
    void mean_variance(const Input& input, const Mean& means, const Variance& variances, i64 ddof = 0) {
        check(all(means.shape() == variances.shape()),
              "The means and variances should have the same shape, but got means={} and variances={}",
              means.shape(), variances.shape());

        mean(input, means);
        auto n_reduced = guts::n_elements_to_reduce(input.shape(), means.shape()) - ddof;

        if (input.device().is_cpu()) {
            auto op = ReduceVariance{static_cast<f64>(n_reduced)};
            reduce_axes_ewise(wrap(input, ni::broadcast(means, input.shape())), f64{}, variances, op);
        } else {
            using real_t = nt::value_type_t<nt::mutable_value_type_t<Input>>;
            auto op = ReduceVariance{static_cast<real_t>(n_reduced)};
            reduce_axes_ewise(wrap(input, ni::broadcast(means, input.shape())), real_t{}, variances, op);
        }
    }

    /// Reduces an array along some dimensions by taking the mean and variance.
    template<typename Input>
    requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto mean_variance(const Input& input, ReduceAxes axes, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::mutable_value_type_twice_t<Input>;

        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
                Array<mean_t>(output_shape, input.options()),
                Array<variance_t>(output_shape, input.options()),
        };
        mean_variance(input, output.first, output.second, ddof);
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
    /// \param[out] mean        Reduced means.
    /// \param[out] stddev      Reduced standard deviations.
    /// \param ddof             Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, ddof=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    template<typename Input, typename Mean, typename Stddev>
    requires (nt::are_varray_of_real_or_complex_v<Input, Mean> and
              nt::is_varray_of_real_v<Stddev> and
              nt::are_varray_of_mutable_v<Mean, Stddev>)
    void mean_stddev(const Input& input, const Mean& means, const Stddev& stddevs, i64 ddof = 0) {
        check(all(means.shape() == stddevs.shape()),
              "The means and stddevs should have the same shape, but got means={} and stddevs={}",
              means.shape(), stddevs.shape());

        mean(input, means);
        auto n_reduced = guts::n_elements_to_reduce(input.shape(), means.shape()) - ddof;

        if (input.device().is_cpu()) {
            auto op = ReduceStddev<f64>{static_cast<f64>(n_reduced)};
            reduce_axes_ewise(wrap(input, ni::broadcast(means, input.shape())), f64{}, stddevs, op);
        } else {
            using real_t = nt::mutable_value_type_twice_t<Input>;
            auto op = ReduceStddev<real_t>{static_cast<real_t>(n_reduced)};
            reduce_axes_ewise(wrap(input, ni::broadcast(means, input.shape())), real_t{}, stddevs, op);
        }
    }

    /// Reduces an array along some dimensions by taking the mean and standard deviation.
    template<typename Input>
    requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto mean_stddev(const Input& input, ReduceAxes axes, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::value_type_t<mean_t>;

        auto output_shape = guts::axes_to_output_shape(input, axes);
        Pair output{
                Array<mean_t>(output_shape, input.options()),
                Array<variance_t>(output_shape, input.options()),
        };
        mean_stddev(input, output.first, output.second, ddof);
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
    template<typename Input, typename Output>
    requires (nt::is_varray_of_real_or_complex_v<Input> and
              nt::is_varray_of_real_v<Output> and
              nt::are_varray_of_mutable_v<Output>)
    void variance(const Input& input, const Output& output, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        Array<mean_t> means(output.shape(), output.options());
        mean_variance(input, means, output, ddof);
    }

    /// Reduces an array along some dimensions by taking the variance.
    template<typename Input>
    requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto variance(const Input& input, ReduceAxes axes, i64 ddof = 0) {
        using value_t = nt::mutable_value_type_t<Input>;
        using variance_t = nt::value_type_t<value_t>;
        Array<variance_t> variances(guts::axes_to_output_shape(input, axes), input.options());
        variance(input, variances, ddof);
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
    template<typename Input, typename Output>
    requires (nt::is_varray_of_real_or_complex_v<Input> and
              nt::is_varray_of_real_v<Output> and
              nt::are_varray_of_mutable_v<Output>)
    void stddev(const Input& input, const Output& output, i64 ddof = 0) {
        using mean_t = nt::mutable_value_type_t<Input>;
        Array<mean_t> means(output.shape(), output.options());
        mean_stddev(input, means, output, ddof);
    }

    /// Reduces an array along some dimensions by taking the standard-deviation.
    template<typename Input>
    requires nt::is_varray_of_real_or_complex_v<Input>
    [[nodiscard]] auto stddev(const Input& input, ReduceAxes axes, i64 ddof = 0) {
        using value_t = nt::mutable_value_type_t<Input>;
        using stddev_t = nt::value_type_t<value_t>;
        Array<stddev_t> stddevs(guts::axes_to_output_shape(input, axes), input.options());
        stddev(input, stddevs, ddof);
        return stddevs;
    }

    /// Reduces an array along some dimensions by taking the maximum value along the reduce axis/axes.
    /// \details Dimensions of the output arrays should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input            Array to reduce.
    /// \param[out] output_values   Array where to save the maximum values, or empty.
    /// \param[out] output_offsets  Array where to save the offsets of the maximum values, or empty.
    /// \note If the maximum value appears more than once, this function makes no guarantee to which one is selected.
    template<typename Input,
             typename Values = View<nt::mutable_value_type_t<Input>>,
             typename Offsets = View<i64>>
    requires (nt::are_varray_of_scalar_v<Input, Values, Offsets> and
              nt::are_varray_of_mutable_v<Values, Offsets>)
    void argmax(
            const Input& input,
            const Values& output_values,
            const Offsets& output_offsets
    ) {
        const bool has_offsets = not output_offsets.is_empty();
        const bool has_values = not output_values.is_empty();
        if (not has_offsets and not has_values)
            return;

        check(not input.is_empty(), "Empty array detected");
        using input_value_t = nt::mutable_value_type_t<Input>;
        using offset_t = nt::value_type_t<Offsets>;
        using pair_t = Pair<input_value_t, offset_t>;
        using op_t = ReduceFirstMax<AccessorI64<const input_value_t, 4>, offset_t, false>;

        // Reorder DHW to rightmost if offsets are not computed.
        auto shape = input.shape();
        auto accessor = input.accessor();
        auto arg_values = output_values.view();
        auto arg_offsets = output_offsets.view();
        if (not has_offsets) {
            const auto order_3d = ni::order(input.strides().pop_front(), shape.pop_front());
            if (any(order_3d != Vec3<i64>{0, 1, 2})) {
                auto order_4d = (order_3d + 1).push_front(0);
                shape = shape.reorder(order_4d);
                accessor.reorder(order_4d);
                arg_values = arg_values.reorder(order_4d);
            }
        }
        auto op = op_t{{accessor}};
        auto reduced = pair_t{std::numeric_limits<input_value_t>::lowest(), offset_t{}};

        auto device = input.device();
        if (has_offsets and has_values) {
            reduce_axes_iwise(
                    shape, device, reduced, wrap(arg_values, arg_offsets), op,
                    input, output_values, output_offsets);
        } else if (has_offsets) {
            reduce_axes_iwise(shape, device, reduced, arg_offsets, op, input, output_offsets);
        } else {
            reduce_axes_iwise(shape, device, reduced, arg_values, op, input, output_values);
        }
    }

    /// Reduces an array along some dimensions by taking the minimum value along the reduce axis/axes.
    /// \details Dimensions of the output arrays should match the input shape, or be 1, indicating the dimension
    ///          should be reduced. Reducing more than one axis at a time is only supported if the reduction
    ///          results to having one value or one value per batch, i.e. the DHW dimensions are empty after reduction.
    /// \param[in] input            Array to reduce.
    /// \param[out] output_values   Array where to save the minimum values, or empty.
    /// \param[out] output_offsets  Array where to save the offsets of the minimum values, or empty.
    /// \note If the minimum value appears more than once, this function makes no guarantee to which one is selected.
    template<typename Input,
             typename Values = View<nt::mutable_value_type_t<Input>>,
             typename Offsets = View<i64>>
    requires (nt::are_varray_of_scalar_v<Input, Values, Offsets> and
              nt::are_varray_of_mutable_v<Values, Offsets>)
    void argmin(
            const Input& input,
            const Values& output_values,
            const Offsets& output_offsets = {}
    ) {
        const bool has_offsets = not output_offsets.is_empty();
        const bool has_values = not output_values.is_empty();
        if (not has_offsets and not has_values)
            return;

        check(not input.is_empty(), "Empty array detected");
        using input_value_t = nt::mutable_value_type_t<Input>;
        using offset_t = nt::value_type_t<Offsets>;
        using pair_t = Pair<input_value_t, offset_t>;
        using op_t = ReduceFirstMin<AccessorI64<const input_value_t, 4>, offset_t, false>;

        // Reorder DHW to rightmost if offsets are not computed.
        auto shape = input.shape();
        auto accessor = input.accessor();
        auto arg_values = output_values.view();
        auto arg_offsets = output_offsets.view();
        if (not has_offsets) {
            const auto order_3d = ni::order(input.strides().pop_front(), shape.pop_front());
            if (any(order_3d != Vec3<i64>{0, 1, 2})) {
                auto order_4d = (order_3d + 1).push_front(0);
                shape = shape.reorder(order_4d);
                accessor.reorder(order_4d);
                arg_values = arg_values.reorder(order_4d);
            }
        }
        auto op = op_t{{accessor}};
        auto reduced = pair_t{std::numeric_limits<input_value_t>::max(), offset_t{}};

        auto device = input.device();
        if (has_offsets and has_values) {
            reduce_axes_iwise(
                    shape, device, reduced, wrap(arg_values, arg_offsets), op,
                    input, output_values, output_offsets);
        } else if (has_offsets) {
            reduce_axes_iwise(shape, device, reduced, arg_offsets, op, input, output_offsets);
        } else {
            reduce_axes_iwise(shape, device, reduced, arg_values, op, input, output_values);
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
    template<typename Input, typename Output>
    requires (nt::are_varray_of_complex_v<Input, Output> or
              nt::are_varray_of_real_v<Input, Output>)
    void normalize(
            const Input& input,
            const Output& output,
            const NormalizeOptions& options = {}
    ) {
        switch (options.mode) {
            case Norm::MIN_MAX: {
                const auto [min, max] = min_max(input);
                ewise(wrap(input, min, max), output, MinusDivide{});
                break;
            }
            case Norm::MEAN_STD: {
                const auto [mean, stddev] = mean_std(input, options.ddof);
                ewise(wrap(input, mean, stddev), output, MinusDivide{});
                break;
            }
            case Norm::L2: {
                const auto norm = l2_norm(input);
                ewise(wrap(input, norm), output, Divide{});
                break;
            }
        }
    }

    /// Normalizes each batch of an array, according to a normalization mode.
    /// Can be in-place or out-of-place.
    template<typename Input, typename Output>
    requires (nt::are_varray_of_complex_v<Input, Output> or
              nt::are_varray_of_real_v<Input, Output>)
    void normalize_per_batch(
            const Input& input,
            const Output& output,
            const NormalizeOptions& options = {}
    ) {
        check(all(input.shape() == output.shape()),
              "The input and output arrays should have the same shape, but got input={} and output={}",
              input.shape(), output.shape());

        const auto axes_to_reduced = ReduceAxes{false, true, true, true};
        switch (options.mode) {
            case Norm::MIN_MAX: {
                const auto mins_maxs = min_max(input, axes_to_reduced);
                ewise(wrap(input, mins_maxs.first, mins_maxs.second), output, NormalizeMinMax{});
                break;
            }
            case Norm::MEAN_STD: {
                const auto [means, stddevs] = mean_stddev(input, axes_to_reduced, options.ddof);
                ewise(wrap(input, means, stddevs), output, NormalizeMeanStddev{});
                break;
            }
            case Norm::L2: {
                const auto l2_norms = l2_norm(input, axes_to_reduced);
                ewise(wrap(input, l2_norms), output, NormalizeNorm{});
                break;
            }
        }
    }
}
