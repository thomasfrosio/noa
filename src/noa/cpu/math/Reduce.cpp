#include "noa/core/Math.hpp"
#include "noa/algorithms/math/AccurateSum.hpp"

#include "noa/core/types/Functors.hpp"
#include "noa/cpu/math/Reduce.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/PtrHost.hpp"
#include "noa/cpu/utils/ReduceUnary.hpp"
#include "noa/cpu/utils/ReduceBinary.hpp"

namespace {
    using namespace noa;

    struct accurate_sum_t {};
    struct accurate_mean_t {};
    struct accurate_variance_t {};
    struct accurate_stddev_t {};

    template<typename Input, typename Reduced, typename Output,
             typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
    constexpr Output reduce_all(
            const Input* input,
            const Strides4<i64>& strides,
            const Shape4<i64>& shape,
            PreProcessOp&& pre_process_op,
            ReduceOp&& reduce_op,
            PostProcessOp&& post_process_op,
            i64 threads,
            i64 ddof = 0,
            Reduced* output_mean = nullptr
    ) {
        namespace nt = noa::traits;

        if constexpr (nt::is_almost_any_v<ReduceOp, noa::min_t, noa::max_t>) {
            const Reduced initial_reduce =
                    nt::is_almost_same_v<ReduceOp, noa::min_t> ?
                    noa::math::Limits<Input>::max() :
                    noa::math::Limits<Input>::lowest();

            Output output{};
            noa::cpu::utils::reduce_unary(
                    input, strides, shape, &output, Strides1<i64>{1},
                    initial_reduce, pre_process_op, reduce_op, post_process_op, threads);
            return output;

        } else if constexpr (nt::is_int_v<Reduced> && nt::is_almost_any_v<ReduceOp, accurate_sum_t, accurate_mean_t>) {
            auto pre_process_op_to_reduced = [&](Input value) {
                return static_cast<Reduced>(pre_process_op(value));
            };
            Reduced output{};
            noa::cpu::utils::reduce_unary(
                    input, strides, shape, &output, Strides1<i64>{1},
                    Reduced{0}, pre_process_op_to_reduced, noa::plus_t{}, {}, threads);
            if constexpr (nt::is_almost_same_v<ReduceOp, accurate_mean_t>) {
                const auto count = static_cast<f64>(shape.elements());
                output = static_cast<Reduced>(noa::math::round(static_cast<f64>(output) / count));
            }
            return post_process_op(output);

        } else if constexpr (nt::is_real_v<Reduced> && nt::is_almost_any_v<ReduceOp, accurate_sum_t, accurate_mean_t>) {
            f64 global_error{0};
            f64 global_sum{0};
            auto pre_process_op_to_reduced = [&](Input value) {
                return static_cast<f64>(pre_process_op(value));
            };
            noa::cpu::utils::reduce_unary(
                    input, strides, shape, &global_sum, Strides1<i64>{1},
                    f64{0}, pre_process_op_to_reduced, noa::algorithm::math::AccuratePlusReal{&global_error},
                    {}, threads);

            global_sum += global_error;
            if constexpr (nt::is_almost_same_v<ReduceOp, accurate_mean_t>)
                global_sum /= static_cast<f64>(shape.elements());
            return post_process_op(static_cast<Reduced>(global_sum));

        } else if constexpr (nt::is_complex_v<Reduced> && nt::is_almost_any_v<ReduceOp, accurate_sum_t, accurate_mean_t>) {
            c64 global_error{0};
            c64 global_sum{0};
            auto pre_process_op_to_reduced = [&](Input value) {
                return static_cast<c64>(pre_process_op(value));
            };
            noa::cpu::utils::reduce_unary(
                    input, strides, shape, &global_sum, Strides1<i64>{1},
                    c64{0}, pre_process_op_to_reduced, noa::algorithm::math::AccuratePlusComplex{&global_error},
                    {}, threads);

            global_sum += global_error;
            if constexpr (nt::is_almost_same_v<ReduceOp, accurate_mean_t>)
                global_sum /= static_cast<f64>(shape.elements());
            return post_process_op(static_cast<Reduced>(global_sum));

        } else if constexpr (nt::is_almost_any_v<ReduceOp, accurate_variance_t, accurate_stddev_t>) {
            using reduced_t = std::conditional_t<nt::is_complex_v<Reduced>, c64, f64>;

            auto pre_process_op_to_reduced = [&](Input value) {
                return static_cast<reduced_t>(pre_process_op(value));
            };
            const auto sum = reduce_all<Input, Reduced, Reduced>(
                    input, strides, shape, pre_process_op_to_reduced,
                    accurate_sum_t{}, noa::copy_t{}, threads);
            if (output_mean) {
                *output_mean = sum / static_cast<noa::traits::value_type_t<Reduced>>(shape.elements());
            }

            const auto count = static_cast<f64>(shape.elements() - ddof);
            const reduced_t mean = static_cast<reduced_t>(sum) / count;
            auto variance_op = [=](Input value) {
                const auto pre_processed_value = static_cast<reduced_t>(pre_process_op(value));
                return noa::algorithm::math::AccurateVariance<reduced_t>{mean}(pre_processed_value);
            };
            f64 variance{};
            noa::cpu::utils::reduce_unary(
                    input, strides, shape, &variance, Strides1<i64>{1},
                    f64{0}, variance_op, noa::plus_t{}, {}, threads);
            variance /= count;
            if constexpr (nt::is_almost_same_v<ReduceOp, accurate_stddev_t>)
                variance = noa::math::sqrt(variance);
            return post_process_op(static_cast<Output>(variance));
        }
    }
}

namespace {
    template<typename Input, typename Reduced, typename Output>
    struct ReduceAxis {
    public:
        static constexpr auto get_reduction_axis_mask_(
                const Shape4<i64>& input_shape,
                const Shape4<i64>& output_shape
        ) -> Vec4<bool> {
            const auto mask = input_shape != output_shape;
            if (noa::any(mask && (output_shape != 1))) {
                NOA_THROW("Dimensions should match the input shape, or be 1, "
                          "indicating the dimension should be reduced to one element. "
                          "Got shape input:{}, output:{}", input_shape, output_shape);
            } else if (noa::math::sum(mask.as<i32>()) > 1 && !noa::all(mask == Vec4<bool>{0, 1, 1, 1})) {
                NOA_THROW("Reducing more than one axis at a time is only supported if the reduction results in "
                          "one value per batch, i.e. the DHW dimensions are empty after reduction. "
                          "Got shape input:{}, output:{}, axis reduction:{}",
                          input_shape, output_shape, mask);
            }
            return mask;
        }

        template<typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
        static constexpr void execute(
                const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                PreProcessOp&& pre_process_op,
                ReduceOp&& reduce_op,
                PostProcessOp&& post_process_op,
                i64 threads, i64 ddof = 0,
                Reduced* output_mean = nullptr, const Strides4<i64>& output_mean_strides = {}
        ) {
            NOA_ASSERT(input && output && all(input_shape > 0) && all(output_shape > 0));
            NOA_ASSERT(reinterpret_cast<std::uintptr_t>(input) != reinterpret_cast<std::uintptr_t>(output));
            const auto axes_to_reduce = get_reduction_axis_mask_(input_shape, output_shape);

            // No reduction.
            if (!noa::any(axes_to_reduce)) {
                if constexpr (noa::traits::is_almost_any_v<ReduceOp, accurate_variance_t, accurate_stddev_t> &&
                              noa::traits::is_complex_v<Input>) {
                    static_assert(noa::traits::is_almost_same_v<PreProcessOp, noa::copy_t>);
                    noa::cpu::utils::ewise_unary(
                            input, input_strides, output, output_strides,
                            output_shape, noa::abs_t{}, threads);
                } else {
                    noa::cpu::utils::ewise_unary(
                            input, input_strides, output, output_strides,
                            output_shape, pre_process_op, threads);
                }
                if (output_mean) {
                    noa::cpu::utils::ewise_unary(
                            input, input_strides, output_mean, output_mean_strides,
                            output_shape, pre_process_op, threads);
                }
                return;
            }

            // Reduce the input to one value or one value per batch.
            const auto axes_empty_or_to_reduce = output_shape == 1 || axes_to_reduce;
            if (axes_empty_or_to_reduce[1] && axes_empty_or_to_reduce[2] && axes_empty_or_to_reduce[3]) {
                auto shape_to_reduce = input_shape;
                if (output_shape[0] > 1)
                    shape_to_reduce[0] = 1;
                for (i64 i = 0; i < output_shape[0]; ++i) {
                    auto* output_mean_i = output_mean ? output_mean + i * output_mean_strides[0] : nullptr;
                    output[i * output_strides[0]] = reduce_all<Input, Reduced, Output>(
                            input + i * input_strides[0], input_strides, shape_to_reduce,
                            pre_process_op, reduce_op, post_process_op, threads, ddof, output_mean_i);
                }
                return;
            }

            // Reduce one axis.
            using input_accessor_t = AccessorRestrict<const Input, 3, i64>;
            using output_accessor_t = AccessorRestrict<Output, 3, i64>;
            using output_mean_accessor_t = AccessorRestrict<Reduced, 3, i64>;
            if (axes_to_reduce[3]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(0, 1, 2));
                const auto output_3d = output_accessor_t(output, output_strides.filter(0, 1, 2));
                const auto output_mean_3d = output_mean_accessor_t(output_mean, output_mean_strides.filter(0, 1, 2));
                const auto shape_3d = input_shape.filter(0, 1, 2);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[3], input_shape[3], ddof,
                             pre_process_op, reduce_op, post_process_op, output_mean_3d);
            } else if (axes_to_reduce[2]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(0, 1, 3));
                const auto output_3d = output_accessor_t(output, output_strides.filter(0, 1, 3));
                const auto output_mean_3d = output_mean_accessor_t(output_mean, output_mean_strides.filter(0, 1, 3));
                const auto shape_3d = input_shape.filter(0, 1, 3);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[2], input_shape[2], ddof,
                             pre_process_op, reduce_op, post_process_op, output_mean_3d);
            } else if (axes_to_reduce[1]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(0, 2, 3));
                const auto output_3d = output_accessor_t(output, output_strides.filter(0, 2, 3));
                const auto output_mean_3d = output_mean_accessor_t(output_mean, output_mean_strides.filter(0, 2, 3));
                const auto shape_3d = input_shape.filter(0, 2, 3);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[1], input_shape[1], ddof,
                             pre_process_op, reduce_op, post_process_op, output_mean_3d);
            } else if (axes_to_reduce[0]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(1, 2, 3));
                const auto output_3d = output_accessor_t(output, output_strides.filter(1, 2, 3));
                const auto output_mean_3d = output_mean_accessor_t(output_mean, output_mean_strides.filter(1, 2, 3));
                const auto shape_3d = input_shape.filter(1, 2, 3);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[0], input_shape[0], ddof,
                             pre_process_op, reduce_op, post_process_op, output_mean_3d);
            }
        }

        template<typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
        constexpr static void execute_axes(
                const AccessorRestrict<const Input, 3, i64>& input,
                const AccessorRestrict<Output, 3, i64>& output,
                const Shape3<i64>& shape, i64 axis_stride, i64 axis_size, i64 ddof,
                PreProcessOp&& pre_process_op,
                ReduceOp&& reduce_op,
                PostProcessOp&& post_process_op,
                const AccessorRestrict<Reduced, 3, i64>& output_mean
        ) {
            for (i64 j = 0; j < shape[0]; ++j) {
                for (i64 k = 0; k < shape[1]; ++k) {
                    for (i64 l = 0; l < shape[2]; ++l) {
                        const auto* axis_ptr = input.offset_pointer(input.get(), j, k, l);
                        auto* mean_ptr = output_mean.is_empty() ? nullptr :
                                         output_mean.offset_pointer(output_mean.get(), j, k, l);
                        const auto reduced = execute_axis(
                                axis_ptr, axis_stride, axis_size, ddof,
                                pre_process_op, reduce_op, post_process_op, mean_ptr);
                        output(j, k, l) = static_cast<Output>(reduced);
                    }
                }
            }
        }

        template<typename PreProcessOp, typename ReduceOp, typename PostProcessOp>
        constexpr static auto execute_axis(
                const Input* axis, i64 strides, i64 size, i64 ddof,
                PreProcessOp&& pre_process_op,
                ReduceOp&& reduce_op,
                PostProcessOp&& post_process_op,
                Reduced* output_mean
        ) {
            namespace nt = noa::traits;

            if constexpr (nt::is_almost_any_v<ReduceOp, noa::min_t, noa::max_t>) {
                Reduced reduced =
                        nt::is_almost_same_v<ReduceOp, noa::min_t> ?
                        noa::math::Limits<Reduced>::max() :
                        noa::math::Limits<Reduced>::lowest();
                for (i64 i = 0; i < size; ++i)
                    reduced = reduce_op(reduced, pre_process_op(axis[i * strides]));
                return post_process_op(reduced);

            } else if constexpr (nt::is_int_v<Reduced> &&
                                 nt::is_almost_any_v<ReduceOp, accurate_sum_t, accurate_mean_t>) {
                Reduced reduced{0};
                for (i64 i = 0; i < size; ++i)
                    reduced += pre_process_op(axis[i * strides]);
                if constexpr (nt::is_almost_same_v<ReduceOp, accurate_mean_t>)
                    reduced = static_cast<Reduced>(noa::math::round(static_cast<f64>(reduced) / static_cast<f64>(size)));
                return post_process_op(reduced);

            } else if constexpr (nt::is_real_v<Reduced> &&
                                 (nt::is_almost_same_v<ReduceOp, accurate_sum_t> ||
                                  nt::is_almost_same_v<ReduceOp, accurate_mean_t>)) {
                // TODO Is it really useful to have the Kahan sum here, given that
                //      the number of elements is supposedly quite small (<10'000)
                //      and we already use double precision?
                noa::algorithm::math::AccuratePlusReal reduction_op;
                f64 reduced{0};
                for (i64 i = 0; i < size; ++i) {
                    const auto tmp = static_cast<f64>(pre_process_op(axis[i * strides]));
                    reduced = reduction_op(reduced, tmp);
                }
                reduced += reduction_op.local_error;
                if constexpr (nt::is_almost_same_v<ReduceOp, accurate_mean_t>)
                    reduced /= static_cast<f64>(size);
                return post_process_op(reduced);

            } else if constexpr (nt::is_complex_v<Reduced> &&
                                 nt::is_almost_any_v<ReduceOp, accurate_sum_t, accurate_mean_t>) {
                noa::algorithm::math::AccuratePlusComplex reduction_op;
                c64 reduced{0};
                for (i64 i = 0; i < size; ++i) {
                    const auto tmp = static_cast<c64>(pre_process_op(axis[i * strides]));
                    reduced = reduction_op(reduced, tmp);
                }
                reduced += reduction_op.local_error;
                if constexpr (nt::is_almost_same_v<ReduceOp, accurate_mean_t>)
                    reduced /= static_cast<f64>(size);
                return post_process_op(reduced);

            } else if constexpr (nt::is_almost_any_v<ReduceOp, accurate_variance_t, accurate_stddev_t>) {
                auto sum = ReduceAxis<Input, Reduced, Reduced>::execute_axis(
                        axis, strides, size, 0,
                        pre_process_op, accurate_sum_t{}, noa::copy_t{}, nullptr);
                if (output_mean)
                    *output_mean = static_cast<Reduced>(sum / static_cast<f64>(size));

                using mean_type = std::conditional_t<nt::is_complex_v<Reduced>, c64, f64>;
                const auto count = static_cast<f64>(size - ddof);
                const auto mean = static_cast<mean_type>(sum) / count;
                const auto transform_op = noa::algorithm::math::AccurateVariance<mean_type>{mean};

                f64 variance = 0;
                for (i64 i = 0; i < size; ++i)
                    variance += transform_op(pre_process_op(axis[i * strides]));
                variance /= count;
                if constexpr (nt::is_almost_same_v<ReduceOp, accurate_stddev_t>)
                    variance = noa::math::sqrt(variance);
                return post_process_op(variance);

            } else {
                static_assert(nt::always_false_v<Input>);
            }
        }
    };
}

namespace noa::cpu::math {
    template<typename Value, typename>
    Value min(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return reduce_all<Value, Value, Value>(
                input, strides, shape, noa::copy_t{}, noa::min_t{}, noa::copy_t{}, threads);
    }

    template<typename Value, typename>
    Value max(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return reduce_all<Value, Value, Value>(
                input, strides, shape, noa::copy_t{}, noa::max_t{}, noa::copy_t{}, threads);
    }

    template<typename Value, typename>
    auto min_max(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads
    ) -> std::pair<Value, Value> {
        using reduced_t = std::pair<Value, Value>;
        const auto initial_reduce = reduced_t(
                noa::math::Limits<Value>::max(),
                noa::math::Limits<Value>::lowest());

        const auto preprocess_operator = [](const Value& value) noexcept {
            return reduced_t(value, value);
        };

        const auto reduction_operator = [](reduced_t reduced, const reduced_t& value) noexcept {
            reduced.first = noa::math::min(reduced.first, value.first);
            reduced.second = noa::math::max(reduced.second, value.second);
            return reduced;
        };

        reduced_t output{};
        noa::cpu::utils::reduce_unary(
                input, strides, shape, &output, Strides1<i64>{1},
                initial_reduce, preprocess_operator, reduction_operator, {}, threads);
        return output;
    }

    template<typename Value, typename PreProcessOp, typename Reduced, typename>
    Reduced sum(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                PreProcessOp pre_process_op, i64 threads) {
        return reduce_all<Value, Reduced, Reduced>(
                input, strides, shape, pre_process_op, accurate_sum_t{}, noa::copy_t{}, threads);
    }

    template<typename Value, typename PreProcessOp, typename Reduced, typename>
    Reduced mean(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                 PreProcessOp pre_process_op, i64 threads) {
        return reduce_all<Value, Reduced, Reduced>(
                input, strides, shape, pre_process_op, accurate_mean_t{}, noa::copy_t{}, threads);
    }

    template<typename Input, typename Output, typename>
    Output norm(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return reduce_all<Input, Output, Output>(
                input, strides, shape, noa::abs_squared_t{}, accurate_sum_t{}, noa::sqrt_t{}, threads);
    }

    template<typename Input, typename Output, typename>
    Output var(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 ddof, i64 threads) {
        return reduce_all<Input, Input, Output>(
                input, strides, shape, noa::copy_t{}, accurate_variance_t{}, noa::copy_t{}, threads, ddof);
    }

    template<typename Input, typename Output, typename>
    Output std(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 ddof, i64 threads) {
        return reduce_all<Input, Input, Output>(
                input, strides, shape, noa::copy_t{}, accurate_stddev_t{}, noa::copy_t{}, threads, ddof);
    }

    template<typename Input, typename Output, typename>
    auto mean_var(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                  i64 ddof, i64 threads) -> std::pair<Input, Output> {
        Input output_mean{};
        const auto output_var = reduce_all<Input, Input, Output>(
                input, strides, shape, noa::copy_t{}, accurate_variance_t{}, noa::copy_t{}, threads,
                ddof, &output_mean);
        return std::pair{output_mean, output_var};
    }

    template<typename Input, typename Output, typename>
    auto mean_std(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                  i64 ddof, i64 threads) -> std::pair<Input, Output> {
        Input output_mean{};
        const auto output_std = reduce_all<Input, Input, Output>(
                input, strides, shape, noa::copy_t{}, accurate_stddev_t{}, noa::copy_t{}, threads,
                ddof, &output_mean);
        return std::pair{output_mean, output_std};
    }

    template<typename Value, typename>
    Value median(Value* input, Strides4<i64> strides, Shape4<i64> shape, bool overwrite) {
        NOA_ASSERT(input && all(shape > 0));

        // Make it in rightmost order.
        const auto order = indexing::order(strides, shape);
        strides = indexing::reorder(strides, order);
        shape = indexing::reorder(shape, order);

        // Allocate buffer only if necessary.
        const auto elements = shape.elements();
        Value* to_sort;
        using buffer_t = typename noa::cpu::memory::PtrHost<Value>::alloc_unique_type ;
        buffer_t buffer;
        if (overwrite && noa::indexing::are_contiguous(strides, shape)) {
            to_sort = input;
        } else {
            buffer = noa::cpu::memory::PtrHost<Value>::alloc(elements);
            noa::cpu::memory::copy(input, strides, buffer.get(), shape.strides(), shape, 1);
            to_sort = buffer.get();
        }

        std::nth_element(to_sort, to_sort + elements / 2, to_sort + elements);
        Value half = to_sort[elements / 2];
        if (elements % 2) {
            return half;
        } else {
            std::nth_element(to_sort, to_sort + (elements - 1) / 2, to_sort + elements);
            return static_cast<Value>(to_sort[(elements - 1) / 2] + half) / Value{2}; // cast to silence integer promotion
        }
    }

    template<typename Value, typename>
    Value rmsd(const Value* lhs, const Strides4<i64>& lhs_strides,
               const Value* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, i64 threads) {

        auto preprocess_op = [](Value a, Value b) {
            const auto c = static_cast<f64>(a) - static_cast<f64>(b);
            return c * c;
        };

        f64 output;
        noa::cpu::utils::reduce_binary(
                lhs, lhs_strides, rhs, rhs_strides, shape, &output, Strides1<i64>{1},
                f64{0}, preprocess_op, noa::plus_t{}, {}, threads);
        output = noa::math::sqrt(output / static_cast<f64>(shape.elements()));
        return static_cast<Value>(output);
    }
}

namespace noa::cpu::math {
    template<typename Value, typename>
    void min(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads) {
        ReduceAxis<Value, Value, Value>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                noa::copy_t{}, noa::min_t{}, noa::copy_t{},
                threads);
    }

    template<typename Value, typename>
    void max(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads) {
        ReduceAxis<Value, Value, Value>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                noa::copy_t{}, noa::max_t{}, noa::copy_t{},
                threads);
    }

    template<typename Value, typename PreProcessOp, typename Reduced, typename>
    void sum(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Reduced* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             PreProcessOp pre_process_op, i64 threads) {
        ReduceAxis<Value, Reduced, Reduced>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                pre_process_op, accurate_sum_t{}, noa::copy_t{},
                threads);
    }

    template<typename Value, typename PreProcessOp, typename Reduced, typename>
    void mean(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
              Reduced* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
              PreProcessOp pre_process_op, i64 threads) {
        ReduceAxis<Value, Reduced, Reduced>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                pre_process_op, accurate_mean_t{}, noa::copy_t{},
                threads);
    }

    template<typename Input, typename Output, typename>
    void norm(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
              Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
              i64 threads) {
        ReduceAxis<Input, Output, Output>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                noa::abs_squared_t{}, accurate_sum_t{}, noa::sqrt_t{},
                threads);
    }

    template<typename Input, typename Output, typename>
    void var(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, i64 threads) {
        ReduceAxis<Input, Input, Output>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                noa::copy_t{}, accurate_variance_t{}, noa::copy_t{},
                threads, ddof);
    }

    template<typename Input, typename Output, typename>
    void std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, i64 threads) {
        ReduceAxis<Input, Input, Output>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                noa::copy_t{}, accurate_stddev_t{}, noa::copy_t{},
                threads, ddof);
    }

    template<typename Input, typename Output, typename>
    void mean_var(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                  Input* mean, const Strides4<i64>& mean_strides,
                  Output* variance, const Strides4<i64>& variance_strides,
                  const Shape4<i64>& output_shape, i64 ddof, i64 threads) {
        ReduceAxis<Input, Input, Output>::execute(
                input, input_strides, input_shape,
                variance, variance_strides, output_shape,
                noa::copy_t{}, accurate_variance_t{}, noa::copy_t{},
                threads, ddof,
                mean, mean_strides);
    }

    template<typename Input, typename Output, typename>
    void mean_std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                  Input* mean, const Strides4<i64>& mean_strides,
                  Output* stddev, const Strides4<i64>& stddev_strides,
                  const Shape4<i64>& output_shape, i64 ddof, i64 threads) {
        ReduceAxis<Input, Input, Output>::execute(
                input, input_strides, input_shape,
                stddev, stddev_strides, output_shape,
                noa::copy_t{}, accurate_stddev_t{}, noa::copy_t{},
                threads, ddof,
                mean, mean_strides);
    }
}

namespace noa::cpu::math {
    #define NOA_INSTANTIATE_MIN_MAX_(T)                                                 \
    template T min<T, void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);   \
    template T max<T, void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);   \
    template std::pair<T,T> min_max<T, void>(                                           \
        const T*, const Strides4<i64>&, const Shape4<i64>&, i64);                       \
    template T median<T, void>(T*, Strides4<i64>, Shape4<i64>, bool);                   \
    template void min<T, void>(                                                         \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                             \
        T*, const Strides4<i64>&, const Shape4<i64>&, i64);                             \
    template void max<T, void>(                                                         \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                             \
        T*, const Strides4<i64>&, const Shape4<i64>&, i64)

    NOA_INSTANTIATE_MIN_MAX_(i16);
    NOA_INSTANTIATE_MIN_MAX_(i32);
    NOA_INSTANTIATE_MIN_MAX_(i64);
    NOA_INSTANTIATE_MIN_MAX_(u16);
    NOA_INSTANTIATE_MIN_MAX_(u32);
    NOA_INSTANTIATE_MIN_MAX_(u64);
    NOA_INSTANTIATE_MIN_MAX_(f16);
    NOA_INSTANTIATE_MIN_MAX_(f32);
    NOA_INSTANTIATE_MIN_MAX_(f64);

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN(T, R, P)                                             \
    template R sum<T,P,R,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, P, i64);     \
    template R mean<T,P,R,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, P, i64);    \
    template void sum<T,P,R,void>(                                                              \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                                     \
        R*, const Strides4<i64>&, const Shape4<i64>&, P, i64);                                  \
    template void mean<T,P,R,void>(                                                             \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                                     \
        R*, const Strides4<i64>&, const Shape4<i64>&, P, i64)

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(T)        \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(T, T, noa::copy_t);   \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(T, T, noa::nonzero_t);\
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(T, T, noa::square_t); \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(T, T, noa::abs_t);    \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(T, T, noa::abs_squared_t)

    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(f32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(f64);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(u32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(u64);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(i32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(i64);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX_ALL(C, R)        \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(C, C, noa::copy_t);      \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(C, R, noa::nonzero_t);   \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(C, C, noa::square_t);    \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(C, R, noa::abs_t);       \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN(C, R, noa::abs_squared_t)

    NOA_INSTANTIATE_REDUCE_COMPLEX_ALL(c32, f32);
    NOA_INSTANTIATE_REDUCE_COMPLEX_ALL(c64, f64);

    #define NOA_INSTANTIATE_VAR_STD_(T,U) \
    template U norm<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);     \
    template U var<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64); \
    template U std<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64); \
    template std::pair<T,U> mean_var<T,U,void>(                                             \
        const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64);                      \
    template std::pair<T,U> mean_std<T,U,void>(                                             \
        const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64);                      \
                                                                                            \
    template void norm<T,U,void>(                                                           \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        U*, const Strides4<i64>&, const Shape4<i64>&, i64);                                 \
    template void var<T,U,void>(                                                            \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        U*, const Strides4<i64>&, const Shape4<i64>&, i64, i64);                            \
    template void std<T,U,void>(                                                            \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        U*, const Strides4<i64>&, const Shape4<i64>&, i64, i64);                            \
    template void mean_var<T,U,void>(                                                       \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        T*, const Strides4<i64>&, U*, const Strides4<i64>&,                                 \
        const Shape4<i64>&, i64, i64);                                                      \
    template void mean_std<T,U,void>(                                                       \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        T*, const Strides4<i64>&, U*, const Strides4<i64>&,                                 \
        const Shape4<i64>&, i64, i64)

    NOA_INSTANTIATE_VAR_STD_(f32, f32);
    NOA_INSTANTIATE_VAR_STD_(f64, f64);
    NOA_INSTANTIATE_VAR_STD_(c32, f32);
    NOA_INSTANTIATE_VAR_STD_(c64, f64);

    #define NOA_INSTANTIATE_RMSD(T)     \
    template T rmsd<T,void>(            \
        const T*, const Strides4<i64>&, \
        const T*, const Strides4<i64>&, \
        const Shape4<i64>&, i64)

    NOA_INSTANTIATE_RMSD(f32);
    NOA_INSTANTIATE_RMSD(f64);
}
