#include "noa/core/Math.hpp"
#include "noa/algorithms/math/AccurateSum.hpp"

#include "noa/cpu/math/Reduce.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/PtrHost.hpp"
#include "noa/cpu/utils/ReduceUnary.hpp"

namespace {
    using namespace noa;

    enum class ReductionMode {
        MIN, MAX, SUM, MEAN, VAR, STD
    };

    template<ReductionMode MODE, typename Input>
    struct ReduceAll {
    public:
        static constexpr auto execute(
                const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                i64 threads, i64 ddof = 0, const Input* sum_ptr = nullptr
        ) {
            constexpr bool IS_SUM_OR_MEAN = MODE == ReductionMode::SUM || MODE == ReductionMode::MEAN;
            if constexpr (MODE == ReductionMode::MIN) {
                Input output{};
                noa::cpu::utils::reduce_unary(
                        input, strides, shape, &output, Strides1<i64>{1},
                        noa::math::Limits<Input>::max(), {}, noa::min_t{}, {}, threads);
                return output;

            } else if constexpr (MODE == ReductionMode::MAX) {
                Input output{};
                noa::cpu::utils::reduce_unary(
                        input, strides, shape, &output, Strides1<i64>{1},
                        noa::math::Limits<Input>::min(), {}, noa::max_t{}, {}, threads);
                return output;

            } else if constexpr (IS_SUM_OR_MEAN && noa::traits::is_int_v<Input>) {
                Input output;
                noa::cpu::utils::reduce_unary(
                        input, strides, shape, &output, Strides1<i64>{1},
                        Input{0}, {}, noa::plus_t{}, {}, threads);
                if constexpr (MODE == ReductionMode::MEAN)
                    output /= static_cast<Input>(shape.elements());
                return output;

            } else if constexpr (IS_SUM_OR_MEAN && noa::traits::is_real_v<Input>) {
                f64 global_error{0};
                f64 global_sum{0};
                noa::cpu::utils::reduce_unary(
                        input, strides, shape, &global_sum, Strides1<i64>{1},
                        f64{0}, [](Input v) { return static_cast<f64>(v); },
                        noa::algorithm::math::AccuratePlusReal{&global_error}, {}, threads);
                global_sum += global_error;
                if constexpr (MODE == ReductionMode::MEAN)
                    global_sum /= static_cast<f64>(shape.elements());
                return static_cast<Input>(global_sum);

            } else if constexpr (IS_SUM_OR_MEAN && noa::traits::is_complex_v<Input>) {
                c64 global_error{0};
                c64 global_sum{0};
                noa::cpu::utils::reduce_unary(
                        input, strides, shape, &global_sum, Strides1<i64>{1},
                        c64{0}, [](Input v) { return static_cast<c64>(v); },
                        noa::algorithm::math::AccuratePlusComplex{&global_error}, {}, threads);
                global_sum += global_error;
                if constexpr (MODE == ReductionMode::MEAN)
                    global_sum /= static_cast<f64>(shape.elements());
                return static_cast<Input>(global_sum);

            } else if constexpr (MODE == ReductionMode::VAR || MODE == ReductionMode::STD) {
                using mean_t = std::conditional_t<noa::traits::is_complex_v<Input>, c64, f64>;
                const auto count = static_cast<f64>(shape.elements() - ddof);
                noa::algorithm::math::AccurateVariance<mean_t> transform_op{0};
                if (sum_ptr) {
                    transform_op.mean = static_cast<mean_t>(*sum_ptr) / count;
                } else {
                    using sum_reduction_t = ReduceAll<ReductionMode::SUM, Input>;
                    const auto sum = sum_reduction_t::execute(input, strides, shape, threads);
                    transform_op.mean = static_cast<mean_t>(sum) / count;
                }

                f64 variance{};
                noa::cpu::utils::reduce_unary(
                        input, strides, shape, &variance, Strides1<i64>{1},
                        f64{0}, transform_op, noa::plus_t{}, {}, threads);
                variance /= count;
                if constexpr (MODE == ReductionMode::STD)
                    variance = noa::math::sqrt(variance);
                return static_cast<noa::traits::value_type_t<Input>>(variance);
            }
        }
    };
}

namespace {
    template<ReductionMode MODE, typename Input, typename Output>
    struct ReduceAxis {
    public:
        static constexpr void
        execute(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                i64 threads, i64 ddof = 0
        ) {
            NOA_ASSERT(input && output && all(input_shape > 0) && all(output_shape > 0));
            NOA_ASSERT(reinterpret_cast<std::uintptr_t>(input) != reinterpret_cast<std::uintptr_t>(output));
            const auto axes_to_reduce = get_reduction_axis_mask_(input_shape, output_shape);

            // No reduction.
            if (!noa::any(axes_to_reduce)) {
                if constexpr (noa::traits::is_complex_v<Input> && noa::traits::is_real_v<Output>) {
                    return noa::cpu::utils::ewise_unary(
                            input, input_strides, output, output_strides,
                            output_shape, noa::abs_t{}, threads);
                } else {
                    return noa::cpu::memory::copy(input, input_strides, output, output_strides, output_shape, threads);
                }
            }

            // Reduce the input to one value or one value per batch.
            const auto axes_empty_or_to_reduce = output_shape == 1 || axes_to_reduce;
            if (axes_empty_or_to_reduce[1] && axes_empty_or_to_reduce[2] && axes_empty_or_to_reduce[3]) {
                auto shape_to_reduce = input_shape;
                if (axes_empty_or_to_reduce[0])
                    shape_to_reduce[0] = 1;
                for (i64 i = 0; i < output_shape[0]; ++i) {
                    output[i * output_strides[0]] = ReduceAll<MODE, Input>::execute(
                            input + i * input_strides[0], input_strides, shape_to_reduce, threads, ddof);
                }
                return;
            }

            // Reduce one axis.
            using input_accessor_t = AccessorRestrict<const Input, 3, i64>;
            using output_accessor_t = AccessorRestrict<Output, 3, i64>;
            if (axes_to_reduce[3]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(0, 1, 2));
                const auto output_3d = output_accessor_t(output, output_strides.filter(0, 1, 2));
                const auto shape_3d = input_shape.filter(0, 1, 2);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[3], input_shape[3]);
            } else if (axes_to_reduce[2]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(0, 1, 3));
                const auto output_3d = output_accessor_t(output, output_strides.filter(0, 1, 3));
                const auto shape_3d = input_shape.filter(0, 1, 3);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[2], input_shape[2]);
            } else if (axes_to_reduce[1]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(0, 2, 3));
                const auto output_3d = output_accessor_t(output, output_strides.filter(0, 2, 3));
                const auto shape_3d = input_shape.filter(0, 2, 3);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[1], input_shape[1]);
            } else if (axes_to_reduce[0]) {
                const auto input_3d = input_accessor_t(input, input_strides.filter(1, 2, 3));
                const auto output_3d = output_accessor_t(output, output_strides.filter(1, 2, 3));
                const auto shape_3d = input_shape.filter(1, 2, 3);
                execute_axes(input_3d, output_3d, shape_3d, input_strides[0], input_shape[0]);
            }
        }

        static constexpr auto get_reduction_axis_mask_(
                const Shape4<i64>& input_shape,
                const Shape4<i64>& output_shape
        ) -> Vec4<bool> {
            const auto mask = input_shape != output_shape;
            if (noa::any(mask && (output_shape != 1))) {
                NOA_THROW("Dimensions should match the input shape, or be 1, "
                          "indicating the dimension should be reduced to one element. "
                          "Got shape input:{}, output:{}", input_shape, output_shape);
            } else if (noa::math::sum(mask.as<i32>()) > 1) {
                NOA_THROW("Reducing more than one axis at a time is only supported if the reduction results in "
                          "one value per batch, i.e. the DHW dimensions are empty after reduction. "
                          "Got shape input:{}, output:{}, axis reduction:{}",
                          input_shape, output_shape, mask);
            }
            return mask;
        }

        constexpr static void execute_axes(
                const AccessorRestrict<const Input, 3, i64>& input,
                const AccessorRestrict<Output, 3, i64>& output,
                const Shape3<i64>& shape, i64 axis_stride, i64 axis_size) {
            for (i64 j = 0; j < shape[0]; ++j) {
                for (i64 k = 0; k < shape[1]; ++k) {
                    for (i64 l = 0; l < shape[2]; ++l) {
                        const auto* axis_ptr = input.offset_pointer(input.get(), j, k, l);
                        output(j, k, l) = static_cast<Output>(execute_axis(axis_ptr, axis_stride, axis_size));
                    }
                }
            }
        }

        constexpr static auto execute_axis(const Input* axis, i64 strides, i64 size, i64 ddof = 0) {
            constexpr bool SUM_OR_MEAN = MODE == ReductionMode::SUM || MODE == ReductionMode::MEAN;
            constexpr bool VAR_OR_STD = MODE == ReductionMode::VAR || MODE == ReductionMode::STD;

            if constexpr (MODE == ReductionMode::MIN && noa::traits::is_scalar_v<Input>) {
                Input min = noa::math::Limits<Input>::max();
                for (i64 i = 0; i < size; ++i)
                    min = noa::math::min(min, axis[i * strides]);
                return min;

            } else if constexpr (MODE == ReductionMode::MAX && noa::traits::is_scalar_v<Input>) {
                Input max = noa::math::Limits<Input>::min();
                for (i64 i = 0; i < size; ++i)
                    max = noa::math::max(max, axis[i * strides]);
                return max;

            } else if constexpr (SUM_OR_MEAN && noa::traits::is_int_v<Input>) {
                Input sum = 0;
                for (i64 i = 0; i < size; ++i)
                    sum += axis[i * strides];
                if constexpr (MODE == ReductionMode::MEAN)
                    sum /= static_cast<Input>(size);
                return sum;

            } else if constexpr (SUM_OR_MEAN && noa::traits::is_real_v<Input>) {
                noa::algorithm::math::AccuratePlusReal reduction_op;
                f64 sum = 0;
                for (i64 i = 0; i < size; ++i) {
                    const auto tmp = static_cast<f64>(axis[i * strides]);
                    sum += reduction_op(sum, tmp);
                }
                sum += reduction_op.local_error;
                if constexpr (MODE == ReductionMode::MEAN)
                    sum /= static_cast<f64>(size);
                return sum;

            } else if constexpr (SUM_OR_MEAN && noa::traits::is_complex_v<Input>) {
                noa::algorithm::math::AccuratePlusComplex reduction_op;
                c64 sum{0};
                for (i64 i = 0; i < size; ++i) {
                    const auto tmp = static_cast<c64>(axis[i * strides]);
                    sum += reduction_op(sum, tmp);
                }
                sum += reduction_op.local_error;
                if constexpr (MODE == ReductionMode::MEAN)
                    sum /= static_cast<f64>(size);
                return sum;

            } else if constexpr (VAR_OR_STD) {
                const auto count = static_cast<f64>(size - ddof);
                using mean_op_type = ReduceAxis<ReductionMode::SUM, Input, Output>;
                using mean_type = std::conditional_t<noa::traits::is_complex_v<Input>, c64, f64>;
                noa::algorithm::math::AccurateVariance<mean_type> transform_op{
                    /*mean=*/ mean_op_type::execute_axis(axis, strides, size) / count};

                f64 variance = 0;
                for (i64 i = 0; i < size; ++i)
                    variance += transform_op(axis[i * strides]);
                variance /= count;
                if constexpr (MODE == ReductionMode::STD)
                    variance = noa::math::sqrt(variance);
                return variance;

            } else {
                static_assert(noa::traits::always_false_v<Input>);
            }
        }
    };
}

namespace noa::cpu::math {
    template<typename Value, typename>
    Value min(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return ReduceAll<ReductionMode::MIN, Value>::execute(input, strides, shape, threads);
    }

    template<typename Value, typename>
    Value max(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return ReduceAll<ReductionMode::MAX, Value>::execute(input, strides, shape, threads);
    }

    template<typename Value, typename>
    Value sum(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return ReduceAll<ReductionMode::SUM, Value>::execute(input, strides, shape, threads);
    }

    template<typename Value, typename>
    Value mean(const Value* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 threads) {
        return ReduceAll<ReductionMode::MEAN, Value>::execute(input, strides, shape, threads);
    }

    template<typename Input, typename Output, typename>
    Output var(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 ddof, i64 threads) {
        return ReduceAll<ReductionMode::VAR, Input>::execute(input, strides, shape, threads, ddof);
    }

    template<typename Input, typename Output, typename>
    auto mean_var(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape,
                  i64 ddof, i64 threads) -> std::pair<Input, Output> {
        const auto output_sum = ReduceAll<ReductionMode::SUM, Input>::execute(input, strides, shape, threads);
        const auto output_var = ReduceAll<ReductionMode::VAR, Input>::execute(input, strides, shape, threads, ddof, &output_sum);
        const auto output_mean = output_sum / static_cast<Output>(shape.elements());
        return std::pair{output_mean, output_var};
    }

    template<typename Input, typename Output, typename>
    Output std(const Input* input, const Strides4<i64>& strides, const Shape4<i64>& shape, i64 ddof, i64 threads) {
        return ReduceAll<ReductionMode::STD, Input>::execute(input, strides, shape, threads, ddof);
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
        Value* to_sort{};
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
}

namespace noa::cpu::math {
    template<typename Value, typename>
    void min(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads) {
        ReduceAxis<ReductionMode::MIN, Value, Value>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                threads);
    }

    template<typename Value, typename>
    void max(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads) {
        ReduceAxis<ReductionMode::MAX, Value, Value>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                threads);
    }

    template<typename Value, typename>
    void sum(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 threads) {
        ReduceAxis<ReductionMode::SUM, Value, Value>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                threads);
    }

    template<typename Value, typename>
    void mean(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
              Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
              i64 threads) {
        ReduceAxis<ReductionMode::MEAN, Value, Value>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                threads);
    }

    template<typename Input, typename Output, typename>
    void var(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, i64 threads) {
        ReduceAxis<ReductionMode::VAR, Input, Output>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                threads, ddof);
    }

    template<typename Input, typename Output, typename>
    void std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, i64 threads) {
        ReduceAxis<ReductionMode::STD, Input, Output>::execute(
                input, input_strides, input_shape,
                output, output_strides, output_shape,
                threads, ddof);
    }
}

namespace noa::cpu::math {
    #define NOA_INSTANTIATE_MIN_MAX_(T)                                                 \
    template T min<T, void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);   \
    template T max<T, void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);   \
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

    #define NOA_INSTANTIATE_SUM_MEAN_(T)                                                \
    template T sum<T, void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);   \
    template T mean<T, void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64);  \
    template void sum<T, void>(                                                         \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                             \
        T*, const Strides4<i64>&, const Shape4<i64>&, i64);                             \
    template void mean<T, void>(                                                        \
        const T*, const Strides4<i64>&, const Shape4<i64>&,                             \
        T*, const Strides4<i64>&, const Shape4<i64>&, i64)

    NOA_INSTANTIATE_SUM_MEAN_(i32);
    NOA_INSTANTIATE_SUM_MEAN_(i64);
    NOA_INSTANTIATE_SUM_MEAN_(u32);
    NOA_INSTANTIATE_SUM_MEAN_(u64);
    NOA_INSTANTIATE_SUM_MEAN_(f32);
    NOA_INSTANTIATE_SUM_MEAN_(f64);
    NOA_INSTANTIATE_SUM_MEAN_(c32);
    NOA_INSTANTIATE_SUM_MEAN_(c64);

    #define NOA_INSTANTIATE_VAR_STD_(T,U)                                                   \
    template U var<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64); \
    template U std<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64); \
    template std::pair<T,U> mean_var<T,U,void>(                                             \
        const T*, const Strides4<i64>&, const Shape4<i64>&, i64, i64);                      \
    template void var<T,U,void>(                                                            \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        U*, const Strides4<i64>&, const Shape4<i64>&, i64, i64);                            \
    template void std<T,U,void>(                                                            \
        const T* , const Strides4<i64>&, const Shape4<i64>&,                                \
        U*, const Strides4<i64>&, const Shape4<i64>&, i64, i64)

    NOA_INSTANTIATE_VAR_STD_(f32, f32);
    NOA_INSTANTIATE_VAR_STD_(f64, f64);
    NOA_INSTANTIATE_VAR_STD_(c32, f32);
    NOA_INSTANTIATE_VAR_STD_(c64, f64);
}
