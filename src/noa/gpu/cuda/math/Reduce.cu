#include "noa/gpu/cuda/Sort.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"

namespace noa::cuda::math {
    template<typename Value, typename>
    Value min(const Value* input,
              const Strides4<i64>& strides,
              const Shape4<i64>& shape,
              Stream& stream) {
        Value output{};
        utils::reduce_unary(
                "math::min",
                input, strides, shape, &output, Strides1<i64>{1},
                noa::math::Limits<Value>::max(),
                {}, noa::min_t{}, {}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Value, typename>
    Value max(const Value* input,
              const Strides4<i64>& strides,
              const Shape4<i64>& shape,
              Stream& stream) {
        Value output{};
        utils::reduce_unary(
                "math::min",
                input, strides, shape, &output, Strides1<i64>{1},
                noa::math::Limits<Value>::lowest(),
                {}, noa::max_t{}, {}, true, true, stream);
        stream.synchronize();
        return output;
    }


    template<typename Value, typename>
    Value sum(const Value* input,
              const Strides4<i64>& strides,
              const Shape4<i64>& shape,
              Stream& stream) {
        Value output{};
        utils::reduce_unary(
                "math::sum",
                input, strides, shape, &output, Strides1<i64>{1}, Value{0},
                {}, noa::plus_t{}, {}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Value, typename>
    Value mean(const Value* input,
               const Strides4<i64>& strides,
               const Shape4<i64>& shape,
               Stream& stream) {
        Value output{};
        utils::reduce_unary(
                "math::mean",
                input, strides, shape, &output, Strides1<i64>{1}, Value{0},
                {}, noa::plus_t{}, {}, true, true, stream);
        stream.synchronize();
        return output / static_cast<Value>(shape.elements());
    }

    template<typename Input, typename Output, typename>
    Output var(const Input* input,
               const Strides4<i64>& strides,
               const Shape4<i64>& shape,
               i64 ddof, Stream& stream) {
        Output output;
        utils::reduce_variance<false>(
                "math::var", input, strides, shape, &output, Strides1<i64>{1},
                ddof, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Input, typename Output, typename _>
    auto mean_var(const Input* input,
                  const Strides4<i64>& strides,
                  const Shape4<i64>& shape,
                  i64 ddof, Stream& stream
    ) -> std::pair<Input, Output> {
        // Get the sum and mean:
        Input output_sum;
        utils::reduce_unary(
                "math::mean_var",
                input, strides, shape, &output_sum, Strides1<i64>{1}, Input{0},
                {}, noa::plus_t{}, {}, true, true, stream);

        stream.synchronize();
        const auto elements = shape.elements();
        Input output_mean = output_sum / static_cast<Output>(elements);

        // Get the variance and stddev:
        const auto count = static_cast<Output>(elements - ddof);
        Input mean = output_sum / count;
        auto preprocess_op = [mean]__device__(Input value) -> Output {
                if constexpr (noa::traits::is_complex_v<Input>) {
                    const Output distance = noa::math::abs(value - mean);
                    return distance * distance;
                } else {
                    const Output distance = value - mean;
                    return distance * distance;
                }
                return Output{0}; // unreachable
        };
        Output output_dist2;
        utils::reduce_unary(
                "math::statistics",
                input, strides, shape,
                 &output_dist2, Strides1<i64>{1}, Output{0},
                preprocess_op, noa::plus_t{}, {}, true, true, stream);
        stream.synchronize();
        Output output_var = output_dist2 / count;

        return std::pair{output_mean, output_var};
    }

    template<typename Input, typename Output, typename>
    Output std(const Input* input,
               const Strides4<i64>& strides,
               const Shape4<i64>& shape,
               i64 ddof, Stream& stream) {
        Output output;
        utils::reduce_variance<true>(
                "math::var", input, strides, shape, &output, Strides1<i64>{1},
                ddof, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Value, typename>
    Value median(Value* input,
                 Strides4<i64> strides,
                 Shape4<i64> shape,
                 bool overwrite,
                 Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0));

        const auto order = noa::indexing::order(strides, shape);
        strides = noa::indexing::reorder(strides, order);
        shape = noa::indexing::reorder(shape, order);

        const auto elements = shape.elements();
        using unique_t = typename noa::cuda::memory::PtrDevice<Value>::unique_type;
        unique_t buffer;
        Value* to_sort{};
        if (overwrite && noa::indexing::are_contiguous(strides, shape)) {
            to_sort = input;
        } else {
            buffer = noa::cuda::memory::PtrDevice<Value>::alloc(elements, stream);
            to_sort = buffer.get();
            noa::cuda::memory::copy(input, strides, to_sort, shape.strides(), shape, stream);
        }

        // Sort the entire contiguous array.
        const auto shape_1d = Shape4<i64>{1, 1, 1, elements};
        noa::cuda::sort(to_sort, shape_1d.strides(), shape_1d, true, -1, stream);

        // Retrieve the median.
        const bool is_even = !(elements % 2);
        Value out[2];
        memory::copy(to_sort + (elements - is_even) / 2, out, 1 + is_even, stream);
        stream.synchronize();

        if (is_even)
            return (out[0] + out[1]) / Value{2};
        else
            return out[0];
    }

    #define NOA_INSTANTIATE_REDUCE_MIN_MAX_(T)                                              \
    template T min<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);    \
    template T max<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);    \
    template T median<T,void>(T*, Strides4<i64>, Shape4<i64>, bool, Stream&)

    NOA_INSTANTIATE_REDUCE_MIN_MAX_(f16);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(f32);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(f64);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(u16);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(u32);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(u64);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(i16);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(i32);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(i64);

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T)                                             \
    template T sum<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);    \
    template T mean<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(f32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(f64);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(u32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(u64);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(i32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(i64);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX(T)                                               \
    template T sum<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);    \
    template T mean<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_REDUCE_COMPLEX(c32);
    NOA_INSTANTIATE_REDUCE_COMPLEX(c64);

    #define NOA_INSTANTIATE_VAR_(T,U)                                                               \
    template U var<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&);     \
    template U std<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&);     \
    template std::pair<T, U> mean_var<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&)

    NOA_INSTANTIATE_VAR_(f32, f32);
    NOA_INSTANTIATE_VAR_(f64, f64);
    NOA_INSTANTIATE_VAR_(c32, f32);
    NOA_INSTANTIATE_VAR_(c64, f64);
}
