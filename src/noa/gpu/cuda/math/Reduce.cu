#include "noa/core/types/Functors.hpp"
#include "noa/gpu/cuda/Sort.hpp"
#include "noa/gpu/cuda/math/Reduce.hpp"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"

namespace noa::cuda::math {
    template<typename Value, typename>
    Value min(const Value* input,
              const Strides4<i64>& strides,
              const Shape4<i64>& shape,
              Stream& stream) {
        Value output{};
        noa::cuda::utils::reduce_unary(
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
        noa::cuda::utils::reduce_unary(
                input, strides, shape, &output, Strides1<i64>{1},
                noa::math::Limits<Value>::lowest(),
                {}, noa::max_t{}, {}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Value, typename _>
    auto min_max(const Value* input,
                 const Strides4<i64>& strides,
                 const Shape4<i64>& shape,
                 Stream& stream
    ) -> std::pair<Value, Value> {
        using reduced_t = Pair<Value, Value>;
        const auto initial_reduce = reduced_t(
                noa::math::Limits<Value>::max(),
                noa::math::Limits<Value>::lowest());

        const auto preprocess_operator = []__device__(const Value& value) noexcept {
            return reduced_t(value, value);
        };

        const auto reduction_operator = []__device__(reduced_t reduced, const reduced_t& value) noexcept {
            reduced.first = noa::math::min(reduced.first, value.first);
            reduced.second = noa::math::max(reduced.second, value.second);
            return reduced;
        };

        reduced_t output{};
        noa::cuda::utils::reduce_unary(
                input, strides, shape, &output, Strides1<i64>{1},
                initial_reduce,
                preprocess_operator, reduction_operator, {}, true, true, stream);
        stream.synchronize();
        return {output.first, output.second};
    }

    template<typename Value, typename PreProcessOp, typename Reduced, typename>
    Reduced sum(const Value* input,
                const Strides4<i64>& strides,
                const Shape4<i64>& shape,
                PreProcessOp pre_process_op,
                Stream& stream) {
        Reduced output{};
        noa::cuda::utils::reduce_unary(
                input, strides, shape, &output, Strides1<i64>{1}, Reduced{0},
                pre_process_op, noa::plus_t{}, {}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Value, typename PreProcessOp, typename Reduced, typename>
    Reduced mean(const Value* input,
                 const Strides4<i64>& strides,
                 const Shape4<i64>& shape,
                 PreProcessOp pre_process_op,
                 Stream& stream) {
        Reduced output{};
        noa::cuda::utils::reduce_unary(
                input, strides, shape, &output, Strides1<i64>{1}, Reduced{0},
                pre_process_op, noa::plus_t{}, {}, true, true, stream);
        stream.synchronize();
        if constexpr (nt::is_int_v<Reduced>) {
            return static_cast<Reduced>(noa::math::round(
                    static_cast<f64>(output) / static_cast<f64>(shape.elements())));
        } else {
            using real_t = nt::value_type_t<Value>;
            return output / static_cast<real_t>(shape.elements());
        }
    }

    template<typename Input, typename Output, typename>
    Output norm(const Input* input,
                const Strides4<i64>& strides,
                const Shape4<i64>& shape,
                Stream& stream) {
        Output output{};
        noa::cuda::utils::reduce_unary(
                input, strides, shape, &output, Strides1<i64>{1}, Output{0},
                noa::abs_squared_t{}, noa::plus_t{}, noa::sqrt_t{}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Input, typename Output, typename>
    Output var(const Input* input,
               const Strides4<i64>& strides,
               const Shape4<i64>& shape,
               i64 ddof, Stream& stream) {
        Input* null{};
        Output output;
        noa::cuda::utils::reduce_variance<false>(
                input, strides, shape,
                null, Strides1<i64>{1},
                &output, Strides1<i64>{1},
                ddof, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename Input, typename Output, typename>
    Output std(const Input* input,
               const Strides4<i64>& strides,
               const Shape4<i64>& shape,
               i64 ddof, Stream& stream) {
        Input* null{};
        Output output;
        noa::cuda::utils::reduce_variance<true>(
                input, strides, shape,
                null, Strides1<i64>{1},
                &output, Strides1<i64>{1},
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
        Input mean;
        Output var;
        noa::cuda::utils::reduce_variance<false>(
                input, strides, shape,
                &mean, Strides1<i64>{1},
                &var, Strides1<i64>{1},
                ddof, true, true, stream);
        stream.synchronize();
        return std::pair{mean, var};
    }

    template<typename Input, typename Output, typename _>
    auto mean_std(const Input* input,
                  const Strides4<i64>& strides,
                  const Shape4<i64>& shape,
                  i64 ddof, Stream& stream
    ) -> std::pair<Input, Output> {
        Input mean;
        Output var;
        noa::cuda::utils::reduce_variance<true>(
                input, strides, shape,
                &mean, Strides1<i64>{1},
                &var, Strides1<i64>{1},
                ddof, true, true, stream);
        stream.synchronize();
        return std::pair{mean, var};
    }

    template<typename Value, typename>
    Value median(
            Value* input,
            Strides4<i64> strides,
            Shape4<i64> shape,
            bool overwrite,
            Stream& stream
    ) {
        NOA_ASSERT(noa::all(shape > 0));

        const auto order = noa::indexing::order(strides, shape);
        strides = noa::indexing::reorder(strides, order);
        shape = noa::indexing::reorder(shape, order);

        const auto elements = shape.elements();
        using unique_t = typename noa::cuda::memory::AllocatorDevice<Value>::unique_type;
        unique_t buffer;
        Value* to_sort{};
        if (overwrite && noa::indexing::are_contiguous(strides, shape)) {
            to_sort = input;
        } else {
            buffer = noa::cuda::memory::AllocatorDevice<Value>::allocate_async(elements, stream);
            to_sort = buffer.get();
            noa::cuda::memory::copy(input, strides, to_sort, shape.strides(), shape, stream);
        }

        // Sort the entire contiguous array.
        const auto shape_1d = Shape4<i64>{1, 1, 1, elements};
        noa::cuda::sort(to_sort, shape_1d.strides(), shape_1d, true, -1, stream);

        // Retrieve the median.
        const bool is_even = !(elements % 2);
        Value out[2];
        noa::cuda::memory::copy(to_sort + (elements - is_even) / 2, out, 1 + is_even, stream);
        stream.synchronize();

        if (is_even)
            return (out[0] + out[1]) / Value{2};
        else
            return out[0];
    }

    template<typename Value, typename _>
    Value rmsd(const Value* lhs, const Strides4<i64>& lhs_strides,
               const Value* rhs, const Strides4<i64>& rhs_strides,
               const Shape4<i64>& shape, Stream& stream) {
        Value output;
        noa::cuda::utils::reduce_binary(
                lhs, lhs_strides, rhs, rhs_strides, shape,
                &output, Strides1<i64>{1}, Value{0},
                noa::dist2_t{}, noa::plus_t{}, {}, true, true, stream);
        stream.synchronize();
        return noa::math::sqrt(output / static_cast<nt::value_type_t<Value>>(shape.elements()));
    }

    #define NOA_INSTANTIATE_REDUCE_MIN_MAX_(T)                                                              \
    template T min<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);                    \
    template T max<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);                    \
    template std::pair<T,T> min_max<T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);   \
    template T median<T,void>(T*, Strides4<i64>, Shape4<i64>, bool, Stream&)

//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(f16);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(f32);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(f64);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(u16);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(u32);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(u64);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(i16);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(i32);
//    NOA_INSTANTIATE_REDUCE_MIN_MAX_(i64);

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T, P)                                              \
    template T sum<T,P,T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, P, Stream&); \
    template T mean<T,P,T,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, P, Stream&)

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(T)      \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T, noa::copy_t);   \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T, noa::nonzero_t);\
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T, noa::square_t); \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T, noa::abs_t);    \
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T, noa::abs_squared_t)

    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(f32);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(f64);
//    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(u32);
//    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(u64);
//    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(i32);
//    NOA_INSTANTIATE_REDUCE_SUM_MEAN_ALL(i64);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX(C, R, P)                                             \
    template R sum<C,P,R,void>(const C*, const Strides4<i64>&, const Shape4<i64>&, P, Stream&); \
    template R mean<C,P,R,void>(const C*, const Strides4<i64>&, const Shape4<i64>&, P, Stream&)

    #define NOA_INSTANTIATE_REDUCE_COMPLEX_ALL(C, R)        \
    NOA_INSTANTIATE_REDUCE_COMPLEX(C, C, noa::copy_t);      \
    NOA_INSTANTIATE_REDUCE_COMPLEX(C, R, noa::nonzero_t);   \
    NOA_INSTANTIATE_REDUCE_COMPLEX(C, C, noa::square_t);    \
    NOA_INSTANTIATE_REDUCE_COMPLEX(C, R, noa::abs_t);       \
    NOA_INSTANTIATE_REDUCE_COMPLEX(C, R, noa::abs_squared_t)

    NOA_INSTANTIATE_REDUCE_COMPLEX_ALL(c32, f32);
    NOA_INSTANTIATE_REDUCE_COMPLEX_ALL(c64, f64);

    #define NOA_INSTANTIATE_VAR_(T,U) \
    template U norm<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, Stream&);                         \
    template U var<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&);                     \
    template U std<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&);                     \
    template std::pair<T, U> mean_var<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&);  \
    template std::pair<T, U> mean_std<T,U,void>(const T*, const Strides4<i64>&, const Shape4<i64>&, i64, Stream&)

    NOA_INSTANTIATE_VAR_(f32, f32);
    NOA_INSTANTIATE_VAR_(f64, f64);
    NOA_INSTANTIATE_VAR_(c32, f32);
    NOA_INSTANTIATE_VAR_(c64, f64);

    #define NOA_INSTANTIATE_RMSD(T)     \
    template T rmsd<T,void>(            \
        const T*, const Strides4<i64>&, \
        const T*, const Strides4<i64>&, \
        const Shape4<i64>&, Stream&)

//    NOA_INSTANTIATE_RMSD(f32);
//    NOA_INSTANTIATE_RMSD(f64);
}
