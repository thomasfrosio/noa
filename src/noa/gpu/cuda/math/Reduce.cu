#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/math/Sort.h"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"

namespace noa::cuda::math {
    template<typename T, typename>
    T min(const shared_t<T[]>& input, size4_t strides, size4_t shape, Stream& stream) {
        T output{};
        T* null{};
        util::reduce("math::min",
                     input.get(), uint4_t(strides), uint4_t(shape),
                     noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                     &output, 1, noa::math::copy_t{}, null, 0, noa::math::copy_t{}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T max(const shared_t<T[]>& input, size4_t strides, size4_t shape, Stream& stream) {
        T output{};
        T* null{};
        util::reduce("math::max",
                     input.get(), uint4_t(strides), uint4_t(shape),
                     noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                     &output, 1, noa::math::copy_t{}, null, 0, noa::math::copy_t{}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T median(const shared_t<T[]>& input, size4_t strides, size4_t shape,
             bool overwrite, Stream& stream) {
        // Make it in rightmost order.
        const size4_t order = indexing::order(strides, shape);
        strides = indexing::reorder(strides, order);
        shape = indexing::reorder(shape, order);

        const size_t elements = shape.elements();
        const shared_t<T[]>* to_sort;
        shared_t<T[]> buffer;
        if (overwrite && indexing::areContiguous(strides, shape)) {
            to_sort = &input;
        } else {
            buffer = memory::PtrDevice<T>::alloc(elements, stream);
            memory::copy(input, strides, buffer, shape.strides(), shape, stream);
            to_sort = &buffer;
        }

        // Sort the entire contiguous array.
        const size4_t shape_1d{1, 1, 1, elements};
        sort(*to_sort, shape_1d.strides(), shape_1d, true, -1, stream);

        // Retrieve the median.
        const bool is_even = !(elements % 2);
        T out[2];
        memory::copy(to_sort->get() + (elements - is_even) / 2, out, 1 + is_even, stream);
        stream.synchronize();

        if (is_even)
            return (out[0] + out[1]) / T{2};
        else
            return out[0];
    }

    template<typename T, typename>
    T sum(const shared_t<T[]>& input, size4_t strides, size4_t shape, Stream& stream) {
        T output{};
        T* null{};
        util::reduce("math::sum",
                     input.get(), uint4_t(strides), uint4_t(shape),
                     noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                     &output, 1, noa::math::copy_t{}, null, 0, noa::math::copy_t{}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename U>
    T mean(const shared_t<T[]>& input, size4_t strides, size4_t shape, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        T output{};
        T* null{};
        const auto inv_count = static_cast<real_t>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v / inv_count; };
        util::reduce("math::mean",
                     input.get(), uint4_t(strides), uint4_t(shape),
                     noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                     &output, 1, sum_to_mean_op, null, 0, noa::math::copy_t{}, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename U, typename>
    U var(const shared_t<T[]>& input, size4_t strides, size4_t shape, int ddof, Stream& stream) {
        U output;
        util::reduceVar<false>("math::var", input.get(), uint4_t(strides), uint4_t(shape), &output, 1,
                               ddof, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename U, typename>
    U std(const shared_t<T[]>& input, size4_t strides, size4_t shape, int ddof, Stream& stream) {
        U output;
        util::reduceVar<true>("math::std", input.get(), uint4_t(strides), uint4_t(shape), &output, 1,
                              ddof, true, true, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename U, typename V>
    std::tuple<T, T, U, U> statistics(const shared_t<T[]>& input, size4_t strides, size4_t shape,
                                      int ddof, Stream& stream) {
        // Get the sum and mean:
        T output_sum, output_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce("math::statistics",
                     input.get(), uint4_t(strides), uint4_t(shape),
                     noa::math::copy_t{}, noa::math::plus_t{}, T{0},
                     &output_sum, 1, noa::math::copy_t{}, &output_mean, 0, sum_to_mean_op,
                     true, true, stream);

        stream.synchronize();
        T mean = output_sum / static_cast<U>(shape.elements() - ddof);

        // Get the variance and stddev:
        auto transform_op = [mean]__device__(T value) -> U {
            if constexpr (noa::traits::is_complex_v<T>) {
                const U distance = noa::math::abs(value - mean);
                return distance * distance;
            } else {
                const U distance = value - mean;
                return distance * distance;
            }
            return U(0); // unreachable
        };
        auto dist2_to_var = [inv_count]__device__(U v) -> U { return v * inv_count; };
        auto var_to_std = []__device__(U v) -> U { return noa::math::sqrt(v); };

        U output_var, output_std;
        util::reduce("math::statistics",
                     input.get(), uint4_t(strides), uint4_t(shape),
                     transform_op, noa::math::plus_t{}, U{0},
                     &output_var, 1, dist2_to_var, &output_std, 0, var_to_std,
                     true, true, stream);
        stream.synchronize();
        return {output_sum, output_mean, output_var, output_std};
    }

    #define NOA_INSTANTIATE_REDUCE_MIN_MAX_(T)                                  \
    template T min<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template T max<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template T median<T,void>(const shared_t<T[]>&, size4_t, size4_t, bool, Stream&)

    NOA_INSTANTIATE_REDUCE_MIN_MAX_(half_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(float);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(double);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(uint16_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(uint32_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(uint64_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(int16_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(int32_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(int64_t);

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T)                                 \
    template T sum<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template T mean<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(float);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(double);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(uint32_t);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(uint64_t);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(int32_t);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(int64_t);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX(T)                                   \
    template T sum<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template T mean<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_REDUCE_COMPLEX(cfloat_t);
    NOA_INSTANTIATE_REDUCE_COMPLEX(cdouble_t);

    #define NOA_INSTANTIATE_VAR_(T,U)                                               \
    template U var<T,U,void>(const shared_t<T[]>&, size4_t, size4_t, int, Stream&); \
    template U std<T,U,void>(const shared_t<T[]>&, size4_t, size4_t, int, Stream&); \
    template std::tuple<T, T, U, U> statistics<T,U,void>(const shared_t<T[]>&, size4_t, size4_t, int, Stream&)

    NOA_INSTANTIATE_VAR_(float, float);
    NOA_INSTANTIATE_VAR_(double, double);
    NOA_INSTANTIATE_VAR_(cfloat_t, float);
    NOA_INSTANTIATE_VAR_(cdouble_t, double);
}
