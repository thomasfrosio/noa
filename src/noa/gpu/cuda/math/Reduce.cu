#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"

namespace noa::cuda::math {
    template<typename T, typename>
    T min(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output{};
        T* null{};
        util::reduce<true>("math::min",
                           input.get(), uint4_t{stride}, uint4_t{shape},
                           noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                           &output, 1, noa::math::copy_t{}, null, 0, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T max(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output{};
        T* null{};
        util::reduce<true>("math::max",
                           input.get(), uint4_t{stride}, uint4_t{shape},
                           noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                           &output, 1, noa::math::copy_t{}, null, 0, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T sum(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output{};
        T* null{};
        util::reduce<true>("math::sum",
                          input.get(), uint4_t{stride}, uint4_t{shape},
                          noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                          &output, 1, noa::math::copy_t{}, null, 0, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename U>
    T mean(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        T output{};
        T* null{};
        const auto inv_count = static_cast<real_t>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v / inv_count; };
        util::reduce<true>("math::mean",
                           input.get(), uint4_t{stride}, uint4_t{shape},
                           noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                           &output, 1, sum_to_mean_op, null, 0, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename>
    U var(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        U output;
        util::reduceVar<DDOF, true, false>(
                "math::var", input.get(), uint4_t{stride}, uint4_t{shape}, &output, 1, stream);
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename>
    U std(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        U output;
        util::reduceVar<DDOF, true, true>(
                "math::std", input.get(), uint4_t{stride}, uint4_t{shape}, &output, 1, stream);
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename V>
    std::tuple<T, T, U, U> statistics(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        // Get the sum and mean:
        T output_sum, output_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true>("math::statistics",
                           input.get(), uint4_t{stride}, uint4_t{shape},
                           noa::math::copy_t{}, noa::math::plus_t{}, T{0},
                           &output_sum, 1, noa::math::copy_t{}, &output_mean, 0, sum_to_mean_op, stream);

        stream.synchronize();
        T mean = output_sum / static_cast<U>(shape.elements() - DDOF);

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
        util::reduce<true>("math::statistics",
                           input.get(), uint4_t{stride}, uint4_t{shape},
                           transform_op, noa::math::plus_t{}, U{0},
                           &output_var, 1, dist2_to_var, &output_std, 0, var_to_std, stream);
        stream.synchronize();
        return {output_sum, output_mean, output_var, output_std};
    }

    #define NOA_INSTANTIATE_REDUCE_MIN_MAX_(T)                                  \
    template T min<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template T max<T,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&)

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

    #define NOA_INSTANTIATE_VAR_(T,U,DDOF)                                          \
    template U var<DDOF,T,U,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&); \
    template U std<DDOF,T,U,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&); \
    template std::tuple<T, T, U, U> statistics<DDOF,T,U,void>(const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_VAR_(float, float, 0);
    NOA_INSTANTIATE_VAR_(double, double, 0);
    NOA_INSTANTIATE_VAR_(float, float, 1);
    NOA_INSTANTIATE_VAR_(double, double, 1);

    NOA_INSTANTIATE_VAR_(cfloat_t, float, 0);
    NOA_INSTANTIATE_VAR_(cdouble_t, double, 0);
    NOA_INSTANTIATE_VAR_(cfloat_t, float, 1);
    NOA_INSTANTIATE_VAR_(cdouble_t, double, 1);
}
