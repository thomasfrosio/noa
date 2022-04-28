#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/util/Reduce.cuh"

namespace noa::cuda::math {
    template<typename T, typename>
    T min(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output;
        util::reduce<true, T, T>("math::min",
                                 input.get(), uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                                 &output, noa::math::copy_t{}, nullptr, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T max(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output;
        util::reduce<true, T, T>("math::max",
                                 input.get(), uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                                 &output, noa::math::copy_t{}, nullptr, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename>
    T sum(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output;
        util::reduce<true, T, T>("math::sum",
                                 input.get(), uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &output, noa::math::copy_t{}, nullptr, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<typename T, typename U>
    T mean(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        T output;
        const auto inv_count = static_cast<real_t>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v / inv_count; };
        util::reduce<true, T, T>("math::mean",
                                 input.get(), uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &output, sum_to_mean_op, nullptr, noa::math::copy_t{}, stream);
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename>
    U var(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        U output;
        util::reduceVar<DDOF>("math::var", input.get(), uint4_t{stride}, uint4_t{shape}, &output, stream);
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename>
    U std(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        U output;
        util::reduceStddev<DDOF>("math::std", input.get(), uint4_t{stride}, uint4_t{shape}, &output, stream);
        stream.synchronize();
        return output;
    }

    template<int DDOF, typename T, typename U, typename V>
    std::tuple<T, T, U, U> statistics(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        // Get the sum and mean:
        T output_sum, output_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true, T, T>("math::statistics",
                                 input.get(), uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &output_sum, noa::math::copy_t{}, &output_mean, sum_to_mean_op, stream);

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
        util::reduce<true, T, U>("math::statistics",
                                 input.get(), uint4_t{stride}, uint4_t{shape},
                                 transform_op, noa::math::plus_t{}, U(0),
                                 &output_var, dist2_to_var, &output_std, var_to_std, stream);
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
