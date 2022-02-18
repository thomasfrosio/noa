#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/math/Reduce.cuh"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Pointers.h"

namespace noa::cuda::math {
    template<typename T>
    void min(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        util::reduce<true, T, T>("math::min", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                                 output, noa::math::copy_t{}, nullptr, noa::math::copy_t{}, stream);
    }

    template<typename T>
    void max(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        util::reduce<true, T, T>("math::max", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                                 output, noa::math::copy_t{}, nullptr, noa::math::copy_t{}, stream);
    }

    template<typename T>
    void sum(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        util::reduce<true, T, T>("math::sum", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 output, noa::math::copy_t{}, nullptr, noa::math::copy_t{}, stream);
    }

    template<typename T>
    void mean(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        using real_t = noa::traits::value_type_t<T>;
        const auto inv_count = static_cast<real_t>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v / inv_count; };
        util::reduce<true, T, T>("math::mean", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 output, sum_to_mean_op, nullptr, noa::math::copy_t{}, stream);
    }

    template<int DDOF, typename T, typename U>
    void var(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Get the mean:
        T h_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements() - DDOF);
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true, T, T>("math::mean", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &h_mean, sum_to_mean_op, nullptr, noa::math::copy_t{}, stream);

        // Get the variance:
        stream.synchronize();
        auto transform_op = [h_mean]__device__(T value) -> U {
            if constexpr (noa::traits::is_complex_v<T>) {
                const U distance = noa::math::abs(value - h_mean);
                return distance * distance;
            } else {
                const U distance = value - h_mean;
                return distance * distance;
            }
            return U(0); // unreachable
        };
        auto dist2_to_var = [inv_count]__device__(U v) -> U { return v * inv_count; };
        util::reduce<true, T, U>("math::var", input, uint4_t{stride}, uint4_t{shape},
                                 transform_op, noa::math::plus_t{}, U(0),
                                 output, dist2_to_var, nullptr, noa::math::copy_t{}, stream);
    }

    template<int DDOF, typename T, typename U>
    void std(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Get the mean:
        T h_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements() - DDOF);
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true, T, T>("math::mean", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &h_mean, sum_to_mean_op, nullptr, noa::math::copy_t{}, stream);

        // Get the variance:
        stream.synchronize();
        auto transform_op = [h_mean]__device__(T value) -> U {
            if constexpr (noa::traits::is_complex_v<T>) {
                const U distance = noa::math::abs(value - h_mean);
                return distance * distance;
            } else {
                const U distance = value - h_mean;
                return distance * distance;
            }
            return U(0); // unreachable
        };
        auto dist2_to_std = [inv_count]__device__(U v) -> U { return noa::math::sqrt(v * inv_count); };
        util::reduce<true, T, U>("math::var", input, uint4_t{stride}, uint4_t{shape},
                                 transform_op, noa::math::plus_t{}, U(0),
                                 output, dist2_to_std, nullptr, noa::math::copy_t{}, stream);
    }

    template<int DDOF, typename T, typename U>
    void statistics(const T* input, size4_t stride, size4_t shape,
                    T* output_sum, T* output_mean,
                    U* output_var, U* output_std,
                    Stream& stream) {
        NOA_PROFILE_FUNCTION();

        // Get the sum and mean:
        const U inv_count = U(1) / static_cast<U>(shape.elements());
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true, T, T>("math::mean", input, uint4_t{stride}, uint4_t{shape},
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 output_sum, noa::math::copy_t{}, output_mean, sum_to_mean_op, stream);

        // This is annoying but since we want to reuse the sum for the variance calculation,
        // we have to wait and possibly transfer the result to the host.
        T mean;
        T* h_output_sum = util::hostPointer(output_sum);
        if (h_output_sum) {
            stream.synchronize();
            mean = *h_output_sum;
        } else {
            memory::copy(output_sum, &mean, 1, stream);
            stream.synchronize();
        }
        mean /= static_cast<U>(shape.elements() - DDOF);

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
        util::reduce<true, T, U>("math::var", input, uint4_t{stride}, uint4_t{shape},
                                 transform_op, noa::math::plus_t{}, U(0),
                                 output_var, dist2_to_var, output_std, var_to_std, stream);
    }

    #define NOA_INSTANTIATE_REDUCE_MIN_MAX_(T)                      \
    template void min<T>(const T*, size4_t, size4_t, T*, Stream&);  \
    template void max<T>(const T*, size4_t, size4_t, T*, Stream&)

    NOA_INSTANTIATE_REDUCE_MIN_MAX_(half_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(float);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(double);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(uint16_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(uint32_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(uint64_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(int16_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(int32_t);
    NOA_INSTANTIATE_REDUCE_MIN_MAX_(int64_t);

    #define NOA_INSTANTIATE_REDUCE_SUM_MEAN_(T)                     \
    template void sum<T>(const T*, size4_t, size4_t, T*, Stream&);  \
    template void mean<T>(const T*, size4_t, size4_t, T*, Stream&)

    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(float);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(double);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(uint32_t);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(uint64_t);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(int32_t);
    NOA_INSTANTIATE_REDUCE_SUM_MEAN_(int64_t);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX(T)                       \
    template void sum<T>(const T*, size4_t, size4_t, T*, Stream&);  \
    template void mean<T>(const T*, size4_t, size4_t, T*, Stream&)

    NOA_INSTANTIATE_REDUCE_COMPLEX(cfloat_t);
    NOA_INSTANTIATE_REDUCE_COMPLEX(cdouble_t);

    #define NOA_INSTANTIATE_VAR_(T,U,DDOF)                                  \
    template void var<DDOF,T,U>(const T*, size4_t, size4_t, U*, Stream&);   \
    template void std<DDOF,T,U>(const T*, size4_t, size4_t, U*, Stream&);   \
    template void statistics<DDOF,T,U>(const T*, size4_t, size4_t, T*, T*, U*, U*, Stream&)

    NOA_INSTANTIATE_VAR_(float, float, 0);
    NOA_INSTANTIATE_VAR_(double, double, 0);
    NOA_INSTANTIATE_VAR_(float, float, 1);
    NOA_INSTANTIATE_VAR_(double, double, 1);

    NOA_INSTANTIATE_VAR_(cfloat_t, float, 0);
    NOA_INSTANTIATE_VAR_(cdouble_t, double, 0);
    NOA_INSTANTIATE_VAR_(cfloat_t, float, 1);
    NOA_INSTANTIATE_VAR_(cdouble_t, double, 1);
}
