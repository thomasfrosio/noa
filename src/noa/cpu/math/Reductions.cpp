#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

#include "noa/cpu/math/Reductions.h"
#include "noa/cpu/memory/Set.h"

namespace {
    using namespace noa;

    template<typename T>
    void accurateMeanDP_(const T* input, size4_t stride, size4_t shape, T* out_sums, size_t threads) {
        const size_t elements_per_batch = shape[1] * shape[2] * shape[3];
        for (size_t batch = 0; batch < shape[0]; ++batch) {
            double sum = 0, err = 0;

            #pragma omp parallel for if (elements_per_batch > 100000) \
            collapse(3) num_threads(threads) reduction(+:sum, err) default(none) \
            shared(input, stride, shape, batch)

            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        auto tmp = static_cast<double>(input[at(batch, j, k, l, stride)]);
                        auto t = sum + tmp;
                        if (noa::math::abs(sum) >= noa::math::abs(tmp)) // Neumaier variation
                            err += (sum - t) + tmp;
                        else
                            err += (tmp - t) + sum;
                        sum = t;
                    }
                }
            }
            out_sums[batch] = static_cast<T>(sum + err);
        }
    }

    template<typename T>
    void accurateMeanDP_(const noa::Complex<T>* input, size4_t stride, size4_t shape,
                         noa::Complex<T>* out_sums, size_t threads) {
        const size_t elements_per_batch = shape[1] * shape[2] * shape[3];
        for (size_t batch = 0; batch < shape[0]; ++batch) {
            double sum_real = 0, sum_imag = 0, err_real = 0, err_imag = 0;

            #pragma omp parallel for if (elements_per_batch > 100000) \
            collapse(3) num_threads(threads) reduction(+:sum_real, sum_imag, err_real, err_imag) \
            default(none) shared(input, stride, shape, batch)

            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        auto tmp = static_cast<cdouble_t>(input[at(batch, j, k, l, stride)]);

                        auto t_real = sum_real + tmp.real;
                        if (noa::math::abs(sum_real) >= noa::math::abs(tmp.real)) // Neumaier variation
                            err_real += (sum_real - t_real) + tmp.real;
                        else
                            err_real += (tmp.real - t_real) + sum_real;
                        sum_real = t_real;

                        auto t_imag = sum_imag + tmp.imag;
                        if (noa::math::abs(sum_imag) >= noa::math::abs(tmp.imag)) // Neumaier variation
                            err_imag += (sum_imag - t_imag) + tmp.imag;
                        else
                            err_imag += (tmp.imag - t_imag) + sum_imag;
                        sum_imag = t_imag;
                    }
                }
            }
            out_sums[batch].real = static_cast<T>(sum_real + err_real);
            out_sums[batch].imag = static_cast<T>(sum_imag + err_imag);
        }
    }

    template<typename T>
    void variance_(const T* input, size4_t stride, size4_t shape, T* means_variances, size_t threads) {
        const auto count = static_cast<double>(shape[1] * shape[2] * shape[3]);
        for (size_t batch = 0; batch < shape[0]; ++batch) {
            const auto mean = static_cast<double>(means_variances[batch]);
            double variance = 0;

            #pragma omp parallel for if (count > 100000) \
            collapse(3) num_threads(threads) reduction(+:variance) default(none) \
            shared(input, stride, shape, batch, mean)

            for (size_t j = 0; j < shape[1]; ++j) {
                for (size_t k = 0; k < shape[2]; ++k) {
                    for (size_t l = 0; l < shape[3]; ++l) {
                        const auto tmp = static_cast<double>(input[at(batch, j, k, l, stride)]);
                        const auto distance = tmp - mean;
                        variance += distance * distance;
                    }
                }
            }
            means_variances[batch] = static_cast<T>(variance / count);
        }
    }
}

namespace noa::cpu::math {
    template<typename T>
    void sum(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>) {
            accurateMeanDP_(input, stride, shape, outputs, stream.threads());
        } else {
            reduce(input, stride, shape, outputs, noa::math::plus_t{}, T(0), stream);
        }
    }

    template<typename T>
    void var(const T* input, size4_t input_stride, size4_t shape, T* outputs, Stream& stream) {
        stream.enqueue([=, &stream]() {
            mean(input, input_stride, shape, outputs, stream);
            variance_(input, input_stride, shape, outputs, stream.threads());
        });
    }
    template void var<float>(const float*, size4_t, size4_t, float*, Stream&);
    template void var<double>(const double*, size4_t, size4_t, double*, Stream&);

    #define NOA_INSTANTIATE_ALL_TYPES_(T) \
    template void sum<T>(const T*, size4_t, size4_t, T*, Stream&)

    NOA_INSTANTIATE_ALL_TYPES_(int);
    NOA_INSTANTIATE_ALL_TYPES_(long);
    NOA_INSTANTIATE_ALL_TYPES_(long long);
    NOA_INSTANTIATE_ALL_TYPES_(unsigned int);
    NOA_INSTANTIATE_ALL_TYPES_(unsigned long);
    NOA_INSTANTIATE_ALL_TYPES_(unsigned long long);
    NOA_INSTANTIATE_ALL_TYPES_(float);
    NOA_INSTANTIATE_ALL_TYPES_(double);
    NOA_INSTANTIATE_ALL_TYPES_(cfloat_t);
    NOA_INSTANTIATE_ALL_TYPES_(cdouble_t);
}
