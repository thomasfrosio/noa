#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/math/Reductions.h"

namespace {
    using namespace noa;

    template<typename T>
    void accurateMeanDP_(const T* inputs, size3_t input_pitch, size3_t shape, T* out_sum,
                         size_t batches, size_t threads) {
        const size_t offset = elements(input_pitch);
        for (size_t batch = 0; batch < batches; ++batch) {
            double sum = 0, err = 0;

            if (elements(shape) < 100000)
                threads = 1;
            #pragma omp parallel for collapse(3) num_threads(threads) reduction(+:sum, err) default(none) \
            shared(inputs, input_pitch, shape, batch, offset)

            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        auto tmp = static_cast<double>(inputs[offset * batch + index(x, y, z, input_pitch)]);
                        auto t = sum + tmp;
                        if (noa::math::abs(sum) >= noa::math::abs(tmp)) // Neumaier variation
                            err += (sum - t) + tmp;
                        else
                            err += (tmp - t) + sum;
                        sum = t;
                    }
                }
            }
            out_sum[batch] = static_cast<T>(sum + err);
        }
    }

    template<typename T>
    void accurateMeanDP_(const noa::Complex<T>* inputs, size3_t input_pitch, size3_t shape, noa::Complex<T>* out_sum,
                         size_t batches, size_t threads) {
        const size_t offset = elements(input_pitch);
        for (size_t batch = 0; batch < batches; ++batch) {
            double sum_real = 0, sum_imag = 0, err_real = 0, err_imag = 0;

            if (elements(shape) < 100000)
                threads = 1;
            #pragma omp parallel for collapse(3) num_threads(threads) reduction(+:sum_real, sum_imag, err_real, err_imag) \
            default(none) shared(inputs, input_pitch, shape, batch, offset)

            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        auto tmp = static_cast<cdouble_t>(inputs[offset * batch + index(x, y, z, input_pitch)]);

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
            out_sum[batch].real = static_cast<T>(sum_real + err_real);
            out_sum[batch].imag = static_cast<T>(sum_imag + err_imag);
        }
    }

    template<typename T>
    void variance_(const T* inputs, size3_t input_pitch, size3_t shape, T* means_variances,
                   size_t batches, size_t threads) {
        const size_t offset = elements(input_pitch);
        const auto count = static_cast<double>(elements(shape));
        for (size_t batch = 0; batch < batches; ++batch) {
            const auto mean = static_cast<double>(means_variances[batch]);
            double variance = 0;

            if (elements(shape) < 100000)
                threads = 1;
            #pragma omp parallel for collapse(3) num_threads(threads) reduction(+:variance) default(none) \
            shared(inputs, input_pitch, shape, batch, offset, mean)
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {
                        const auto tmp = static_cast<double>(inputs[offset * batch + index(x, y, z, input_pitch)]);
                        const auto distance = tmp - mean;
                        variance += distance * distance;
                    }
                }
            }
            means_variances[batch] = static_cast<T>(variance / count);
        }
    }

    template<bool DO_MEAN, typename T>
    void reduceAddMean_(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                        size_t nb_to_reduce, size_t batches, size_t threads) {
        NOA_PROFILE_FUNCTION();
        const size_t iffset = elements(input_pitch);
        const size_t offset = elements(output_pitch);
        using value_t = noa::traits::value_type_t<T>;
        [[maybe_unused]] const auto weight = static_cast<value_t>(nb_to_reduce);

        if (elements(shape) < 100000)
            threads = 1;
        #pragma omp parallel for collapse(4) num_threads(threads) default(none) \
            shared(inputs, input_pitch, outputs, output_pitch, shape, nb_to_reduce, batches, iffset, offset, weight)

        for (size_t batch = 0; batch < batches; ++batch) {
            for (size_t z = 0; z < shape.z; ++z) {
                for (size_t y = 0; y < shape.y; ++y) {
                    for (size_t x = 0; x < shape.x; ++x) {

                        const T* input = inputs + iffset * nb_to_reduce * batch + index(x, y, z, input_pitch);
                        T* output = outputs + offset * batch + index(x, y, z, output_pitch);

                        T sum = 0;
                        for (size_t count = 0; count < nb_to_reduce; ++count)
                            sum += input[iffset * count];
                        if constexpr (DO_MEAN)
                            *output = sum / weight;
                        else
                            *output = sum;
                    }
                }
            }
        }
    }
}

namespace noa::cpu::math {
    template<typename T>
    void sum(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        if constexpr (noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>) {
            accurateMeanDP_(inputs, input_pitch, shape, outputs, batches, stream.threads());
        } else {
            reduce(inputs, input_pitch, shape, outputs, batches, noa::math::plus_t{}, T(0), stream);
        }
    }

    template<typename T>
    void var(const T* inputs, size3_t input_pitch, size3_t shape,
             T* outputs, size_t batches, Stream& stream) {
        stream.enqueue([=, &stream]() {
            mean(inputs, input_pitch, shape, outputs, batches, stream);
            variance_(inputs, input_pitch, shape, outputs, batches, stream.threads());
        });
    }

    template<typename T>
    void reduceAdd(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                   size_t nb_to_reduce, size_t batches, Stream& stream) {
        stream.enqueue(reduceAddMean_<false, T>, inputs, input_pitch, outputs, output_pitch,
                       shape, nb_to_reduce, batches, stream.threads());
    }

    template<typename T>
    void reduceMean(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                    size_t nb_to_reduce, size_t batches, Stream& stream) {
        stream.enqueue(reduceAddMean_<true, T>, inputs, input_pitch, outputs, output_pitch,
                       shape, nb_to_reduce, batches, stream.threads());
    }

    template<typename T, typename U>
    void reduceMeanWeighted(const T* inputs, size3_t input_pitch, const U* weights, size3_t weight_pitch,
                            T* outputs, size3_t output_pitch, size3_t shape,
                            size_t nb_to_reduce, size_t batches, Stream& stream) {
        const size_t sthreads = stream.threads();
        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            const size_t iffset = elements(input_pitch);
            const size_t wffset = elements(weight_pitch);
            const size_t offset = elements(output_pitch);
            const size_t threads = elements(shape) < 100000 ? 1 : sthreads;

            #pragma omp parallel for collapse(4) num_threads(threads) default(none) \
            shared(inputs, input_pitch, weights, weight_pitch, outputs, output_pitch, \
                   shape, nb_to_reduce, batches, iffset, wffset, offset)

            for (size_t batch = 0; batch < batches; ++batch) {
                for (size_t z = 0; z < shape.z; ++z) {
                    for (size_t y = 0; y < shape.y; ++y) {
                        for (size_t x = 0; x < shape.x; ++x) {

                            const T* input = inputs + iffset * nb_to_reduce * batch + index(x, y, z, input_pitch);
                            const U* weight = weights + index(x, y, z, weight_pitch);
                            T* output = outputs + offset * batch + index(x, y, z, output_pitch);

                            T sum = 0;
                            U sum_of_weights = 0;
                            for (size_t count = 0; count < nb_to_reduce; ++count) {
                                const U w = weight[wffset * count];
                                sum_of_weights += w;
                                sum += input[offset * count] * w;
                            }
                            if (sum_of_weights != 0)
                                *output = sum / sum_of_weights;
                            else
                                *output = 0;
                        }
                    }
                }
            }
        });
    }
}

namespace noa::cpu::math {
    #define NOA_INSTANTIATE_ALL_TYPES_(T)                                   \
    template void sum<T>(const T*, size3_t, size3_t, T*, size_t, Stream&);  \
    template void reduceAdd<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, size_t, Stream&);       \
    template void reduceMean<T>(const T*, size3_t, T*, size3_t, size3_t, size_t, size_t, Stream&)

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

    template void var<float>(const float*, size3_t, size3_t, float*, size_t, Stream&);
    template void var<double>(const double*, size3_t, size3_t, double*, size_t, Stream&);

    #define NOA_INSTANTIATE_ALL_INT_FLOAT_(T)                                   \
    template void reduceMeanWeighted<T, T>(const T*, size3_t, const T*, size3_t, T*, size3_t, size3_t, size_t, size_t, Stream&)

    NOA_INSTANTIATE_ALL_INT_FLOAT_(int);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(long long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(unsigned int);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(unsigned long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(unsigned long long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(float);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(double);

    template void reduceMeanWeighted<cfloat_t, float>(const cfloat_t*, size3_t, const float*, size3_t,
                                                      cfloat_t*, size3_t, size3_t, size_t, size_t, Stream&);
    template void reduceMeanWeighted<cdouble_t, double>(const cdouble_t*, size3_t, const double*, size3_t,
                                                        cdouble_t*, size3_t, size3_t, size_t, size_t, Stream&);
}
