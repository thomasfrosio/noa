#include <numeric>

#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/cpu/math/Reductions.h"

namespace {
    using namespace noa;

    template<typename T>
    void defaultMinMaxSum_(const T* input, size_t elements, T* out_min, T* out_max, T* out_sum) {
        T min = *input, max = *input, sum = *input;
        for (size_t i = 1; i < elements; ++i) {
            T tmp = input[i];
            min = math::min(tmp, min);
            max = math::max(tmp, max);
            sum += tmp;
        }
        *out_min = min;
        *out_max = max;
        *out_sum = sum;
    }

    template<typename T>
    void accurateMeanDP_(const T* input, size_t elements, double* out_sum, double* out_mean) {
        double sum = 0.0;
        double c = 0.0;
        for (size_t i = 0; i < elements; ++i) {
            auto tmp = static_cast<double>(input[i]);
            double t = sum + tmp;
            if (math::abs(sum) >= math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }

    template<typename T>
    void accurateMeanDP_(const T* input, size_t elements, cdouble_t* out_sum, cdouble_t* out_mean) {
        cdouble_t sum = 0.0;
        double& sum_real = sum[0];
        double& sum_imag = sum[1];
        cdouble_t c = 0.0;
        double& c_real = c[0];
        double& c_imag = c[1];

        for (size_t i = 0; i < elements; ++i) {
            auto tmp = static_cast<cdouble_t>(input[i]);
            cdouble_t t = sum + tmp;

            if (math::abs(sum_real) >= math::abs(tmp[0])) // Neumaier variation
                c_real += (sum_real - t[0]) + tmp[0];
            else
                c_real += (tmp[0] - t[0]) + sum_real;

            if (math::abs(sum_imag) >= math::abs(tmp[1])) // Neumaier variation
                c_imag += (sum_imag - t[1]) + tmp[1];
            else
                c_imag += (tmp[1] - t[1]) + sum_imag;

            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }

    template<typename T>
    void accurateMeanDPAndMinMax_(const T* input, size_t elements,
                                  double* out_sum, double* out_mean, T* out_min, T* out_max) {
        double sum = 0.0;
        double c = 0.0;
        *out_min = *input;
        *out_max = *input;
        for (size_t i = 0; i < elements; ++i) {
            *out_min = math::min(input[i], *out_min);
            *out_max = math::max(input[i], *out_max);
            auto tmp = static_cast<double>(input[i]);
            double t = sum + tmp;
            if (math::abs(sum) >= math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
}

namespace noa::cpu::math {
    template<typename T>
    void sumMean(const T* inputs, T* output_sums, T* output_means, size_t elements, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            if constexpr (noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>) {
                using double_precision = std::conditional_t<noa::traits::is_float_v<T>, double, cdouble_t>;
                double_precision sum, mean;
                accurateMeanDP_(input, elements, &sum, &mean);
                if (output_sums)
                    output_sums[batch] = static_cast<T>(sum);
                if (output_means)
                    output_means[batch] = static_cast<T>(mean);
            } else {
                T sum = std::reduce(input, input + elements);
                if (output_sums)
                    output_sums[batch] = sum;
                if (output_means)
                    output_means[batch] = sum / static_cast<T>(elements);
            }
        }
    }

    template<typename T>
    void minMaxSumMean(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size_t elements, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements;
            T* output_min = output_mins + batch;
            T* output_max = output_maxs + batch;

            if constexpr (noa::traits::is_float_v<T>) {
                double sum, mean;
                accurateMeanDPAndMinMax_(input, elements, &sum, &mean, output_min, output_max);
                if (output_sums)
                    output_sums[batch] = static_cast<T>(sum);
                if (output_means)
                    output_means[batch] = static_cast<T>(mean);
            } else {
                T sum;
                defaultMinMaxSum_(input, elements, output_min, output_max, &sum);
                if (output_sums)
                    output_sums[batch] = sum;
                if (output_means)
                    output_means[batch] = static_cast<T>(sum / static_cast<T>(elements));
            }
        }
    }

    template<typename T>
    void sumMeanVarianceStddev(const T* inputs, T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                               size_t elements, size_t batches) {
        NOA_PROFILE_FUNCTION();

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* start = inputs + elements * batch;
            const T* end = start + elements;

            double sum, mean;
            accurateMeanDP_(start, elements, &sum, &mean);
            if (output_sums)
                output_sums[batch] = static_cast<T>(sum);
            if (output_means)
                output_means[batch] = static_cast<T>(mean);

            double distance, variance = 0.0;
            while (start < end) {
                distance = static_cast<double>(*start++) - mean;
                variance += distance * distance;
            }
            variance /= static_cast<double>(elements);

            if (output_variances)
                output_variances[batch] = static_cast<T>(variance);
            if (output_stddevs)
                output_stddevs[batch] = static_cast<T>(noa::math::sqrt(variance));
        }
    }

    template<typename T>
    void varianceStddev(const T* inputs, const T* input_means, T* output_variances, T* output_stddevs,
                        size_t elements, size_t batches) {
        NOA_PROFILE_FUNCTION();

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* start = inputs + elements * batch;

            double distance, variance = 0.0, mean = static_cast<double>(input_means[batch]);
            for (size_t i = 0; i < elements; ++i) {
                distance = static_cast<double>(start[i]) - mean;
                variance += distance * distance;
            }
            variance /= static_cast<double>(elements);

            if (output_variances)
                output_variances[batch] = static_cast<T>(variance);
            if (output_stddevs)
                output_stddevs[batch] = static_cast<T>(noa::math::sqrt(variance));
        }
    }

    template<typename T>
    void statistics(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                    T* output_variances, T* output_stddevs, size_t elements, size_t batches) {
        NOA_PROFILE_FUNCTION();

        for (size_t batch = 0; batch < batches; ++batch) {
            const T* start = inputs + elements * batch;

            double sum, mean;
            accurateMeanDPAndMinMax_(start, elements, &sum, &mean, output_mins + batch, output_maxs + batch);
            if (output_sums)
                output_sums[batch] = static_cast<T>(sum);
            if (output_means)
                output_means[batch] = static_cast<T>(mean);

            double distance, variance = 0.0;
            for (size_t i = 0; i < elements; ++i) {
                distance = static_cast<double>(start[i]) - mean;
                variance += distance * distance;
            }
            variance /= static_cast<double>(elements);

            if (output_variances)
                output_variances[batch] = static_cast<T>(variance);
            if (output_stddevs)
                output_stddevs[batch] = static_cast<T>(noa::math::sqrt(variance));
        }
    }

    template<typename T>
    void reduceAdd(const T* inputs, T* outputs, size_t elements, size_t nb_to_reduce, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = elements * batch;
            for (size_t idx = 0; idx < elements; ++idx) {
                T sum = 0;
                for (size_t vector = 0; vector < nb_to_reduce; ++vector)
                    sum += inputs[offset * nb_to_reduce + elements * vector + idx];
                outputs[offset + idx] = sum;
            }
        }
    }

    template<typename T>
    void reduceMean(const T* inputs, T* outputs, size_t elements, size_t nb_to_reduce, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = elements * batch;
            for (size_t idx = 0; idx < elements; ++idx) {
                T sum = 0;
                for (size_t vector = 0; vector < nb_to_reduce; ++vector)
                    sum += inputs[offset * nb_to_reduce + elements * vector + idx];
                outputs[offset + idx] = sum / static_cast<noa::traits::value_type_t<T>>(nb_to_reduce);
            }
        }
    }

    template<typename T, typename U>
    void reduceMeanWeighted(const T* inputs, const U* weights, T* outputs, size_t elements,
                            size_t nb_to_reduce, size_t batches) {
        NOA_PROFILE_FUNCTION();
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * nb_to_reduce * batch;
            for (size_t idx = 0; idx < elements; ++idx) {
                T sum = 0;
                U sum_of_weights = 0;
                for (size_t vector = 0; vector < nb_to_reduce; vector++) {
                    U weight = weights[vector * elements + idx];
                    sum_of_weights += weight;
                    sum += inputs[batch_offset + vector * elements + idx] * weight;
                }
                if (sum_of_weights != 0)
                    outputs[elements * batch + idx] = sum / sum_of_weights;
                else
                    outputs[elements * batch + idx] = 0;
            }
        }
    }
}

namespace noa::cpu::math {
    #define NOA_INSTANTIATE_ALL_TYPES_(T)                               \
    template void sumMean<T>(const T*, T*, T*, size_t, size_t);         \
    template void reduceAdd<T>(const T*, T*, size_t, size_t, size_t);   \
    template void reduceMean<T>(const T*, T*, size_t, size_t, size_t)

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

    #define NOA_INSTANTIATE_ALL_INT_FLOAT_(T)                                   \
    template void minMaxSumMean<T>(const T*, T*, T*, T*, T*, size_t, size_t);   \
    template void reduceMeanWeighted<T, T>(const T*, const T*, T*, size_t, size_t, size_t)

    NOA_INSTANTIATE_ALL_INT_FLOAT_(int);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(long long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(unsigned int);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(unsigned long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(unsigned long long);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(float);
    NOA_INSTANTIATE_ALL_INT_FLOAT_(double);

    template void reduceMeanWeighted<cfloat_t, float>(const cfloat_t*, const float*, cfloat_t*, size_t, size_t, size_t);
    template void reduceMeanWeighted<cdouble_t, double>(const cdouble_t*, const double*, cdouble_t*, size_t, size_t, size_t);

    #define NOA_INSTANTIATE_ALL_FLOAT_(T)                                               \
    template void sumMeanVarianceStddev<T>(const T*, T*, T*, T*, T*, size_t, size_t);   \
    template void varianceStddev<T>(const T*, const T*, T*, T*, size_t, size_t);        \
    template void statistics<T>(const T*, T*, T*, T*, T*, T*, T*, size_t, size_t)

    NOA_INSTANTIATE_ALL_FLOAT_(float);
    NOA_INSTANTIATE_ALL_FLOAT_(double);
}
