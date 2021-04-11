#include "noa/cpu/math/Reductions.h"

namespace Noa::Math::Details {
    template<class T>
    void defaultMinMaxSum(T* input, size_t elements, T* out_min, T* out_max, T* out_sum) {
        T* end = input + elements;
        T min = *input, max = *input, sum = 0;
        while (input < end) {
            T tmp = *input++;
            min = Math::min(tmp, min);
            max = Math::max(tmp, max);
            sum += tmp;
        }
        *out_min = min;
        *out_max = max;
        *out_sum = sum;
    }
    template void defaultMinMaxSum<int>(int*, size_t, int*, int*, int*);
    template void defaultMinMaxSum<uint>(uint*, size_t, uint*, uint*, uint*);

    template<class T>
    void accurateMeanDP(T* input, size_t elements, double* out_sum, double* out_mean) {
        double sum = 0.0;
        double c = 0.0;
        T* end = input + elements;
        while (input < end) {
            auto tmp = static_cast<double>(*input++);
            double t = sum + tmp;
            if (Math::abs(sum) >= Math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
    template void accurateMeanDP<float>(float*, size_t, double*, double*);
    template void accurateMeanDP<double>(double*, size_t, double*, double*);

    template<class T>
    void accurateMeanDP(T* input, size_t elements, cdouble_t* out_sum, cdouble_t* out_mean) {
        cdouble_t sum = 0.0;
        cdouble_t c = 0.0;
        T* end = input + elements;
        while (input < end) {
            auto tmp = static_cast<cdouble_t>(*input++);
            cdouble_t t = sum + tmp;
            if (Math::abs(sum) >= Math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
    template void accurateMeanDP<cfloat_t>(cfloat_t*, size_t, cdouble_t*, cdouble_t*);
    template void accurateMeanDP<cdouble_t>(cdouble_t*, size_t, cdouble_t*, cdouble_t*);

    template<class T>
    void accurateMeanDPAndMinMax(T* input, size_t elements, double* out_sum, double* out_mean, T* out_min, T* out_max) {
        double sum = 0.0;
        double c = 0.0;
        T* end = input + elements;
        *out_min = *input;
        *out_max = *input;
        while (input < end) {
            *out_min = Math::min(*input, *out_min);
            *out_max = Math::max(*input, *out_max);
            auto tmp = static_cast<double>(*input++);
            double t = sum + tmp;
            if (Math::abs(sum) >= Math::abs(tmp)) // Neumaier variation
                c += (sum - t) + tmp;
            else
                c += (tmp - t) + sum;
            sum = t;
        }
        sum += c;
        *out_sum = sum;
        *out_mean = sum / static_cast<double>(elements);
    }
    template void accurateMeanDPAndMinMax<float>(float*, size_t, double*, double*, float*, float*);
    template void accurateMeanDPAndMinMax<double>(double*, size_t, double*, double*, double*, double*);
}

namespace Noa::Math {
    template<typename T>
    void sumMean(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + elements * batch;
            if constexpr (Noa::Traits::is_float_v<T> || Noa::Traits::is_complex_v<T>) {
                using double_precision = std::conditional_t<Noa::Traits::is_float_v<T>, double, cdouble_t>;
                double_precision sum, mean;
                Details::accurateMeanDP(input, elements, &sum, &mean);
                if (output_sums)
                    output_sums[batch] = static_cast<T>(sum);
                if (output_means)
                    output_means[batch] = static_cast<T>(sum / static_cast<double>(elements));
            } else {
                T sum = std::reduce(std::execution::par_unseq, input, input + elements);
                if (output_sums)
                    output_sums[batch] = sum;
                if (output_means)
                    output_means[batch] = sum / static_cast<T>(elements);
            }
        }
    }
    template void sumMean<float>(float*, float*, float*, size_t, uint);
    template void sumMean<double>(double*, double*, double*, size_t, uint);
    template void sumMean<cfloat_t>(cfloat_t*, cfloat_t*, cfloat_t*, size_t, uint);
    template void sumMean<cdouble_t>(cdouble_t*, cdouble_t*, cdouble_t*, size_t, uint);
    template void sumMean<int>(int*, int*, int*, size_t, uint);
    template void sumMean<uint>(uint*, uint*, uint*, size_t, uint);

    template<typename T>
    void minMaxSumMean(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + batch * elements;
            T* output_min = output_mins + batch;
            T* output_max = output_maxs + batch;

            if constexpr (Noa::Traits::is_float_v<T>) {
                double sum, mean;
                Details::accurateMeanDPAndMinMax(input, elements, &sum, &mean, output_min, output_max);
                if (output_sums)
                    output_sums[batch] = static_cast<T>(sum);
                if (output_means)
                    output_means[batch] = static_cast<T>(mean);
            } else {
                T sum;
                Details::defaultMinMaxSum(input, elements, output_min, output_max, &sum);
                if (output_sums)
                    output_sums[batch] = sum;
                if (output_means)
                    output_means[batch] = static_cast<T>(sum / static_cast<T>(elements));
            }
        }
    }
    template void minMaxSumMean<int>(int*, int*, int*, int*, int*, size_t, uint);
    template void minMaxSumMean<uint>(uint*, uint*, uint*, uint*, uint*, size_t, uint);
    template void minMaxSumMean<float>(float*, float*, float*, float*, float*, size_t, uint);
    template void minMaxSumMean<double>(double*, double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void sumMeanVarianceStddev(T* inputs, T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                               size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
            T* start = inputs + elements * batch;
            T* end = start + elements;

            double sum, mean;
            Details::accurateMeanDP(start, elements, &sum, &mean);
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
                output_stddevs[batch] = static_cast<T>(Math::sqrt(variance));
        }
    }
    template void sumMeanVarianceStddev<float>(float*, float*, float*, float*, float*, size_t, uint);
    template void sumMeanVarianceStddev<double>(double*, double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void varianceStddev(T* inputs, T* input_means, T* output_variances, T* output_stddevs,
                        size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
            T* start = inputs + elements * batch;
            T* end = start + elements;
            double distance, variance = 0.0, mean = static_cast<double>(input_means[batch]);
            while (start < end) {
                distance = static_cast<double>(*start++) - mean;
                variance += distance * distance;
            }
            variance /= static_cast<double>(elements);

            if (output_variances)
                output_variances[batch] = static_cast<T>(variance);
            if (output_stddevs)
                output_stddevs[batch] = static_cast<T>(Math::sqrt(variance));
        }
    }
    template void varianceStddev<float>(float*, float*, float*, float*, size_t, uint);
    template void varianceStddev<double>(double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void statistics(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means, T* output_variances,
                    T* output_stddevs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
            T* start = inputs + elements * batch;
            T* end = start + elements;

            double sum, mean;
            Details::accurateMeanDPAndMinMax(start, elements, &sum, &mean, output_mins + batch, output_maxs + batch);
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
                output_stddevs[batch] = static_cast<T>(Math::sqrt(variance));
        }
    }
    template void statistics<float>(float*, float*, float*, float*, float*, float*, float*, size_t, uint);
    template void statistics<double>(double*, double*, double*, double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void reduceAdd(T* inputs, T* outputs, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            for (size_t idx = 0; idx < elements; ++idx) {
                T sum = 0;
                for (uint vector = 0; vector < vectors; ++vector)
                    sum += inputs[elements * vectors * batch + elements * vector + idx];
                outputs[elements * batch + idx] = sum;
            }
        }
    }
    template void reduceAdd<int>(int*, int*, size_t, uint, uint);
    template void reduceAdd<uint>(uint*, uint*, size_t, uint, uint);
    template void reduceAdd<float>(float*, float*, size_t, uint, uint);
    template void reduceAdd<double>(double*, double*, size_t, uint, uint);
    template void reduceAdd<cfloat_t>(cfloat_t*, cfloat_t*, size_t, uint, uint);
    template void reduceAdd<cdouble_t>(cdouble_t*, cdouble_t*, size_t, uint, uint);

    template<typename T>
    void reduceMean(T* inputs, T* outputs, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            for (size_t idx = 0; idx < elements; ++idx) {
                T sum = 0;
                for (uint vector = 0; vector < vectors; ++vector)
                    sum += inputs[elements * vectors * batch + elements * vector + idx];
                outputs[elements * batch + idx] = sum / static_cast<Noa::Traits::value_type_t<T>>(vectors);
            }
        }
    }
    template void reduceMean<int>(int*, int*, size_t, uint, uint);
    template void reduceMean<uint>(uint*, uint*, size_t, uint, uint);
    template void reduceMean<float>(float*, float*, size_t, uint, uint);
    template void reduceMean<double>(double*, double*, size_t, uint, uint);
    template void reduceMean<cfloat_t>(cfloat_t*, cfloat_t*, size_t, uint, uint);
    template void reduceMean<cdouble_t>(cdouble_t*, cdouble_t*, size_t, uint, uint);

    template<typename T, typename U>
    void reduceMeanWeighted(T* inputs, U* weights, T* output, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * vectors * batch;
            for (size_t idx{0}; idx < elements; ++idx) {
                T sum = 0;
                U sum_of_weights = 0;
                for (uint vector = 0; vector < vectors; vector++) {
                    U weight = weights[batch_offset + vector * elements + idx];
                    sum_of_weights += weight;
                    sum += inputs[batch_offset + vector * elements + idx] * weight;
                }
                if (sum_of_weights != 0)
                    output[elements * batch + idx] = sum / sum_of_weights;
                else
                    output[elements * batch + idx] = 0;
            }
        }
    }
    template void reduceMeanWeighted<int, int>(int*, int*, int*, size_t, uint, uint);
    template void reduceMeanWeighted<uint, uint>(uint*, uint*, uint*, size_t, uint, uint);
    template void reduceMeanWeighted<float, float>(float*, float*, float*, size_t, uint, uint);
    template void reduceMeanWeighted<double, double>(double*, double*, double*, size_t, uint, uint);
    template void reduceMeanWeighted<cfloat_t, float>(cfloat_t*, float*, cfloat_t*, size_t, uint, uint);
    template void reduceMeanWeighted<cdouble_t, double>(cdouble_t*, double*, cdouble_t*, size_t, uint, uint);
}
