#include "noa/Math.h"
#include "noa/cpu/math/Reductions.h"

namespace {
    using namespace noa;

    template<class T>
    void defaultMinMaxSum_(const T* input, size_t elements, T* out_min, T* out_max, T* out_sum) {
        const T* end = input + elements;
        T min = *input, max = *input, sum = 0;
        while (input < end) {
            T tmp = *input++;
            min = math::min(tmp, min);
            max = math::max(tmp, max);
            sum += tmp;
        }
        *out_min = min;
        *out_max = max;
        *out_sum = sum;
    }
    template void defaultMinMaxSum_<int>(const int*, size_t, int*, int*, int*);
    template void defaultMinMaxSum_<uint>(const uint*, size_t, uint*, uint*, uint*);

    template<class T>
    void accurateMeanDP_(const T* input, size_t elements, double* out_sum, double* out_mean) {
        double sum = 0.0;
        double c = 0.0;
        const T* end = input + elements;
        while (input < end) {
            auto tmp = static_cast<double>(*input++);
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
    template void accurateMeanDP_<float>(const float*, size_t, double*, double*);
    template void accurateMeanDP_<double>(const double*, size_t, double*, double*);

    template<class T>
    void accurateMeanDP_(const T* input, size_t elements, cdouble_t* out_sum, cdouble_t* out_mean) {
        cdouble_t sum = 0.0;
        cdouble_t c = 0.0;
        const T* end = input + elements;
        while (input < end) {
            auto tmp = static_cast<cdouble_t>(*input++);
            cdouble_t t = sum + tmp;
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
    template void accurateMeanDP_<cfloat_t>(const cfloat_t*, size_t, cdouble_t*, cdouble_t*);
    template void accurateMeanDP_<cdouble_t>(const cdouble_t*, size_t, cdouble_t*, cdouble_t*);

    template<class T>
    void accurateMeanDPAndMinMax_(const T* input, size_t elements,
                                  double* out_sum, double* out_mean, T* out_min, T* out_max) {
        double sum = 0.0;
        double c = 0.0;
        const T* end = input + elements;
        *out_min = *input;
        *out_max = *input;
        while (input < end) {
            *out_min = math::min(*input, *out_min);
            *out_max = math::max(*input, *out_max);
            auto tmp = static_cast<double>(*input++);
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
    template void accurateMeanDPAndMinMax_<float>(const float*, size_t, double*, double*, float*, float*);
    template void accurateMeanDPAndMinMax_<double>(const double*, size_t, double*, double*, double*, double*);
}

namespace noa::math {
    template<typename T>
    void sumMean(const T* inputs, T* output_sums, T* output_means, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + elements * batch;
            if constexpr (noa::traits::is_float_v<T> || noa::traits::is_complex_v<T>) {
                using double_precision = std::conditional_t<noa::traits::is_float_v<T>, double, cdouble_t>;
                double_precision sum, mean;
                accurateMeanDP_(input, elements, &sum, &mean);
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
    template void sumMean<float>(const float*, float*, float*, size_t, uint);
    template void sumMean<double>(const double*, double*, double*, size_t, uint);
    template void sumMean<cfloat_t>(const cfloat_t*, cfloat_t*, cfloat_t*, size_t, uint);
    template void sumMean<cdouble_t>(const cdouble_t*, cdouble_t*, cdouble_t*, size_t, uint);
    template void sumMean<int>(const int*, int*, int*, size_t, uint);
    template void sumMean<uint>(const uint*, uint*, uint*, size_t, uint);

    template<typename T>
    void minMaxSumMean(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                       size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
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
    template void minMaxSumMean<int>(const int*, int*, int*, int*, int*, size_t, uint);
    template void minMaxSumMean<uint>(const uint*, uint*, uint*, uint*, uint*, size_t, uint);
    template void minMaxSumMean<float>(const float*, float*, float*, float*, float*, size_t, uint);
    template void minMaxSumMean<double>(const double*, double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void sumMeanVarianceStddev(const T* inputs, T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                               size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
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
                output_stddevs[batch] = static_cast<T>(math::sqrt(variance));
        }
    }
    template void sumMeanVarianceStddev<float>(const float*, float*, float*, float*, float*, size_t, uint);
    template void sumMeanVarianceStddev<double>(const double*, double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void varianceStddev(const T* inputs, const T* input_means, T* output_variances, T* output_stddevs,
                        size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
            const T* start = inputs + elements * batch;
            const T* end = start + elements;
            double distance, variance = 0.0, mean = static_cast<double>(input_means[batch]);
            while (start < end) {
                distance = static_cast<double>(*start++) - mean;
                variance += distance * distance;
            }
            variance /= static_cast<double>(elements);

            if (output_variances)
                output_variances[batch] = static_cast<T>(variance);
            if (output_stddevs)
                output_stddevs[batch] = static_cast<T>(math::sqrt(variance));
        }
    }
    template void varianceStddev<float>(const float*, const float*, float*, float*, size_t, uint);
    template void varianceStddev<double>(const double*, const double*, double*, double*, size_t, uint);

    template<typename T>
    void statistics(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                    T* output_variances, T* output_stddevs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
            const T* start = inputs + elements * batch;
            const T* end = start + elements;

            double sum, mean;
            accurateMeanDPAndMinMax_(start, elements, &sum, &mean, output_mins + batch, output_maxs + batch);
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
                output_stddevs[batch] = static_cast<T>(math::sqrt(variance));
        }
    }
    template void statistics<float>(const float*, float*, float*, float*, float*, float*, float*, size_t, uint);
    template void statistics<double>(const double*, double*, double*, double*, double*, double*, double*, size_t, uint);

    template<typename T>
    void reduceAdd(const T* inputs, T* outputs, size_t elements, uint vectors, uint batches) {
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
    template void reduceAdd<int>(const int*, int*, size_t, uint, uint);
    template void reduceAdd<uint>(const uint*, uint*, size_t, uint, uint);
    template void reduceAdd<float>(const float*, float*, size_t, uint, uint);
    template void reduceAdd<double>(const double*, double*, size_t, uint, uint);
    template void reduceAdd<cfloat_t>(const cfloat_t*, cfloat_t*, size_t, uint, uint);
    template void reduceAdd<cdouble_t>(const cdouble_t*, cdouble_t*, size_t, uint, uint);

    template<typename T>
    void reduceMean(const T* inputs, T* outputs, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            for (size_t idx = 0; idx < elements; ++idx) {
                T sum = 0;
                for (uint vector = 0; vector < vectors; ++vector)
                    sum += inputs[elements * vectors * batch + elements * vector + idx];
                outputs[elements * batch + idx] = sum / static_cast<noa::traits::value_type_t<T>>(vectors);
            }
        }
    }
    template void reduceMean<int>(const int*, int*, size_t, uint, uint);
    template void reduceMean<uint>(const uint*, uint*, size_t, uint, uint);
    template void reduceMean<float>(const float*, float*, size_t, uint, uint);
    template void reduceMean<double>(const double*, double*, size_t, uint, uint);
    template void reduceMean<cfloat_t>(const cfloat_t*, cfloat_t*, size_t, uint, uint);
    template void reduceMean<cdouble_t>(const cdouble_t*, cdouble_t*, size_t, uint, uint);

    template<typename T, typename U>
    void reduceMeanWeighted(const T* inputs, const U* weights, T* output, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * vectors * batch;
            for (size_t idx{0}; idx < elements; ++idx) {
                T sum = 0;
                U sum_of_weights = 0;
                for (uint vector = 0; vector < vectors; vector++) {
                    U weight = weights[vector * elements + idx];
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
    template void reduceMeanWeighted<int, int>(const int*, const int*, int*, size_t, uint, uint);
    template void reduceMeanWeighted<uint, uint>(const uint*, const uint*, uint*, size_t, uint, uint);
    template void reduceMeanWeighted<float, float>(const float*, const float*, float*, size_t, uint, uint);
    template void reduceMeanWeighted<double, double>(const double*, const double*, double*, size_t, uint, uint);
    template void reduceMeanWeighted<cfloat_t, float>(const cfloat_t*, const float*, cfloat_t*, size_t, uint, uint);
    template void reduceMeanWeighted<cdouble_t, double>(const cdouble_t*, const double*, cdouble_t*, size_t, uint, uint);
}
