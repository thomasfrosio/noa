#pragma once

#include <algorithm>
#include <execution>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

namespace Noa::Math::Details {
    template<class T> NOA_HOST void accurateMeanDP(T* input, size_t elements, double* out_sum, double* out_mean);
    template<class T> NOA_HOST void accurateMeanDP(T* input, size_t elements, cdouble_t* out_sum, cdouble_t* out_mean);
    template<class T> NOA_HOST void accurateMeanDPAndMinMax(T* input, size_t elements,
                                                            double* out_sum, double* out_mean,
                                                            T* out_min, T* out_max);

    template<class T>
    NOA_HOST void defaultMinMaxSum(T* input, size_t elements, T* out_min, T* out_max, T* out_sum) {
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
}

namespace Noa::Math {
    /**
     * For each batch, returns the minimum value of an input array, i.e. outputs[b] = Math::min(inputs[b]).
     * @tparam T                Any type with `T operator<(T, T)` defined.
     * @param[in] inputs        Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_mins  Minimum values. One per batch.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     */
    template<typename T>
    NOA_IH void min(T* inputs, T* output_mins, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + batch * elements;
            output_mins[batch] = *std::min_element(std::execution::par_unseq, input, input + elements);
        }
    }

    /**
     * For each batch, returns the maximum value of an input array, i.e. outputs[b] = Math::max(inputs[b]).
     * @tparam T                Any type with `T operator<(T, T)` defined.
     * @param[in] inputs        Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_maxs  Maximum values. One per batch.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     */
    template<typename T>
    NOA_IH void max(T* inputs, T* output_maxs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + batch * elements;
            output_maxs[batch] = *std::max_element(std::execution::par_unseq, input, input + elements);
        }
    }

    /**
     * For each batch, returns the minimum and maximum value of an input array.
     * @tparam T                Any type with `T operator<(T, T)` defined.
     * @param[in] inputs        Input data. Should be at least `@a elements * @a batches` elements.
     * @param[out] output_mins  Minimum values. One per batch.
     * @param[out] output_maxs  Maximum values. One per batch.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     */
    template<typename T>
    NOA_IH void minMax(T* inputs, T* output_mins, T* output_maxs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            T* input = inputs + batch * elements;
            auto[it_min, it_max] = std::minmax_element(std::execution::par_unseq, input, input + elements);
            output_mins[batch] = *it_min;
            output_maxs[batch] = *it_max;
        }
    }

    /**
     * For each batch, returns the sum and/or average of the elements in the input array.
     * @tparam T                Any type with `T operator+(T, T)` defined.
     * @param[in] inputs        Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] output_sums  Output sums. One per batch. If nullptr, ignore it.
     * @param[out] output_means Output means. One per batch. If nullptr, ignore it.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     *
     * @details For floating-point types, the Neumaier variation of the Kahan summation algorithm is used. It also uses
     *          double precision for accumulation and compensation. For complex types, the Kahan summation algorithm
     *          is used. Note that -ffast-math optimizes it away.
     */
    template<typename T>
    NOA_HOST void sumMean(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches) {
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

    /// For each batch, returns the sum of the elements in the input array. @see sumMean for more details.
    template<typename T>
    NOA_IH void sum(T* inputs, T* output_sums, size_t elements, uint batches) {
        sumMean<T>(inputs, output_sums, nullptr, elements, batches);
    }

    /// For each batch, returns the mean of the elements in the input array. @see sumMean for more details.
    template<typename T>
    NOA_IH void mean(T* inputs, T* output_means, size_t elements, uint batches) {
        sumMean<T>(inputs, nullptr, output_means, elements, batches);
    }

    /**
     * For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
     * @tparam T                    Any type with the basic arithmetic and comparison operators defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] out_mins         Output minimum values. One per batch.
     * @param[out] out_maxs         Output maximum values. One per batch.
     * @param[out] out_sums         Output sum values. One per batch. If nullptr, ignore it.
     * @param[out] out_means        Output mean values. One per batch. If nullptr, ignore it.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void minMaxSumMean(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
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
                Details::defaultMinMaxSum(input, elements, output_min, output_max, sum);
                if (output_sums)
                    output_sums[batch] = sum;
                if (output_means)
                    output_means[batch] = static_cast<T>(sum / static_cast<T>(elements));
            }
        }
    }

    /**
     * For each batch, returns the sum, mean, variance and stddev of the elements in the input array.
     * @tparam T                        Float or double.
     * @param[in] inputs                Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] output_sums          Output sums. One per batch. If nullptr, ignore it.
     * @param[out] output_means         Output means. One per batch. If nullptr, ignore it.
     * @param[out] output_variances     Output variances. One per batch. If nullptr, ignore it.
     * @param[out] output_stddevs       Output stddevs. One per batch. If nullptr, ignore it.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(T* inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size_t elements, uint batches) {
        static_assert(Noa::Traits::is_float_v<T>);
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

    /**
     * For each batch, returns the variance and/or stddev of the elements in the input array.
     * @tparam T                        Float or double.
     * @param[in] inputs                Array of least @a elements * @a batches elements.
     * @param[in] means                 Means used to compute the variance. One per batch.
     * @param[out] output_variances     Output variances. One per batch. If nullptr, ignore it.
     * @param[out] output_stddevs       Output stddevs. One per batch. If nullptr, ignore it.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void varianceStddev(T* inputs, T* means, T* output_variances, T* output_stddevs,
                                 size_t elements, uint batches) {
        static_assert(Noa::Traits::is_float_v<T>);
        NOA_PROFILE_FUNCTION();

        for (uint batch = 0; batch < batches; ++batch) {
            T* start = inputs + elements * batch;
            T* end = start + elements;
            double distance, variance = 0.0, mean = static_cast<double>(means[batch]);
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

    /// For each batch, returns the variance of the elements in the input array.
    /// @see varianceStddev for more details.
    template<typename T>
    NOA_IH void variance(T* inputs, T* means, T* output_variances, size_t elements, uint batches) {
        varianceStddev<T>(inputs, means, output_variances, nullptr, elements, batches);
    }

    /// For each batch, returns the stddev of the elements in the input array.
    /// @see varianceStddev for more details.
    template<typename T>
    NOA_IH void stddev(T* inputs, T* means, T* output_stddevs, size_t elements, uint batches) {
        varianceStddev<T>(inputs, means, nullptr, output_stddevs, elements, batches);
    }

    /// For each batch, returns the variance of the elements in the input array.
    /// @see sumMeanVarianceStddev for more details.
    template<typename T>
    NOA_IH void variance(T* inputs, T* output_variances, size_t elements, uint batches) {
        sumMeanVarianceStddev<T>(inputs, nullptr, nullptr, output_variances, nullptr, elements, batches);
    }

    /// For each batch, returns the stddev of the elements in the input array.
    /// @see sumMeanVarianceStddev for more details.
    template<typename T>
    NOA_IH void stddev(T* inputs, T* output_stddevs, size_t elements, uint batches) {
        sumMeanVarianceStddev<T>(inputs, nullptr, nullptr, nullptr, output_stddevs, elements, batches);
    }

    /**
     * For each batch, returns the full statistics of the elements of the input array, as described in Stats<T>.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] out_mins         Output minimum values. One per batch.
     * @param[out] out_maxs         Output maximum values. One per batch.
     * @param[out] out_sums         Output sum values. One per batch. If nullptr, ignore it.
     * @param[out] out_means        Output mean values. One per batch. If nullptr, ignore it.
     * @param[out] out_variances    Output variance values. One per batch. If nullptr, ignore it.
     * @param[out] out_stddevs      Output stddev values. One per batch. If nullptr, ignore it.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_IH void statistics(T* inputs,
                           T* output_mins, T* output_maxs,
                           T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                           size_t elements, uint batches) {
        static_assert(Noa::Traits::is_float_v<T>);
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

    /**
     * For each batch, computes the sum over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] outputs  Reduced vectors. One per batch. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to sum over.
     * @param batches       Number of vector sets to reduce independently.
     */
    template<typename T>
    NOA_IH void reduceAdd(T* inputs, T* outputs, size_t elements, uint vectors, uint batches) {
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

    /**
     * For each batch, computes the average over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] outputs  Reduced vectors. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to average over.
     * @param batches       Number of vector sets to reduce independently.
     */
    template<typename T>
    NOA_IH void reduceMean(T* inputs, T* outputs, size_t elements, uint vectors, uint batches) {
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

    /**
     * For each batch, computes the averages over multiple vectors with individual weights for all values and vectors.
     * If the sum of the weights for a given element is 0, the resulting average is 0.
     *
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[in] weights   Weights. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] output   Reduced vectors. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to average over.
     * @param batches       Number of vector sets to reduce independently.
     */
    template<typename T>
    NOA_HOST void reduceMeanWeighted(T* inputs, T* weights, T* output, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * vectors * batch;
            for (size_t idx{0}; idx < elements; ++idx) {
                T sum = 0;
                T sum_of_weights = 0;
                for (uint vector = 0; vector < vectors; vector++) {
                    T weight = weights[batch_offset + vector * elements + idx];
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
}
