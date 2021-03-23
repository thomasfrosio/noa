#pragma once

#include <algorithm>
#include <execution>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/util/Profiler.h"

#include "noa/cpu/math/ArithmeticsComposite.h" // squaredDistanceFromValue to calculate the variance.

namespace Noa::Math {
    /* -------------------------- */
    /* --- "Batched" versions --- */
    /* -------------------------- */

    /**
     * For each batch, returns the minimum value of an input array, i.e. outputs[b] = min(inputs[b]).
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_min_values    Minimum values. One per batch. Should be at least @a batches elements.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_IH void min(T* inputs, T* output_min_values, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch)
            output_min_values[batch] = *std::min_element(
                    std::execution::par_unseq, inputs + batch * elements, inputs + batch * elements + elements);
    }

    /**
     * For each batch, returns the maximum value of an input array, i.e. outputs[b] = max(inputs[b]).
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_max_values    Maximum values. One per batch. Should be at least @a batches elements.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_IH void max(T* inputs, T* output_max_values, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch)
            output_max_values[batch] = *std::max_element(
                    std::execution::par_unseq, inputs + batch * elements, inputs + batch * elements + elements);
    }

    /**
     * For each batch, returns the minimum and maximum value of an input array:
     * output_min_values[b] = min(inputs[b]);
     * output_max_values[b] = max(inputs[b]);
     *
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input data. Should be at least `@a elements * @a batches` elements.
     * @param[out] output_min_values    Output minimum values. One per batch. Should be at least @a batches elements.
     * @param[out] output_max_values    Output maximum values. One per batch. Should be at least @a batches elements.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_IH void minMax(T* inputs, T* output_min_values, T* output_max_values, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch) {
            auto[it_min, it_max] = std::minmax_element(
                    std::execution::par_unseq, inputs + elements * batch, inputs + elements * batch + elements);
            output_min_values[batch] = *it_min;
            output_max_values[batch] = *it_max;
        }
    }

    /**
     * For each batch, returns the sum of the elements in the input array.
     * @tparam T                Any type with `T operator+(T, T)` defined.
     * @param[in] inputs        Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] output_sums  Output sums. One per batch. Should be at least @a batches elements.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     */
    template<typename T>
    NOA_IH void sum(T* inputs, T* output_sums, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch)
            output_sums[batch] = std::reduce(std::execution::par_unseq,
                                             inputs + elements * batch, inputs + elements * batch + elements);
    }

    /**
     * For each batch, returns the minimum, maximum and sum of the elements of the input array.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] out_mins         Output minimum values. One per batch.
     * @param[out] out_maxs         Output maximum values. One per batch.
     * @param[out] out_sums         Output sum values. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_IH void minMaxSum(T* inputs, T* out_mins, T* out_maxs, T* out_sums, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch) {
            out_sums[batch] = sum(inputs, elements);
            std::tie(out_mins[batch], out_maxs[batch]) = minMax(inputs, elements);
        }
    }

    /**
     * For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] out_mins         Output minimum values. One per batch.
     * @param[out] out_maxs         Output maximum values. One per batch.
     * @param[out] out_sums         Output sum values. One per batch.
     * @param[out] out_means        Output mean values. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_IH void minMaxSumMean(T* inputs, T* out_mins, T* out_maxs, T* out_sums, T* out_means,
                              size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch) {
            out_sums[batch] = sum(inputs, elements);
            out_means[batch] = out_sums[batch] / static_cast<Noa::Traits::value_type_t<T>>(elements);
            std::tie(out_mins[batch], out_maxs[batch]) = minMax(inputs, elements);
        }
    }

    /**
     * For each batch, returns the mean of the elements in the input array.
     * @tparam T                    Any type with `T operator+(T, T)` defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] output_means     Output means. One per batch. Should be at least @a batches elements.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_IH void mean(T* inputs, T* output_means, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        sum(inputs, output_means, elements);
        for (uint batch = 0; batch < batches; ++batch)
            output_means[batch] /= static_cast<Noa::Traits::value_type_t<T>>(elements);
    }

    /**
     * For each batch, returns the mean of the elements in the input array.
     * @tparam T                    Any type with `T operator+(T, T)` defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] output_means     Output means. One per batch. Should be at least @a batches elements.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_IH void variance(T* inputs, T* means, T* output_variances, T* output_stddevs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        squaredDistanceFromValue(inputs, means, output_variances, elements, batches);
        for (uint batch = 0; batch < batches; ++batch) {
            output_variances[batch] /= static_cast<Noa::Traits::value_type_t<T>>(elements);
            output_stddevs[batch] = Math::sqrt(output_variances[batch]);
        }
    }

    /**
     * For each batch, returns the full statistics of the elements of the input array, as described in Stats<T>.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] out_mins         Output minimum values. One per batch.
     * @param[out] out_maxs         Output maximum values. One per batch.
     * @param[out] out_sums         Output sum values. One per batch.
     * @param[out] out_means        Output mean values. One per batch.
     * @param[out] out_variances    Output variance values. One per batch.
     * @param[out] out_stddevs      Output stddev values. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_IH void statistics(T* inputs, T* out_mins, T* out_maxs, T* out_sums, T* out_means, T* out_variances,
                           T* out_stddevs, size_t elements, uint batches) {
        minMaxSumMean(inputs, out_mins, out_maxs, out_sums, out_means, elements, batches);
        variance(inputs, out_means, out_variances, out_stddevs, elements, batches);
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
        NOA_PROFILE_FUNCTION("cpu,arith");
        size_t batch_offset;
        T sum;
        for (uint batch = 0; batch < batches; ++batch) {
            batch_offset = elements * vectors * batches;
            for (size_t idx = 0; idx < elements; ++elements) {
                sum = 0;
                for (uint vector = 0; vector < vectors; ++vector)
                    sum += inputs[batch_offset + elements * vector + idx];
                outputs[batch_offset + idx] = sum;
            }
        }
    }

    /**
     * For each batch, computes the average over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] output   Reduced vectors. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to average over.
     * @param batches       Number of vector sets to reduce independently.
     */
    template<typename T>
    NOA_IH void reduceMean(T* inputs, T* output, size_t elements, uint vectors, uint batches) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        size_t batch_offset;
        T sum;
        for (uint batch = 0; batch < batches; ++batch) {
            batch_offset = elements * vectors * batches;
            for (size_t idx = 0; idx < elements; ++elements) {
                sum = 0;
                for (uint vector = 0; vector < vectors; ++vector)
                    sum += inputs[batch_offset + elements * vector + idx];
                output[batch_offset + idx] = sum / static_cast<Noa::Traits::value_type_t<T>>(vectors);
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
        NOA_PROFILE_FUNCTION("cpu,arith");
        for (uint batch = 0; batch < batches; ++batch) {
            size_t batch_offset = elements * vectors * batches;
            for (size_t idx{0}; idx < elements; ++elements) {
                T sum = 0;
                T sum_of_weights = 0;
                for (uint vector = 0; vector < vectors; vector++) {
                    T weight = weights[batch_offset + vector * elements + idx];
                    sum_of_weights += weight;
                    sum += inputs[batch_offset + vector * elements + idx] * weight;
                }
                if (sum_of_weights != 0)
                    output[batch_offset + idx] = sum / sum_of_weights;
                else
                    output[batch_offset + idx] = 0;
            }
        }
    }

    /* ---------------------------- */
    /* --- "One array" versions --- */
    /* ---------------------------- */

    /**
     * Returns the minimum value of the input array.
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input array with at least @a element elements.
     * @param[out] output_min_value     Output minimum value.
     * @param elements                  Number of elements.
     */
    template<typename T>
    NOA_IH void min(T* input, T* output_min_value, size_t elements) {
        min(input, output_min_value, elements, 1);
    }

    /**
     * Returns the maximum value of the input array.
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input array with at least @a element elements.
     * @param[out] output_max_value     Output maximum value.
     * @param elements                  Number of elements.
     */
    template<typename T>
    NOA_IH void max(T* input, T* output_max_value, size_t elements) {
        max(input, output_max_value, elements, 1);
    }

    /**
     * Returns the minimum and maximum value of the input array.
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] input                 Input array with at least @a element elements.
     * @param[out] output_min_value     Output minimum value.
     * @param[out] output_max_value     Output maximum value.
     * @param elements                  Number of elements.
     */
    template<typename T>
    NOA_IH void minMax(T* input, T* output_min_value, T* output_max_value, size_t elements) {
        minMax(input, output_min_value, output_max_value, elements, 1);
    }

    /**
     * Returns the sum of the elements in the input array.
     * @tparam T                Any type with `T operator+(T, T)` defined.
     * @param[in] input         Input array with at least @a element elements.
     * @param[out] output_sum   Output sum.
     * @param elements          Number of elements.
     */
    template<typename T>
    NOA_IH void sum(T* input, T* output_sum, size_t elements) {
        sum(input, output_sum, elements, 1);
    }

    /**
     * Returns the minimum, maximum and sum of the elements of the input array.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] input             Input array with at least @a element elements.
     * @param[out] out_min          Output minimum values.
     * @param[out] out_max          Output maximum values.
     * @param[out] out_sum          Output sum values.
     * @param elements              Number of elements.
     */
    template<typename T>
    NOA_IH void minMaxSum(T* input, T* out_min, T* out_max, T* out_sum, size_t elements) {
        minMaxSum(input, out_min, out_max, out_sum, elements, 1);
    }

    /**
     * Returns the minimum, maximum, sum and the mean of the elements of the input array.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] input             Input array with at least @a element elements.
     * @param[out] out_min          Output minimum values.
     * @param[out] out_max          Output maximum values.
     * @param[out] out_sum          Output sum values.
     * @param[out] out_mean         Output mean values.
     * @param elements              Number of elements.
     */
    template<typename T>
    NOA_IH void minMaxSumMean(T* input, T* out_min, T* out_max, T* out_sum, T* out_mean, size_t elements) {
        minMaxSumMean(input, out_min, out_max, out_sum, out_mean, elements, 1);
    }

    /**
     * Returns the mean of the elements in the input array.
     * @tparam T                    Any type with `T operator+(T, T)` defined.
     * @param[in] input             Input array with at least @a element elements.
     * @param[out] output_mean      Output mean.
     * @param elements              Number of elements.
     */
    template<typename T>
    NOA_IH void mean(T* input, T* output_mean, size_t elements) {
        mean(input, output_mean, elements, 1);
    }

    /**
     * Returns the mean of the elements in the input array.
     * @tparam T                    Any type with `T operator+(T, T)` defined.
     * @param[in] input             Input array with at least @a element elements.
     * @param[out] output_mean      Output mean.
     * @param elements              Number of elements.
     */
    template<typename T>
    NOA_IH void variance(T* input, T* mean, T* output_variance, T* output_stddev, size_t elements) {
        variance(input, mean, output_variance, output_stddev, elements, 1);
    }

    /**
     * Returns the full statistics of the elements of the input array, as described in Stats<T>.
     * @tparam T                    Any type with the basic arithmetic operators defined.
     * @param[in] inputs            Input array with at least @a element elements.
     * @param[out] out_min          Output minimum value.
     * @param[out] out_max          Output maximum value.
     * @param[out] out_sum          Output sum value.
     * @param[out] out_mean         Output mean value.
     * @param[out] out_variance     Output variance value.
     * @param[out] out_stddev       Output stddev value.
     * @param elements              Number of elements.
     */
    template<typename T>
    NOA_IH void statistics(T* input, T* out_min, T* out_max, T* out_sum, T* out_mean, T* out_variance,
                           T* out_stddev, size_t elements) {
        statistics(input, out_min, out_max, out_sum, out_mean, out_mean, out_variance, out_stddev, elements, 1);
    }

    /**
     * Computes the sum over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors to reduce into one single vector. Should contain at least @a elements * @a vectors elements.
     * @param[out] output   Reduced vector with at least @a elements elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to sum over.
     */
    template<typename T>
    NOA_IH void reduceAdd(T* inputs, T* output, size_t elements, uint vectors) {
        reduceAdd(inputs, output, elements, vectors, 1);
    }

    /**
     * Computes the average over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors to reduce into one single vector. Should contain at least @a elements * @a vectors elements.
     * @param[out] output   Reduced vector with at least @a elements elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to average over.
     */
    template<typename T>
    NOA_IH void reduceMean(T* inputs, T* output, size_t elements, uint vectors) {
        reduceMean(inputs, output, elements, vectors, 1);
    }

    /**
     * Computes the average over multiple vectors with individual weights for all values and vectors.
     * If the sum of the weights for a given element is 0, the resulting average is 0.
     *
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors. Should contain at least @a elements * @a vectors elements.
     * @param[in] weights   Weights. Should contain at least @a elements * @a vectors elements.
     * @param[out] output   Reduced vector with at least @a elements elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to average over.
     */
    template<typename T>
    NOA_IH void reduceMeanWeighted(T* inputs, T* weights, T* output, size_t elements, uint vectors) {
        reduceMeanWeighted(inputs, weights, output, elements, vectors, 1);
    }
}
