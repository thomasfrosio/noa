#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace Noa::CUDA::Math {
    /**
     * For each batch, returns the minimum value of an input array, i.e. outputs[b] = min(inputs[b]).
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_min_values    Minimum values. One per batch. Should be at least @a batches elements.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void min(T* inputs, T* output_min_values, size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void min(T* inputs, size_t pitch_inputs, T* output_min_values,
                      size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, returns the maximum value of an input array, i.e. outputs[b] = max(inputs[b]).
     * @tparam T                        Any type with `T operator<(T, T)` defined.
     * @param[in] inputs                Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_max_values    Maximum values. One per batch. Should be at least @a batches elements.
     * @param elements                  Number of elements per batch.
     * @param batches                   Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void max(T* inputs, T* output_max_values, size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void max(T* inputs, size_t pitch_inputs, T* output_min_values,
                      size3_t shape, uint batches, Stream& stream);

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
    NOA_HOST void minMax(T* inputs, T* output_min_values, T* output_max_values,
                         size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void minMax(T* inputs, size_t pitch_inputs, T* output_min_values, T* output_max_values,
                         size3_t shape, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void sumMean(T* inputs, T* output_sums, T* output_means, size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void sumMean(T* inputs, size_t pitch_inputs, T* output_sums, T* output_means, size3_t shape, uint batches);

    template<typename T>
    NOA_IH void sum(T* inputs, T* output_sums, size_t elements, uint batches, Stream& stream) {
        sumMean<T>(inputs, output_sums, nullptr, elements, batches, stream);
    }

    template<typename T>
    NOA_IH void sum(T* inputs, size_t pitch_inputs, T* output_sums, size3_t shape, uint batches, Stream& stream) {
        sumMean<T>(inputs, pitch_inputs, output_sums, nullptr, shape, batches, stream);
    }

    template<typename T>
    NOA_IH void mean(T* inputs, T* output_means, size_t elements, uint batches, Stream& stream) {
        sumMean<T>(inputs, nullptr, output_means, elements, batches, stream);
    }

    template<typename T>
    NOA_IH void mean(T* inputs, size_t pitch_inputs, T* output_means, size3_t shape, uint batches, Stream& stream) {
        sumMean<T>(inputs, pitch_inputs, nullptr, output_means, shape, batches, stream);
    }

    template<typename T>
    NOA_HOST void minMaxSumMean(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void minMaxSumMean(T* inputs, size_t pitch_inputs,
                                T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, returns the mean of the elements in the input array.
     * @tparam T                    Any type with `T operator+(T, T)` defined.
     * @param[in] inputs            Input arrays. Should be at least @a elements * @a batches elements.
     * @param[out] output_means     Output means. One per batch. Should be at least @a batches elements.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(T* inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(T* inputs, size_t pitch_inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size3_t shape, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void varianceStddev(T* inputs, T* means, T* output_variances, T* output_stddevs,
                                 size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void varianceStddev(T* inputs, size_t pitch_inputs, T* means, T* output_variances, T* output_stddevs,
                                 size3_t shape, uint batches, Stream& stream);

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
     * @param[out] out_sums         Output sum values. One per batch.
     * @param[out] out_means        Output mean values. One per batch.
     * @param[out] out_variances    Output variance values. One per batch.
     * @param[out] out_stddevs      Output stddev values. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     */
    template<typename T>
    NOA_HOST void statistics(T* inputs, T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size_t elements, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void statistics(T* inputs, size_t pitch_inputs,
                             T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size3_t shape, uint batches, Stream& stream);

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
    NOA_HOST void reduceAdd(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void reduceAdd(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                            size3_t shape, uint vectors, uint batches, Stream& stream);

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
    NOA_HOST void reduceMean(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void reduceMean(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                             size3_t shape, uint vectors, uint batches, Stream& stream);

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
    NOA_HOST void reduceMeanWeighted(T* inputs, T* weights, T* output,
                                     size_t elements, uint vectors, uint batches, Stream& stream);

    template<typename T>
    NOA_HOST void reduceMeanWeighted(T* inputs, size_t pitch_inputs,
                                     T* weights, size_t pitch_weights,
                                     T* output, size_t pitch_output,
                                     size3_t shape, uint vectors, uint batches, Stream& stream);
}
