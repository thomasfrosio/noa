#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Stream.h"

// Implementation details
namespace Noa::CUDA::Math::Details {
    enum : int { REDUCTION_MIN, REDUCTION_MAX, REDUCTION_SUM };

    template<int REDUCTION, typename T>
    void minOrMax(T* inputs, T* output_values, size_t elements, uint batches, Stream& stream);

    template<int REDUCTION, typename T>
    void minOrMax(T* inputs, size_t pitch_inputs, T* output_values, size3_t shape, uint batches, Stream& stream);
}

namespace Noa::CUDA::Math {
    /**
     * For each batch, returns the minimum value of an input array, i.e. outputs[b] = Math::min(inputs[b]).
     * @see For more tails, see the corresponding function in the CPU backend.
     *
     * @tparam T                (uint), (u)char, float or double.
     * @param[in] inputs        Input arrays. One per batch. Should be at least @a element * @a batches elements.
     * @param[out] output_mins  Minimum values. One per batch. Should be at least @a batches elements.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_IH void min(T* inputs, T* output_mins, size_t elements, uint batches, Stream& stream) {
        Details::minOrMax<Details::REDUCTION_MIN>(inputs, output_mins, elements, batches, stream);
    }

    /**
     * For each batch, returns the minimum value of an input array, i.e. outputs[b] = Math::min(inputs[b]).
     * @see For more tails, see the corresponding function in the CPU backend.
     *
     * @tparam T                (uint), (u)char, float or double.
     * @param[in] inputs        Input arrays. One per batch.
     * @param pitch_inputs      Pitch of @a inputs, in elements.
     * @param[out] output_mins  Minimum values. One per batch. Should be at least @a batches elements.
     * @param shape             Logical {fast, medium, slow} shape of an input array.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_IH void min(T* inputs, size_t pitch_inputs, T* output_mins, size3_t shape, uint batches, Stream& stream) {
        Details::minOrMax<Details::REDUCTION_MIN>(inputs, pitch_inputs, output_mins, shape, batches, stream);
    }

    /// For each batch, returns the maximum value of an input array, i.e. outputs[b] = Math::max(inputs[b]).
    /// @see For more tails, see the corresponding Noa::CUDA::Math::min() function.
    template<typename T>
    NOA_IH void max(T* inputs, T* output_maxs, size_t elements, uint batches, Stream& stream) {
        Details::minOrMax<Details::REDUCTION_MAX>(inputs, output_maxs, elements, batches, stream);
    }

    /// For each batch, returns the maximum value of an input array, i.e. outputs[b] = Math::max(inputs[b]).
    /// @see For more tails, see the corresponding Noa::CUDA::Math::min() function.
    template<typename T>
    NOA_IH void max(T* inputs, size_t pitch_inputs, T* output_maxs, size3_t shape, uint batches, Stream& stream) {
        Details::minOrMax<Details::REDUCTION_MAX>(inputs, pitch_inputs, output_maxs, shape, batches, stream);
    }

    /**
     * For each batch, returns the minimum and maximum value of an input array.
     * @tparam T                (uint), (u)char, float or double.
     * @param[in] inputs        Input arrays. One per batch.
     * @param[out] output_mins  Minimum values. One per batch.
     * @param[out] output_maxs  Maximum values. One per batch.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void minMax(T* inputs, T* output_mins, T* output_maxs, size_t elements, uint batches, Stream& stream);

    /**
     * For each batch, returns the minimum and maximum value of an input array.
     * @tparam T                (uint), (u)char, float or double.
     * @param[in] inputs        Input arrays. One per batch.
     * @param pitch_inputs      Pitch of @a inputs, in elements.
     * @param[out] output_mins  Minimum values. One per batch.
     * @param[out] output_maxs  Maximum values. One per batch.
     * @param shape             Logical {fast, medium, slow} shape of an input array.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void minMax(T* inputs, size_t pitch_inputs, T* output_mins, T* output_maxs,
                         size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, returns the sum and/or average of the elements in the input array.
     * @tparam T                (u)int, float, double, cfloat or cdouble_t.
     * @param[in] inputs        Input arrays. One per batch.
     * @param[out] output_sums  Output minimum values. One per batch. If nullptr, it is ignored.
     * @param[out] output_means Output maximum values. One per batch. If nullptr, it is ignored.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *                          This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void sumMean(T* inputs, T* output_sums, T* output_means,
                          size_t elements, uint batches, Stream& stream);

    /**
     * For each batch, returns the sum and/or average of the elements in the input array.
     * @tparam T                (u)int, float, double, cfloat or cdouble_t.
     * @param[in] inputs        Input arrays. One per batch.
     * @param pitch_inputs      Pitch of @a inputs, in batch.
     * @param[out] output_sums  Output minimum values. One per batch. If nullptr, it is ignored.
     * @param[out] output_means Output maximum values. One per batch. If nullptr, it is ignored.
     * @param shape             Logical {fast, medium, slow} shape of an input array.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *                          This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void sumMean(T* inputs, size_t pitch_inputs, T* output_sums, T* output_means,
                          size3_t shape, uint batches, Stream& stream);

    /// For each batch, returns the sum of an input array. @see Noa::CUDA::sumMean for more details.
    template<typename T>
    NOA_IH void sum(T* inputs, T* output_sums, size_t elements, uint batches, Stream& stream) {
        sumMean<T>(inputs, output_sums, nullptr, elements, batches, stream);
    }

    /// For each batch, returns the sum of an input array with a given pitch. @see Noa::CUDA::sumMean for more details.
    template<typename T>
    NOA_IH void sum(T* inputs, size_t pitch_inputs, T* output_sums, size3_t shape, uint batches, Stream& stream) {
        sumMean<T>(inputs, pitch_inputs, output_sums, nullptr, shape, batches, stream);
    }

    /// For each batch, returns the mean of an input array. @see Noa::CUDA::sumMean for more details.
    template<typename T>
    NOA_IH void mean(T* inputs, T* output_means, size_t elements, uint batches, Stream& stream) {
        sumMean<T>(inputs, nullptr, output_means, elements, batches, stream);
    }

    /// For each batch, returns the mean of an input array with a given pitch. @see Noa::CUDA::sumMean for more details.
    template<typename T>
    NOA_IH void mean(T* inputs, size_t pitch_inputs, T* output_means, size3_t shape, uint batches, Stream& stream) {
        sumMean<T>(inputs, pitch_inputs, nullptr, output_means, shape, batches, stream);
    }

    /**
     * For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
     * @tparam T                (u)int, float or double.
     * @param[in] inputs        Input arrays. One per batch.
     * @param[out] out_mins     Output minimum values. One per batch.
     * @param[out] out_maxs     Output maximum values. One per batch.
     * @param[out] out_sums     Output sum values. One per batch. If nullptr, it is ignored.
     * @param[out] out_means    Output mean values. One per batch. If nullptr, it is ignored.
     * @param elements          Number of elements per batch.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *                          This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void minMaxSumMean(T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size_t elements, uint batches, Stream& stream);

    /**
     * For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
     * @tparam T                (u)int, float or double.
     * @param[in] inputs        Input arrays. One per batch.
     * @param pitch_inputs      Pitch of @a inputs, in elements.
     * @param[out] out_mins     Output minimum values. One per batch.
     * @param[out] out_maxs     Output maximum values. One per batch.
     * @param[out] out_sums     Output sum values. One per batch. If nullptr, it is ignored.
     * @param[out] out_means    Output mean values. One per batch. If nullptr, it is ignored.
     * @param shape             Logical {fast, medium, slow} shape of an input array.
     * @param batches           Number of batches to compute.
     * @param stream            Stream on which to enqueue this function.
     *                          This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void minMaxSumMean(T* inputs, size_t pitch_inputs,
                                T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, returns the sum, mean, variance and stddev of the elements in the input array.
     * @tparam T                    float or double.
     * @param[in] inputs            Input arrays. One per batch.
     * @param[out] output_sums      Output sums. One per batch. If nullptr, it is ignored.
     * @param[out] output_means     Output means. One per batch. If nullptr, it is ignored.
     * @param[out] output_variances Output variances. One per batch. If nullptr, it is ignored.
     * @param[out] output_stddevs   Output stddevs. One per batch. If nullptr, it is ignored.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     * @param stream                Stream on which to enqueue this function.
     *                              This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(T* inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size_t elements, uint batches, Stream& stream);

    /**
     * For each batch, returns the sum, mean, variance and stddev of the elements in the input array.
     * @tparam T                    float or double.
     * @param[in] inputs            Input arrays. One per batch.
     * @param pitch_inputs          Pitch of @a inputs, in elements.
     * @param[out] output_sums      Output sums. One per batch. If nullptr, it is ignored.
     * @param[out] output_means     Output means. One per batch. If nullptr, it is ignored.
     * @param[out] output_variances Output variances. One per batch. If nullptr, it is ignored.
     * @param[out] output_stddevs   Output stddevs. One per batch. If nullptr, it is ignored.
     * @param shape                 Logical {fast, medium, slow} shape of an input array.
     * @param batches               Number of batches to compute.
     * @param stream                Stream on which to enqueue this function.
     *                              This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(T* inputs, size_t pitch_inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, returns the variance and/or stddev of the input array.
     * @tparam T                    float or double.
     * @param[in] inputs            Input arrays. One per batch.
     * @param[out] means            Means used to compute the variance. One per batch.
     * @param[out] output_variances Output variances. One per batch. If nullptr, it is ignored.
     * @param[out] output_stddevs   Output stddevs. One per batch. If nullptr, it is ignored.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     * @param stream                Stream on which to enqueue this function.
     *                              This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void varianceStddev(T* inputs, T* means, T* output_variances, T* output_stddevs,
                                 size_t elements, uint batches, Stream& stream);

    /**
     * For each batch, returns the variance and/or stddev of the input array.
     * @tparam T                    float or double.
     * @param[in] inputs            Input arrays. One per batch.
     * @param pitch_inputs          Pitch of @a inputs, in elements.
     * @param[out] means            Means used to compute the variance. One per batch.
     * @param[out] output_variances Output variances. One per batch. If nullptr, it is ignored.
     * @param[out] output_stddevs   Output stddevs. One per batch. If nullptr, it is ignored.
     * @param shape                 Logical {fast, medium, slow} shape of an input array.
     * @param batches               Number of batches to compute.
     * @param stream                Stream on which to enqueue this function.
     *                              This stream will be synchronized when the function returns.
     */
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
     * For each batch, returns the statistics of the input array, as described in Noa::Stats<T>.
     * @tparam T                    float or double.
     * @param[in] inputs            Input arrays.    One per batch.
     * @param[out] out_mins         Minimum values.  One per batch.
     * @param[out] out_maxs         Maximum values.  One per batch.
     * @param[out] out_sums         Sum values.      One per batch.
     * @param[out] out_means        Mean values.     One per batch.
     * @param[out] out_variances    Variance values. One per batch.
     * @param[out] out_stddevs      Stddev values.   One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches to compute.
     * @param stream                Stream on which to enqueue this function.
     *                              This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void statistics(T* inputs, T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size_t elements, uint batches, Stream& stream);

    /**
     * For each batch, returns the statistics of the input array, as described in Noa::Stats<T>.
     * @tparam T                    float or double.
     * @param[in] inputs            Input arrays.    One per batch.
     * @param pitch_inputs          Pitch of @a inputs, in elements.
     * @param[out] out_mins         Minimum values.  One per batch.
     * @param[out] out_maxs         Maximum values.  One per batch.
     * @param[out] out_sums         Sum values.      One per batch.
     * @param[out] out_means        Mean values.     One per batch.
     * @param[out] out_variances    Variance values. One per batch.
     * @param[out] out_stddevs      Stddev values.   One per batch.
     * @param shape                 Logical {fast, medium, slow} shape of an input array.
     * @param batches               Number of batches to compute.
     * @param stream                Stream on which to enqueue this function.
     *                              This stream will be synchronized when the function returns.
     */
    template<typename T>
    NOA_HOST void statistics(T* inputs, size_t pitch_inputs,
                             T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size3_t shape, uint batches, Stream& stream);

    /**
     * For each batch, computes the sum over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors to reduce. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] outputs  Reduced vectors. One per batch. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to sum over.
     * @param batches       Number of vector sets to reduce independently. Should be less than 65535.
     */
    template<typename T>
    NOA_HOST void reduceAdd(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream);


    /**
     * For each batch, computes the sums over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors to reduce.
     * @param pitch_inputs  Pitch of @a inputs, in elements.
     * @param[out] outputs  Reduced vectors.
     * @param pitch_outputs Pitch of @a outputs, in elements.
     * @param shape         Logical {fast, medium, slow} shape of a vector.
     * @param vectors       Number of vectors to sum over.
     * @param batches       Number of vector sets to reduce independently. Should be less than 65535.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void reduceAdd(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                            size3_t shape, uint vectors, uint batches, Stream& stream);

    /**
     * For each batch, computes the means over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors to reduce. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] outputs  Reduced vectors. One per batch. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to sum over.
     * @param batches       Number of vector sets to reduce independently. Should be less than 65535.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void reduceMean(T* inputs, T* outputs, size_t elements, uint vectors, uint batches, Stream& stream);

    /**
     * For each batch, computes the sums over multiple vectors.
     * @tparam T            Integer or floating-point types.
     * @param[in] inputs    Vectors to reduce. Should contain @a vectors vectors of @a elements elements per batch.
     * @param pitch_inputs  Pitch of @a inputs, in elements.
     * @param[out] outputs  Reduced vectors. One per batch.
     * @param pitch_outputs Pitch of @a outputs, in elements.
     * @param shape         Logical {fast, medium, slow} shape of a vector.
     * @param vectors       Number of vectors to sum over.
     * @param batches       Number of vector sets to reduce independently. Should be less than 65535.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T>
    NOA_HOST void reduceMean(T* inputs, size_t pitch_inputs, T* outputs, size_t pitch_outputs,
                             size3_t shape, uint vectors, uint batches, Stream& stream);

    /**
     * For each batch, computes the averages over multiple vectors with individual weights for all values and vectors.
     * If the sum of the weights for a given element is 0, the resulting average is 0.
     *
     * @tparam T            (u)int, float, double, cfloat_t or cdouble_t.
     * @tparam U            Traits::value_t<@a T>.
     * @param[in] inputs    Vectors. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[in] weights   Weights. Should contain at least @a elements * @a vectors * @a batches elements.
     * @param[out] outputs  Reduced vectors. Should be at least @a elements * @a batches elements.
     * @param elements      Number of elements in a vector.
     * @param vectors       Number of vectors to average over.
     * @param batches       Number of vector sets to reduce independently. Should be less than 65535.
     *
     * @warning This function is asynchronous with respect to the host and may return before completion.
     */
    template<typename T, typename U>
    NOA_HOST void reduceMeanWeighted(T* inputs, U* weights, T* outputs,
                                     size_t elements, uint vectors, uint batches, Stream& stream);

    template<typename T, typename U>
    NOA_HOST void reduceMeanWeighted(T* inputs, size_t pitch_inputs,
                                     U* weights, size_t pitch_weights,
                                     T* output, size_t pitch_output,
                                     size3_t shape, uint vectors, uint batches, Stream& stream);
}
