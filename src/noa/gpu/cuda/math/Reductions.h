/// \file noa/cpu/math/Reductions.h
/// \brief Reduction operations for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// Implementation details
namespace noa::cuda::math::details {
    enum : int { REDUCTION_MIN, REDUCTION_MAX, REDUCTION_SUM };

    template<int REDUCTION, typename T>
    void minOrMax(const T* inputs, T* output_values, size_t elements, size_t batches, Stream& stream);

    template<int REDUCTION, typename T>
    void minOrMax(const T* inputs, size_t input_pitch, T* output_values, size3_t shape, size_t batches, Stream& stream);
}

namespace noa::cuda::math {
    /// For each batch, returns the minimum value of an input array.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float or double.
    /// \param[in] inputs       On the \b device. Contiguous input arrays. One per batch.
    /// \param input_pitch      Pitch of \p inputs, in elements.
    /// \param[out] output_mins On the \b device. Minimum values. One value per batch.
    /// \param elements         Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches          Number of contiguous batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         This stream will be synchronized when the function returns.
    template<typename T>
    NOA_IH void min(const T* inputs, size_t input_pitch, T* output_mins,
                    size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::minOrMax<details::REDUCTION_MIN>(inputs, input_pitch, output_mins, shape, batches, stream);
    }

    /// For each batch, returns the minimum value of an input array. Version for contiguous layouts.
    template<typename T>
    NOA_IH void min(const T* inputs, T* output_mins, size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::minOrMax<details::REDUCTION_MIN>(inputs, output_mins, elements, batches, stream);
    }

    /// For each batch, returns the maximum value of an input array.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float or double.
    /// \param[in] inputs       On the \b device. Contiguous input arrays. One per batch.
    /// \param input_pitch      Pitch of \p inputs, in elements.
    /// \param[out] output_maxs On the \b device. Maximum values. One value per batch.
    /// \param elements         Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches          Number of contiguous batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         This stream will be synchronized when the function returns.
    template<typename T>
    NOA_IH void max(const T* inputs, size_t input_pitch, T* output_maxs,
                    size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::minOrMax<details::REDUCTION_MAX>(inputs, input_pitch, output_maxs, shape, batches, stream);
    }

    /// For each batch, returns the maximum value of an input array. Version for contiguous layouts.
    template<typename T>
    NOA_IH void max(const T* inputs, T* output_maxs, size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::minOrMax<details::REDUCTION_MAX>(inputs, output_maxs, elements, batches, stream);
    }

    /// For each batch, returns the minimum and maximum value of an input array.On the \b device.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float or double.
    /// \param[in] inputs       On the \b device. Input arrays. One per batch.
    /// \param[out] output_mins On the \b device. Minimum values. One value per batch.
    /// \param[out] output_maxs On the \b device. Maximum values. One value per batch.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         This stream will be synchronized when the function returns.
    template<typename T>
    NOA_HOST void minMax(const T* inputs, size_t input_pitch, T* output_mins, T* output_maxs,
                         size3_t shape, size_t batches, Stream& stream);

    /// For each batch, returns the minimum and maximum value of an input array. Version for contiguous layouts.
    template<typename T>
    NOA_HOST void minMax(const T* inputs, T* output_mins, T* output_maxs,
                         size_t elements, size_t batches, Stream& stream);

    /// For each batch, returns the sum and/or average of the elements in the input array.
    /// \tparam T                   (u)int, (u)long, (u)long long, float, double, cfloat_t or cdouble_t.
    /// \param[in] inputs           On the \b device. Input arrays. One per batch.
    /// \param input_pitch          Pitch of \p inputs, in batch.
    /// \param[out] output_sums     On the \b device. Output minimum values. One per batch. If nullptr, it is ignored.
    /// \param[out] output_means    On the \b device. Output maximum values. One per batch. If nullptr, it is ignored.
    /// \param shape                Logical {fast, medium, slow} shape of an input array.
    /// \param batches              Number of batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             This stream will be synchronized when the function returns.
    template<typename T>
    NOA_HOST void sumMean(const T* inputs, size_t input_pitch, T* output_sums, T* output_means,
                          size3_t shape, size_t batches, Stream& stream);

    /// For each batch, returns the sum and/or average of the elements in the input array. Version for contiguous layouts.
    template<typename T>
    NOA_HOST void sumMean(const T* inputs, T* output_sums, T* output_means,
                          size_t elements, size_t batches, Stream& stream);

    /// For each batch, returns the sum of the elements in \p inputs.
    /// \see sumMean for more details.
    template<typename T>
    NOA_IH void sum(const T* inputs, size_t input_pitch, T* output_sums,
                    size3_t shape, size_t batches, Stream& stream) {
        sumMean<T>(inputs, input_pitch, output_sums, nullptr, shape, batches, stream);
    }

    /// For each batch, returns the sum of the elements in \p inputs. Version for contiguous layouts.
    /// \see sumMean for more details.
    template<typename T>
    NOA_IH void sum(const T* inputs, T* output_sums, size_t elements, size_t batches, Stream& stream) {
        sumMean<T>(inputs, output_sums, nullptr, elements, batches, stream);
    }

    /// For each batch, returns the mean of the elements in \p inputs.
    /// \see sumMean for more details.
    template<typename T>
    NOA_IH void mean(const T* inputs, size_t input_pitch, T* output_means,
                     size3_t shape, size_t batches, Stream& stream) {
        sumMean<T>(inputs, input_pitch, nullptr, output_means, shape, batches, stream);
    }

    /// For each batch, returns the mean of the elements in \p inputs. Version for contiguous layouts.
    /// \see sumMean for more details.
    template<typename T>
    NOA_IH void mean(const T* inputs, T* output_means, size_t elements, size_t batches, Stream& stream) {
        sumMean<T>(inputs, nullptr, output_means, elements, batches, stream);
    }

    /// For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b device. Input arrays. One per batch.
    /// \param input_pitch      Pitch of \p inputs, in elements.
    /// \param[out] out_mins    On the \b device. Output minimum values. One per batch.
    /// \param[out] out_maxs    On the \b device. Output maximum values. One per batch.
    /// \param[out] out_sums    On the \b device. Output sum values.     One per batch. If nullptr, ignore it.
    /// \param[out] out_means   On the \b device. Output mean values.    One per batch. If nullptr, ignore it.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches          Number of batches to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         This stream will be synchronized when the function returns.
    template<typename T>
    NOA_HOST void minMaxSumMean(const T* inputs, size_t input_pitch,
                                T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size3_t shape, size_t batches, Stream& stream);

    /// For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
    /// Version for contiguous layouts.
    template<typename T>
    NOA_HOST void minMaxSumMean(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size_t elements, size_t batches, Stream& stream);

    /// For each batch, returns the sum, mean, variance and stddev of the elements in the input array.
    /// \tparam T                       float or double.
    /// \param[in] inputs               On the \b device. Input arrays. One per batch.
    /// \param input_pitch              Pitch of \p inputs, in elements.
    /// \param[out] output_sums         On the \b device. Output sums.      One per batch. If nullptr, ignore it.
    /// \param[out] output_means        On the \b device. Output means.     One per batch. If nullptr, ignore it.
    /// \param[out] output_variances    On the \b device. Output variances. One per batch. If nullptr, ignore it.
    /// \param[out] output_stddevs      On the \b device. Output stddevs.   One per batch. If nullptr, ignore it.
    /// \param shape                    Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches                  Number of batches to compute.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    ///                                 This stream will be synchronized when the function returns.
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(const T* inputs, size_t input_pitch,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size3_t shape, size_t batches, Stream& stream);

    /// For each batch, returns the sum, mean, variance and stddev of the elements in the input array.
    /// Version for contiguous layouts.
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(const T* inputs,
                                        T* output_sums, T* output_means, T* output_variances, T* output_stddevs,
                                        size_t elements, size_t batches, Stream& stream);

    /// For each batch, returns the variance and/or stddev of the input array.
    /// \tparam T                       float or double.
    /// \param[in] inputs               On the \b device. Input arrays. One per batch.
    /// \param input_pitch              Pitch of \p inputs, in elements.
    /// \param[in] means                On the \b device. Means used to compute the variance. One value per batch.
    /// \param[out] output_variances    On the \b device. Output variances. One per batch. If nullptr, ignore it.
    /// \param[out] output_stddevs      On the \b device. Output stddevs. One per batch. If nullptr, ignore it.
    /// \param shape                    Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches                  Number of batches to compute.
    /// \param[in,out] stream           Stream on which to enqueue this function.
    ///                                 This stream will be synchronized when the function returns.
    template<typename T>
    NOA_HOST void varianceStddev(const T* inputs, size_t input_pitch, const T* means,
                                 T* output_variances, T* output_stddevs,
                                 size3_t shape, size_t batches, Stream& stream);

    /// For each batch, returns the variance and/or stddev of the input array. Version for contiguous layouts.
    template<typename T>
    NOA_HOST void varianceStddev(const T* inputs, const T* means, T* output_variances, T* output_stddevs,
                                 size_t elements, size_t batches, Stream& stream);

    /// For each batch, returns the variance of the elements in the input array. Version for contiguous layouts.
    /// \see varianceStddev for more details.
    template<typename T>
    NOA_IH void variance(const T* inputs, const T* means, T* output_variances, size_t elements, size_t batches) {
        varianceStddev<T>(inputs, means, output_variances, nullptr, elements, batches);
    }

    /// For each batch, returns the stddev of the elements in the input array. Version for contiguous layouts.
    /// \see varianceStddev for more details.
    template<typename T>
    NOA_IH void stddev(const T* inputs, const T* means, T* output_stddevs, size_t elements, size_t batches) {
        varianceStddev<T>(inputs, means, nullptr, output_stddevs, elements, batches);
    }

    /// For each batch, returns the variance of the elements in the input array. Version for contiguous layouts.
    /// \see sumMeanVarianceStddev for more details.
    template<typename T>
    NOA_IH void variance(const T* inputs, T* output_variances, size_t elements, size_t batches) {
        sumMeanVarianceStddev<T>(inputs, nullptr, nullptr, output_variances, nullptr, elements, batches);
    }

    /// For each batch, returns the stddev of the elements in the input array. Version for contiguous layouts.
    /// \see sumMeanVarianceStddev for more details.
    template<typename T>
    NOA_IH void stddev(const T* inputs, T* output_stddevs, size_t elements, size_t batches) {
        sumMeanVarianceStddev<T>(inputs, nullptr, nullptr, nullptr, output_stddevs, elements, batches);
    }

    /// For each batch, returns the statistics of the input array, as described in noa::Stats<T>.
    /// \tparam T                   float or double.
    /// \param[in] inputs           On the \b device. Input arrays. One per batch.
    /// \param input_pitch          Pitch of \p inputs, in elements.
    /// \param[out] out_mins        On the \b device. Output minimum values.  One value per batch.
    /// \param[out] out_maxs        On the \b device. Output maximum values.  One value per batch.
    /// \param[out] out_sums        On the \b device. Output sum values.      One value per batch. If nullptr, ignore it.
    /// \param[out] out_means       On the \b device. Output mean values.     One value per batch. If nullptr, ignore it.
    /// \param[out] out_variances   On the \b device. Output variance values. One value per batch. If nullptr, ignore it.
    /// \param[out] out_stddevs     On the \b device. Output stddev values.   One value per batch. If nullptr, ignore it.
    /// \param shape                Logical {fast, medium, slow} shape of \p inputs, ignoring the batches.
    /// \param batches              Number of batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             This stream will be synchronized when the function returns.
    template<typename T>
    NOA_HOST void statistics(const T* inputs, size_t input_pitch,
                             T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size3_t shape, size_t batches, Stream& stream);

    /// For each batch, returns the statistics of the input array, as described in noa::Stats<T>.
    /// Version for contiguous layouts.
    template<typename T>
    NOA_HOST void statistics(const T* inputs, T* output_mins, T* output_maxs, T* output_sums,
                             T* output_means, T* output_variances, T* output_stddevs,
                             size_t elements, size_t batches, Stream& stream);

    /// For each batch, computes the sums over multiple arrays.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Sets of arrays to reduce. One set per batch.
    /// \param input_pitch      Pitch of \p inputs, in elements.
    /// \param[out] outputs     On the \b device. Reduced arrays. One per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch of \p outputs, in elements.
    /// \param shape            Logical {fast, medium, slow} shape of an array.
    /// \param nb_to_reduce     Number of arrays to sum over (i.e. in one set).
    /// \param batches          Number of array sets to reduce independently. Should be less than 65535.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void reduceAdd(const T* inputs, size_t input_pitch, T* outputs, size_t outputs_pitch,
                            size3_t shape, size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the sum over multiple arrays. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void reduceAdd(const T* inputs, T* outputs, size_t elements,
                            size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the means over multiple arrays.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b device. Sets of arrays to reduce. One set per batch.
    /// \param input_pitch      Pitch of \p inputs, in elements.
    /// \param[out] outputs     On the \b device. Reduced arrays. One per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch of \p outputs, in elements.
    /// \param shape            Logical {fast, medium, slow} shape of an array.
    /// \param nb_to_reduce     Number of arrays to average over (i.e. in one set).
    /// \param batches          Number of array sets to reduce independently. Should be less than 65535.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void reduceMean(const T* inputs, size_t input_pitch, T* outputs, size_t outputs_pitch,
                             size3_t shape, size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the means over multiple arrays. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void reduceMean(const T* inputs, T* outputs, size_t elements,
                             size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the averages over multiple arrays with individual weights for all values and arrays.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, \p U should be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] inputs       On the \b device. Contiguous sets of arrays to reduce. One set per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[in] weights      On the \b device. Contiguous array of weights. The same weights are used for every batch.
    /// \param weights_pitch    Pitch, in elements, of \p weights.
    /// \param[out] outputs     On the \b device. Reduced arrays. One per batch. Can be equal to \p inputs.
    /// \param outputs_pitch    Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of an array.
    /// \param nb_to_reduce     Number of arrays to average over (i.e. in one set).
    /// \param batches          Number of array sets to reduce independently. Should be less than 65535.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void reduceMeanWeighted(const T* inputs, size_t input_pitch,
                                     const U* weights, size_t weights_pitch,
                                     T* outputs, size_t outputs_pitch,
                                     size3_t shape, size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the averages over multiple vectors with individual weights for all values and vectors.
    /// Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void reduceMeanWeighted(const T* inputs, const U* weights, T* outputs,
                                     size_t elements, size_t nb_to_reduce, size_t batches, Stream& stream);
}
