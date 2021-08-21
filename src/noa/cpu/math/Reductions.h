/// \file noa/cpu/math/Reductions.h
/// \brief Reduction operations for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::cpu::math {
    /// For each batch, returns the minimum value of an input array.
    /// \tparam T               Any type with `T operator<(T, T)` defined.
    /// \param[in] inputs       On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] output_mins On the \b host. Minimum values. One per batch.
    /// \param elements         Number of elements to compute per batch.
    /// \param batches          Number of batches to compute.
    template<typename T>
    NOA_IH void min(const T* inputs, T* output_mins, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements;
            output_mins[batch] = *std::min_element(input, input + elements);
        }
    }

    /// For each batch, returns the maximum value of an input array.
    /// \tparam T               Any type with `T operator<(T, T)` defined.
    /// \param[in] inputs       On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] output_maxs On the \b host. Maximum values. One per batch.
    /// \param elements         Number of elements to compute per batch.
    /// \param batches          Number of batches to compute.
    template<typename T>
    NOA_IH void max(const T* inputs, T* output_maxs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements;
            output_maxs[batch] = *std::max_element(input, input + elements);
        }
    }

    /// For each batch, returns the minimum and maximum value of an input array.
    /// \tparam T               Any type with `T operator<(T, T)` defined.
    /// \param[in] inputs       On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] output_mins On the \b host. Minimum values. One per batch.
    /// \param[out] output_maxs On the \b host. Maximum values. One per batch.
    /// \param elements         Number of elements to compute per batch.
    /// \param batches          Number of batches to compute.
    template<typename T>
    NOA_IH void minMax(const T* inputs, T* output_mins, T* output_maxs, size_t elements, uint batches) {
        NOA_PROFILE_FUNCTION();
        for (uint batch = 0; batch < batches; ++batch) {
            const T* input = inputs + batch * elements;
            auto[it_min, it_max] = std::minmax_element(input, input + elements);
            output_mins[batch] = *it_min;
            output_maxs[batch] = *it_max;
        }
    }

    /// For each batch, returns the sum and/or average of the elements in \p inputs.
    /// \tparam T                   (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs           On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] output_sums     On the \b host. Output sums. One per batch. If nullptr, ignore it.
    /// \param[out] output_means    On the \b host. Output means. One per batch. If nullptr, ignore it.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of batches to compute.
    ///
    /// \details For floating-point types, the Neumaier variation of the Kahan summation algorithm is used. It also uses
    ///          double precision for accumulation and compensation. For complex types, the Kahan summation algorithm
    ///          is used. Note that -ffast-math optimizes it away.
    template<typename T>
    NOA_HOST void sumMean(const T* inputs, T* output_sums, T* output_means, size_t elements, uint batches);

    /// For each batch, returns the sum of the elements in \p inputs. \see sumMean for more details.
    template<typename T>
    NOA_IH void sum(const T* inputs, T* output_sums, size_t elements, uint batches) {
        sumMean<T>(inputs, output_sums, nullptr, elements, batches);
    }

    /// For each batch, returns the mean of the elements in \p inputs. \see sumMean for more details.
    template<typename T>
    NOA_IH void mean(const T* inputs, T* output_means, size_t elements, uint batches) {
        sumMean<T>(inputs, nullptr, output_means, elements, batches);
    }

    /// For each batch, returns the minimum, maximum, sum and the mean of the elements of the input array.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double.
    /// \param[in] inputs       On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] out_mins    On the \b host. Output minimum values. One per batch.
    /// \param[out] out_maxs    On the \b host. Output maximum values. One per batch.
    /// \param[out] out_sums    On the \b host. Output sum values.     One per batch. If nullptr, ignore it.
    /// \param[out] out_means   On the \b host. Output mean values.    One per batch. If nullptr, ignore it.
    /// \param elements         Number of elements to compute per batch.
    /// \param batches          Number of batches to compute.
    template<typename T>
    NOA_HOST void minMaxSumMean(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                                size_t elements, uint batches);

    /// For each batch, returns the sum, mean, variance and stddev of the elements in the input array.
    /// \tparam T                       float or double.
    /// \param[in] inputs               On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] output_sums         On the \b host. Output sums.      One per batch. If nullptr, ignore it.
    /// \param[out] output_means        On the \b host. Output means.     One per batch. If nullptr, ignore it.
    /// \param[out] output_variances    On the \b host. Output variances. One per batch. If nullptr, ignore it.
    /// \param[out] output_stddevs      On the \b host. Output stddevs.   One per batch. If nullptr, ignore it.
    /// \param elements                 Number of elements to compute per batch.
    /// \param batches                  Number of batches to compute.
    template<typename T>
    NOA_HOST void sumMeanVarianceStddev(const T* inputs, T* output_sums, T* output_means,
                                        T* output_variances, T* output_stddevs,
                                        size_t elements, uint batches);

    /// For each batch, returns the variance and/or stddev of the elements in the input array.
    /// \tparam T                       float or double.
    /// \param[in] inputs               On the \b host. Contiguous input arrays. One per batch.
    /// \param[in] means                On the \b host. Means used to compute the variance. One value per batch.
    /// \param[out] output_variances    On the \b host. Output variances. One per batch. If nullptr, ignore it.
    /// \param[out] output_stddevs      On the \b host. Output stddevs. One per batch. If nullptr, ignore it.
    /// \param elements                 Number of elements to compute per batch.
    /// \param batches                  Number of batches to compute.
    template<typename T>
    NOA_HOST void varianceStddev(const T* inputs, const T* input_means, T* output_variances, T* output_stddevs,
                                 size_t elements, uint batches);

    /// For each batch, returns the variance of the elements in the input array.
    template<typename T>
    NOA_IH void variance(const T* inputs, const T* input_means, T* output_variances, size_t elements, uint batches) {
        varianceStddev<T>(inputs, input_means, output_variances, nullptr, elements, batches);
    }

    /// For each batch, returns the stddev of the elements in the input array.
    template<typename T>
    NOA_IH void stddev(const T* inputs, const T* input_means, T* output_stddevs, size_t elements, uint batches) {
        varianceStddev<T>(inputs, input_means, nullptr, output_stddevs, elements, batches);
    }

    /// For each batch, returns the variance of the elements in the input array.
    template<typename T>
    NOA_IH void variance(const T* inputs, T* output_variances, size_t elements, uint batches) {
        sumMeanVarianceStddev<T>(inputs, nullptr, nullptr, output_variances, nullptr, elements, batches);
    }

    /// For each batch, returns the stddev of the elements in the input array.
    template<typename T>
    NOA_IH void stddev(const T* inputs, T* output_stddevs, size_t elements, uint batches) {
        sumMeanVarianceStddev<T>(inputs, nullptr, nullptr, nullptr, output_stddevs, elements, batches);
    }

    /// For each batch, returns the full statistics of the elements of the input array, as described in Stats<T>.
    /// \tparam T                   float or double.
    /// \param[in] inputs           On the \b host. Contiguous input arrays. One per batch.
    /// \param[out] out_mins        On the \b host. Output minimum values.   One value per batch.
    /// \param[out] out_maxs        On the \b host. Output maximum values.   One value per batch.
    /// \param[out] out_sums        On the \b host. Output sum values.       One value per batch. If nullptr, ignore it.
    /// \param[out] out_means       On the \b host. Output mean values.      One value per batch. If nullptr, ignore it.
    /// \param[out] out_variances   On the \b host. Output variance values.  One value per batch. If nullptr, ignore it.
    /// \param[out] out_stddevs     On the \b host. Output stddev values.    One value per batch. If nullptr, ignore it.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of batches to compute.
    template<typename T>
    NOA_HOST void statistics(const T* inputs, T* output_mins, T* output_maxs, T* output_sums, T* output_means,
                             T* output_variances, T* output_stddevs, size_t elements, uint batches);

    /// For each batch, computes the sum over multiple arrays.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Contiguous sets of arrays to reduce. One set per batch.
    /// \param[out] outputs On the \b host. Reduced arrays. One per batch.
    /// \param elements     Number of elements in a vector.
    /// \param nb_to_reduce Number of arrays to sum over.
    /// \param batches      Number of array sets to reduce independently.
    template<typename T>
    NOA_HOST void reduceAdd(const T* inputs, T* outputs, size_t elements, uint nb_to_reduce, uint batches);

    /// For each batch, computes the average over multiple arrays.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs   On the \b host. Contiguous sets of arrays to reduce. One set per batch.
    /// \param[out] outputs On the \b host. Reduced arrays. One per batch.
    /// \param elements     Number of elements in a vector.
    /// \param nb_to_reduce Number of arrays to average over.
    /// \param batches      Number of array sets to reduce independently.
    template<typename T>
    NOA_HOST void reduceMean(const T* inputs, T* outputs, size_t elements, uint nb_to_reduce, uint batches);

    /// For each batch, computes the averages over multiple arrays with individual weights for all values and arrays.
    /// \tparam T           (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U           If \p T is complex, \p U should be the corresponding value type. Otherwise, same as \p T.
    /// \param[in] inputs   On the \b host. Contiguous sets of arrays to reduce. One set per batch.
    /// \param[in] weights  On the \b host. Contiguous array of weights. The same weights are used for every batch.
    /// \param[out] output  On the \b host. Contiguous reduced arrays. One per batch.
    /// \param elements     Number of elements in an array.
    /// \param nb_to_reduce Number of arrays to average over.
    /// \param batches      Number of array sets to reduce independently.
    template<typename T, typename U>
    NOA_HOST void reduceMeanWeighted(const T* inputs, const U* weights, T* output,
                                     size_t elements, uint nb_to_reduce, uint batches);
}
