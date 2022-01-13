/// \file noa/cpu/math/Reductions.h
/// \brief Reduction operations for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/common/Math.h"
#include "noa/common/Functors.h"

#include "noa/cpu/Stream.h"

namespace noa::cpu::math {
    /// Reduces the input array(s) using a binary operator()(\p T, \p T) -> \p T.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce.
    /// \param input_pitch      Pitch, in elements of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Reduced value(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param binary_op        Binary operation function object that will be applied.
    /// \param init             Initial value for the reduction.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename BinaryOp>
    NOA_HOST void reduce(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches,
                         BinaryOp binary_op, T init, Stream& stream);

    /// Returns the minimum value of the input array(s).
    /// \tparam T               Any type with a noa::math::min() overload defined.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Minimum value(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void min(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream);

    /// Returns the maximum value of the input array(s).
    /// \tparam T               Any type with a noa::math::max() overload defined.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Maximum value(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void max(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream);

    /// Returns the sum of the input array(s).
    /// \tparam T               Any type with `T operator+(T, T)` defined.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Sum(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void sum(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream);

    /// Returns the mean of the input array(s).
    /// \tparam T               Any type with `T operator+(T, T)` defined.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Mean(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void mean(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream);

    /// Returns the variance of the input array(s).
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Variance(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void var(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream);

    /// Returns the standard-deviation of the input array(s).
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] outputs     On the \b host. Minimum value(s). One per batch.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_IH void std(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream);

    /// Returns some statistics of the input array(s).
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input array(s) to reduce. One per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs.
    /// \param[out] out_mins    On the \b host. Output minimum values.   One value per batch. If nullptr, ignore it.
    /// \param[out] out_maxs    On the \b host. Output maximum values.   One value per batch. If nullptr, ignore it.
    /// \param[out] out_sums    On the \b host. Output sum values.       One value per batch. If nullptr, ignore it.
    /// \param[out] out_means   On the \b host. Output mean values.      One value per batch. If nullptr, ignore it.
    /// \param[out] out_vars    On the \b host. Output variance values.  One value per batch. If nullptr, ignore it.
    /// \param[out] out_stds    On the \b host. Output stddev values.    One value per batch. If nullptr, ignore it.
    /// \param batches          Number of batches to reduce.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void statistics(const T* inputs, size3_t input_pitch, size3_t shape,
                             T* output_mins, T* output_maxs,
                             T* output_sums, T* output_means,
                             T* output_vars, T* output_stds,
                             size_t batches, Stream& stream);

    /// For each batch, computes the sum over multiple arrays.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Sets of arrays to reduce. One set per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Reduced arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p outputs.
    /// \param nb_to_reduce     Number of arrays (in a set) to sum over.
    /// \param batches          Number of array sets to reduce independently.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void reduceAdd(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                            size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the average over multiple arrays.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] inputs       On the \b host. Sets of arrays to reduce. One set per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Reduced arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs and \p outputs.
    /// \param nb_to_reduce     Number of arrays (in a set) to average over.
    /// \param batches          Number of array sets to reduce independently.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void reduceMean(const T* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch, size3_t shape,
                             size_t nb_to_reduce, size_t batches, Stream& stream);

    /// For each batch, computes the averages over multiple arrays with individual weights for all values and arrays.
    /// \tparam T               (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \tparam U               Same as \p or if \p T is complex, \p U should be the corresponding real type.
    /// \param[in] inputs       On the \b host. Set(s) of arrays to reduce. One set per batch.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[in] weights      On the \b host. Set(s) of arrays of weights. The same weights are used for every batch.
    /// \param weight_pitch     Pitch, in elements, of \p weights. If any dimension is set to 0, the same weights
    ///                         will be used for the entire set. In any case, the weights are reused for every batch.
    /// \param[out] outputs     On the \b host. Reduced arrays. One per batch. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast,medium,slow} shape of \p inputs, \p weights and \p outputs.
    /// \param nb_to_reduce     Number of arrays (in a set) to average over.
    /// \param batches          Number of array sets to reduce independently.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename U>
    NOA_HOST void reduceMeanWeighted(const T* inputs, size3_t input_pitch,
                                     const U* weights, size3_t weight_pitch,
                                     T* outputs, size3_t output_pitch, size3_t shape,
                                     size_t nb_to_reduce, size_t batches, Stream& stream);
}

#define NOA_REDUCTIONS_INL_
#include "noa/cpu/math/Reductions.inl"
#undef NOA_REDUCTIONS_INL_
