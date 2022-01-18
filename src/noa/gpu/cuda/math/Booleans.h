/// \file noa/gpu/cuda/math/Booleans.h
/// \brief Boolean "operators" for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math {
    /// Computes whether \p input is less than \p threshold, element-wise, i.e. x < threshold.
    /// \tparam T               float, double or any integral type.
    /// \tparam U               Any type allowed for \p T or bool.
    /// \param[in] input        On the \b device. Input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param threshold        Value to use as threshold.
    /// \param[out] output      On the \b device. Output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isLess(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                         size3_t shape, Stream& stream);

    /// Computes whether \p input is less than \p threshold, element-wise, i.e. x < threshold. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isLess(const T* input, T threshold, U* output, size_t elements, Stream& stream);

    /// Computes whether \p input is greater than \p threshold, element-wise, i.e. threshold < x.
    /// \tparam T               float, double or any integral type.
    /// \tparam U               Any type allowed for \p T or bool.
    /// \param[in] input        On the \b device. Input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param threshold        Value to use as threshold.
    /// \param[out] output      On the \b device. Output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isGreater(const T* input, size_t input_pitch, T threshold, U* output, size_t output_pitch,
                            size3_t shape, Stream& stream);

    /// Computes whether \p input is less than \p threshold, element-wise, i.e. x < threshold.
    /// Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isGreater(const T* input, T threshold, U* output, size_t elements, Stream& stream);

    /// Computes whether \p input is within \p low and \p high, element-wise, i.e. x < high && low < x.
    /// \tparam T               float, double or any integral type.
    /// \tparam U               Any type allowed for \p T or bool.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param low              Low threshold.
    /// \param high             High threshold.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isWithin(const T* input, size_t input_pitch, T low, T high, U* output, size_t output_pitch,
                           size3_t shape, Stream& stream);

    /// Computes whether \p input is within \p low and \p high, element-wise, i.e. x < high && low < x.
    /// Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void isWithin(const T* input, T low, T high, U* output, size_t elements, Stream& stream);

    /// Computes the logical NOT, element-wise, i.e. !x.
    /// \tparam T               Any integral type.
    /// \tparam U               Any type allowed for \p T or bool.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void logicNOT(const T* input, size_t input_pitch, U* output, size_t output_pitch,
                           size3_t shape, Stream& stream);

    /// Computes the logical NOT, element-wise, i.e. !x. Version for contiguous layouts.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename U>
    NOA_HOST void logicNOT(const T* input, U* output, size_t elements, Stream& stream);
}
