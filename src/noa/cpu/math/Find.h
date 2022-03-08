/// \file noa/cpu/math/Find.h
/// \brief Find index of min/max.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// TODO Add find() to find the (x,y,z) index of the min (or max, or zero) element.

namespace noa::cpu::math {
    /// Returns the index of the first minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void firstMin(const T* inputs, size_t input_pitch, size_t* output_indexes,
                           size_t elements, size_t batches, Stream& stream);

    /// Returns the index of the first maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void firstMax(const T* inputs, size_t input_pitch, size_t* output_indexes,
                           size_t elements, size_t batches, Stream& stream);

    /// Returns the index of the last minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void lastMin(const T* inputs, size_t input_pitch, size_t* output_indexes,
                          size_t elements, size_t batches, Stream& stream);

    /// Returns the index of the last maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param input_pitch          Pitch, in elements, of \p inputs.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T>
    NOA_HOST void lastMax(const T* inputs, size_t input_pitch, size_t* output_indexes,
                          size_t elements, size_t batches, Stream& stream);
}
