/// \file noa/cpu/math/Indexes.h
/// \brief Find indexes for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::math {
    /// Returns the index of the first minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            Input array with at least `\a elements * sizeof(T)` bytes.
    /// \param[out] output_indexes  Output indexes. One per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of batches.
    template<typename T>
    NOA_HOST void firstMin(const T* inputs, size_t* output_indexes, size_t elements, uint batches);

    /// Returns the index of the first maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            Input array with at least `\a elements * sizeof(T)` bytes.
    /// \param[out] output_indexes  Output indexes. One per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of batches.
    template<typename T>
    NOA_HOST void firstMax(const T* inputs, size_t* output_indexes, size_t elements, uint batches);

    /// Returns the index of the last minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            Input array with at least `\a elements * sizeof(T)` bytes.
    /// \param[out] output_indexes  Output indexes. One per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of batches.
    template<typename T>
    NOA_HOST void lastMin(const T* inputs, size_t* output_indexes, size_t elements, uint batches);

    /// Returns the index of the last maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            Input array with at least `\a elements * sizeof(T)` bytes.
    /// \param[out] output_indexes  Output indexes. One per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of batches.
    template<typename T>
    NOA_HOST void lastMax(const T* inputs, size_t* output_indexes, size_t elements, uint batches);
}
