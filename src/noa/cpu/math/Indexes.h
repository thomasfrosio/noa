/// \file noa/cpu/math/Indexes.h
/// \brief Find indexes for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::cpu::math {
    /// Returns the index of the first minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of contiguous batches to compute.
    template<typename T>
    NOA_HOST void firstMin(const T* inputs, size_t* output_indexes, size_t elements, size_t batches);

    /// Returns the index of the first maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of contiguous batches to compute.
    template<typename T>
    NOA_HOST void firstMax(const T* inputs, size_t* output_indexes, size_t elements, size_t batches);

    /// Returns the index of the last minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of contiguous batches to compute.
    template<typename T>
    NOA_HOST void lastMin(const T* inputs, size_t* output_indexes, size_t elements, size_t batches);

    /// Returns the index of the last maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] input            On the \b host. Contiguous input array.
    /// \param[out] output_indexes  On the \b host. Output indexes. One value per batch.
    /// \param elements             Number of elements to compute per batch.
    /// \param batches              Number of contiguous batches to compute.
    template<typename T>
    NOA_HOST void lastMax(const T* inputs, size_t* output_indexes, size_t elements, size_t batches);
}
