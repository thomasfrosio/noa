/// \file noa/gpu/cuda/math/Find.h
/// \brief Find indexes for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

// TODO This is an old piece of code. Update this to support more reductions, better launch configs
//      and vectorized loads/stores.

namespace noa::cuda::math::details {
    enum : int {
        FIRST_MIN, FIRST_MAX, LAST_MIN, LAST_MAX
    };

    template<int SEARCH_FOR, typename T>
    NOA_HOST void find(const T* inputs, size_t* output_indexes, size_t elements, size_t batches, Stream& stream);
}

namespace noa::cuda::math {
    /// Returns the index of the first minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs           On the \b device. Input arrays. One per batch.
    /// \param[out] output_indexes  On the \b device. Indexes of the first minimum values. One value per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of contiguous batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when this function returns.
    /// \note This function has an optimization for arrays with <= 4096 elements.
    ///       It also assumes that if \a batches > 8, \a elements is relatively small (in the thousands).
    template<typename T>
    NOA_IH void firstMin(const T* inputs, size_t* output_indexes, size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::find<details::FIRST_MIN>(inputs, output_indexes, elements, batches, stream);
    }

    /// Returns the index of the first maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs           On the \b device. Input arrays. One per batch.
    /// \param[out] output_indexes  On the \b device. Indexes of the first maximum values. One value per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of contiguous batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when this function returns.
    /// \note This function has an optimization for arrays with <= 4096 elements.
    ///       It also assumes that if \a batches > 8, \a elements is relatively small (in the thousands).
    template<typename T>
    NOA_IH void firstMax(const T* inputs, size_t* output_indexes, size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::find<details::FIRST_MAX>(inputs, output_indexes, elements, batches, stream);
    }

    /// Returns the index of the last minimum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs           On the \b device. Input arrays. One per batch.
    /// \param[out] output_indexes  On the \b device. Indexes of the last minimum values. One value per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of contiguous batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when this function returns.
    /// \note This function has an optimization for arrays with <= 4096 elements.
    ///       It also assumes that if \a batches > 8, \a elements is relatively small (in the thousands).
    template<typename T>
    NOA_IH void lastMin(const T* inputs, size_t* output_indexes, size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::find<details::LAST_MIN>(inputs, output_indexes, elements, batches, stream);
    }

    /// Returns the index of the last maximum value.
    /// \tparam T                   (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs           On the \b device. Input arrays. One per batch.
    /// \param[out] output_indexes  On the \b device. Indexes of the last maximum values. One value per batch.
    /// \param elements             Number of elements per batch.
    /// \param batches              Number of contiguous batches to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    ///                             The stream is synchronized when this function returns.
    /// \note This function has an optimization for arrays with <= 4096 elements.
    ///       It also assumes that if \a batches > 8, \a elements is relatively small (in the thousands).
    template<typename T>
    NOA_IH void lastMax(const T* inputs, size_t* output_indexes, size_t elements, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::find<details::LAST_MAX>(inputs, output_indexes, elements, batches, stream);
    }
}
