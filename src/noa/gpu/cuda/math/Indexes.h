/// \file noa/gpu/cuda/math/Indexes.h
/// \brief Find indexes for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

namespace noa::cuda::math::details {
    enum : int { FIRST_MIN, FIRST_MAX, LAST_MIN, LAST_MAX };

    template<int SEARCH_FOR, typename T>
    NOA_HOST void find(const T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream);
}

namespace noa::cuda::math {
    /// For each batch, returns the index of the first minimum.
    /// \tparam T                    (u)char, (u)short, (u)int, (u)long, (u)long long.
    /// \param[in] inputs            Input array. One per batch. Should be at least `batches * elements * sizeof(T)` bytes.
    /// \param[out] output_indexes   Indexes of the first minimum values. One per batch.
    /// \param elements              Number of elements per batch.
    /// \param batches               Number of batches.
    /// \param stream                Stream on which to enqueue this function.
    ///                              The stream is synchronized when this function returns.
    /// \note This function has an optimization for arrays with <= 4096 elements.
    ///       It also assumes that if \a batches > 8, \a elements is relatively small (in the thousands).
    template<typename T>
    NOA_IH void firstMin(const T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        details::find<details::FIRST_MIN>(inputs, output_indexes, elements, batches, stream);
    }

    /// For each batch, returns the index of the first maximum. \see noa::cuda::math::firstMin() for more details.
    template<typename T>
    NOA_IH void firstMax(const T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        details::find<details::FIRST_MAX>(inputs, output_indexes, elements, batches, stream);
    }

    /// For each batch, returns the index of the last minimum. \see noa::cuda::math::firstMin() for more details.
    template<typename T>
    NOA_IH void lastMin(const T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        details::find<details::LAST_MIN>(inputs, output_indexes, elements, batches, stream);
    }

    /// For each batch, returns the index of the last maximum. \see noa::cuda::math::firstMin() for more details.
    template<typename T>
    NOA_IH void lastMax(const T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        details::find<details::LAST_MAX>(inputs, output_indexes, elements, batches, stream);
    }
}
