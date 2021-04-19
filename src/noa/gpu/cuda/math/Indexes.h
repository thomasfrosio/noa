#pragma once

#include "noa/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Stream.h"

// Implementation details:
namespace Noa::CUDA::Math::Details {
    enum : int { FIRST_MIN, FIRST_MAX, LAST_MIN, LAST_MAX };

    template<int SEARCH_FOR, typename T>
    NOA_HOST void find(T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream);
}

namespace Noa::CUDA::Math {
    /**
     * For each batch, returns the index of the first minimum.
     * @tparam T                    (u)char, (u)int16_t, (u)int32_t.
     * @param[in] inputs            Input array. One per batch. Should be at least `batches * elements * sizeof(T)` bytes.
     * @param[out] output_indexes   Indexes of the first minimum values. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches.
     * @param stream                Stream on which to enqueue this function.
     *                              The stream is synchronized when this function returns.
     *
     * @note This function has an optimization for arrays with <= 4096 elements. It is also assumed that if
     *       @a batches > 8, @a elements is relatively small (in the thousands).
     */
    template<typename T>
    NOA_IH void firstMin(T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        Details::find<Details::FIRST_MIN>(inputs, output_indexes, elements, batches, stream);
    }

    /// For each batch, returns the index of the first maximum. @see Noa::CUDA::Math::firstMin() for more details.
    template<typename T>
    NOA_IH void firstMax(T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        Details::find<Details::FIRST_MAX>(inputs, output_indexes, elements, batches, stream);
    }

    /// For each batch, returns the index of the last minimum. @see Noa::CUDA::Math::firstMin() for more details.
    template<typename T>
    NOA_IH void lastMin(T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        Details::find<Details::LAST_MIN>(inputs, output_indexes, elements, batches, stream);
    }

    /// For each batch, returns the index of the last maximum. @see Noa::CUDA::Math::firstMin() for more details.
    template<typename T>
    NOA_IH void lastMax(T* inputs, size_t* output_indexes, size_t elements, uint batches, Stream& stream) {
        Details::find<Details::LAST_MAX>(inputs, output_indexes, elements, batches, stream);
    }
}
