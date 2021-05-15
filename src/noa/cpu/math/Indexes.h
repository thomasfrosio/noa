#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

namespace Noa::Math {
    /**
     * Returns the index of the first minimum value.
     * @tparam T                    (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] input             Input array with at least `@a elements * sizeof(T)` bytes.
     * @param[out] output_indexes   Output indexes. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches.
     */
    template<typename T>
    NOA_HOST void firstMin(T* inputs, size_t* output_indexes, size_t elements, uint batches);

    /**
     * Returns the index of the first maximum value.
     * @tparam T                    (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] input             Input array with at least `@a elements * sizeof(T)` bytes.
     * @param[out] output_indexes   Output indexes. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches.
     */
    template<typename T>
    NOA_HOST void firstMax(T* inputs, size_t* output_indexes, size_t elements, uint batches);

    /**
     * Returns the index of the last minimum value.
     * @tparam T                    (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] input             Input array with at least `@a elements * sizeof(T)` bytes.
     * @param[out] output_indexes   Output indexes. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches.
     */
    template<typename T>
    NOA_HOST void lastMin(T* inputs, size_t* output_indexes, size_t elements, uint batches);

    /**
     * Returns the index of the last maximum value.
     * @tparam T                    (u)char, (u)short, (u)int, (u)long, (u)long long.
     * @param[in] input             Input array with at least `@a elements * sizeof(T)` bytes.
     * @param[out] output_indexes   Output indexes. One per batch.
     * @param elements              Number of elements per batch.
     * @param batches               Number of batches.
     */
    template<typename T>
    NOA_HOST void lastMax(T* inputs, size_t* output_indexes, size_t elements, uint batches);
}
