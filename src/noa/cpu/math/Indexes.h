#pragma once

#include <utility>

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

namespace Noa::Math {
    /**
     * Returns the first minimum value and its index.
     * @param[in] input     Input array with at least `@a elements * sizeof(T)` bytes.
     * @param elements      Number of elements to consider.
     * @return              {1: index of the minimum value, 2: minimum value}
     */
    template<typename T>
    NOA_HOST std::pair<size_t, T> firstMin(T* input, size_t elements);

    /**
     * Returns the first maximum value and its index.
     * @param[in] input     Input array with at least `@a elements * sizeof(T)` bytes.
     * @param elements      Number of elements to consider.
     * @return              {1: index of the maximum value, 2: maximum value}
     */
    template<typename T>
    NOA_HOST std::pair<size_t, T> firstMax(T* input, size_t elements);
}
