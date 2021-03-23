#pragma once

#include <pair>

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/util/Profiler.h"

namespace Noa::Math {
    /**
     * Returns the first minimum value and its index.
     * @param[in] input     Input array with at least `@a elements * sizeof(T)` bytes.
     * @param elements      Number of elements to consider.
     * @return              {1: index of the minimum value, 2: minimum value}
     */
    template<typename T>
    NOA_HOST std::pair<size_t, T> firstMin(T* input, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        size_t min_index = 0;
        T min_value = input[0];
        for (size_t idx = 1; idx < elements; ++idx) {
            if (input[idx] < min_value) {
                min_value = input[idx];
                min_index = idx;
            }
        }
        return {min_index, min_value};
    }

    /**
     * Returns the first maximum value and its index.
     * @param[in] input     Input array with at least `@a elements * sizeof(T)` bytes.
     * @param elements      Number of elements to consider.
     * @return              {1: index of the maximum value, 2: maximum value}
     */
    template<typename T>
    NOA_HOST std::pair<size_t, T> firstMax(T* input, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        size_t max_index = 0;
        T max_value = input[0];
        for (size_t idx = 1; idx < elements; ++idx) {
            if (max_value < input[idx]) {
                max_value = input[idx];
                max_index = idx;
            }
        }
        return {max_index, max_value};
    }
}
