#pragma once

#include <algorithm>
#include <execution>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/util/Profiler.h"

namespace Noa::Math {
    /**
     * Computes the boolean input[x] < threshold, for every x from 0 to @a elements.
     * @tparam T            Any type with `bool operator<(T,T)` defined.
     * @tparam U            Any type that can be casted from bool. Can be equal to @a T.
     * @param[in] input     Input array. Should be at least `@a elements * sizeof(T)` bytes.
     * @param threshold     Value to use as threshold.
     * @param[out] output   Output array. Should be at least `@a elements * sizeof(U)` bytes.
     * @param elements      Number of elements to compute.
     */
    template<typename T, typename U>
    NOA_HOST void isLess(T* input, T threshold, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq,
                       input, input + elements, output,
                       [threshold](const T& element) -> U { return static_cast<U>(element < threshold); });
    }

    /**
     * Computes the boolean input[x] < threshold, for every x from 0 to @a elements.
     * @tparam T            Any type with `bool operator<(T,T)` defined.
     * @tparam U            Any type that can be casted from bool. Can be equal to @a T.
     * @param[in] input     Input array. Should be at least `@a elements * sizeof(T)` bytes.
     * @param threshold     Value to use as threshold.
     * @param[out] output   Output array. Should be at least `@a elements * sizeof(U)` bytes.
     * @param elements      Number of elements to compute.
     */
    template<typename T, typename U>
    NOA_HOST void isGreater(T* input, T threshold, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq,
                       input, input + elements, output,
                       [threshold](const T& element) -> U { return static_cast<U>(threshold < element); });
    }

    /**
     * Computes the boolean (input[x] < high && low < input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `bool operator<(T,T)` defined.
     * @tparam U            Any type that can be casted from bool. Can be equal to @a T.
     * @param[in] input     Input array. Should be at least `@a elements * sizeof(T)` bytes.
     * @param low           Low threshold.
     * @param high          High threshold.
     * @param[out] output   Output array. Should be at least `@a elements * sizeof(U)` bytes.
     * @param elements      Number of elements to compute.
     */
    template<typename T, typename U>
    NOA_HOST void isWithin(T* input, T low, T high, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq,
                       input, input + elements, output,
                       [threshold](const T& element) -> U {
                           return static_cast<U>(element < high && low < element);
                       });
    }

    /**
     * Computes the logical NOT, i.e. output[x] = !x, for every x from 0 to @a elements.
     * @tparam T            Any integral type.
     * @param[in] input     Input array.
     * @param[out] output   Output array.
     * @param elements      Number of elements to compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    NOA_HOST void logicNOT(T* input, T* output, size_t elements) {
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [](T element) -> T { return !element; });
    }
}
