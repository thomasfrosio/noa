#pragma once

#include <algorithm>
#include <execution>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

namespace Noa::Math {
    /**
     * Computes output[x] = input[x] < threshold, for every x from 0 to @a elements.
     * @tparam T            Any type with `bool operator<(T,T)` defined.
     * @tparam U            Any type that can be casted from bool. Can be equal to @a T.
     * @param[in] input     Input array. Should be at least `@a elements * sizeof(T)` bytes.
     * @param threshold     Value to use as threshold.
     * @param[out] output   Output array. Should be at least `@a elements * sizeof(U)` bytes.
     * @param elements      Number of elements to compute.
     */
    template<typename T, typename U>
    NOA_IH void isLess(T* input, T threshold, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::par_unseq,
                       input, input + elements, output,
                       [threshold](T element) -> U { return static_cast<U>(element < threshold); });
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
    NOA_IH void isGreater(T* input, T threshold, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::par_unseq,
                       input, input + elements, output,
                       [threshold](T element) -> U { return static_cast<U>(threshold < element); });
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
    NOA_IH void isWithin(T* input, T low, T high, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::par_unseq,
                       input, input + elements, output,
                       [low, high](T element) -> U {
                           return static_cast<U>(element < high && low < element);
                       });
    }

    /**
     * Computes the logical NOT, i.e. output[x] = !x, for every x from 0 to @a elements.
     * @tparam T            Any integral type.
     * @tparam U            Any type that can be casted from bool. Can be equal to @a T.
     * @param[in] input     Input array.
     * @param[out] output   Output array.
     * @param elements      Number of elements to compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T>>>
    NOA_IH void logicNOT(T* input, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [](T element) -> U { return static_cast<U>(!element); });
    }
}
