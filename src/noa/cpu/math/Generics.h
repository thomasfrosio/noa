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
     * Computes one minus, i.e. output[x] = T(1) - input[x], for every x from 0 to @a elements.
     * @tparam T            Any type with `T operator-(T, T)` defined.
     * @param[in] input     Right operand (i.e. subtrahends).
     * @param[out] output   Results. Can be equal to @a input (i.e. in-place).
     * @param elements      Number of elements to compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void oneMinus(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq,
                       input, input + elements, output, [](const T& element) { return T(1) - element; });
    }

    /**
     * Computes the inverse, i.e. output[x] = T(1) / input[x], for every x from 0 to @a elements.
     * @tparam T            Any type with `T operator/(T, T)` defined.
     * @param[in] input     Right operands (i.e. divisors).
     * @param[out] output   Results. Can be equal to @a input (i.e. in-place).
     * @param elements      Number of elements to compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void inverse(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [](const T& element) { return T(1) / element; });
    }

    /**
     * Computes the square, i.e. output[x] = input[x] * input[x], for every x from 0 to @a elements.
     * @tparam T            Any type with `T operator*(T, T)` defined.
     * @param[in] input     Input data. Should be at least `@a elements * sizeof(T)` bytes.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     */
    template<typename T>
    NOA_IH void square(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq,
                       input, input + elements, output, [](const T& element) { return element * element; });
    }

    /**
     * Computes the square root, i.e. output[x] = Math::sqrt(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::sqrt<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     */
    template<typename T>
    NOA_IH void sqrt(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::sqrt<T>);
    }

    /**
     * Computes the inverse square root, i.e. output[x] = Math::rsqrt(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::rsqrt<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void rsqrt(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::rsqrt<T>);
    }

    /**
     * Computes the power, i.e. output[x] = Math::pow(input[x], exponent), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::pow<T>` defined.
     * @param[in] input     Input data.
     * @param exponent      Exponent.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void pow(T* input, T exponent, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [exponent](const T& element) { return Math::pow(element, exponent); });
    }

    /**
     * Computes the exp, i.e. output[x] = Math::exp(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::exp<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void exp(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::exp<T>);
    }

    /**
     * Computes the log, i.e. output[x] = Math::log(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::log<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void log(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::log<T>);
    }

    /**
     * Computes the abs, i.e. output[x] = Math::abs(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::abs<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void abs(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::abs<T>);
    }

    /**
     * Computes the cos, i.e. output[x] = Math::cos(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::cos<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void cos(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::cos<T>);
    }

    /**
     * Computes the sin, i.e. output[x] = Math::sin(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::sin<T>` defined.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void sin(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::sin<T>);
    }

    /**
     * Normalizes, i.e. output[x] = Math::normalize(input[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::normalize<T>` defined. Usually cfloat_t or cdouble_t.
     * @param[in] input     Input data.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void normalize(T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output, Math::normalize<T>);
    }

    /**
     * Computes the min, i.e. output[x] = Math::min(lhs[x], rhs[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::min(T, T)` defined.
     * @param[in] lhs       Input data.
     * @param[in] rhs       Input data.
     * @param[out] output   Output data. Can be equal to @a lhs or @a rhs.
     * @param elements      Number of elements compute.
     * @note @a lhs, @a rhs and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void min(T* lhs, T* rhs, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, lhs, lhs + elements, rhs, output, Math::min<T>);
    }

    /**
     * Sets the maximum value of an array, i.e. output[x] = Math::min(input[x], threshold), for every x from 0 to @a elements.
     * @tparam T            Any type with `Math::min(T, T)` defined.
     * @param[in] input     Input data.
     * @param[in] threshold Threshold value.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void min(T* input, T threshold, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [threshold](const T& element) { return Math::min(element, threshold); });
    }

    /**
     * Computes the max, i.e. output[x] = Math::max(lhs[x], rhs[x]), for every x from 0 to @a elements.
     * @tparam T            Any type with `Math::max(T, T)` defined.
     * @param[in] lhs       Input data.
     * @param[in] rhs       Input data.
     * @param[out] output   Output data. Can be equal to @a lhs or @a rhs.
     * @param elements      Number of elements compute.
     * @note @a lhs, @a rhs and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void max(T* lhs, T* rhs, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, lhs, lhs + elements, rhs, output, Math::max<T>);
    }

    /**
     * Sets the minimum value, i.e. output[x] = Math::max(input[x], threshold), for every x from 0 to @a elements.
     * @tparam T            Any type with `Math::max(T, T)` defined.
     * @param[in] input     Input data.
     * @param[in] threshold Threshold value.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void max(T* input, T threshold, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [threshold](const T& element) { return Math::max(element, threshold); });
    }

    /**
     * Clamps, i.e. output[x] = Math::clamp(input[x], low, high), for every x from 0 to @a elements.
     * @tparam T            Any type with `T Math::clamp(T, T)` defined.
     * @param[in] input     Input data.
     * @param[in] low       Lowest value allowed.
     * @param[in] high      Highest value allowed.
     * @param[out] output   Output data. Can be equal to @a input.
     * @param elements      Number of elements compute.
     * @note @a input and @a output should be at least `@a elements * sizeof(T)` bytes.
     */
    template<typename T>
    NOA_IH void clamp(T* input, T low, T high, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION("cpu,arith");
        std::transform(std::execution::par_unseq, input, input + elements, output,
                       [low, high](const T& element) { return Math::clamp(element, low, high); });
    }
}
