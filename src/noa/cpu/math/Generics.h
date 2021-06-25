/// \file noa/cpu/math/Generics.h
/// \brief Generic math functions for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>
#include <execution>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

// TODO Allow to modify the execution policy via a template parameter?

namespace noa::math {
    /// Computes one minus, i.e. output[x] = T(1) - input[x], for every x from 0 to \a elements.
    /// \tparam T           Any type with `T operator-(T, T)` defined.
    /// \param[in] input    Right operand (i.e. subtrahends).
    /// \param[out] output  Results. Can be equal to \a input (i.e. in-place).
    /// \param elements     Number of elements to compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void oneMinus(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq,
                       input, input + elements, output, [](T element) { return T(1) - element; });
    }

    /// Computes the inverse, i.e. output[x] = T(1) / input[x], for every x from 0 to \a elements.
    /// \tparam T           Any type with `T operator/(T, T)` defined.
    /// \param[in] input    Right operands (i.e. divisors).
    /// \param[out] output  Results. Can be equal to \a input (i.e. in-place).
    /// \param elements     Number of elements to compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void inverse(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output,
                       [](T element) { return T(1) / element; });
    }

    /// Computes the square, i.e. output[x] = input[x] * input[x], for every x from 0 to \a elements.
    /// \tparam T           Any type with `T operator*(T, T)` defined.
    /// \param[in] input    Input data. Should be at least `\a elements * sizeof(T)` bytes.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void square(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq,
                       input, input + elements, output, [](T element) { return element * element; });
    }

    /// Computes the square root, i.e. output[x] = math::sqrt(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::sqrt<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void sqrt(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::sqrt(a); });
    }

    /// Computes the inverse square root, i.e. output[x] = math::rsqrt(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::rsqrt<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void rsqrt(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::rsqrt(a); });
    }

    /// Computes the power, i.e. output[x] = math::pow(input[x], exponent), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::pow<T>` defined.
    /// \param[in] input    Input data.
    /// \param exponent     Exponent.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void pow(const T* input, T exponent, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output,
                       [exponent](T element) { return math::pow(element, exponent); });
    }

    /// Computes the exp, i.e. output[x] = math::exp(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::exp<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void exp(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::exp(a); });
    }

    /// Computes the log, i.e. output[x] = math::log(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::log<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void log(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::log(a); });
    }

    /// Computes the abs, i.e. output[x] = math::abs(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::abs<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void abs(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::abs(a); });
    }

    /// Computes the cos, i.e. output[x] = math::cos(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::cos<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void cos(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::cos(a); });
    }

    /// Computes the sin, i.e. output[x] = math::sin(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::sin<T>` defined.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void sin(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output, [](T a) { return math::sin(a); });
    }

    /// Normalizes, i.e. output[x] = math::normalize(input[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::normalize<T>` defined. Usually cfloat_t or cdouble_t.
    /// \param[in] input    Input data.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void normalize(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq,
                       input, input + elements, output, [](T a) { return math::normalize(a); });
    }

    /// Computes the min, i.e. output[x] = math::min(lhs[x], rhs[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::min(T, T)` defined.
    /// \param[in] lhs      Input data.
    /// \param[in] rhs      Input data.
    /// \param[out] output  Output data. Can be equal to \a lhs or \a rhs.
    /// \param elements     Number of elements compute.
    /// \note \a lhs, \a rhs and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void min(const T* lhs, const T* rhs, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq,
                       lhs, lhs + elements, rhs, output, [](T a, T b) { return math::min(a, b); });
    }

    /// Sets the maximum value of an array, i.e. output[x] = math::min(input[x], threshold), for every x from 0 to \a elements.
    /// \tparam T               Any type with `math::min(T, T)` defined.
    /// \param[in] input        Input data.
    /// \param[in] threshold    Threshold value.
    /// \param[out] output      Output data. Can be equal to \a input.
    /// \param elements         Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void min(const T* input, T threshold, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output,
                       [threshold](T element) { return math::min(element, threshold); });
    }

    /// Computes the max, i.e. output[x] = math::max(lhs[x], rhs[x]), for every x from 0 to \a elements.
    /// \tparam T           Any type with `math::max(T, T)` defined.
    /// \param[in] lhs      Input data.
    /// \param[in] rhs      Input data.
    /// \param[out] output  Output data. Can be equal to \a lhs or \a rhs.
    /// \param elements     Number of elements compute.
    /// \note \a lhs, \a rhs and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void max(const T* lhs, const T* rhs, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq,
                       lhs, lhs + elements, rhs, output, [](T a, T b) { return math::max(a, b); });
    }

    /// Sets the minimum value, i.e. output[x] = math::max(input[x], threshold), for every x from 0 to \a elements.
    /// \tparam T               Any type with `math::max(T, T)` defined.
    /// \param[in] input        Input data.
    /// \param[in] threshold    Threshold value.
    /// \param[out] output      Output data. Can be equal to \a input.
    /// \param elements         Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void max(const T* input, T threshold, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output,
                       [threshold](T element) { return math::max(element, threshold); });
    }

    /// Clamps, i.e. output[x] = math::clamp(input[x], low, high), for every x from 0 to \a elements.
    /// \tparam T           Any type with `T math::clamp(T, T)` defined.
    /// \param[in] input    Input data.
    /// \param[in] low      Lowest value allowed.
    /// \param[in] high     Highest value allowed.
    /// \param[out] output  Output data. Can be equal to \a input.
    /// \param elements     Number of elements compute.
    /// \note \a input and \a output should be at least `\a elements * sizeof(T)` bytes.
    template<typename T>
    NOA_IH void clamp(const T* input, T low, T high, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(std::execution::seq, input, input + elements, output,
                       [low, high](T element) { return math::clamp(element, low, high); });
    }
}
