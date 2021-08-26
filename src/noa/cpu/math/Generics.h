/// \file noa/cpu/math/Generics.h
/// \brief Generic math functions for arrays.
/// \puthor Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::cpu::math {
    /// Subtracts \p input to one, i.e. T(1) - x.
    /// \tparam T           Any type with `T operator-(T, T)` defined.
    /// \param[in] input    On the \b host. Contiguous array with the right operands (i.e. subtrahends).
    /// \param[out] output  On the \b host. Contiguous array with the results. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void oneMinus(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return T(1) - a; });
    }

    /// Computes the inverse of \p input, i.e. T(1) / x.
    /// \tparam T           Any type with `T operator/(T, T)` defined.
    /// \param[in] input    On the \b host. Contiguous array with the right operands (i.e. divisors).
    /// \param[out] output  On the \b host. Contiguous array with the results. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void inverse(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [](const T& a) { return T(1) / a; });
    }

    /// Computes the square of \p input, i.e. x * x.
    /// \tparam T           Any type with `T operator*(T, T)` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void square(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return a * a; });
    }

    /// Computes the square root of \p input, i.e. math::sqrt(x).
    /// \tparam T           Any type with `T math::sqrt<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void sqrt(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::sqrt(a); });
    }

    /// Computes the square root of \p input, i.e. math::sqrt(x).
    /// \tparam T           Any type with `T math::rsqrt<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void rsqrt(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::rsqrt(a); });
    }

    /// Computes the power of \p input, i.e. math::pow(x, exponent).
    /// \tparam T           Any type with `T math::pow<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param exponent     Exponent.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void pow(const T* input, T exponent, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [exponent](const T& a) { return noa::math::pow(a, exponent); });
    }

    /// Computes the exponential of \p input, i.e. math::exp(x).
    /// \tparam T           Any type with `T math::exp<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void exp(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::exp(a); });
    }

    /// Computes the log of \p input, i.e. math::log(x).
    /// \tparam T           Any type with `T math::log<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void log(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::log(a); });
    }

    /// Computes the abs of \p input, i.e. math::abs(x).
    /// \tparam T           Any type with `R math::abs<T>` defined.
    /// \tparam R           If \p T is complex, \p R should be the corresponding value type, otherwise, same as \p T.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T, typename R>
    NOA_IH void abs(const T* input, R* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) -> R { return noa::math::abs(a); });
    }

    /// Computes the cosine of \p input, i.e. math::cos(x).
    /// \tparam T           Any type with `T math::cos<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void cos(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::cos(a); });
    }

    /// Computes the sin of \p input, i.e. math::sin(x).
    /// \tparam T           Any type with `T math::sin<T>` defined.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void sin(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::sin(a); });
    }
}

namespace noa::cpu::math {
    ///  Returns the length-normalized of complex numbers to 1, reducing them to their phase.
    /// \tparam T           Any type with `T math::normalize<T>` defined. Usually cfloat_t or cdouble_t.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void normalize(const T* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const T& a) { return noa::math::normalize(a); });
    }

    ///  Extracts the real parts of complex numbers.
    /// \tparam T           float or double.
    /// \param[in] input    On the \b host. Input complex data.
    /// \param[out] output  On the \b host. Output real data.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void real(const noa::Complex<T>* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const noa::Complex<T>& a) { return noa::math::real(a); });
    }

    ///Extracts the imaginary parts of complex numbers.
    /// \tparam T           float or double.
    /// \param[in] input    On the \b host. Input complex data.
    /// \param[out] output  On the \b host. Output imaginary data.
    /// \param elements     Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void imag(const noa::Complex<T>* input, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output, [](const noa::Complex<T>& a) { return noa::math::imag(a); });
    }

    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T               float or double.
    /// \param[in] input        On the \b host. Input complex data.
    /// \param[out] output_real On the \b host. Output real data.
    /// \param[out] output_imag On the \b host. Output imaginary data.
    /// \param elements         Number of elements in \p input to compute.
    template<typename T>
    NOA_IH void realAndImag(const noa::Complex<T>* input, T* output_real, T* output_imag, size_t elements) {
        NOA_PROFILE_FUNCTION();
        for (size_t idx = 0; idx < elements; ++idx) {
            output_real[idx] = input[idx].real;
            output_imag[idx] = input[idx].imag;
        }
    }

    /// Gathers the real and imaginary parts into complex numbers.
    /// \tparam T               float or double.
    /// \param[in] input_real   On the \b host. Input real data.
    /// \param[in] input_imag   On the \b host. Input imaginary data.
    /// \param[out] output      On the \b host. Output complex data.
    /// \param elements         Number of elements in \p input_real and \p input_imag to compute.
    template<typename T>
    NOA_IH void complex(const T* input_real, const T* input_imag, noa::Complex<T>* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input_real, input_real + elements, input_imag, output,
                       [](const T& real, const T& imag) { return Complex<T>(real, imag); });
    }
}

namespace noa::cpu::math {
    /// Element-wise min comparison between \p lhs and \p rhs, saving the minimum values in \p output.
    /// \tparam T           Any type with `T math::min(T, T)` defined.
    /// \param[in] lhs      On the \b host. Contiguous input data.
    /// \param[in] rhs      On the \b host. Contiguous input data.
    /// \param[out] output  On the \b host. Contiguous output data. Can be equal to \p lhs or \p rhs.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void min(const T* lhs, const T* rhs, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(lhs, lhs + elements, rhs, output, [](const T& a, const T& b) { return noa::math::min(a, b); });
    }

    /// Sets the values in \p output to the values in \p input, with the maximum value allowed being \p threshold.
    /// \tparam T           Any type with `math::min(T, T)` defined.
    /// \param[in] input    On the \b host. Contiguous input data.
    /// \param threshold    Threshold value.
    /// \param[out] output  On the \b host. Contiguous output data. Can be equal to \p input.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void min(const T* input, T threshold, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [threshold](const T& a) { return noa::math::min(a, threshold); });
    }

    /// Element-wise max comparison between \p lhs and \p rhs, saving the maximum values in \p output.
    /// \tparam T           Any type with `math::max(T, T)` defined.
    /// \param[in] lhs      On the \b host. Contiguous input data.
    /// \param[in] rhs      On the \b host. Contiguous input data.
    /// \param[out] output  On the \b host. Contiguous output data. Can be equal to \p lhs or \p rhs.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void max(const T* lhs, const T* rhs, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(lhs, lhs + elements, rhs, output, [](const T& a, const T& b) { return noa::math::max(a, b); });
    }

    /// Sets the values in \p output to the values in \p input, with the minimum value allowed being \p threshold.
    /// \tparam T           Any type with `math::max(T, T)` defined.
    /// \param[in] input    On the \b host. Contiguous input data.
    /// \param threshold    Threshold value.
    /// \param[out] output  On the \b host. Contiguous output data. Can be equal to \p input.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void max(const T* input, T threshold, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [threshold](const T& a) { return noa::math::max(a, threshold); });
    }

    /// Clamps the \p input with a minimum and maximum value.
    /// \tparam T           Any type with `T math::clamp(T, T)` defined.
    /// \param[in] input    On the \b host. Contiguous input data.
    /// \param low          Minimum value allowed in \p output.
    /// \param high         Maximum value allowed in \p output.
    /// \param[out] output  On the \b host. Contiguous output data. Can be equal to \p input.
    /// \param elements     Number of elements compute.
    template<typename T>
    NOA_IH void clamp(const T* input, T low, T high, T* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [low, high](const T& a) { return noa::math::clamp(a, low, high); });
    }
}
