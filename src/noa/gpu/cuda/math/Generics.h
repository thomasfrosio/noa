/// \file noa/gpu/cuda/math/Generics.h
/// \brief Generic math functions for arrays.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::math::details {
    enum : int {
        GEN_ONE_MINUS, GEN_INVERSE, GEN_SQUARE, GEN_SQRT, GEN_RSQRT, GEN_EXP, GEN_LOG, GEN_ABS, GEN_COS, GEN_SIN,
        GEN_NORMALIZE, GEN_REAL, GEN_IMAG, GEN_COMPLEX, GEN_POW, GEN_MIN, GEN_MAX
    };

    template<int GEN, typename T, typename R>
    NOA_HOST void generic(const T* input, R* output, size_t elements, Stream& stream);

    template<int GEN, typename T, typename R>
    NOA_HOST void genericWithValue(const T* input, T value, R* output, size_t elements, Stream& stream);

    template<int GEN, typename T, typename R>
    NOA_HOST void genericWithArray(const T* input, const T* array, R* output, size_t elements, Stream& stream);

    template<int GEN, typename T, typename R>
    NOA_HOST void generic(const T* input, size_t input_pitch, R* output, size_t output_pitch,
                          size3_t shape, Stream& stream);

    template<int GEN, typename T, typename R>
    NOA_HOST void genericWithValue(const T* input, size_t input_pitch, T value, R* output, size_t output_pitch,
                                   size3_t shape, Stream& stream);

    template<int GEN, typename T, typename R>
    NOA_HOST void genericWithArray(const T* input, size_t input_pitch, const T* array, size_t array_pitch,
                                   R* output, size_t output_pitch, size3_t shape, Stream& stream);
}

namespace noa::cuda::math {
    /// Subtracts \p input to one, i.e. T(1) - x.
    /// \tparam T               short, int, long, long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void oneMinus(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_ONE_MINUS>(input, output, elements, stream);
    }

    /// Subtracts \p input to one, i.e. T(1) - x.
    /// \tparam T               short, int, long, long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void oneMinus(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                         size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_ONE_MINUS>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the inverse of \p input, i.e. T(1) / x.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void inverse(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_INVERSE>(input, output, elements, stream);
    }

    /// Computes the inverse of \p input, i.e. T(1) / x.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void inverse(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                        size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_INVERSE>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the square of \p input, i.e. x * x.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void square(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_SQUARE>(input, output, elements, stream);
    }

    /// Computes the square of \p input, i.e. x * x.
    /// \tparam T               (u)short, (u)int, (u)long, (u)long long, float, double, cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void square(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                       size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_SQUARE>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the square root of \p input, i.e. math::sqrt(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void sqrt(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_SQRT>(input, output, elements, stream);
    }

    /// Computes the square root of \p input, i.e. math::sqrt(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void sqrt(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                     size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_SQRT>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the inverse square root of \p input, i.e. math::rsqrt(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void rsqrt(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_RSQRT>(input, output, elements, stream);
    }

    /// Computes the inverse square root of \p input, i.e. math::rsqrt(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void rsqrt(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                      size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_RSQRT>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the power of \p input, i.e. math::pow(x, exponent).
    /// \tparam T       float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param exponent         Exponent.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void pow(const T* input, T exponent, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithValue<details::GEN_POW>(input, exponent, output, elements, stream);
    }

    /// Computes the power of \p input, i.e. math::pow(x, exponent).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param exponent         Exponent.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void pow(const T* input, size_t input_pitch, T exponent, T* output, size_t output_pitch,
                    size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithValue<details::GEN_POW>(input, input_pitch, exponent, output, output_pitch, shape, stream);
    }

    /// Computes the exponential of \p input, i.e. math::exp(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void exp(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_EXP>(input, output, elements, stream);
    }

    /// Computes the exponential of \p input, i.e. math::exp(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void exp(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                    size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_EXP>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the log of \p input, i.e. math::log(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void log(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_LOG>(input, output, elements, stream);
    }

    /// Computes the log of \p input, i.e. math::log(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void log(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                    size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_LOG>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the abs of \p input, i.e. math::abs(x).
    /// \tparam T               short, int, long, long long, float, double, cfloat, cdouble_t.
    /// \tparam T               If \p T is complex, \p R is the corresponding value type, otherwise, same as \p T.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename R>
    NOA_IH void abs(const T* input, R* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_ABS>(input, output, elements, stream);
    }

    /// Computes the abs of \p input, i.e. math::abs(x).
    /// \tparam T               short, int, long, long long, float, double, cfloat, cdouble_t
    /// \tparam T               If \p T is complex, \p R is the corresponding value type, otherwise, same as \p T.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T, typename R>
    NOA_IH void abs(const T* input, size_t input_pitch, R* output, size_t output_pitch,
                    size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_ABS>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the cosine of \p input, i.e. math::cos(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void cos(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_COS>(input, output, elements, stream);
    }

    /// Computes the cosine of \p input, i.e. math::cos(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void cos(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                    size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_COS>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Computes the sin of \p input, i.e. math::sin(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void sin(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_SIN>(input, output, elements, stream);
    }

    /// Computes the sin of \p input, i.e. math::sin(x).
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void sin(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                    size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_SIN>(input, input_pitch, output, output_pitch, shape, stream);
    }
}

namespace noa::cuda::math {
    /// Returns the length-normalized of complex numbers to 1, reducing them to their phase.
    /// \tparam T               cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void normalize(const T* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_NORMALIZE>(input, output, elements, stream);
    }

    /// Returns the length-normalized of complex numbers to 1, reducing them to their phase.
    /// \tparam T               cfloat_t, cdouble_t.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] output       On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void normalize(const T* input, size_t input_pitch, T* output, size_t output_pitch,
                          size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_NORMALIZE>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Extracts the real parts of complex numbers.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array with the real parts.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void real(const Complex<T>* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_REAL>(input, output, elements, stream);
    }

    /// Extracts the real parts of complex numbers.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] output      On the \b device. Contiguous output array with the real parts.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void real(const Complex<T>* input, size_t input_pitch, T* output, size_t output_pitch,
                     size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_REAL>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Extracts the imaginary parts of complex numbers.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array with the imaginary parts.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void imag(const Complex<T>* input, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_IMAG>(input, output, elements, stream);
    }

    /// Extracts the imaginary parts of complex numbers.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] output      On the \b device. Contiguous output array with the imaginary parts.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void imag(const Complex<T>* input, size_t input_pitch, T* output, size_t output_pitch,
                     size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::generic<details::GEN_IMAG>(input, input_pitch, output, output_pitch, shape, stream);
    }

    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T                   float, double.
    /// \param[in] input            On the \b device. Contiguous input array.
    /// \param[out] output_real     On the \b device. Contiguous output array with the real parts.
    /// \param[out] output_imag     On the \b device. Contiguous output array with the imaginary parts.
    /// \param elements             Number of elements in \p input to compute.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void realAndImag(const Complex<T>* input, T* output_real, T* output_imag, size_t elements,
                              Stream& stream);

    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T                   float, double.
    /// \param input                On the \b device. Contiguous input array.
    /// \param[in] input_pitch      Pitch, in elements, of \p input.
    /// \param[out] output_real     On the \b device. Contiguous output array with the real parts.
    /// \param output_real_pitch    Pitch, in elements, of \p output_real.
    /// \param[out] output_imag     On the \b device. Contiguous output array with the imaginary parts.
    /// \param output_imag_pitch    Pitch, in elements, of \p output_imag.
    /// \param shape                Logical {fast, medium, slow} shape of \p input, \p output_real and \p output_imag.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void realAndImag(const Complex<T>* input, size_t input_pitch,
                              T* output_real, size_t output_real_pitch,
                              T* output_imag, size_t output_imag_pitch,
                              size3_t shape, Stream& stream);

    /// Gathers the real and imaginary parts into complex numbers.
    /// \tparam T               float, double.
    /// \param[in] input_real   On the \b device. Contiguous input array with the real parts.
    /// \param[in] input_imag   On the \b device. Contiguous input array with the imaginary parts.
    /// \param[out] output      On the \b device. Contiguous output array.
    /// \param elements         Number of elements in \p input_real and \p input_imag to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void complex(const T* input_real, const T* input_imag,
                        Complex<T>* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithArray<details::GEN_COMPLEX>(input_real, input_imag, output, elements, stream);
    }

    /// Gathers the real and imaginary parts into complex numbers.
    /// \tparam T               float, double.
    /// \param[in] input_real   On the \b device. Contiguous input array with the real parts.
    /// \param input_real_pitch Pitch, in elements, of \p input_real.
    /// \param[in] input_imag   On the \b device. Contiguous input array with the imaginary parts.
    /// \param input_imag_pitch Pitch, in elements, of \p input_imag.
    /// \param[out] output      On the \b device. Contiguous output array.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input_real, \p input_imag and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void complex(const T* input_real, size_t input_real_pitch, const T* input_imag, size_t input_imag_pitch,
                        Complex<T>* output, size_t output_pitch, size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithArray<details::GEN_COMPLEX>(input_real, input_real_pitch, input_imag, input_imag_pitch,
                                                        output, output_pitch, shape, stream);
    }
}

namespace noa::cuda::math {
    /// Element-wise min comparison between \p lhs and \p rhs, saving the minimum values in \p output.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] lhs          On the \b device. Contiguous input array.
    /// \param[in] rhs          On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array with the minimum values.
    ///                         Can be equal to \p lhs or \p rhs.
    /// \param elements         Number of elements in \p lhs and \p rhs to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void min(const T* lhs, const T* rhs, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithArray<details::GEN_MIN>(lhs, rhs, output, elements, stream);
    }

    /// Element-wise min comparison between \p lhs and \p rhs, saving the minimum values in \p output.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] lhs          On the \b device. Contiguous input array.
    /// \param lhs_pitch        Pitch, in elements, of \p lhs.
    /// \param[in] rhs          On the \b device. Contiguous input array.
    /// \param rhs_pitch        Pitch, in elements, of \p rhs.
    /// \param[out] output      On the \b device. Contiguous output array with the minimum values.
    ///                         Can be equal to \p lhs or \p rhs.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p lhs, \p rhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void min(const T* lhs, size_t lhs_pitch, const T* rhs, size_t rhs_pitch,
                    T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithArray<details::GEN_MIN>(lhs, lhs_pitch, rhs, rhs_pitch, output, output_pitch,
                                                    shape, stream);
    }

    /// Sets the values in \p output to the values in \p input, with the maximum value allowed being \p threshold.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param threshold        Maximum value allowed in \p output.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void min(const T* input, T threshold, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithValue<details::GEN_MIN>(input, threshold, output, elements, stream);
    }

    /// Sets the values in \p output to the values in \p input, with the maximum value allowed being \p threshold.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param threshold        Maximum value allowed in \p output.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void min(const T* input, size_t input_pitch, T threshold,
                    T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithValue<details::GEN_MIN>(input, input_pitch, threshold, output, output_pitch, shape, stream);
    }

    /// Element-wise max comparison between \p lhs and \p rhs, saving the maximum values in \p output.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] lhs          On the \b device. Contiguous input array.
    /// \param[in] rhs          On the \b device. Contiguous input array.
    /// \param[out] output      On the \b device. Contiguous output array with the maximum values.
    ///                         Can be equal to \p lhs or \p rhs.
    /// \param elements         Number of elements in \p lhs and \p rhs to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void max(const T* lhs, const T* rhs, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithArray<details::GEN_MAX>(lhs, rhs, output, elements, stream);
    }

    /// Element-wise max comparison between \p lhs and \p rhs, saving the maximum values in \p output.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] lhs          On the \b device. Contiguous input array.
    /// \param lhs_pitch        Pitch, in elements, of \p lhs.
    /// \param[in] rhs          On the \b device. Contiguous input array.
    /// \param rhs_pitch        Pitch, in elements, of \p rhs.
    /// \param[out] output      On the \b device. Contiguous output array with the maximum values.
    ///                         Can be equal to \p lhs or \p rhs.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p lhs, \p rhs and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void max(const T* lhs, size_t lhs_pitch, const T* rhs, size_t rhs_pitch,
                    T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithArray<details::GEN_MAX>(lhs, lhs_pitch, rhs, rhs_pitch, output, output_pitch,
                                                    shape, stream);
    }

    /// Sets the values in \p output to the values in \p input, with the minimum value allowed being \p threshold.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param threshold        Minimum value allowed in \p output.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void max(const T* input, T threshold, T* output, size_t elements, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithValue<details::GEN_MAX>(input, threshold, output, elements, stream);
    }

    /// Sets the values in \p output to the values in \p input, with the minimum value allowed being \p threshold.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param threshold        Minimum value allowed in \p output.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void max(const T* input, size_t input_pitch, T threshold,
                    T* output, size_t output_pitch, size3_t shape, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        details::genericWithValue<details::GEN_MAX>(input, input_pitch, threshold, output, output_pitch, shape, stream);
    }

    /// Clamps the \p input with a minimum and maximum value.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param low              Minimum value allowed in \p output.
    /// \param high             Maximum value allowed in \p output.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param elements         Number of elements in \p input to compute.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void clamp(const T* input, T low, T high, T* output, size_t elements, Stream& stream);

    /// Clamps the \p input with a minimum and maximum value.
    /// \tparam T               (u)char, (u)short, (u)int, (u)long, (u)long long, float, double.
    /// \param[in] input        On the \b device. Contiguous input array.
    /// \param input_pitch      Pitch, in elements, of \p input.
    /// \param low              Minimum value allowed in \p output.
    /// \param high             Maximum value allowed in \p output.
    /// \param[out] output      On the \b device. Contiguous output array. Can be equal to \p input.
    /// \param output_pitch     Pitch, in elements, of \p output.
    /// \param shape            Logical {fast, medium, slow} shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_HOST void clamp(const T* input, size_t input_pitch, T low, T high,
                        T* output, size_t output_pitch, size3_t shape, Stream& stream);
}
