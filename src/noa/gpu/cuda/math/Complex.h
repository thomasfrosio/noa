/// \file noa/gpu/cuda/math/Complex.h
/// \brief Decompose complex numbers into real and imaginary components.
/// \author Thomas - ffyr2w
/// \date 2 Feb 2022

#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/math/Ewise.h"

namespace noa::cuda::math {
    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b device. Complex array to decompose.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] real        On the \b device. Real elements.
    /// \param real_stride      Rightmost stride, in elements, of \p real.
    /// \param[out] imag        On the \b device. Imaginary elements.
    /// \param imag_stride      Rightmost stride, in elements, of \p imag.
    /// \param shape            Rightmost shape of \p input, \p real and \p imag.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void decompose(const noa::Complex<T>* input, size4_t input_stride,
                            T* real, size4_t real_stride,
                            T* imag, size4_t imag_stride,
                            size4_t shape, Stream& stream);

    /// Extracts the real part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b device. Complex array to decompose.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] real        On the \b device. Real elements.
    /// \param real_stride      Rightmost stride, in elements, of \p real.
    /// \param shape            Rightmost shape of \p input and \p real.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void real(const noa::Complex<T>* input, size4_t input_stride,
                     T* real, size4_t real_stride,
                     size4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_stride, real, real_stride, shape, noa::math::real_t{}, stream);
    }

    /// Extracts the imaginary part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b device. Complex array to decompose.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] imag        On the \b device. Imaginary elements.
    /// \param imag_stride      Rightmost stride, in elements, of \p imag.
    /// \param shape            Rightmost shape of \p input and \p imag.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void imag(const noa::Complex<T>* input, size4_t input_stride,
                     T* imag, size4_t imag_stride,
                     size4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_stride, imag, imag_stride, shape, noa::math::imag_t{}, stream);
    }

    /// Fuses the real and imaginary components.
    /// \tparam T               half_t, float, double.
    /// \param[in] real         On the \b device. Real elements to interleave.
    /// \param real_stride      Rightmost strides, in elements, of \p real.
    /// \param[in] imag         On the \b device. Imaginary elements to interleave.
    /// \param imag_stride      Rightmost strides, in elements, of \p imag.
    /// \param output           On the \b device. Complex array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p real, \p imag and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_HOST void complex(const T* real, size4_t real_stride,
                          const T* imag, size4_t imag_stride,
                          noa::Complex<T>* output, size4_t output_stride,
                          size4_t shape, Stream& stream);
}
