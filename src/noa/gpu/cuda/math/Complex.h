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
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param[out] real        On the \b device. Real elements.
    /// \param real_strides     Strides, in elements, of \p real.
    /// \param[out] imag        On the \b device. Imaginary elements.
    /// \param imag_strides     Strides, in elements, of \p imag.
    /// \param shape            Shape of \p input, \p real and \p imag.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void decompose(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                   const shared_t<T[]>& real, size4_t real_strides,
                   const shared_t<T[]>& imag, size4_t imag_strides,
                   size4_t shape, Stream& stream);

    /// Extracts the real part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b device. Complex array to decompose.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param[out] real        On the \b device. Real elements.
    /// \param real_strides     Strides, in elements, of \p real.
    /// \param shape            Shape of \p input and \p real.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void real(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                     const shared_t<T[]>& real, size4_t real_strides,
                     size4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_strides, real, real_strides, shape, noa::math::real_t{}, stream);
    }

    /// Extracts the imaginary part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b device. Complex array to decompose.
    /// \param input_strides    Strides, in elements, of \p input.
    /// \param[out] imag        On the \b device. Imaginary elements.
    /// \param imag_strides     Strides, in elements, of \p imag.
    /// \param shape            Shape of \p input and \p imag.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void imag(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                     const shared_t<T[]>& imag, size4_t imag_strides,
                     size4_t shape, Stream& stream) {
        cuda::math::ewise(input, input_strides, imag, imag_strides, shape, noa::math::imag_t{}, stream);
    }

    /// Fuses the real and imaginary components.
    /// \tparam T               half_t, float, double.
    /// \param[in] real         On the \b device. Real elements to interleave.
    /// \param real_strides     Strides, in elements, of \p real.
    /// \param[in] imag         On the \b device. Imaginary elements to interleave.
    /// \param imag_strides     Strides, in elements, of \p imag.
    /// \param output           On the \b device. Complex array.
    /// \param output_strides   Strides, in elements, of \p output.
    /// \param shape            Shape of \p real, \p imag and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void complex(const shared_t<T[]>& real, size4_t real_strides,
                 const shared_t<T[]>& imag, size4_t imag_strides,
                 const shared_t<Complex<T>[]>& output, size4_t output_strides,
                 size4_t shape, Stream& stream);
}
