/// \file noa/cpu/math/Complex.h
/// \brief Decompose complex numbers into real and imaginary components.
/// \author Thomas - ffyr2w
/// \date 12 Jan 2022

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

#include "noa/cpu/Stream.h"
#include "noa/cpu/math/Ewise.h"

namespace noa::cpu::math {
    /// Extracts the real and imaginary part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b host. Complex array to decompose.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] real        On the \b host. Real elements.
    /// \param real_stride      Rightmost stride, in elements, of \p real.
    /// \param[out] imag        On the \b host. Imaginary elements.
    /// \param imag_stride      Rightmost stride, in elements, of \p imag.
    /// \param shape            Rightmost shape of \p input, \p real and \p imag.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void decompose(const shared_t<Complex<T>[]>& input, size4_t input_stride,
                   const shared_t<T[]>& real, size4_t real_stride,
                   const shared_t<T[]>& imag, size4_t imag_stride,
                   size4_t shape, Stream& stream) {
        stream.enqueue([=]() {
            const Complex<T> ptr = input.get();
            T* rptr = real.get();
            T* iptr = imag.get();
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        for (size_t l = 0; l < shape[3]; ++l) {
                            const size_t offset = indexing::at(i, j, k, l, input_stride);
                            rptr[indexing::at(i, j, k, l, real_stride)] = ptr[offset].real;
                            iptr[indexing::at(i, j, k, l, imag_stride)] = ptr[offset].imag;
                        }
                    }
                }
            }
        });
    }

    /// Extracts the real part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b host. Complex array to decompose.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] real        On the \b host. Real elements.
    /// \param real_stride      Rightmost stride, in elements, of \p real.
    /// \param shape            Rightmost shape of \p input and \p real.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    NOA_IH void real(const shared_t<Complex<T>[]>& input, size4_t input_stride,
                     const shared_t<T[]>& real, size4_t real_stride,
                     size4_t shape, Stream& stream) {
        cpu::math::ewise(input, input_stride, real, real_stride, shape, noa::math::real_t{}, stream);
    }

    /// Extracts the imaginary part of complex numbers.
    /// \tparam T               half_t, float, double.
    /// \param[in] input        On the \b host. Complex array to decompose.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] imag        On the \b host. Imaginary elements.
    /// \param imag_stride      Rightmost stride, in elements, of \p imag.
    /// \param shape            Rightmost shape of \p input and \p imag.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    NOA_IH void imag(const shared_t<Complex<T>[]>& input, size4_t input_stride,
                     const shared_t<T[]>& imag, size4_t imag_stride,
                     size4_t shape, Stream& stream) {
        cpu::math::ewise(input, input_stride, imag, imag_stride, shape, noa::math::imag_t{}, stream);
    }

    /// Fuses the real and imaginary components.
    /// \tparam T               half_t, float, double.
    /// \param[in] real         On the \b host. Real elements to interleave.
    /// \param real_stride      Rightmost strides, in elements, of \p real.
    /// \param[in] imag         On the \b host. Imaginary elements to interleave.
    /// \param imag_stride      Rightmost strides, in elements, of \p imag.
    /// \param output           On the \b host. Complex array.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape of \p real, \p imag and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    NOA_IH void complex(const shared_t<T[]>& real, size4_t real_stride,
                        const shared_t<T[]>& imag, size4_t imag_stride,
                        const shared_t<Complex<T>[]>& output, size4_t output_stride,
                        size4_t shape, Stream& stream) {
        return math::ewise(real, real_stride, imag, imag_stride, output, output_stride, shape,
                           [](const T& r, const T& i) { return noa::Complex<T>(r, i); }, stream);
    }
}
