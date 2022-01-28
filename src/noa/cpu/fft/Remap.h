/// \file noa/cpu/fft/Remap.h
/// \brief Remap FFTs.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Exception.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/memory/Copy.h"

namespace noa::cpu::fft::details {
    template<typename T>
    NOA_HOST void hc2h(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void h2hc(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void fc2f(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void f2fc(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void h2f(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void f2h(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void fc2h(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void hc2f(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);

    template<typename T>
    NOA_HOST void f2hc(const T* input, size4_t input_stride, T* output, size4_t output_stride, size4_t shape);
}

namespace noa::cpu::fft {
    using Remap = ::noa::fft::Remap;

    /// Remaps FFT(s).
    /// \tparam T               float, double, cfloat_t or cdouble_t.
    /// \param remap            Remapping operation. \p H2FC is not supported. See noa::fft::Remap for more details.
    /// \param[in] input        On the \b host. Input FFT to remap.
    /// \param input_stride     Rightmost strides, in elements, of \p input.
    /// \param[out] output      On the \b host. Remapped FFT.
    /// \param output_stride    Rightmost strides, in elements, of \p output.
    /// \param shape            Rightmost shape, in elements.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note If no remapping is done, e.g. H2H, a copy is performed for if \p input is not equal to \p output.
    /// \note If \p remap is \c H2HC, \p input can be equal to \p output, only iff \p shape[2] is even,
    ///       and \p shape[1] is even or 1.
    template<typename T>
    NOA_IH void remap(Remap remap,
                      const T* input, size4_t input_stride,
                      T* output, size4_t output_stride,
                      size4_t shape, Stream& stream) {
        switch (remap) {
            case Remap::H2H:
            case Remap::HC2HC:
                if (input != output)
                    memory::copy(input, input_stride, output, output_stride, shape.fft(), stream);
                break;
            case Remap::F2F:
            case Remap::FC2FC:
                if (input != output)
                    memory::copy(input, input_stride, output, output_stride, shape, stream);
                break;
            case Remap::H2HC:
                return stream.enqueue(details::h2hc<T>, input, input_stride, output, output_stride, shape);
            case Remap::HC2H:
                return stream.enqueue(details::hc2h<T>, input, input_stride, output, output_stride, shape);
            case Remap::H2F:
                return stream.enqueue(details::h2f<T>, input, input_stride, output, output_stride, shape);
            case Remap::F2H:
                return stream.enqueue(details::f2h<T>, input, input_stride, output, output_stride, shape);
            case Remap::F2FC:
                return stream.enqueue(details::f2fc<T>, input, input_stride, output, output_stride, shape);
            case Remap::FC2F:
                return stream.enqueue(details::fc2f<T>, input, input_stride, output, output_stride, shape);
            case Remap::HC2F:
                return stream.enqueue(details::hc2f<T>, input, input_stride, output, output_stride, shape);
            case Remap::F2HC:
                return stream.enqueue(details::f2hc<T>, input, input_stride, output, output_stride, shape);
            case Remap::FC2H:
                return stream.enqueue(details::fc2h<T>, input, input_stride, output, output_stride, shape);
            case Remap::H2FC:
                NOA_THROW("{} is currently not supported", Remap::H2FC);
                // TODO H2FC is missing, since it seems a bit more complicated and it would be surprising
                //      if we ever use it. Moreover, the same can be achieved with h2f and then f2fc.
        }
    }
}
