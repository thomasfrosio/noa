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
#include "noa/cpu/memory/PtrHost.h"

namespace noa::cpu::fft::details {
    template<typename T>
    void h2f(AccessorRestrict<const T, 4, dim_t> input,
             AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void h2hc(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void hc2h(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void hc2f(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void f2h(AccessorRestrict<const T, 4, dim_t> input,
             AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void f2hc(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void f2fc(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void fc2h(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void fc2hc(AccessorRestrict<const T, 4, dim_t> input,
               AccessorRestrict<T, 4, dim_t> output, dim4_t shape);

    template<typename T>
    void fc2f(AccessorRestrict<const T, 4, dim_t> input,
              AccessorRestrict<T, 4, dim_t> output, dim4_t shape);
}

namespace noa::cpu::fft {
    using Remap = ::noa::fft::Remap;

    // Remaps FFT(s).
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T> || traits::is_complex_v<T>>>
    void remap(Remap remap,
               const shared_t<T[]>& input, dim4_t input_strides,
               const shared_t<T[]>& output, dim4_t output_strides,
               dim4_t shape, Stream& stream) {
        switch (remap) {
            case Remap::H2H:
            case Remap::HC2HC:
                if (input != output)
                    memory::copy(input, input_strides, output, output_strides, shape.fft(), stream);
                break;
            case Remap::F2F:
            case Remap::FC2FC:
                if (input != output)
                    memory::copy(input, input_strides, output, output_strides, shape, stream);
                break;
            case Remap::H2HC:
                return stream.enqueue([=]() {
                    details::h2hc<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::HC2H:
                return stream.enqueue([=]() {
                    details::hc2h<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::H2F:
                return stream.enqueue([=]() {
                    details::h2f<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::F2H:
                return stream.enqueue([=]() {
                    details::f2h<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::F2FC:
                return stream.enqueue([=]() {
                    details::f2fc<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::FC2F:
                return stream.enqueue([=]() {
                    details::fc2f<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::HC2F:
                return stream.enqueue([=]() {
                    details::hc2f<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::F2HC:
                return stream.enqueue([=]() {
                    details::f2hc<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::FC2H:
                return stream.enqueue([=]() {
                    details::fc2h<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case noa::fft::FC2HC:
                return stream.enqueue([=]() {
                    details::fc2hc<T>({input.get(), input_strides}, {output.get(), output_strides}, shape);
                });
            case Remap::HC2FC:
                return stream.enqueue([=]() {
                    memory::PtrHost<T> tmp{shape.elements()};
                    details::hc2f<T>({input.get(), input_strides}, {tmp.get(), shape.strides()}, shape);
                    details::f2fc<T>({tmp.get(), shape.strides()}, {output.get(), output_strides}, shape);
                });
            case Remap::H2FC:
                return stream.enqueue([=]() {
                    memory::PtrHost<T> tmp{shape.elements()};
                    details::h2f<T>({input.get(), input_strides}, {tmp.get(), shape.strides()}, shape);
                    details::f2fc<T>({tmp.get(), shape.strides()}, {output.get(), output_strides}, shape);
                });
        }
    }
}
