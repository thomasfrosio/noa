#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::fft::details {
    using Remap = ::noa::fft::Remap;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_resize = (traits::is_float_v<T> || traits::is_complex_v<T>) &&
                                     (REMAP == Remap::H2H || REMAP == Remap::F2F);

    template<typename T>
    void cropH2H(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                 AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape);
    template<typename T>
    void cropF2F(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                 AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape);
    template<typename T>
    void padH2H(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape);
    template<typename T>
    void padF2F(AccessorRestrict<const T, 4, dim_t> input, dim4_t input_shape,
                AccessorRestrict<T, 4, dim_t> output, dim4_t output_shape);
}

namespace noa::cpu::fft {
    using Remap = ::noa::fft::Remap;

    // Crops or zero-pads a FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_resize<REMAP, T>>>
    inline void resize(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                       const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        if (all(input_shape >= output_shape)) {
            if constexpr (REMAP == Remap::H2H) {
                stream.enqueue([=]() {
                    details::cropH2H<T>({input.get(), input_strides}, input_shape,
                                        {output.get(), output_strides}, output_shape);
                });
            } else if constexpr (REMAP == Remap::F2F) {
                stream.enqueue([=]() {
                    details::cropF2F<T>({input.get(), input_strides}, input_shape,
                                        {output.get(), output_strides}, output_shape);
                });
            } else {
                static_assert(traits::always_false_v<T>);
            }
        } else {
            if constexpr (REMAP == Remap::H2H) {
                stream.enqueue([=]() {
                    details::padH2H<T>({input.get(), input_strides}, input_shape,
                                       {output.get(), output_strides}, output_shape);
                });
            } else if constexpr (REMAP == Remap::F2F) {
                stream.enqueue([=]() {
                    details::padF2F<T>({input.get(), input_strides}, input_shape,
                                       {output.get(), output_strides}, output_shape);
                });
            } else {
                static_assert(traits::always_false_v<T>);
            }
        }
    }
}
