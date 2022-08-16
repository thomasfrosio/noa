#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::fft::details {
    template<typename T>
    void cropH2H(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape, Stream& stream);
    template<typename T>
    void cropF2F(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape, Stream& stream);
    template<typename T>
    void padH2H(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape, Stream& stream);
    template<typename T>
    void padF2F(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape, Stream& stream);

    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_remap_v =
            (traits::is_float_v<T> || traits::is_complex_v<T>) && (REMAP == H2H || REMAP == F2F);
}

namespace noa::cuda::fft {
    using Remap = ::noa::fft::Remap;

    // Crops or zero-pads a FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_remap_v<REMAP, T>>>
    inline void resize(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                       const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape, Stream& stream) {
        if (all(input_shape >= output_shape)) {
            if constexpr (REMAP == Remap::H2H)
                details::cropH2H<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else if constexpr (REMAP == Remap::F2F)
                details::cropF2F<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else
                static_assert(traits::always_false_v<T>);
        } else {
            if constexpr (REMAP == Remap::H2H)
                details::padH2H<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else if constexpr (REMAP == Remap::F2F)
                details::padF2F<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else
                static_assert(traits::always_false_v<T>);
        }
    }
}
