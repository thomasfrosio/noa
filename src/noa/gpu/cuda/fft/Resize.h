#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"

namespace noa::cuda::fft::details {
    template<typename T>
    void cropH2H(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);
    template<typename T>
    void cropF2F(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);
    template<typename T>
    void padH2H(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);
    template<typename T>
    void padF2F(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream);

    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_remap_v =
            (traits::is_float_v<T> || traits::is_complex_v<T>) && (REMAP == H2H || REMAP == F2F);
}

namespace noa::cuda::fft {
    using Remap = ::noa::fft::Remap;

    // Crops or zero-pads a FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_remap_v<REMAP, T>>>
    inline void resize(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                       const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        NOA_ASSERT(input && output && input.get() != output.get() && all(input_shape > 0) && all(input_shape > 0));
        NOA_ASSERT(input_shape[0] == output_shape[0]);

        if (all(input_shape >= output_shape)) {
            if constexpr (REMAP == Remap::H2H)
                details::cropH2H<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else if constexpr (REMAP == Remap::F2F)
                details::cropF2F<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else
                static_assert(traits::always_false_v<T>);
        } else if (all(input_shape <= output_shape)) {
            if constexpr (REMAP == Remap::H2H)
                details::padH2H<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else if constexpr (REMAP == Remap::F2F)
                details::padF2F<T>(input, input_strides, input_shape, output, output_strides, output_shape, stream);
            else
                static_assert(traits::always_false_v<T>);
        } else {
            NOA_THROW("Cannot crop and pad at the same time");
        }
    }
}
