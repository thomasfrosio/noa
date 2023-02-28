#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_pass_v =
            noa::traits::is_real_or_complex_v<T> &&
            (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void lowpass(const T* input, const Strides4<i64>& input_strides,
                 T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                 f32 cutoff, f32 width, Stream& stream);

    template<noa::fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void highpass(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  f32 cutoff, f32 width, Stream& stream);

    template<noa::fft::Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void bandpass(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  f32 cutoff_high, f32 cutoff_low, f32 width_high, f32 width_low, Stream& stream);
}
