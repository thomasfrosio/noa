#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

namespace noa::cpu::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_pass_v = (traits::is_float_v<T> || traits::is_complex_v<T>) &&
                                     (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::cpu::signal::fft {
    using noa::fft::Remap;

    // Lowpass FFTs.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void lowpass(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float cutoff, float width, Stream& stream);

    // Highpass FFTs.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void highpass(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float cutoff, float width, Stream& stream);

    // Bandpass FFTs.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_pass_v<REMAP, T>>>
    void bandpass(const shared_t<T[]>& input, dim4_t input_strides,
                  const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                  float cutoff1, float cutoff2, float width1, float width2, Stream& stream);
}
