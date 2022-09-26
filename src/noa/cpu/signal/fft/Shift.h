#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"

// TODO(TF) Add all remaining layouts

namespace noa::cpu::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_shift_v =
            traits::is_any_v<T, cfloat_t, cdouble_t> &&
            (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::cpu::signal::fft {
    using Remap = noa::fft::Remap;

    // Phase-shifts a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 const shared_t<float2_t[]>& shifts, float cutoff, Stream& stream);

    // Phase-shifts a non-redundant 2D (batched) FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float2_t shift, float cutoff, Stream& stream);

    // Phase-shifts a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift3D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 const shared_t<float3_t[]>& shifts, float cutoff, Stream& stream);

    //  Phase-shifts a non-redundant 3D (batched) FFT.
    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void shift3D(const shared_t<T[]>& input, dim4_t input_strides,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t shape,
                 float3_t shift, float cutoff, Stream& stream);
}
