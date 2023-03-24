#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

// TODO(TF) Add all remaining layouts

namespace noa::cuda::signal::fft::details {
    using namespace ::noa::fft;
    template<Remap REMAP, typename T>
    constexpr bool is_valid_shift_v =
            noa::traits::is_any_v<T, c32, c64> &&
            (REMAP == H2H || REMAP == H2HC || REMAP == HC2H || REMAP == HC2HC);
}

namespace noa::cuda::signal::fft {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void phase_shift_2d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec2<f32>* shifts, f32 cutoff, Stream& stream);

    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void phase_shift_2d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec2<f32>& shift, f32 cutoff, Stream& stream);

    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void phase_shift_3d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec3<f32>* shifts, f32 cutoff, Stream& stream);

    template<Remap REMAP, typename T, typename = std::enable_if_t<details::is_valid_shift_v<REMAP, T>>>
    void phase_shift_3d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec3<f32>& shift, f32 cutoff, Stream& stream);
}

