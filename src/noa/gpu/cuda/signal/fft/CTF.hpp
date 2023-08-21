#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTFIsotropic>
    void ctf_isotropic(
            const Input* input, Strides4<i64> input_strides,
            Output* output, Strides4<i64> output_strides, Shape4<i64> shape,
            const CTFIsotropic& ctf, bool ctf_abs, bool ctf_square, Stream& stream);

    template<noa::fft::Remap REMAP, typename Output, typename CTFIsotropic>
    void ctf_isotropic(
            Output* output, Strides4<i64> output_strides, Shape4<i64> shape,
            const CTFIsotropic& ctf, bool ctf_abs, bool ctf_square,
            const Vec2<f32>& fftfreq_range, bool fftfreq_range_endpoint, Stream& stream);

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTFAnisotropic>
    void ctf_anisotropic(
            const Input* input, const Strides4<i64>& input_strides,
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const CTFAnisotropic& ctf, bool ctf_abs, bool ctf_square, Stream& stream);

    template<noa::fft::Remap REMAP, typename Output, typename CTFAnisotropic>
    void ctf_anisotropic(
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
            const CTFAnisotropic& ctf, bool ctf_abs, bool ctf_square,
            const Vec2<f32>& fftfreq_range, bool fftfreq_range_endpoint, Stream& stream);
}
