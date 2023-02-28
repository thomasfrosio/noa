#include "noa/algorithms/signal/Bandpass.hpp"
#include "noa/core/Math.hpp"
#include "noa/gpu/cuda/fft/Exception.hpp"
#include "noa/gpu/cuda/signal/fft/Bandpass.h"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace noa::cuda::signal::fft {
    template<noa::fft::Remap REMAP, typename T, typename>
    void lowpass(const T* input, const Strides4<i64>& input_strides,
                 T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                 f32 cutoff, f32 width, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto i_shape = shape.as_safe<i32>();
        if (width > 1e-6f) {
            auto kernel = noa::algorithm::signal::lowpass<REMAP, true>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    i_shape, cutoff, width);
            noa::cuda::utils::iwise_4d("lowpass", i_shape.fft(), kernel, stream);
        } else {
            auto kernel = noa::algorithm::signal::lowpass<REMAP, false>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    i_shape, cutoff, width);
            noa::cuda::utils::iwise_4d("lowpass", i_shape.fft(), kernel, stream);
        }
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void highpass(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  f32 cutoff, f32 width, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto i_shape = shape.as_safe<i32>();
        if (width > 1e-6f) {
            auto kernel = noa::algorithm::signal::highpass<REMAP, true>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    i_shape, cutoff, width);
            noa::cuda::utils::iwise_4d("highpass", i_shape.fft(), kernel, stream);
        } else {
            auto kernel = noa::algorithm::signal::highpass<REMAP, false>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    i_shape, cutoff, width);
            noa::cuda::utils::iwise_4d("highpass", i_shape.fft(), kernel, stream);
        }
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void bandpass(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  f32 cutoff_high, f32 cutoff_low, f32 width_high, f32 width_low, Stream& stream) {
        using Layout = ::noa::fft::Layout;
        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_OR_NULL_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        const auto i_shape = shape.as_safe<i32>();
        if (width_high > 1e-6f || width_low > 1e-6f) {
            auto kernel = noa::algorithm::signal::bandpass<REMAP, true>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(), i_shape,
                    cutoff_high, cutoff_low, width_high, width_low);
            noa::cuda::utils::iwise_4d("bandpass", i_shape.fft(), kernel, stream);
        } else {
            auto kernel = noa::algorithm::signal::bandpass<REMAP, false>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(), i_shape,
                    cutoff_high, cutoff_low, width_high, width_low);
            noa::cuda::utils::iwise_4d("bandpass", i_shape.fft(), kernel, stream);
        }
    }

    #define NOA_INSTANTIATE_FILTERS_(R, T)                                                                                              \
    template void lowpass<R, T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, f32, f32, Stream&);  \
    template void highpass<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, f32, f32, Stream&);  \
    template void bandpass<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, f32, f32, f32, f32, Stream&)

    #define NOA_INSTANTIATE_FILTERS_ALL(T)                  \
    NOA_INSTANTIATE_FILTERS_(noa::fft::Remap::H2H, T);      \
    NOA_INSTANTIATE_FILTERS_(noa::fft::Remap::H2HC, T);     \
    NOA_INSTANTIATE_FILTERS_(noa::fft::Remap::HC2H, T);     \
    NOA_INSTANTIATE_FILTERS_(noa::fft::Remap::HC2HC, T)

    NOA_INSTANTIATE_FILTERS_ALL(f16);
    NOA_INSTANTIATE_FILTERS_ALL(f32);
    NOA_INSTANTIATE_FILTERS_ALL(f64);
    NOA_INSTANTIATE_FILTERS_ALL(c16);
    NOA_INSTANTIATE_FILTERS_ALL(c32);
    NOA_INSTANTIATE_FILTERS_ALL(c64);
}
