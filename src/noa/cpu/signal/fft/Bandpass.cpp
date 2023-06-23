#include "noa/algorithms/signal/Bandpass.hpp"
#include "noa/cpu/signal/fft/Bandpass.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::signal::fft {
    template<noa::fft::Remap REMAP, typename T, typename>
    void lowpass(const T* input, const Strides4<i64>& input_strides,
                 T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                 f32 cutoff, f32 width, i64 threads) {
        if (width > 1e-6f) {
            auto kernel = noa::algorithm::signal::lowpass<REMAP, true>(
                    input, input_strides, output, output_strides, shape, cutoff, width);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        } else {
            auto kernel = noa::algorithm::signal::lowpass<REMAP, false>(
                    input, input_strides, output, output_strides, shape, cutoff, width);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void highpass(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  f32 cutoff, f32 width, i64 threads) {
        if (width > 1e-6f) {
            auto kernel = noa::algorithm::signal::highpass<REMAP, true>(
                    input, input_strides, output, output_strides, shape, cutoff, width);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        } else {
            auto kernel = noa::algorithm::signal::highpass<REMAP, false>(
                    input, input_strides, output, output_strides, shape, cutoff, width);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void bandpass(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                  f32 cutoff_high, f32 cutoff_low, f32 width_high, f32 width_low, i64 threads) {
        if (width_high > 1e-6f || width_low > 1e-6f) {
            auto kernel = noa::algorithm::signal::bandpass<REMAP, true>(
                    input, input_strides, output, output_strides, shape,
                    cutoff_high, cutoff_low, width_high, width_low);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        } else {
            auto kernel = noa::algorithm::signal::bandpass<REMAP, false>(
                    input, input_strides, output, output_strides, shape,
                    cutoff_high, cutoff_low, width_high, width_low);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        }
    }

    #define NOA_INSTANTIATE_FILTERS_(R, T)                                                                                          \
    template void lowpass<R, T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, f32, f32, i64);  \
    template void highpass<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, f32, f32, i64);  \
    template void bandpass<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, f32, f32, f32, f32, i64)

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
