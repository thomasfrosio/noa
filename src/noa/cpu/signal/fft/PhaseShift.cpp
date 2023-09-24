#include "noa/algorithms/signal/PhaseShift.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/Set.hpp"
#include "noa/cpu/fft/Remap.hpp"
#include "noa/cpu/signal/fft/PhaseShift.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace {
    using namespace ::noa;

    template<fft::Remap REMAP, typename T>
    void no_shift_(const T* input, const Strides4<i64>& input_strides,
                  T* output, const Strides4<i64>& output_strides,
                  const Shape4<i64>& shape, i64 threads) {
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        constexpr bool NO_REMAP = bool(u8_REMAP & fft::Layout::SRC_CENTERED) == bool(u8_REMAP & fft::Layout::DST_CENTERED);

        if (!input) {
            noa::cpu::memory::set(output, output_strides, shape.rfft(), T{1, 0}, threads);
        } else {
            if constexpr (NO_REMAP)
                noa::cpu::memory::copy(input, input_strides, output, output_strides, shape.rfft(), threads);
            else
                noa::cpu::fft::remap(REMAP, input, input_strides, output, output_strides, shape, threads);
        }
    }
}

namespace noa::cpu::signal::fft {
    using Layout = ::noa::fft::Layout;

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_2d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec2<f32>* shifts, f32 cutoff, i64 threads) {
        NOA_ASSERT(shape[1] == 1);

        auto kernel = noa::algorithm::signal::phase_shift<REMAP, 2>(
                input, input_strides, output, output_strides, shape, shifts, cutoff);
        noa::cpu::utils::iwise_3d(shape.filter(0, 2, 3).rfft(), kernel, threads);
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_2d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec2<f32>& shift, f32 cutoff, i64 threads) {
        NOA_ASSERT(shape[1] == 1);

        if (noa::all(shift == 0))
            return no_shift_<REMAP>(input, input_strides, output, output_strides, shape, threads);

        const bool is_half_shift = noa::all(noa::math::are_almost_equal(
                noa::math::abs(shift), shape.filter(2, 3).vec().as<f32>() / 2));

        if (is_half_shift && cutoff >= noa::math::sqrt(0.5f)) {
            auto kernel = noa::algorithm::signal::phase_shift_half<REMAP>(
                    input, input_strides, output, output_strides, shape);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        } else {
            auto kernel = noa::algorithm::signal::phase_shift<REMAP, 2>(
                    input, input_strides, output, output_strides, shape, shift, cutoff);
            noa::cpu::utils::iwise_3d(shape.filter(0, 2, 3).rfft(), kernel, threads);
        }
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_3d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec3<f32>* shifts, f32 cutoff, i64 threads) {
        auto kernel = noa::algorithm::signal::phase_shift<REMAP, 3>(
                input, input_strides, output, output_strides, shape, shifts, cutoff);
        noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_3d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec3<f32>& shift, f32 cutoff, i64 threads) {
        if (noa::all(shift == 0))
            return no_shift_<REMAP>(input, input_strides, output, output_strides, shape, threads);

        const bool is_half_shift = noa::all(noa::math::are_almost_equal(
                noa::math::abs(shift), shape.filter(1, 2, 3).vec().as<f32>() / 2));

        if (is_half_shift && cutoff >= noa::math::sqrt(0.5f)) {
            auto kernel = noa::algorithm::signal::phase_shift_half<REMAP>(
                    input, input_strides, output, output_strides, shape);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        } else {
            auto kernel = noa::algorithm::signal::phase_shift<REMAP, 3>(
                    input, input_strides, output, output_strides, shape, shift, cutoff);
            noa::cpu::utils::iwise_4d(shape.rfft(), kernel, threads);
        }
    }

    #define NOA_INSTANTIATE_SHIFT_(R, T)                                                                                                                \
    template void phase_shift_2d<R, T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec2<f32>&, f32, i64);  \
    template void phase_shift_2d<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec2<f32>*, f32, i64);   \
    template void phase_shift_3d<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec3<f32>&, f32, i64);   \
    template void phase_shift_3d<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec3<f32>*, f32, i64)

    #define NOA_INSTANTIATE_SHIFT_ALL(T)                  \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::H2H, T);      \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::H2HC, T);     \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::HC2H, T);     \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::HC2HC, T)

    NOA_INSTANTIATE_SHIFT_ALL(c32);
//    NOA_INSTANTIATE_SHIFT_ALL(c64);
}
