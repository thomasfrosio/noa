#include "noa/algorithms/signal/PhaseShift.hpp"
#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/fft/Remap.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/memory/Set.hpp"
#include "noa/gpu/cuda/signal/fft/PhaseShift.hpp"
#include "noa/gpu/cuda/utils/Iwise.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace {
    using namespace ::noa;

    template<fft::Remap REMAP, typename T>
    void no_shift_(const T* input, const Strides4<i64>& input_strides,
                   T* output, const Strides4<i64>& output_strides,
                   const Shape4<i64>& shape, cuda::Stream& stream) {
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        constexpr bool NO_REMAP = (u8_REMAP & fft::Layout::SRC_CENTERED) == (u8_REMAP & fft::Layout::DST_CENTERED);

        if (!input) {
            noa::cuda::memory::set(output, output_strides, shape.rfft(), T{1, 0}, stream);
        } else {
            if constexpr (NO_REMAP)
                noa::cuda::memory::copy(input, input_strides, output, output_strides, shape.rfft(), stream);
            else
                noa::cuda::fft::remap(REMAP, input, input_strides, output, output_strides, shape, stream);
        }
    }
}

namespace noa::cuda::signal::fft {
    using Layout = ::noa::fft::Layout;

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_2d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec2<f32>* shifts, f32 cutoff, Stream& stream) {
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));
        NOA_ASSERT(shape[1] == 1);

        const auto i_shape = shape.as_safe<i32>();
        auto kernel = noa::algorithm::signal::phase_shift<REMAP, 2>(
                input, input_strides.as_safe<u32>(),
                output, output_strides.as_safe<u32>(),
                i_shape, shifts, cutoff);
        noa::cuda::utils::iwise_3d("phase_shift_2d", i_shape.filter(0, 2, 3).rfft(), kernel, stream);
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_2d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec2<f32>& shift, f32 cutoff, Stream& stream) {
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));
        NOA_ASSERT(shape[1] == 1);

        if (noa::all(shift == 0))
            return no_shift_<REMAP>(input, input_strides, output, output_strides, shape, stream);

        const bool is_half_shift = noa::all(noa::math::are_almost_equal(
                noa::math::abs(shift), shape.filter(2, 3).vec().as<f32>() / 2));

        const auto i_shape = shape.as_safe<i32>();
        if (is_half_shift && cutoff >= noa::math::sqrt(0.5f)) {
            auto kernel = noa::algorithm::signal::phase_shift_half<REMAP>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    i_shape);
            noa::cuda::utils::iwise_4d("phase_shift_2d", i_shape.rfft(), kernel, stream);
        } else {
            auto kernel = noa::algorithm::signal::phase_shift<REMAP, 2>(
                    input, input_strides.as_safe<u32>(),
                    output, output_strides.as_safe<u32>(),
                    i_shape, shift, cutoff);
            noa::cuda::utils::iwise_3d("phase_shift_2d", i_shape.filter(0, 2, 3).rfft(), kernel, stream);
        }
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_3d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec3<f32>* shifts, f32 cutoff, Stream& stream) {
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));

        const auto i_shape = shape.as_safe<i32>();
        auto kernel = noa::algorithm::signal::phase_shift<REMAP, 3>(
                input, input_strides.as_safe<u32>(),
                output, output_strides.as_safe<u32>(), i_shape, shifts, cutoff);
        noa::cuda::utils::iwise_4d("phase_shift_3d", i_shape.rfft(), kernel, stream);
    }

    template<noa::fft::Remap REMAP, typename T, typename>
    void phase_shift_3d(const T* input, const Strides4<i64>& input_strides,
                        T* output, const Strides4<i64>& output_strides, const Shape4<i64>& shape,
                        const Vec3<f32>& shift, f32 cutoff, Stream& stream) {
        constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static_assert(u8_REMAP & Layout::SRC_HALF && u8_REMAP & Layout::DST_HALF);
        NOA_ASSERT(input != output || ((u8_REMAP & Layout::SRC_CENTERED) == (u8_REMAP & Layout::DST_CENTERED)));

        if (noa::all(shift == 0))
            return no_shift_<REMAP>(input, input_strides, output, output_strides, shape, stream);

        const bool is_half_shift = noa::all(noa::math::are_almost_equal(
                noa::math::abs(shift), shape.filter(1, 2, 3).vec().as<f32>() / 2));

        const auto i_shape = shape.as_safe<i32>();
        if (is_half_shift && cutoff >= noa::math::sqrt(0.5f)) {
            auto kernel = noa::algorithm::signal::phase_shift_half<REMAP>(
                    input, input_strides.as_safe<i32>(),
                    output, output_strides.as_safe<i32>(), i_shape);
            noa::cuda::utils::iwise_4d("phase_shift_3d", i_shape.rfft(), kernel, stream);
        } else {
            auto kernel = noa::algorithm::signal::phase_shift<REMAP, 3>(
                    input, input_strides.as_safe<i32>(),
                    output, output_strides.as_safe<i32>(), i_shape, shift, cutoff);
            noa::cuda::utils::iwise_4d("phase_shift_3d", i_shape.rfft(), kernel, stream);
        }
    }

    #define NOA_INSTANTIATE_SHIFT_(R, T)                                                                                                                    \
    template void phase_shift_2d<R, T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec2<f32>&, f32, Stream&);  \
    template void phase_shift_2d<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec2<f32>*, f32, Stream&);   \
    template void phase_shift_3d<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec3<f32>&, f32, Stream&);   \
    template void phase_shift_3d<R,T,void>(const T*, const Strides4<i64>&, T*, const Strides4<i64>&, const Shape4<i64>&, const Vec3<f32>*, f32, Stream&)

    #define NOA_INSTANTIATE_SHIFT_ALL(T)                  \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::H2H, T);      \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::H2HC, T);     \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::HC2H, T);     \
    NOA_INSTANTIATE_SHIFT_(noa::fft::Remap::HC2HC, T)

    NOA_INSTANTIATE_SHIFT_ALL(c32);
    NOA_INSTANTIATE_SHIFT_ALL(c64);
}
