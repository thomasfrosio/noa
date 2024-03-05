#pragma once

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/kernels/SubpixelRegistration.cuh"

namespace noa::cuda::signal {
    template<noa::fft::Remap REMAP, typename Real, typename Coord>
    void xpeak_1d(
            Real* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const i64* xmap_max_offsets,
            Vec1<Coord>* output_peak_coordinates, Real* output_peak_values,
            noa::signal::CorrelationRegistration peak_mode, Vec1<i64> peak_radius,
            Stream& stream
    ) {
        const bool is_column = shape[3] == 1;
        const auto u_shape = shape.as_safe<u32>();
        const auto u_strides = strides.as_safe<u32>();
        const auto u_strides_1d = u_strides.filter(0, 3 - is_column);
        const auto i_shape_1d = u_shape.filter(3 - is_column).as_safe<i32>();

        const auto peak_window_size = (peak_radius * 2 + 1).as_clamp<u32>()[0];
        const auto xmap_accessor = AccessorRestrict<const Real, 2, u32>(xmap, u_strides_1d);
        const auto peak_offsets_accessor = AccessorRestrictContiguousI32<const i64, 1>(xmap_max_offsets);
        const auto output_peak_coordinates_accessor = AccessorRestrictContiguousI32<Vec<Coord, 1>, 1>(output_peak_coordinates);
        const auto output_peak_values_accessor = AccessorRestrictContiguousI32<Real, 1>(output_peak_values);

        switch (peak_mode) {
            case noa::signal::CorrelationRegistration::PARABOLA_1D: {
                const u32 n_threads = next_multiple_of(peak_window_size, Constant::WARP_SIZE);
                const auto config = LaunchConfig{
                        .n_blocks=u_shape[0], .n_threads=n_threads,
                        .n_bytes_of_shared_memory=n_threads * sizeof(Real),
                };
                stream.enqueue(guts::subpixel_registration_parabola_1d<1, REMAP, Real>, config,
                               xmap_accessor, i_shape_1d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
            case noa::signal::CorrelationRegistration::COM: {
                const auto config = LaunchConfig{
                        .n_blocks=u_shape[0], .n_threads=Constant::WARP_SIZE,
                        .n_bytes_of_shared_memory=peak_window_size * sizeof(Real),
                };
                stream.enqueue(guts::subpixel_registration_com<1, REMAP, Real>, config,
                               xmap_accessor, i_shape_1d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Real, typename Coord>
    void xpeak_2d(
            Real* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const i64* xmap_max_offsets,
            Vec2<Coord>* output_peak_coordinates, Real* output_peak_values,
            noa::signal::CorrelationRegistration peak_mode, const Vec2<i64>& peak_radius,
            Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto u_strides = strides.as_safe<u32>();
        const auto u_strides_2d = u_strides.filter(0, 2, 3);
        const auto i_shape_2d = u_shape.filter(2, 3).as_safe<i32>();

        const auto peak_window_shape = (peak_radius * 2 + 1).as_clamp<u32>();
        const auto xmap_accessor = AccessorRestrict<const Real, 3, u32>(xmap, u_strides_2d);
        const auto peak_offsets_accessor = AccessorRestrictContiguousI32<const i64, 1>(xmap_max_offsets);
        const auto output_peak_coordinates_accessor = AccessorRestrictContiguousI32<Vec<Coord, 2>, 1>(output_peak_coordinates);
        const auto output_peak_values_accessor = AccessorRestrictContiguousI32<Real, 1>(output_peak_values);

        switch (peak_mode) {
            case noa::signal::CorrelationRegistration::PARABOLA_1D: {
                const u32 threads = next_multiple_of(max(peak_window_shape), Constant::WARP_SIZE);
                const u32 size_shared_memory = max(threads, peak_window_shape[1]) + peak_window_shape[0];
                const u32 blocks = u_shape[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                return stream.enqueue(
                        guts::subpixel_registration_parabola_1d<2, REMAP, Real>, config,
                        xmap_accessor, i_shape_2d, peak_offsets_accessor, peak_radius.as<i32>(),
                        output_peak_coordinates_accessor, output_peak_values_accessor);
            }
            case noa::signal::CorrelationRegistration::COM: {
                const u32 threads = Constant::WARP_SIZE;
                const u32 blocks = u_shape[0];
                const u32 size_shared_memory = product(peak_window_shape);
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                return stream.enqueue(
                        guts::subpixel_registration_com<2, REMAP, Real>, config,
                        xmap_accessor, i_shape_2d, peak_offsets_accessor, peak_radius.as<i32>(),
                        output_peak_coordinates_accessor, output_peak_values_accessor);
            }
        }
    }

    template<noa::fft::Remap REMAP, typename Real, typename Coord>
    void xpeak_3d(
            Real* xmap, const Strides4<i64>& strides, const Shape4<i64>& shape,
            const i64* xmap_max_offsets,
            Vec3<Coord>* output_peak_coordinates, Real* output_peak_values,
            noa::signal::CorrelationRegistration peak_mode, const Vec3<i64>& peak_radius,
            Stream& stream
    ) {
        const auto u_shape = shape.as_safe<u32>();
        const auto u_strides = strides.as_safe<u32>();
        const auto i_shape_3d = u_shape.filter(1, 2, 3).as_safe<i32>();

        const auto peak_window_shape = (peak_radius * 2 + 1).as_clamp<u32>();
        const auto xmap_accessor = AccessorRestrict<const Real, 4, u32>(xmap, u_strides);
        const auto peak_offsets_accessor = AccessorRestrictContiguousI32<const i64, 1>(xmap_max_offsets);
        const auto output_peak_coordinates_accessor = AccessorRestrictContiguousI32<Vec3<Coord>, 1>(output_peak_coordinates);
        const auto output_peak_values_accessor = AccessorRestrictContiguousI32<Real, 1>(output_peak_values);

        switch (peak_mode) {
            case noa::signal::CorrelationRegistration::PARABOLA_1D: {
                const u32 n_threads = next_multiple_of(max(peak_window_shape), Constant::WARP_SIZE);
                const u32 size_shared_memory =
                        max(n_threads, peak_window_shape[2]) +
                        peak_window_shape[1] +
                        peak_window_shape[0];
                const auto config = LaunchConfig{
                    .n_blocks=u_shape[0], .n_threads=n_threads,
                    .n_bytes_of_shared_memory=size_shared_memory * sizeof(Real),
                };
                return stream.enqueue(
                        guts::subpixel_registration_parabola_1d<3, REMAP, Real>, config,
                        xmap_accessor, i_shape_3d, peak_offsets_accessor, peak_radius.as<i32>(),
                        output_peak_coordinates_accessor, output_peak_values_accessor);
            }
            case noa::signal::CorrelationRegistration::COM: {
                NOA_ASSERT(all(peak_radius <= 2));
                const u32 threads = Constant::WARP_SIZE;
                const u32 blocks = u_shape[0];
                const u32 size_shared_memory = product(peak_window_shape);
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                return stream.enqueue(
                        guts::subpixel_registration_com<3, REMAP, Real>, config,
                        xmap_accessor, i_shape_3d, peak_offsets_accessor, peak_radius.as<i32>(),
                        output_peak_coordinates_accessor, output_peak_values_accessor);
            }
        }
    }
}
