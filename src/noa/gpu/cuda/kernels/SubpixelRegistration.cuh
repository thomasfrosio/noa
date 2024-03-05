#pragma once

#include "noa/core/signal/Correlation.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/core/math/LeastSquare.hpp"
#include "noa/gpu/cuda/kernels/Warp.cuh"
#include "noa/gpu/cuda/kernels/Block.cuh"

namespace noa::cuda::signal::guts {
    // Copy the window of the peak. The number of threads should be >= than the window size.
    // The read to global memory only coalesces if the stride is 1.
    template<fft::Remap REMAP, typename Real>
    constexpr NOA_FD Real copy_window_parabola_1d(
            const Real* xmap, u32 xmap_stride, i32 xmap_size,
            i32 peak_index, i32 peak_radius, i32 tid
    ) {
        Real value = 0;
        const i32 elements = peak_radius * 2 + 1;
        if (tid < elements) {
            const i32 tid_offset = tid - peak_radius;
            if constexpr (REMAP == fft::F2F) {
                const i32 tid_frequency = noa::fft::index2frequency<false>(peak_index, xmap_size) + tid_offset;
                if (-xmap_size / 2 <= tid_frequency and tid_frequency <= (xmap_size - 1) / 2) {
                    const i32 tid_index = noa::fft::frequency2index<false>(tid_frequency, xmap_size);
                    value = xmap[static_cast<u32>(tid_index) * xmap_stride];
                }
            } else {
                const i32 tid_index = peak_index + tid_offset;
                if (0 <= tid_index and tid_index < xmap_size)
                    value = xmap[static_cast<u32>(tid_index) * xmap_stride];
            }
        }
        return value;
    }

    template<size_t N, fft::Remap REMAP, typename Real, typename Coord>
    __global__ void subpixel_registration_parabola_1d(
            AccessorRestrict<const Real, N + 1, u32> xmap_batched, Shape<i32, N> xmap_shape,
            AccessorRestrictContiguousI32<const i64, 1> peak_offset, Vec<i32, N> peak_window_radius,
            AccessorRestrictContiguousI32<Vec<Coord, N>, 1> output_peak_coordinate,
            AccessorRestrictContiguousI32<Real, 1> output_peak_value
    ) {
        const u32 batch = blockIdx.x;
        const auto thread_index = static_cast<i32>(threadIdx.x);

        const auto xmap = xmap_batched[batch];
        const auto xmap_strides = xmap_batched.strides().pop_front();

        extern __shared__ Real buffer[];
        auto peak_index = Vec<i32, N>(ni::offset2index<true>(
                static_cast<u32>(peak_offset[batch]), xmap_strides, xmap_shape.template as<u32>()));

        #pragma unroll
        for (i32 i = 0; i < N; ++i) {
            auto xmap_ptr = xmap.get();
            i32 buffer_offset{};
            if (i >= 1)
                buffer_offset += peak_window_radius[0] * 2 + 1;
            if (i == 2)
                buffer_offset += peak_window_radius[1] * 2 + 1;

            if (i != 0)
                xmap_ptr += peak_index[0] * xmap_strides[0];
            if (i != 1 and N > 1)
                xmap_ptr += peak_index[1] * xmap_strides[1];
            if (i != 2 and N > 2)
                xmap_ptr += peak_index[2] * xmap_strides[2];

            buffer[buffer_offset + thread_index] =
                    copy_window_parabola_1d<REMAP>(
                            xmap_ptr, xmap_strides[i], xmap_shape[i],
                            peak_index[i], peak_window_radius[i], thread_index);
            noa::cuda::guts::block_synchronize();
        }

        // Fit parabola and compute vertex.
        // Not a lot of parallelism here, but while we could use wrap reduction, for most cases,
        // the peak radius is quite small, and it's just simpler (and maybe faster) to use a single thread.
        if (thread_index == 0) {
            if constexpr (REMAP == noa::fft::F2F)
                peak_index = noa::fft::fftshift(peak_index, xmap_shape);

            const auto [peak_value, peak_subpixel_coordinate] =
                    noa::signal::guts::peak_parabola_1d(buffer, peak_window_radius);
            const auto peak_coordinate = peak_subpixel_coordinate + peak_index;
            if (output_peak_coordinate)
                output_peak_coordinate[batch] = peak_coordinate.template as<Coord>();
            if (output_peak_value)
                output_peak_value[batch] = static_cast<Real>(peak_value);
        }
    }

    // The block must be a single warp.
    // Shared memory size is the size of the peak window.
    // Elements are collected and saved in the shared buffer.
    template<size_t N, fft::Remap REMAP, typename Real>
    __global__ void __launch_bounds__(cuda::Constant::WARP_SIZE)
    subpixel_registration_com(
            AccessorRestrict<const Real, N + 1, u32> xmap_batched, Shape<i32, N> xmap_shape,
            AccessorRestrictContiguous<const u32, 1, u32> peak_offset, Vec<i32, N> peak_window_radius,
            AccessorRestrictContiguous<Vec<f32, N>, 1, u32> output_peak_coordinate,
            AccessorRestrictContiguous<Real, 1, u32> output_peak_value
    ) {
        const u32 batch = blockIdx.x;
        const auto xmap = xmap_batched[batch];
        const auto xmap_strides = xmap_batched.strides().pop_front();

        auto peak_index = Vec<i32, N>(ni::offset2index<true>(
                peak_offset[batch], xmap_strides, xmap_shape.template as<u32>()));
        const auto peak_window_shape = Shape<i32, N>(peak_window_radius * 2 + 1);
        const auto peak_window_elements = peak_window_shape.elements();

        // Collect the elements within the peak window.
        // Values are stored in the rightmost order.
        extern __shared__ Real peak_window_values[];

        // We'll need to subtract by the peak window min.
        // Here, each thread collects the min of the values it processed.
        Real thread_peak_window_min{0};

        const auto thread_index = static_cast<i32>(threadIdx.x);
        constexpr i32 WARP_SIZE = noa::cuda::Constant::WARP_SIZE;
        NOA_ASSERT(WARP_SIZE == blockDim.x);

        if constexpr (REMAP == fft::FC2FC) {
            for (i32 i = thread_index; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = ni::offset2index(i, peak_window_shape);
                const auto relative_index = indexes - peak_window_radius;
                const auto current_index = peak_index + relative_index;

                // Collect the value in the peak window at these indexes.
                // If the indexes are OOB, we still need to initialize the shared array with 0.
                Real value{0};
                if (noa::all(current_index >= 0 && current_index < xmap_shape.vec())) {
                    value = xmap(current_index);
                    thread_peak_window_min = min(thread_peak_window_min, value);
                }
                peak_window_values[i] = value;
            }
        } else if constexpr (REMAP == fft::F2F) {
            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            const auto frequency_min = -xmap_shape.vec() / 2;
            const auto frequency_max = (xmap_shape.vec() - 1) / 2;
            const auto peak_frequency = noa::fft::index2frequency<false>(peak_index, xmap_shape);

            for (i32 i = thread_index; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = ni::offset2index(i, peak_window_shape);
                const auto relative_index = indexes - peak_window_radius;
                const auto current_frequency = peak_frequency + relative_index;

                Real value{0};
                if (noa::all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                    const auto current_index = noa::fft::frequency2index<false>(current_frequency, xmap_shape);
                    value = xmap(current_index);
                    thread_peak_window_min = min(thread_peak_window_min, value);
                }
                peak_window_values[i] = value;
            }
        } else {
            static_assert(nt::always_false_v<Real>);
        }

        // Compute the min of the peak window.
        using namespace noa::cuda::guts;
        thread_peak_window_min = warp_reduce(Less{}, thread_peak_window_min);
        warp_shuffle(thread_peak_window_min, 0);

        // Set the min to 0 and compute the partial COM.
        block_synchronize();
        Vec<f64, N> tid_com{0};
        f64 tid_com_total{0};
        for (i32 i = thread_index; i < peak_window_elements; i += WARP_SIZE) {
            const auto indexes = ni::offset2index(i, peak_window_shape);
            const auto relative_index = Vec<f64, N>(indexes - peak_window_radius);
            const auto value = static_cast<f64>(peak_window_values[i] - thread_peak_window_min);
            tid_com += value * relative_index;
            tid_com_total += value;
        }

        // Then the first thread collects everything.
        for (i32 dim = 0; dim < N; ++dim)
            tid_com[dim] = warp_reduce(Plus{}, tid_com[dim]);
        tid_com_total = warp_reduce(Plus{}, tid_com_total);

        // From that point, only the value in the thread 0 matters.
        if (thread_index == 0) {
            if (output_peak_coordinate) {
                if constexpr (REMAP == noa::fft::F2F)
                    peak_index = noa::fft::fftshift(peak_index, xmap_shape);
                if (tid_com_total != 0)
                    tid_com /= tid_com_total;
                tid_com += Vec<f64, N>(peak_index);
                output_peak_coordinate[batch] = tid_com.template as<f32>();
            }
            if (output_peak_value) {
                // FIXME For now, just take the closest value...
                const auto offset = ni::offset_at(peak_window_radius, peak_window_shape.strides());
                output_peak_value[batch] = peak_window_values[offset];
            }
        }
    }
}
