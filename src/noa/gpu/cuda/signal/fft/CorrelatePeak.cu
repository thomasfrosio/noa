#include "noa/algorithms/signal/CorrelationPeak.hpp"
#include "noa/algorithms/Utilities.hpp"
#include "noa/core/geometry/Shape.hpp"
#include "noa/core/math/LeastSquare.hpp"

#include "noa/gpu/cuda/Find.hpp"
#include "noa/gpu/cuda/memory/PtrPinned.hpp"
#include "noa/gpu/cuda/memory/PtrDevice.hpp"
#include "noa/gpu/cuda/signal/fft/Correlate.hpp"
#include "noa/gpu/cuda/geometry/fft/Shape.hpp"

#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/Warp.cuh"
#include "noa/gpu/cuda/utils/Block.cuh"

// TODO If centered, select subregion within xmap_ellipse_radius.

namespace {
    using namespace ::noa;

    template<fft::Remap REMAP, typename T>
    __global__ void mask_ellipse_1d_(
            Accessor<T, 2, u32> xmap, u32 size,
            noa::geometry::LineSmooth<T> line) {
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= size)
            return;
        const f32 coord = REMAP == fft::FC2FC ? gid : noa::math::fft_shift(gid, size);
        xmap(blockIdx.y, gid) *= line(coord);
    }

    template<fft::Remap REMAP, typename T>
    void enforce_max_radius_inplace_1d_(
            T* xmap, f32 max_radius,
            const Shape2<u32>& shape_1d, const Strides2<u32>& strides_1d,
            cuda::Stream& stream) {
        const auto center = static_cast<f32>(shape_1d[1] / 2);
        const auto edge_size = static_cast<f32>(shape_1d[1]) * 0.05f;
        const auto accessor = Accessor<T, 2, u32>(xmap, strides_1d);
        const auto line = noa::geometry::LineSmooth<T>(center, max_radius, edge_size);

        const u32 threads = noa::math::min(128u, math::next_multiple_of(shape_1d[1], 32u));
        const dim3 blocks(noa::math::divide_up(shape_1d[1], threads), shape_1d[0]);
        const cuda::LaunchConfig config{blocks, threads};
        stream.enqueue("xpeak_1d_mask", mask_ellipse_1d_<REMAP, T>, config,
                       accessor, shape_1d[1], line);
    }
}

namespace {
    using namespace ::noa;

    // Copy the window of the peak. The number of threads should be >= than the window size.
    // The read to global memory only coalesces if the stride is 1.
    template<fft::Remap REMAP, typename Real>
    constexpr NOA_FD Real copy_window_parabola_1d_(
            const Real* xmap, u32 xmap_stride, i32 xmap_size,
            i32 peak_index, i32 peak_radius, i32 tid) {
        Real value = 0;
        const i32 elements = peak_radius * 2 + 1;
        if (tid < elements) {
            const i32 tid_offset = tid - peak_radius;
            if constexpr (REMAP == fft::F2F) {
                const i32 tid_frequency = noa::algorithm::index2frequency<false>(peak_index, xmap_size) + tid_offset;
                if (-xmap_size / 2 <= tid_frequency && tid_frequency <= (xmap_size - 1) / 2) {
                    const i32 tid_index = noa::algorithm::frequency2index<false>(tid_frequency, xmap_size);
                    value = xmap[tid_index * xmap_stride];
                }
            } else {
                const i32 tid_index = peak_index + tid_offset;
                if (0 <= tid_index && tid_index < xmap_size)
                    value = xmap[tid_index * xmap_stride];
            }
        }
        return value;
    }

    template<size_t N, fft::Remap REMAP, typename Real>
    __global__ void subpixel_registration_parabola_1d_(
            AccessorRestrict<const Real, N + 1, u32> xmap_batched, Shape<i32, N> xmap_shape,
            AccessorRestrictContiguous<const u32, 1, u32> peak_offset, Vec<i32, N> peak_window_radius,
            AccessorRestrictContiguous<Vec<f32, N>, 1, u32> output_peak_coordinate,
            AccessorRestrictContiguous<Real, 1, u32> output_peak_value) {
        const u32 batch = blockIdx.x;
        const auto thread_index = static_cast<i32>(threadIdx.x);

        const auto xmap = xmap_batched[batch];
        const auto xmap_strides = xmap_batched.strides().pop_front();

        Real* buffer = noa::cuda::utils::block_dynamic_shared_resource<Real>();
        auto peak_index = Vec<i32, N>(noa::indexing::offset2index<true>(
                peak_offset[batch], xmap_strides, xmap_shape.template as<u32>()));

        #pragma unroll
        for (i32 i = 0; i < N; ++i) {
            auto xmap_ptr = xmap.get();
            i32 buffer_offset{0};
            if (i >= 1)
                buffer_offset += peak_window_radius[0] * 2 + 1;
            if (i == 2)
                buffer_offset += peak_window_radius[1] * 2 + 1;

            if (i != 0)
                xmap_ptr += peak_index[0] * xmap_strides[0];
            if (i != 1 && N > 1)
                xmap_ptr += peak_index[1] * xmap_strides[1];
            if (i != 2 && N > 2)
                xmap_ptr += peak_index[2] * xmap_strides[2];

            buffer[buffer_offset + thread_index] =
                    copy_window_parabola_1d_<REMAP>(
                            xmap_ptr, xmap_strides[i], xmap_shape[i],
                            peak_index[i], peak_window_radius[i], thread_index);
            noa::cuda::utils::block_synchronize();
        }

        // Fit parabola and compute vertex.
        // Not a lot of parallelism here, but while we could use wrap reduction, for most cases,
        // the peak radius is quite small, and it's just simpler (and maybe faster) to use a single thread.
        if (thread_index == 0) {
            if constexpr (REMAP == noa::fft::F2F)
                peak_index = noa::math::fft_shift(peak_index, xmap_shape);

            const auto [peak_value, peak_subpixel_coordinate] =
                    noa::algorithm::signal::peak_parabola_1d(buffer, peak_window_radius);
            const auto peak_coordinate =
                    peak_subpixel_coordinate.template as<f32>() + peak_index.template as<f32>();
            if (output_peak_coordinate)
                output_peak_coordinate[batch] = peak_coordinate;
            if (output_peak_value)
                output_peak_value[batch] = static_cast<Real>(peak_value);
        }
    }

    // The block must be a single warp.
    // Shared memory size is the size of the peak window.
    // Elements are collected and saved in the shared buffer.
    template<size_t N, fft::Remap REMAP, typename Real>
    __global__ void __launch_bounds__(cuda::Constant::WARP_SIZE)
    subpixel_registration_com_(
            AccessorRestrict<const Real, N + 1, u32> xmap_batched, Shape<i32, N> xmap_shape,
            AccessorRestrictContiguous<const u32, 1, u32> peak_offset, Vec<i32, N> peak_window_radius,
            AccessorRestrictContiguous<Vec<f32, N>, 1, u32> output_peak_coordinate,
            AccessorRestrictContiguous<Real, 1, u32> output_peak_value) {

        const u32 batch = blockIdx.x;
        const auto xmap = xmap_batched[batch];
        const auto xmap_strides = xmap_batched.strides().pop_front();

        auto peak_index = Vec<i32, N>(noa::indexing::offset2index<true>(
                peak_offset[batch], xmap_strides, xmap_shape.template as<u32>()));
        const auto peak_window_shape = Shape<i32, N>(peak_window_radius * 2 + 1);
        const auto peak_window_elements = peak_window_shape.elements();

        // Collect the elements within the peak window.
        // Values are stored in the rightmost order.
        Real* peak_window_values = noa::cuda::utils::block_dynamic_shared_resource<Real>();

        // We'll need to subtract by the peak window min.
        // Here, each thread collects the min of the values it processed.
        Real thread_peak_window_min{0};

        const auto thread_index = static_cast<i32>(threadIdx.x);
        constexpr i32 WARP_SIZE = cuda::Constant::WARP_SIZE;
        NOA_ASSERT(WARP_SIZE == blockDim.x);

        if constexpr (REMAP == fft::FC2FC) {
            for (i32 i = thread_index; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = noa::indexing::offset2index(i, peak_window_shape);
                const auto relative_index = indexes - peak_window_radius;
                const auto current_index = peak_index + relative_index;

                // Collect the value in the peak window at these indexes.
                // If the indexes are OOB, we still need to initialize the shared array with 0.
                Real value{0};
                if (noa::all(current_index >= 0 && current_index < xmap_shape.vec())) {
                    value = xmap(current_index);
                    thread_peak_window_min = noa::math::min(thread_peak_window_min, value);
                }
                peak_window_values[i] = value;
            }
        } else if constexpr (REMAP == fft::F2F) {
            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            const auto frequency_min = -xmap_shape.vec() / 2;
            const auto frequency_max = (xmap_shape.vec() - 1) / 2;
            const auto peak_frequency = noa::algorithm::index2frequency<false>(peak_index, xmap_shape);

            for (i32 i = thread_index; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = noa::indexing::offset2index(i, peak_window_shape);
                const auto relative_index = indexes - peak_window_radius;
                const auto current_frequency = peak_frequency + relative_index;

                Real value{0};
                if (noa::all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                    const auto current_index = noa::algorithm::frequency2index<false>(current_frequency, xmap_shape);
                    value = xmap(current_index);
                    thread_peak_window_min = noa::math::min(thread_peak_window_min, value);
                }
                peak_window_values[i] = value;
            }
        } else {
            static_assert(noa::traits::always_false_v<Real>);
        }

        // Compute the min of the peak window.
        thread_peak_window_min = noa::cuda::utils::warp_reduce(thread_peak_window_min, noa::less_t{});
        noa::cuda::utils::warp_shuffle(thread_peak_window_min, 0);

        // Set the min to 0 and compute the partial COM.
        noa::cuda::utils::block_synchronize();
        Vec<f64, N> tid_com{0};
        f64 tid_com_total{0};
        for (i32 i = thread_index; i < peak_window_elements; i += WARP_SIZE) {
            const auto indexes = noa::indexing::offset2index(i, peak_window_shape);
            const auto relative_index = Vec<f64, N>(indexes - peak_window_radius);
            const auto value = static_cast<f64>(peak_window_values[i] - thread_peak_window_min);
            tid_com += value * relative_index;
            tid_com_total += value;
        }

        // Then the first thread collects everything.
        for (i32 dim = 0; dim < N; ++dim)
            tid_com[dim] = noa::cuda::utils::warp_reduce(tid_com[dim], noa::plus_t{});
        tid_com_total = noa::cuda::utils::warp_reduce(tid_com_total, noa::plus_t{});

        // From that point, only the value in the thread 0 matters.
        if (thread_index == 0) {
            if (output_peak_coordinate) {
                if constexpr (REMAP == noa::fft::F2F)
                    peak_index = noa::math::fft_shift(peak_index, xmap_shape);
                if (tid_com_total != 0)
                    tid_com /= tid_com_total;
                tid_com += Vec<f64, N>(peak_index);
                output_peak_coordinate[batch] = tid_com.template as<f32>();
            }
            if (output_peak_value) {
                // FIXME For now, just take the closest value...
                const auto offset = noa::indexing::at(peak_window_radius, peak_window_shape.strides());
                output_peak_value[batch] = peak_window_values[offset];
            }
        }
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  Vec1<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, Vec1<i64> peak_radius, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(peak_radius[0] > 0);
        NOA_ASSERT(strides[3 - is_column] > 0);
        NOA_ASSERT(noa::all(shape > 0) && shape.ndim() == 1);
        NOA_ASSERT_DEVICE_PTR(xmap, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_peak_coordinates, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_peak_values, stream.device());

        const auto u_shape = shape.as_safe<u32>();
        const auto u_strides = strides.as_safe<u32>();
        const auto u_strides_1d = u_strides.filter(0, 3 - is_column);
        const auto i_shape_1d = u_shape.filter(3 - is_column).as_safe<i32>();

        // In-place mask on the phase-correlation map.
        if (xmap_ellipse_radius[0] > 0) {
            enforce_max_radius_inplace_1d_<REMAP>(
                    xmap, xmap_ellipse_radius[0], u_shape.filter(0, 3 - is_column), u_strides_1d, stream);
        }

        // Find the maximum value.
        const auto peak_offsets = cuda::memory::PtrDevice<u32>::alloc(shape[0], stream);
        noa::cuda::find_offsets(noa::first_max_t{}, xmap, strides, shape, peak_offsets.get(), false, true, stream);

        const auto peak_window_size = (peak_radius * 2 + 1).as_clamp<u32>()[0];
        const auto xmap_accessor = AccessorRestrict<const Real, 2, u32>(xmap, u_strides_1d);
        const auto peak_offsets_accessor = AccessorRestrictContiguous<const u32, 1, u32>(peak_offsets.get());
        const auto output_peak_coordinates_accessor = AccessorRestrictContiguous<Vec<f32, 1>, 1, u32>(output_peak_coordinates);
        const auto output_peak_values_accessor = AccessorRestrictContiguous<Real, 1, u32>(output_peak_values);

        switch (peak_mode) {
            case noa::signal::PeakMode::PARABOLA_1D: {
                const u32 threads = noa::math::next_multiple_of(peak_window_size, Constant::WARP_SIZE);
                const u32 blocks = u_shape[0];
                const LaunchConfig config{blocks, threads, threads * sizeof(Real)};
                stream.enqueue("xpeak_1d_parabola_1d",
                               subpixel_registration_parabola_1d_<1, REMAP, Real>, config,
                               xmap_accessor, i_shape_1d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
            case noa::signal::PeakMode::COM: {
                const u32 threads = Constant::WARP_SIZE;
                const u32 blocks = u_shape[0];
                const LaunchConfig config{blocks, threads, peak_window_size * sizeof(Real)};
                stream.enqueue("xpeak_1d_com",
                               subpixel_registration_com_<1, REMAP, Real>, config,
                               xmap_accessor, i_shape_1d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
        }
    }

    template<Remap REMAP, typename Real, typename>
    auto xpeak_1d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, Vec1<f32> xmap_ellipse_radius,
                  PeakMode peak_mode, Vec1<i64> peak_radius, Stream& stream) -> std::pair<Vec1<f32>, Real> {
        NOA_CHECK(shape[0] == 1, "This overload does not support batched arrays, but got {} batches", shape[0]);

        const auto peak_coordinate = memory::PtrPinned<Vec1<f32>>::alloc(1);
        const auto peak_value = memory::PtrPinned<Real>::alloc(1);

        xpeak_1d<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                        peak_coordinate.get(), peak_value.get(),
                        peak_mode, peak_radius, stream);
        stream.synchronize();
        return {peak_coordinate[0], peak_value[0]};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  Vec2<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, Stream& stream) {
        NOA_ASSERT(noa::all(peak_radius > 0));
        NOA_ASSERT(noa::all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(xmap, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_peak_coordinates, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_peak_values, stream.device());

        const auto u_shape = shape.as_safe<u32>();
        const auto u_strides = strides.as_safe<u32>();
        const auto u_strides_2d = u_strides.filter(0, 2, 3);
        const auto i_shape_2d = u_shape.filter(2, 3).as_safe<i32>();

        // In-place mask on the phase-correlation map.
        if (noa::any(xmap_ellipse_radius > 0)) {
            const auto center = (i_shape_2d / 2).vec().as<f32>();
            const auto edge_size = static_cast<f32>(noa::math::max(i_shape_2d)) * 0.05f;
            const auto cvalue = Real{1};
            noa::cuda::geometry::fft::ellipse<REMAP>(
                    xmap, strides, xmap, strides, shape,
                    center, xmap_ellipse_radius, edge_size,
                    Float22{}, noa::multiply_t{}, cvalue, false, stream);
        }

        // Find the maximum value.
        const auto peak_offsets = cuda::memory::PtrDevice<u32>::alloc(shape[0], stream);
        noa::cuda::find_offsets(noa::first_max_t{}, xmap, strides, shape, peak_offsets.get(), false, true, stream);

        const auto peak_window_shape = (peak_radius * 2 + 1).as_clamp<u32>();
        const auto xmap_accessor = AccessorRestrict<const Real, 3, u32>(xmap, u_strides_2d);
        const auto peak_offsets_accessor = AccessorRestrictContiguous<const u32, 1, u32>(peak_offsets.get());
        const auto output_peak_coordinates_accessor = AccessorRestrictContiguous<Vec<f32, 2>, 1, u32>(output_peak_coordinates);
        const auto output_peak_values_accessor = AccessorRestrictContiguous<Real, 1, u32>(output_peak_values);

        switch (peak_mode) {
            case noa::signal::PeakMode::PARABOLA_1D: {
                const u32 threads = noa::math::next_multiple_of(noa::math::max(peak_window_shape), Constant::WARP_SIZE);
                const u32 size_shared_memory = noa::math::max(threads, peak_window_shape[1]) + peak_window_shape[0];
                const u32 blocks = u_shape[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("xpeak_2d_parabola_1d",
                               subpixel_registration_parabola_1d_<2, REMAP, Real>, config,
                               xmap_accessor, i_shape_2d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
            case noa::signal::PeakMode::COM: {
                const u32 threads = Constant::WARP_SIZE;
                const u32 blocks = u_shape[0];
                const u32 size_shared_memory = noa::math::product(peak_window_shape);
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("xpeak_2d_com",
                               subpixel_registration_com_<2, REMAP, Real>, config,
                               xmap_accessor, i_shape_2d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
        }
    }

    template<Remap REMAP, typename Real, typename>
    auto xpeak_2d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec2<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec2<i64>& peak_radius, Stream& stream) -> std::pair<Vec2<f32>, Real> {
        NOA_CHECK(shape[0] == 1, "This overload does not support batched arrays, but got {} batches", shape[0]);

        const auto peak_coordinate = memory::PtrPinned<Vec2<f32>>::alloc(1);
        const auto peak_value = memory::PtrPinned<Real>::alloc(1);

        xpeak_2d<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                        peak_coordinate.get(), peak_value.get(),
                        peak_mode, peak_radius, stream);
        stream.synchronize();
        return {peak_coordinate[0], peak_value[0]};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  Vec3<f32>* output_peak_coordinates, Real* output_peak_values,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, Stream& stream) {
        NOA_ASSERT(noa::all(peak_radius > 0));
        NOA_ASSERT(noa::all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(xmap, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_peak_coordinates, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_peak_values, stream.device());

        const auto u_shape = shape.as_safe<u32>();
        const auto u_strides = strides.as_safe<u32>();
        const auto i_shape_3d = u_shape.filter(1, 2, 3).as_safe<i32>();

        // In-place mask on the phase-correlation map.
        if (noa::any(xmap_ellipse_radius > 0)) {
            const auto center = (i_shape_3d / 2).vec().as<f32>();
            const auto edge_size = static_cast<f32>(noa::math::max(i_shape_3d)) * 0.05f;
            const auto cvalue = Real{1};
            noa::cuda::geometry::fft::ellipse<REMAP>(
                    xmap, strides, xmap, strides, shape,
                    center, xmap_ellipse_radius, edge_size,
                    Float33{}, noa::multiply_t{}, cvalue, false, stream);
        }

        // Find the maximum value.
        const auto peak_offsets = cuda::memory::PtrDevice<u32>::alloc(shape[0], stream);
        noa::cuda::find_offsets(noa::first_max_t{}, xmap, strides, shape, peak_offsets.get(), false, true, stream);

        const auto peak_window_shape = (peak_radius * 2 + 1).as_clamp<u32>();
        const auto xmap_accessor = AccessorRestrict<const Real, 4, u32>(xmap, u_strides);
        const auto peak_offsets_accessor = AccessorRestrictContiguous<const u32, 1, u32>(peak_offsets.get());
        const auto output_peak_coordinates_accessor = AccessorRestrictContiguous<Vec<f32, 3>, 1, u32>(output_peak_coordinates);
        const auto output_peak_values_accessor = AccessorRestrictContiguous<Real, 1, u32>(output_peak_values);

        switch (peak_mode) {
            case noa::signal::PeakMode::PARABOLA_1D: {
                const u32 threads = noa::math::next_multiple_of(noa::math::max(peak_window_shape), Constant::WARP_SIZE);
                const u32 blocks = u_shape[0];
                const u32 size_shared_memory =
                        noa::math::max(threads, peak_window_shape[2]) +
                        peak_window_shape[1] +
                        peak_window_shape[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("xpeak_3d_parabola_1d",
                               subpixel_registration_parabola_1d_<3, REMAP, Real>, config,
                               xmap_accessor, i_shape_3d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
            case noa::signal::PeakMode::COM: {
                NOA_ASSERT(all(peak_radius <= 2));
                const u32 threads = Constant::WARP_SIZE;
                const u32 blocks = u_shape[0];
                const u32 size_shared_memory = noa::math::product(peak_window_shape);
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("xpeak_3d_com",
                               subpixel_registration_com_<3, REMAP, Real>, config,
                               xmap_accessor, i_shape_3d, peak_offsets_accessor, peak_radius.as<i32>(),
                               output_peak_coordinates_accessor, output_peak_values_accessor);
                break;
            }
        }
    }

    template<Remap REMAP, typename Real, typename>
    auto xpeak_3d(Real* xmap, const Strides4<i64>& strides,
                  const Shape4<i64>& shape, const Vec3<f32>& xmap_ellipse_radius,
                  PeakMode peak_mode, const Vec3<i64>& peak_radius, Stream& stream) -> std::pair<Vec3<f32>, Real> {
        NOA_CHECK(shape[0] == 1, "This overload does not support batched arrays, but got {} batches", shape[0]);

        const auto peak_coordinate = memory::PtrPinned<Vec3<f32>>::alloc(1);
        const auto peak_value = memory::PtrPinned<Real>::alloc(1);

        xpeak_3d<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                        peak_coordinate.get(), peak_value.get(),
                        peak_mode, peak_radius, stream);
        stream.synchronize();
        return {peak_coordinate[0], peak_value[0]};
    }

    #define NOA_INSTANTIATE_XPEAK(R, T)         \
    template void xpeak_1d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, Vec1<f32>,          \
        Vec1<f32>*, T*,                         \
        PeakMode, Vec1<i64>, Stream&);          \
    template auto xpeak_1d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, Vec1<f32>,          \
        PeakMode, Vec1<i64>, Stream&)           \
        -> std::pair<Vec1<f32>, T>;             \
    template void xpeak_2d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec2<f32>&,   \
        Vec2<f32>*, T*,                         \
        PeakMode, const Vec2<i64>&, Stream&);   \
    template auto xpeak_2d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec2<f32>&,   \
        PeakMode, const Vec2<i64>&, Stream&)    \
        -> std::pair<Vec2<f32>, T>;             \
    template void xpeak_3d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec3<f32>&,   \
        Vec3<f32>*, T*,                         \
        PeakMode, const Vec3<i64>&, Stream&);   \
    template auto xpeak_3d<R, T, void>(         \
        T*, const Strides4<i64>&,               \
        const Shape4<i64>&, const Vec3<f32>&,   \
        PeakMode, const Vec3<i64>&, Stream&)    \
        -> std::pair<Vec3<f32>, T>

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(f32);
    NOA_INSTANTIATE_XPEAK_ALL(f64);
}
