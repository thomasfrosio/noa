#include "noa/common/signal/Shape.h"
#include "noa/common/signal/details/FourierCorrelationPeak.h"
#include "noa/common/math/LeastSquare.h"

#include "noa/gpu/cuda/math/Find.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#include "noa/gpu/cuda/signal/fft/Shape.h"

#include "noa/gpu/cuda/utils/Pointers.h"
#include "noa/gpu/cuda/utils/Warp.cuh"
#include "noa/gpu/cuda/utils/Block.cuh"

// TODO If centered, select subregion within max_radius.
// TODO Try to merge the 1D, 2D and 3D cases in to a single function?
//      The annoying bit right now is that the library doesn't have a 1D vector.

namespace {
    using namespace ::noa;

    template<fft::Remap REMAP, typename T, typename line_t>
    __global__ void maskEllipse1D_(Accessor<T, 2, uint32_t> xmap, uint32_t size, line_t line) {
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= size)
            return;
        const float coord = REMAP == fft::FC2FC ? gid : math::FFTShift(gid, size);
        xmap(blockIdx.y, gid) *= line(coord);
    }

    template<fft::Remap REMAP, typename T>
    void enforceMaxRadiusInPlace1D_(T* xmap, float max_radius,
                                    uint2_t shape_1d, uint2_t strides_1d,
                                    cuda::Stream& stream) {
        const auto center = static_cast<float>(shape_1d[1] / 2);
        const float edge_size = static_cast<float>(shape_1d[1]) * 0.05f;

        const Accessor<T, 2, uint32_t> accessor(xmap, strides_1d);

        using real_t = traits::value_type_t<T>;
        using line_t = noa::signal::LineSmooth<real_t>;
        const line_t line(center, max_radius, edge_size);

        const uint32_t threads = noa::math::min(128u, math::nextMultipleOf(shape_1d[1], 32u));
        const dim3 blocks(noa::math::divideUp(shape_1d[1], threads), shape_1d[0]);
        const cuda::LaunchConfig config{blocks, threads};
        stream.enqueue("signal::xpeak1D_mask", maskEllipse1D_<REMAP, T, line_t>, config,
                       accessor, shape_1d[1], line);
    }
}

namespace {
    using namespace ::noa;

    // Copy the window of the peak. The number of threads should be >= than the window size.
    // The read to global memory only coalesces if the stride is 1.
    template<fft::Remap REMAP, typename Real>
    constexpr NOA_FD Real copyWindowParabola1D_(const Real* xmap, uint32_t xmap_stride, int32_t xmap_size,
                                                int32_t peak_index, int32_t peak_radius, int32_t tid) {
        Real value = 0;
        const int32_t elements = peak_radius * 2 + 1;
        if (tid < elements) {
            const int32_t tid_offset = tid - peak_radius;
            if constexpr (REMAP == fft::F2F) {
                using namespace noa::signal::fft::details;
                const int32_t tid_frequency = nonCenteredIndex2Frequency(peak_index, xmap_size) + tid_offset;
                if (-xmap_size / 2 <= tid_frequency && tid_frequency <= (xmap_size - 1) / 2) {
                    const int32_t tid_index = frequency2NonCenteredIndex(tid_frequency, xmap_size);
                    value = xmap[tid_index * xmap_stride];
                }
            } else {
                const int32_t tid_index = peak_index + tid_offset;
                if (0 <= tid_index && tid_index < xmap_size)
                    value = xmap[tid_index * xmap_stride];
            }
        }
        return value;
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void subpixelRegistrationParabola1D_1D(
            const Real* __restrict__ xmap, uint2_t xmap_strides, int32_t xmap_size,
            const uint32_t* __restrict__ peak_offset,
            float* __restrict__ peak_coord,
            Real* __restrict__ peak_value,
            int32_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const uint32_t argmax = peak_offset[batch];
        Real* peak_window = block::dynamicSharedResource<Real>();
        peak_window[tidx] = copyWindowParabola1D_<REMAP>(
                xmap, xmap_strides[1], xmap_size, argmax, peak_radius, tidx);
        block::synchronize();

        // Fit parabola and compute vertex.
        // Not a lot of parallelism here, but while we could use wrap reduction, for most cases,
        // the peak radius is quite small, and it's just simpler (and maybe faster) to use a single
        // thread for the rest of this.
        if (tidx == 0) {
            auto peak_coord_2d = float2_t{
                REMAP == fft::F2F ? math::FFTShift(static_cast<int32_t>(argmax), xmap_size) : argmax, 0};
            const auto value = signal::fft::details::vertexParabola1D<1>(peak_window, int2_t{peak_radius}, peak_coord_2d);
            if (peak_coord)
                peak_coord[batch] = peak_coord_2d[0];
            if (peak_value)
                peak_value[batch] = static_cast<Real>(value);
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void subpixelRegistrationParabola1D_2D(
            const Real* __restrict__ xmap, uint3_t xmap_strides, int2_t xmap_shape,
            const uint32_t* __restrict__ peak_offset,
            float2_t* __restrict__ peak_coord,
            Real* __restrict__ peak_value,
            int2_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const auto xmap_strides_2d = uint2_t(xmap_strides.get(1));
        const auto peak_index = int2_t(indexing::indexes(peak_offset[batch], xmap_strides_2d, uint2_t(xmap_shape)));
        Real* peak_window = block::dynamicSharedResource<Real>();
        const int2_t peak_window_offset = {0, peak_radius[0] * 2 + 1};

        peak_window[tidx + peak_window_offset[0]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[1] * xmap_strides_2d[1], xmap_strides_2d[0], xmap_shape[0],
                peak_index[0], peak_radius[0], tidx);
        block::synchronize();
        peak_window[tidx + peak_window_offset[1]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[0] * xmap_strides_2d[0], xmap_strides_2d[1], xmap_shape[1],
                peak_index[1], peak_radius[1], tidx);
        block::synchronize();

        if (tidx == 0) {
            auto peak_coord_2d = float2_t(REMAP == fft::F2F ? math::FFTShift(peak_index, xmap_shape) : peak_index);
            const auto value = signal::fft::details::vertexParabola1D<2>(peak_window, peak_radius, peak_coord_2d);
            if (peak_coord)
                peak_coord[batch] = peak_coord_2d;
            if (peak_value)
                peak_value[batch] = static_cast<Real>(value);
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void subpixelRegistrationParabola1D_3D(
            const Real* __restrict__ xmap, uint4_t xmap_strides, int3_t xmap_shape,
            const uint32_t* __restrict__ peak_offset,
            float3_t* __restrict__ peak_coord,
            Real* __restrict__ peak_value,
            int3_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const auto xmap_strides_3d = uint3_t(xmap_strides.get(1));
        const auto peak_index = int3_t(indexing::indexes(peak_offset[batch], xmap_strides_3d, uint3_t(xmap_shape)));
        Real* peak_window = block::dynamicSharedResource<Real>();
        int3_t peak_window_offset;
        peak_window_offset[1] = peak_radius[0] * 2 + 1;
        peak_window_offset[2] = peak_radius[1] * 2 + 1 + peak_window_offset[1];

        peak_window[tidx + peak_window_offset[0]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[1] * xmap_strides_3d[1] + peak_index[2] * xmap_strides_3d[2],
                xmap_strides_3d[0], xmap_shape[0], peak_index[0], peak_radius[0], tidx);
        block::synchronize();
        peak_window[tidx + peak_window_offset[1]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[0] * xmap_strides_3d[0] + peak_index[2] * xmap_strides_3d[2],
                xmap_strides_3d[1], xmap_shape[1], peak_index[1], peak_radius[1], tidx);
        block::synchronize();
        peak_window[tidx + peak_window_offset[2]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[0] * xmap_strides_3d[0] + peak_index[1] * xmap_strides_3d[1],
                xmap_strides_3d[2], xmap_shape[2], peak_index[2], peak_radius[2], tidx);
        block::synchronize();

        if (tidx == 0) {
            auto peak_coord_3d = float3_t(REMAP == fft::F2F ? math::FFTShift(peak_index, xmap_shape) : peak_index);
            const auto value = signal::fft::details::vertexParabola1D<3>(peak_window, peak_radius, peak_coord_3d);
            if (peak_coord)
                peak_coord[batch] = peak_coord_3d;
            if (peak_value)
                peak_value[batch] = static_cast<Real>(value);
        }
    }

    template<int32_t NDIM, typename UIntN, typename IntN>
    constexpr NOA_FD auto peakOffsetToIndex_(uint32_t peak_offset, UIntN xmap_strides, IntN xmap_shape) {
        if constexpr (NDIM == 1) {
            return static_cast<IntN>(peak_offset / xmap_strides);
        } else { // NDIM == 3
            return static_cast<IntN>(indexing::indexes(peak_offset, xmap_strides, UIntN(xmap_shape)));
        }
    }

    template<int32_t NDIM, typename IntN>
    constexpr NOA_FD auto peakWindowOffsetToIndex_(int32_t offset, IntN peak_width) {
        if constexpr (NDIM == 1) {
            return offset;
        } else if constexpr (NDIM == 2) {
            return indexing::indexes(offset, peak_width[1]);
        } else { // NDIM == 3
            return indexing::indexes(offset, peak_width[1], peak_width[2]);
        }
    }

    // The block must be a single warp.
    // Shared memory size is the size of the peak window.
    // Elements are collected and saved in the shared buffer.
    template<fft::Remap REMAP, int32_t NDIM, typename Real,
             typename UIntN, typename IntN, typename FloatN>
    __global__ void __launch_bounds__(cuda::Limits::WARP_SIZE)
    subpixelRegistrationCOM_ND(
            const Real* __restrict__ xmap,
            uint32_t xmap_stride_batch, UIntN xmap_strides, IntN xmap_shape,
            const uint32_t* __restrict__ peak_offset,
            FloatN* __restrict__ peak_coords,
            Real* __restrict__ peak_values,
            IntN peak_radius, int32_t peak_window_elements) {

        constexpr int32_t WARP_SIZE = cuda::Limits::WARP_SIZE;
        NOA_ASSERT(WARP_SIZE == blockDim.x);

        const uint32_t batch = blockIdx.x;
        xmap += xmap_stride_batch * batch;

        const auto peak_index = peakOffsetToIndex_<NDIM>(peak_offset[batch], xmap_strides, xmap_shape);
        const IntN peak_width = peak_radius * 2 + 1;

        // Collect the elements within the peak window.
        // Values are stored in the rightmost order.
        using namespace cuda::utils;
        Real* peak_window_values = block::dynamicSharedResource<Real>();

        // We'll need to subtract by the peak window min.
        // Here, each thread collects the min of the values it processed.
        Real tid_peak_window_min{0};

        const auto tid = static_cast<int32_t>(threadIdx.x);
        if constexpr (REMAP == fft::FC2FC) {
            for (int32_t i = tid; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = peakWindowOffsetToIndex_<NDIM>(i, peak_width);
                const auto relative_index = indexes - peak_radius;
                const auto current_index = peak_index + relative_index;

                // Collect the value in the peak window at these indexes.
                // If the indexes are OOB, we still need to initialize the shared array with 0.
                Real value{0};
                if (noa::all(current_index >= 0 && current_index < xmap_shape)) {
                    value = xmap[indexing::at(current_index, xmap_strides)];
                    tid_peak_window_min = math::min(tid_peak_window_min, value);
                }
                peak_window_values[i] = value;
            }
        } else if constexpr (REMAP == fft::F2F) {
            // The peak window can be split across two separate quadrant.
            // Retrieve the frequency and if it is a valid frequency,
            // convert back to an index and compute the memory offset.
            using namespace noa::signal::fft::details;
            const IntN frequency_min = -xmap_shape / 2;
            const IntN frequency_max = (xmap_shape - 1) / 2;
            const IntN peak_frequency = nonCenteredIndex2Frequency(peak_index, xmap_shape);

            for (int32_t i = tid; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = peakWindowOffsetToIndex_<NDIM>(i, peak_width);
                const auto relative_index = indexes - peak_radius;
                const auto current_frequency = peak_frequency + relative_index;

                Real value{0};
                if (noa::all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                    const IntN current_index = frequency2NonCenteredIndex(current_frequency, xmap_shape);
                    value = xmap[indexing::at(current_index, xmap_strides)];
                    tid_peak_window_min = math::min(tid_peak_window_min, value);
                }
                peak_window_values[i] = value;
            }
        } else {
            static_assert(noa::traits::always_false_v<Real>);
        }

        // Compute the min of the peak window.
        tid_peak_window_min = warp::reduce(tid_peak_window_min, math::less_t{});
        warp::shuffle(tid_peak_window_min, 0);

        // Set the min to 0 and compute the partial COM.
        block::synchronize();

        // FIXME We don't have 1D support. We need to add it (Vector<N,T>).
        //       In the meantime, have a special case for 1D...
        if constexpr (NDIM == 1) {
            double tid_com{0};
            double tid_com_total{0};
            for (int32_t i = tid; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = peakWindowOffsetToIndex_<NDIM>(i, peak_width);
                const auto relative_index = static_cast<double>(indexes - peak_radius);
                const auto value = static_cast<double>(peak_window_values[i] - tid_peak_window_min);
                tid_com += value * relative_index;
                tid_com_total += value;
            }

            // Then the first thread collects everything.
            tid_com = warp::reduce(tid_com, math::plus_t{});
            tid_com_total = warp::reduce(tid_com_total, math::plus_t{});

            // From that point, only the value in the thread 0 matters.
            if (tid == 0) {
                if (peak_coords) {
                    if (tid_com_total != 0)
                        tid_com /= tid_com_total;
                    const auto f_peak_index =
                            REMAP == noa::fft::FC2FC ?
                            peak_index : noa::math::FFTShift(peak_index, xmap_shape);
                    tid_com += static_cast<double>(f_peak_index);
                    peak_coords[batch] = static_cast<float>(tid_com);
                }
                if (peak_values)
                    peak_values[batch] = peak_window_values[peak_radius];
            }

        } else {
            using doubleN_t = std::conditional_t<NDIM == 2, double2_t, double3_t>;
            doubleN_t tid_com{0};
            double tid_com_total{0};
            for (int32_t i = tid; i < peak_window_elements; i += WARP_SIZE) {
                const auto indexes = peakWindowOffsetToIndex_<NDIM>(i, peak_width);
                const auto relative_index = doubleN_t(indexes - peak_radius);
                const auto value = static_cast<double>(peak_window_values[i] - tid_peak_window_min);
                tid_com += value * relative_index;
                tid_com_total += value;
            }

            // Then the first thread collects everything.
            for (int32_t dim = 0; dim < NDIM; ++dim)
                tid_com[dim] = warp::reduce(tid_com[dim], math::plus_t{});
            tid_com_total = warp::reduce(tid_com_total, math::plus_t{});

            // From that point, only the value in the thread 0 matters.
            if (tid == 0) {
                if (peak_coords) {
                    if (tid_com_total != 0)
                        tid_com /= tid_com_total;
                    const auto f_peak_index =
                            REMAP == noa::fft::FC2FC ?
                            peak_index : noa::math::FFTShift(peak_index, xmap_shape);
                    tid_com += static_cast<doubleN_t>(f_peak_index);
                    peak_coords[batch] = static_cast<FloatN>(tid_com);
                }
                if (peak_values) {
                    const auto offset = noa::indexing::at(peak_radius, peak_width.strides());
                    peak_values[batch] = peak_window_values[offset];
                }
            }
        }
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                 const shared_t<float[]>& peak_coordinates, const shared_t<Real[]>& peak_values,
                 PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(peak_radius > 0);
        NOA_ASSERT(strides[3 - is_column] > 0);
        NOA_ASSERT(all(shape > 0) && dim3_t(shape.get(1)).ndim() == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        NOA_ASSERT(peak_coordinates == nullptr || ::noa::cuda::utils::devicePointer(peak_coordinates.get(), stream.device()));
        NOA_ASSERT(peak_values == nullptr || ::noa::cuda::utils::devicePointer(peak_values.get(), stream.device()));

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_1d{u_shape[0], u_shape[3 - is_column]};
        const uint2_t u_strides_1d{u_strides[0], u_strides[3 - is_column]};

        // In-place mask on the phase-correlation map.
        if (xmap_ellipse_radius > 0)
            enforceMaxRadiusInPlace1D_<REMAP>(xmap.get(), xmap_ellipse_radius, u_shape_1d, u_strides_1d, stream);

        // Find the maximum value.
        const auto peak_offsets = memory::PtrPinned<uint32_t>(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, peak_offsets.share(), true, true, stream);

        const auto peak_window_size = clamp_cast<uint32_t>(peak_radius * 2 + 1);
        switch (peak_mode) {
            case noa::signal::PEAK_PARABOLA_1D: {
                const uint32_t threads = noa::math::nextMultipleOf(peak_window_size, Limits::WARP_SIZE);
                const uint32_t blocks = u_shape_1d[0];
                const LaunchConfig config{blocks, threads, threads * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak1D_parabola1D",
                               subpixelRegistrationParabola1D_1D<REMAP, Real>, config,
                               xmap.get(), u_strides_1d, u_shape_1d[1],
                               peak_offsets.get(), peak_coordinates.get(), peak_values.get(), peak_radius);
                break;
            }
            case noa::signal::PEAK_COM: {
                const uint32_t threads = Limits::WARP_SIZE;
                const uint32_t blocks = u_shape_1d[0];
                const LaunchConfig config{blocks, threads, peak_window_size * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak2D_COM",
                               subpixelRegistrationCOM_ND<REMAP, 1, Real, uint32_t, int32_t, float>, config,
                               xmap.get(), u_strides_1d[0], u_strides_1d[1], u_shape_1d[1],
                               peak_offsets.get(), peak_coordinates.get(), peak_values.get(),
                               peak_radius, peak_window_size);
                break;
            }
        }
        stream.attach(xmap, peak_coordinates, peak_values);
    }

    template<Remap REMAP, typename Real, typename>
    [[nodiscard]] std::pair<float, Real>
    xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
            PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        const auto peak_coordinate = memory::PtrPinned<float>::alloc(1);
        const auto peak_value = memory::PtrPinned<Real>::alloc(1);

        const shared_t<float[]> coordinate_ptr(xmap, peak_coordinate.get());
        const shared_t<Real[]> value_ptr(xmap, peak_value.get());
        xpeak1D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       coordinate_ptr, value_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return {peak_coordinate[0], peak_value[0]};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                 const shared_t<float2_t[]>& peak_coordinates, const shared_t<Real[]>& peak_values,
                 PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(peak_radius > 0));
        NOA_ASSERT(all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        NOA_ASSERT(peak_coordinates == nullptr || ::noa::cuda::utils::devicePointer(peak_coordinates.get(), stream.device()));
        NOA_ASSERT(peak_values == nullptr || ::noa::cuda::utils::devicePointer(peak_values.get(), stream.device()));

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const auto u_strides_2d = uint3_t{u_strides[0], u_strides[2], u_strides[3]};
        const auto i_shape_2d = int2_t{u_shape[2], u_shape[3]};

        // In-place mask on the phase-correlation map.
        if (any(xmap_ellipse_radius > 0)) {
            const float2_t center(i_shape_2d / 2);
            const float edge_size = static_cast<float>(noa::math::max(i_shape_2d)) * 0.05f;
            ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                           center, xmap_ellipse_radius, edge_size,
                           float22_t{}, noa::math::multiply_t{}, Real{1}, false, stream);
        }

        // Find the maximum value.
        const auto peak_offsets = memory::PtrPinned<uint32_t>(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, peak_offsets.share(), true, true, stream);

        const auto peak_window_size = clamp_cast<uint2_t>(peak_radius * 2 + 1);
        switch (peak_mode) {
            case noa::signal::PEAK_PARABOLA_1D: {
                const uint32_t threads = noa::math::nextMultipleOf(noa::math::max(peak_window_size), Limits::WARP_SIZE);
                const uint32_t size_shared_memory = noa::math::max(threads, peak_window_size[1]) + peak_window_size[0];
                const uint32_t blocks = u_shape[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak2D_parabola1D",
                               subpixelRegistrationParabola1D_2D<REMAP, Real>, config,
                               xmap.get(), u_strides_2d, i_shape_2d,
                               peak_offsets.get(), peak_coordinates.get(), peak_values.get(), int2_t(peak_radius));
                break;
            }
            case noa::signal::PEAK_COM: {
                const uint32_t threads = Limits::WARP_SIZE;
                const uint32_t blocks = u_shape[0];
                const uint32_t size_shared_memory = noa::math::prod(peak_window_size);
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak2D_COM",
                               subpixelRegistrationCOM_ND<REMAP, 2, Real, uint2_t, int2_t, float2_t>, config,
                               xmap.get(), u_strides_2d[0], uint2_t(u_strides_2d[1], u_strides_2d[2]), i_shape_2d,
                               peak_offsets.get(), peak_coordinates.get(), peak_values.get(), int2_t(peak_radius),
                               size_shared_memory);
                break;
            }
        }
        stream.attach(xmap, peak_coordinates, peak_values);
    }

    template<Remap REMAP, typename Real, typename>
    [[nodiscard]] std::pair<float2_t, Real>
    xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
            PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        const auto peak_coordinate = memory::PtrPinned<float2_t>::alloc(1);
        const auto peak_value = memory::PtrPinned<Real>::alloc(1);

        const shared_t<float2_t[]> coordinate_ptr(xmap, peak_coordinate.get());
        const shared_t<Real[]> value_ptr(xmap, peak_value.get());
        xpeak2D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       coordinate_ptr, value_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return {peak_coordinate[0], peak_value[0]};
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                 const shared_t<float3_t[]>& peak_coordinates, const shared_t<Real[]>& peak_values,
                 PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(peak_radius > 0));
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        NOA_ASSERT(peak_coordinates == nullptr || ::noa::cuda::utils::devicePointer(peak_coordinates.get(), stream.device()));
        NOA_ASSERT(peak_values == nullptr || ::noa::cuda::utils::devicePointer(peak_values.get(), stream.device()));

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const auto u_strides_3d = uint3_t(u_strides.get(1));
        const auto i_shape_3d = int3_t{u_shape[1], u_shape[2], u_shape[3]};

        // In-place mask on the phase-correlation map.
        if (any(xmap_ellipse_radius > 0)) {
            const float3_t center(i_shape_3d / 2);
            const float edge_size = static_cast<float>(noa::math::max(i_shape_3d)) * 0.05f;
            ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                           center, xmap_ellipse_radius, edge_size,
                           float33_t{}, noa::math::multiply_t{}, Real{1}, false, stream);
        }

        // Find the maximum value.
        const auto peak_offsets = memory::PtrPinned<uint32_t>(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, peak_offsets.share(), true, true, stream);

        const auto peak_window_size = clamp_cast<uint3_t>(peak_radius * 2 + 1);
        switch (peak_mode) {
            case noa::signal::PEAK_PARABOLA_1D: {
                const uint32_t threads = noa::math::nextMultipleOf(noa::math::max(peak_window_size), Limits::WARP_SIZE);
                const uint32_t blocks = u_shape[0];
                const uint32_t size_shared_memory =
                        noa::math::max(threads, peak_window_size[2]) +
                        peak_window_size[1] +
                        peak_window_size[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak3D_parabola1D",
                               subpixelRegistrationParabola1D_3D<REMAP, Real>, config,
                               xmap.get(), u_strides, i_shape_3d,
                               peak_offsets.get(), peak_coordinates.get(), peak_values.get(), int3_t(peak_radius));
                break;
            }
            case noa::signal::PEAK_COM: {
                NOA_ASSERT(all(peak_radius <= 2));
                const uint32_t threads = Limits::WARP_SIZE;
                const uint32_t blocks = u_shape[0];
                const uint32_t size_shared_memory = noa::math::prod(peak_window_size);
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak2D_COM",
                               subpixelRegistrationCOM_ND<REMAP, 3, Real, uint3_t, int3_t, float3_t>, config,
                               xmap.get(), u_strides[0], u_strides_3d, i_shape_3d,
                               peak_offsets.get(), peak_coordinates.get(), peak_values.get(), int3_t(peak_radius),
                               size_shared_memory);
                break;
            }
        }
        stream.attach(xmap, peak_coordinates, peak_values);
    }

    template<Remap REMAP, typename Real, typename>
    [[nodiscard]] std::pair<float3_t, Real>
    xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
            PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        const auto peak_coordinate = memory::PtrPinned<float3_t>::alloc(1);
        const auto peak_value = memory::PtrPinned<Real>::alloc(1);

        const shared_t<float3_t[]> coordinate_ptr(xmap, peak_coordinate.get());
        const shared_t<Real[]> value_ptr(xmap, peak_value.get());
        xpeak3D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       coordinate_ptr, value_ptr, peak_mode, peak_radius, stream);
        stream.synchronize();
        return {peak_coordinate[0], peak_value[0]};
    }

    #define NOA_INSTANTIATE_XPEAK(R, T) \
    template void xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, const shared_t<float[]>&, const shared_t<T[]>&, PeakMode, int64_t, Stream&);         \
    template void xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, const shared_t<float2_t[]>&, const shared_t<T[]>&, PeakMode, long2_t, Stream&);   \
    template void xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, const shared_t<float3_t[]>&, const shared_t<T[]>&, PeakMode, long3_t, Stream&);   \
    template std::pair<float,T> xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, PeakMode, int64_t, Stream&);                                           \
    template std::pair<float2_t,T> xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, PeakMode, long2_t, Stream&);                                     \
    template std::pair<float3_t,T> xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, PeakMode, long3_t, Stream&)

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(float);
    NOA_INSTANTIATE_XPEAK_ALL(double);
}
