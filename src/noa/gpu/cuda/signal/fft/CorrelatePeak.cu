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
            const Real* xmap, uint2_t xmap_strides, int32_t xmap_size,
            const uint32_t* peak_offset, float* peak_coord, int32_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const uint32_t argmax = peak_offset[batch];
        Real* peak_window = block::dynamicSharedResource<Real>();
        peak_window[tidx] = copyWindowParabola1D_<REMAP>(
                xmap, xmap_strides[1], xmap_size, argmax, peak_radius, tidx);

        // Fit parabola and compute vertex.
        // Not a lot of parallelism here, but while we could use wrap reduction, for most cases,
        // the peak radius is quite small, and it's just simpler (and maybe faster) to use a single
        // thread for the rest of this.
        block::synchronize();
        if (tidx == 0) {
            auto peak_coord_2d = float2_t{
                REMAP == fft::F2F ? math::FFTShift(static_cast<int32_t>(argmax), xmap_size) : argmax, 0};
            signal::fft::details::addSubpixelCoordParabola1D<1>(
                    peak_window, int2_t{peak_radius}, peak_coord_2d);
            peak_coord[batch] = peak_coord_2d[0];
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void subpixelRegistrationParabola1D_2D(
            const Real* xmap, uint3_t xmap_strides, int2_t xmap_shape,
            const uint32_t* peak_offset, float2_t* peak_coord, int2_t peak_radius) {
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
        peak_window[tidx + peak_window_offset[1]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[0] * xmap_strides_2d[0], xmap_strides_2d[1], xmap_shape[1],
                peak_index[1], peak_radius[1], tidx);

        block::synchronize();
        if (tidx == 0) {
            auto peak_coord_2d = float2_t(REMAP == fft::F2F ? math::FFTShift(peak_index, xmap_shape) : peak_index);
            signal::fft::details::addSubpixelCoordParabola1D<2>(
                    peak_window, peak_radius, peak_coord_2d);
            peak_coord[batch] = peak_coord_2d;
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void subpixelRegistrationParabola1D_3D(
            const Real* xmap, uint4_t xmap_strides, int3_t xmap_shape,
            const uint32_t* peak_offset, float3_t* peak_coord, int3_t peak_radius) {
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
        peak_window[tidx + peak_window_offset[1]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[0] * xmap_strides_3d[0] + peak_index[2] * xmap_strides_3d[2],
                xmap_strides_3d[1], xmap_shape[1], peak_index[1], peak_radius[1], tidx);
        peak_window[tidx + peak_window_offset[2]] = copyWindowParabola1D_<REMAP>(
                xmap + peak_index[0] * xmap_strides_3d[0] + peak_index[1] * xmap_strides_3d[1],
                xmap_strides_3d[2], xmap_shape[2], peak_index[2], peak_radius[2], tidx);

        block::synchronize();
        if (tidx == 0) {
            auto peak_coord_3d = float3_t(REMAP == fft::F2F ? math::FFTShift(peak_index, xmap_shape) : peak_index);
            signal::fft::details::addSubpixelCoordParabola1D<3>(
                    peak_window, peak_radius, peak_coord_3d);
            peak_coord[batch] = peak_coord_3d;
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void __launch_bounds__(cuda::Limits::WARP_SIZE)
    subpixelRegistrationCOM_1D(
            const Real* xmap, uint2_t xmap_strides, int32_t xmap_size,
            const uint32_t* peak_offset, float* peak_coords, int32_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const auto xmap_stride = static_cast<uint32_t>(xmap_strides[1]);
        const auto peak_index = static_cast<int32_t>(peak_offset[batch] / xmap_stride);
        const int32_t peak_width = peak_radius * 2 + 1;

        double tid_com{0}, tid_com_total{0};
        // Each thread collects its partial 3D COM...
        if constexpr (REMAP == fft::FC2FC) {
            for (int32_t l = tidx; l < peak_width; l += cuda::Limits::WARP_SIZE) {
                const auto relative_offset = l - peak_radius;
                const auto current_index = peak_index + relative_offset;
                const auto f_relative_offset = static_cast<double>(relative_offset);

                if (current_index >= 0 && current_index < xmap_size) {
                    const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_stride)]);
                    tid_com += value * f_relative_offset;
                    tid_com_total += value;
                }
            }
        } else if constexpr (REMAP == fft::F2F) {
            using namespace noa::signal::fft::details;
            const int32_t frequency_min = -xmap_size / 2;
            const int32_t frequency_max = (xmap_size - 1) / 2;
            const int32_t peak_frequency = nonCenteredIndex2Frequency(peak_index, xmap_size);

            for (int32_t l = tidx; l < peak_width; l += cuda::Limits::WARP_SIZE) {
                const auto relative_offset = l - peak_radius;
                const auto current_frequency = peak_frequency + relative_offset;
                const auto f_relative_offset = static_cast<double>(relative_offset);

                if (frequency_min <= current_frequency && current_frequency <= frequency_max) {
                    using namespace noa::signal::fft::details;
                    const int32_t current_index = frequency2NonCenteredIndex(current_frequency, xmap_size);
                    const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_stride)]);
                    tid_com += value * f_relative_offset;
                    tid_com_total += value;
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<Real>);
        }

        // ... and then the first thread collects everything.
        tid_com = warp::reduce(tid_com, math::plus_t{});
        tid_com_total = warp::reduce(tid_com_total, math::plus_t{});

        if (tidx == 0) {
            if (tid_com_total != 0)
                tid_com /= tid_com_total;
            tid_com += static_cast<double>(REMAP == fft::FC2FC ? peak_index : math::FFTShift(peak_index, xmap_size));
            peak_coords[batch] = static_cast<float>(tid_com);
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void __launch_bounds__(cuda::Limits::WARP_SIZE)
    subpixelRegistrationCOM_2D(
            const Real* xmap, uint3_t xmap_strides, int2_t xmap_shape,
            const uint32_t* peak_offset, float2_t* peak_coords, int2_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const auto xmap_strides_2d = uint2_t(xmap_strides.get(1));
        const auto peak_index = int2_t(indexing::indexes(peak_offset[batch], xmap_strides_2d, uint2_t(xmap_shape)));
        const int32_t peak_width = peak_radius[1] * 2 + 1;

        double2_t tid_com;
        double tid_com_total{0};
        // Each thread collects its partial 3D COM...
        if constexpr (REMAP == fft::FC2FC) {
            for (int32_t k = -peak_radius[0]; k <= peak_radius[0]; ++k) {
                for (int32_t l = tidx; l < peak_width; l += cuda::Limits::WARP_SIZE) {
                    const auto relative_offset = int2_t{k, l - peak_radius[1]};
                    const auto current_index = peak_index + relative_offset;
                    const auto f_relative_offset = double2_t(relative_offset);

                    if (all(current_index >= 0 && current_index < xmap_shape)) {
                        const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_strides_2d)]);
                        tid_com += value * f_relative_offset;
                        tid_com_total += value;
                    }
                }
            }
        } else if constexpr (REMAP == fft::F2F) {
            using namespace noa::signal::fft::details;
            const int2_t frequency_min = -xmap_shape / 2;
            const int2_t frequency_max = (xmap_shape - 1) / 2;
            const int2_t peak_frequency = nonCenteredIndex2Frequency(peak_index, xmap_shape);

            for (int32_t k = -peak_radius[0]; k <= peak_radius[0]; ++k) {
                for (int32_t l = tidx; l < peak_width; l += cuda::Limits::WARP_SIZE) {
                    const auto relative_offset = int2_t{k, l - peak_radius[1]};
                    const auto current_frequency = peak_frequency + relative_offset;
                    const auto f_relative_offset = double2_t(relative_offset);

                    if (all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                        using namespace noa::signal::fft::details;
                        const int2_t current_index = frequency2NonCenteredIndex(current_frequency, xmap_shape);
                        const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_strides_2d)]);
                        tid_com += value * f_relative_offset;
                        tid_com_total += value;
                    }
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<Real>);
        }

        // ... and then the first thread collects everything.
        for (int32_t dim = 0; dim < 2; ++dim)
            tid_com[dim] = warp::reduce(tid_com[dim], math::plus_t{});
        tid_com_total = warp::reduce(tid_com_total, math::plus_t{});

        if (tidx == 0) {
            if (tid_com_total != 0)
                tid_com /= tid_com_total;
            tid_com += double2_t(REMAP == fft::FC2FC ? peak_index : math::FFTShift(peak_index, xmap_shape));
            peak_coords[batch] = float2_t(tid_com);
        }
    }

    template<fft::Remap REMAP, typename Real>
    __global__ void __launch_bounds__(cuda::Limits::WARP_SIZE)
    subpixelRegistrationCOM_3D(
            const Real* xmap, uint4_t xmap_strides, int3_t xmap_shape,
            const uint32_t* peak_offset, float3_t* peak_coords, int3_t peak_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        xmap += xmap_strides[0] * batch;

        const auto xmap_strides_3d = uint3_t(xmap_strides.get(1));
        const auto peak_index = int3_t(indexing::indexes(peak_offset[batch], xmap_strides_3d, uint3_t(xmap_shape)));
        const int32_t peak_width = peak_radius[2] * 2 + 1;

        double3_t tid_com;
        double tid_com_total{};
        // Each thread collects its partial 3D COM...
        if constexpr (REMAP == fft::FC2FC) {
            for (int32_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (int32_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (int32_t l = tidx; l < peak_width; l += cuda::Limits::WARP_SIZE) {
                        const auto relative_offset = int3_t{j, k, l - peak_radius[2]};
                        const auto current_index = peak_index + relative_offset;
                        const auto f_relative_offset = double3_t(relative_offset);

                        if (all(current_index >= 0 && current_index < xmap_shape)) {
                            const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_strides_3d)]);
                            tid_com += value * f_relative_offset;
                            tid_com_total += value;
                        }
                    }
                }
            }
        } else if constexpr (REMAP == fft::F2F) {
            using namespace noa::signal::fft::details;
            const int3_t frequency_min = -xmap_shape / 2;
            const int3_t frequency_max = (xmap_shape - 1) / 2;
            const int3_t peak_frequency = nonCenteredIndex2Frequency(peak_index, xmap_shape);

            for (int32_t j = -peak_radius[0]; j <= peak_radius[0]; ++j) {
                for (int32_t k = -peak_radius[1]; k <= peak_radius[1]; ++k) {
                    for (int32_t l = tidx; l < peak_width; l += cuda::Limits::WARP_SIZE) {
                        const auto relative_offset = int3_t{j, k, l - peak_radius[2]};
                        const auto current_frequency = peak_frequency + relative_offset;
                        const auto f_relative_offset = double3_t(relative_offset);

                        if (all(frequency_min <= current_frequency && current_frequency <= frequency_max)) {
                            using namespace noa::signal::fft::details;
                            const int3_t current_index = frequency2NonCenteredIndex(current_frequency, xmap_shape);
                            const auto value = static_cast<double>(xmap[indexing::at(current_index, xmap_strides_3d)]);
                            tid_com += value * f_relative_offset;
                            tid_com_total += value;
                        }
                    }
                }
            }
        } else {
            static_assert(noa::traits::always_false_v<Real>);
        }

        // ... and then the first thread collects everything.
        for (int32_t dim = 0; dim < 3; ++dim)
            tid_com[dim] = warp::reduce(tid_com[dim], math::plus_t{});
        tid_com_total = warp::reduce(tid_com_total, math::plus_t{});

        if (tidx == 0) {
            if (tid_com_total != 0)
                tid_com /= tid_com_total;
            tid_com += double3_t(REMAP == fft::FC2FC ? peak_index : math::FFTShift(peak_index, xmap_shape));
            peak_coords[batch] = float3_t(tid_com);
        }
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename Real, typename>
    void xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                 const shared_t<float[]>& peak_coordinates, PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(strides[3 - is_column] > 0);
        NOA_ASSERT(all(shape > 0) && dim3_t(shape.get(1)).ndim() == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(peak_coordinates.get(), stream.device());

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_1d{u_shape[0], u_shape[3 - is_column]};
        const uint2_t u_strides_1d{u_strides[0], u_strides[3 - is_column]};

        // In-place mask on the phase-correlation map.
        if (xmap_ellipse_radius > 0)
            enforceMaxRadiusInPlace1D_<REMAP>(xmap.get(), xmap_ellipse_radius, u_shape_1d, u_strides_1d, stream);

        // Find the maximum value.
        memory::PtrPinned<uint32_t> peak_offsets(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, peak_offsets.share(), true, true, stream);

        switch (peak_mode) {
            case noa::signal::PEAK_PARABOLA_1D: {
                const auto peak_mode_size = clamp_cast<uint32_t>(peak_radius * 2 + 1);
                const uint32_t threads = noa::math::nextMultipleOf(peak_mode_size, Limits::WARP_SIZE);
                const uint32_t blocks = u_shape_1d[0];
                const LaunchConfig config{blocks, threads, threads * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak1D_parabola1D",
                               subpixelRegistrationParabola1D_1D<REMAP, Real>, config,
                               xmap.get(), u_strides_1d, u_shape_1d[1],
                               peak_offsets.get(), peak_coordinates.get(), peak_radius);
                break;
            }
            case noa::signal::PEAK_COM: {
                const uint32_t threads = Limits::WARP_SIZE;
                const uint32_t blocks = u_shape_1d[0];
                const LaunchConfig config{blocks, threads};
                stream.enqueue("signal::fft::xpeak1D_COM",
                               subpixelRegistrationCOM_1D<REMAP, Real>, config,
                               xmap.get(), u_strides_1d, u_shape_1d[1],
                               peak_offsets.get(), peak_coordinates.get(), peak_radius);
                break;
            }
        }
        stream.attach(xmap, peak_coordinates);
    }

    template<Remap REMAP, typename Real, typename>
    float xpeak1D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float xmap_ellipse_radius,
                  PeakMode peak_mode, int64_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        memory::PtrPinned<float> peak_coordinate(1);
        xpeak1D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       peak_coordinate.share(), peak_mode, peak_radius, stream);
        stream.synchronize();
        return peak_coordinate[0];
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                 const shared_t<float2_t[]>& peak_coordinates, PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(peak_coordinates.get(), stream.device());

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
        memory::PtrPinned<uint32_t> peak_offsets(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, peak_offsets.share(), true, true, stream);

        switch (peak_mode) {
            case noa::signal::PEAK_PARABOLA_1D: {
                const auto peak_mode_size = clamp_cast<uint2_t>(peak_radius * 2 + 1);
                const uint32_t threads = noa::math::nextMultipleOf(noa::math::max(peak_mode_size), Limits::WARP_SIZE);
                const uint32_t size_shared_memory = noa::math::max(threads, peak_mode_size[1]) + peak_mode_size[0];
                const uint32_t blocks = u_shape[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak2D_parabola1D",
                               subpixelRegistrationParabola1D_2D<REMAP, Real>, config,
                               xmap.get(), u_strides_2d, i_shape_2d,
                               peak_offsets.get(), peak_coordinates.get(), int2_t(peak_radius));
                break;
            }
            case noa::signal::PEAK_COM: {
                const uint32_t threads = Limits::WARP_SIZE;
                const uint32_t blocks = u_shape[0];
                const LaunchConfig config{blocks, threads};
                stream.enqueue("signal::fft::xpeak1D_COM",
                               subpixelRegistrationCOM_2D<REMAP, Real>, config,
                               xmap.get(), u_strides_2d, i_shape_2d,
                               peak_offsets.get(), peak_coordinates.get(), int2_t(peak_radius));
                break;
            }
        }
        stream.attach(xmap, peak_coordinates);
    }

    template<Remap REMAP, typename Real, typename>
    float2_t xpeak2D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float2_t xmap_ellipse_radius,
                     PeakMode peak_mode, long2_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        memory::PtrPinned<float2_t> peak_coordinate(1);
        xpeak2D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       peak_coordinate.share(), peak_mode, peak_radius, stream);
        stream.synchronize();
        return peak_coordinate[0];
    }

    template<Remap REMAP, typename Real, typename>
    void xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                 const shared_t<float3_t[]>& peak_coordinates, PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(peak_coordinates.get(), stream.device());

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
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
        memory::PtrPinned<uint32_t> peak_offsets(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, peak_offsets.share(), true, true, stream);

        switch (peak_mode) {
            case noa::signal::PEAK_PARABOLA_1D: {
                const auto peak_mode_size = clamp_cast<uint3_t>(peak_radius * 2 + 1);
                const uint32_t threads = noa::math::nextMultipleOf(noa::math::max(peak_mode_size), Limits::WARP_SIZE);
                const uint32_t blocks = u_shape[0];
                const uint32_t size_shared_memory =
                        noa::math::max(threads, peak_mode_size[2]) +
                        peak_mode_size[1] +
                        peak_mode_size[0];
                const LaunchConfig config{blocks, threads, size_shared_memory * sizeof(Real)};
                stream.enqueue("signal::fft::xpeak3D_parabola1D",
                               subpixelRegistrationParabola1D_3D<REMAP, Real>, config,
                               xmap.get(), u_strides, i_shape_3d,
                               peak_offsets.get(), peak_coordinates.get(), int3_t(peak_radius));
                break;
            }
            case noa::signal::PEAK_COM: {
                const uint32_t threads = Limits::WARP_SIZE;
                const uint32_t blocks = u_shape[0];
                const LaunchConfig config{blocks, threads};
                stream.enqueue("signal::fft::xpeak2D_COM",
                               subpixelRegistrationCOM_3D<REMAP, Real>, config,
                               xmap.get(), u_strides, i_shape_3d,
                               peak_offsets.get(), peak_coordinates.get(), int3_t(peak_radius));
                break;
            }
        }
        stream.attach(xmap, peak_coordinates);
    }

    template<Remap REMAP, typename Real, typename>
    float3_t xpeak3D(const shared_t<Real[]>& xmap, dim4_t strides, dim4_t shape, float3_t xmap_ellipse_radius,
                     PeakMode peak_mode, long3_t peak_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        memory::PtrPinned<float3_t> peak_coordinate(1);
        xpeak3D<REMAP>(xmap, strides, shape, xmap_ellipse_radius,
                       peak_coordinate.share(), peak_mode, peak_radius, stream);
        stream.synchronize();
        return peak_coordinate[0];
    }

    #define NOA_INSTANTIATE_XPEAK(R, T) \
    template void xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, const shared_t<float[]>&, PeakMode, int64_t, Stream&);         \
    template void xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, const shared_t<float2_t[]>&, PeakMode, long2_t, Stream&);   \
    template void xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, const shared_t<float3_t[]>&, PeakMode, long3_t, Stream&);   \
    template float xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, PeakMode, int64_t, Stream&);                                  \
    template float2_t xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, PeakMode, long2_t, Stream&);                            \
    template float3_t xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, PeakMode, long3_t, Stream&)

    #define NOA_INSTANTIATE_XPEAK_ALL(T)    \
    NOA_INSTANTIATE_XPEAK(Remap::F2F, T);   \
    NOA_INSTANTIATE_XPEAK(Remap::FC2FC, T)

    NOA_INSTANTIATE_XPEAK_ALL(float);
    NOA_INSTANTIATE_XPEAK_ALL(double);
}
