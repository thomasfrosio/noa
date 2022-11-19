#include "noa/common/signal/Shape.h"
#include "noa/common/math/LeastSquare.h"

#include "noa/gpu/cuda/fft/Transforms.h"
#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/math/Find.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#include "noa/gpu/cuda/signal/fft/Shape.h"
#include "noa/gpu/cuda/signal/fft/Shift.h"

#include "noa/gpu/cuda/utils/EwiseBinary.cuh"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"
#include "noa/gpu/cuda/utils/ReduceBinary.cuh"
#include "noa/gpu/cuda/utils/Warp.cuh"

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
    template<fft::Remap REMAP, typename T>
    constexpr NOA_FD T copyWindowParabola1D_(const T* input, uint32_t stride, int32_t size,
                                             int32_t peak, int32_t radius, int32_t tid) {
        auto nonCenteredIndex2Frequency_ = [] (int32_t index, int32_t dim_size) {
            return index < (dim_size + 1) / 2 ? index : index - dim_size;
        };

        auto frequency2NonCenteredIndex_ = [] (int32_t frequency, int32_t dim_size) {
            return frequency < 0 ? dim_size + frequency : frequency;
        };

        T value = 0;
        const int32_t elements = radius * 2 + 1;
        if (tid < elements) {
            const int32_t thread_offset = tid - radius;
            if constexpr (REMAP == fft::F2F) {
                const int32_t tid_frequency = nonCenteredIndex2Frequency_(peak, size) + thread_offset;
                if (-size / 2 <= tid_frequency && tid_frequency <= (size - 1) / 2) {
                    const int64_t tid_index = frequency2NonCenteredIndex_(tid_frequency, size);
                    value = input[tid_index * stride];
                }
            } else {
                const int32_t index = peak + thread_offset;
                if (0 <= index && index < size)
                    value = input[index * stride];
            }
        }
        return value;
    }

    template<fft::Remap REMAP, typename T>
    __global__ void copy1DWindowParabola_(const T* __restrict__ input, uint2_t strides, int32_t size,
                                          const uint32_t* __restrict__ peaks,
                                          T* __restrict__ output_window, int32_t window_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        input += strides[0] * batch;

        output_window[tidx] = copyWindowParabola1D_<REMAP>(input, strides[1], size, peaks[batch], window_radius, tidx);
    }

    template<fft::Remap REMAP, typename T>
    __global__ void copy2DWindowParabola_(const T* __restrict__ input, uint3_t strides, int2_t shape,
                                          const uint32_t* __restrict__ max_value_offsets,
                                          T* __restrict__ output_window, int2_t window_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        input += strides[0] * batch;

        const uint32_t max_value_offset = max_value_offsets[batch];
        const auto peak = int2_t(indexing::indexes(max_value_offset, uint2_t(strides.get(1)), uint2_t(shape)));
        output_window[tidx] =
                copyWindowParabola1D_<REMAP>(input + peak[1] * strides[2], strides[1],
                                             shape[0], peak[0], window_radius[0], tidx);
        output_window[tidx + window_radius[0] * 2 + 1] =
                copyWindowParabola1D_<REMAP>(input + peak[0] * strides[1], strides[2],
                                             shape[1], peak[1], window_radius[1], tidx);
    }

    template<fft::Remap REMAP, typename T>
    __global__ void copy3DWindowParabola_(const T* __restrict__ input, uint4_t strides, int3_t shape,
                                          const uint32_t* __restrict__ max_value_offsets,
                                          T* __restrict__ output_window, int3_t window_radius) {
        using namespace cuda::utils;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int32_t>(threadIdx.x);
        input += strides[0] * batch;

        const uint32_t max_value_offset = max_value_offsets[batch];
        const auto peak = int3_t(indexing::indexes(max_value_offset, uint3_t(strides.get(1)), uint3_t(shape)));
        output_window[tidx] =
                copyWindowParabola1D_<REMAP>(input + peak[1] * strides[2] + peak[2] * strides[3],
                                             strides[1], shape[0], peak[0], window_radius[0], tidx);
        output_window[tidx + window_radius[0] * 2 + 1] =
                copyWindowParabola1D_<REMAP>(input + peak[0] * strides[1] + peak[2] * strides[3],
                                             strides[2], shape[1], peak[1], window_radius[1], tidx);
        output_window[tidx + window_radius[0] * 2 + 1 + window_radius[1] * 2 + 1] =
                copyWindowParabola1D_<REMAP>(input + peak[0] * strides[1] + peak[1] * strides[2],
                                             strides[3], shape[2], peak[2], window_radius[2], tidx);
    }

    // This is equivalent to math::lstsqFitQuadratic(), but slightly faster
    // (I don't think it is significantly faster though).
    template<typename T>
    inline T getParabolicVertex3Points_(T y0, T y1, T y2) noexcept {
        // From IMOD/libcfshr/filtxcorr.c::parabolicFitPosition
        const T d = 2 * (y0 + y2 - 2 * y1);
        T x = 0;
        if (math::abs(d) > math::abs(static_cast<T>(1e-2) * (y0 - y2)))
            x = (y0 - y2) / d;
        if (x > T{0.5})
            x = T{0.5};
        if (x < T{-0.5})
            x = T{-0.5};
        return x;
    }

    template<fft::Remap REMAP, typename T>
    float addVertexOffsetParabola1D(const T* window, int32_t window_radius, int32_t size, int32_t peak) {
        static_assert(REMAP == fft::F2F || REMAP == fft::FC2FC);

        const auto window_elements = static_cast<size_t>(window_radius) * 2 + 1;

        // Add sub-pixel position by fitting a 1D parabola to the peak and its adjacent points.
        float vertex_offset;
        if (window_radius == 1) {
            vertex_offset = static_cast<float>(getParabolicVertex3Points_(window[0], window[1], window[2]));
        } else {
            const auto [a, b, _] = math::lstsqFitQuadratic(window, window_elements);
            NOA_ASSERT(a != 0); // This can only happen if all values in output are equal.
            const auto radius_ = static_cast<double>(window_radius);
            vertex_offset = static_cast<float>(std::clamp(-b / (2 * a) - radius_, -radius_ + 0.5, radius_ - 0.5));
        }
        if constexpr (REMAP == fft::F2F)
            peak = math::FFTShift(peak, size);
        return static_cast<float>(peak) + vertex_offset;
    }

    template<fft::Remap REMAP, typename T>
    inline float subpixelRegistration1DParabola(const T* window, int32_t window_radius, int32_t size, int32_t peak) {
        // TODO Check assembly to see if this helps and is optimized.
        switch (window_radius) {
            case 1:
                return addVertexOffsetParabola1D<REMAP>(window, 1, size, peak);
            case 2:
                return addVertexOffsetParabola1D<REMAP>(window, 2, size, peak);
            case 3:
                return addVertexOffsetParabola1D<REMAP>(window, 3, size, peak);
            default:
                return addVertexOffsetParabola1D<REMAP>(window, window_radius, size, peak);
        }
    }

    template<fft::Remap REMAP, typename T>
    inline float2_t subpixelRegistration2DParabola(const T* window, int2_t window_radius, int2_t shape, int2_t peak) {
        return {
                subpixelRegistration1DParabola<REMAP>(
                        window, window_radius[0], shape[0], peak[0]),
                subpixelRegistration1DParabola<REMAP>(
                        window + window_radius[0] * 2 + 1, window_radius[1], shape[1], peak[1])
        };
    }

    template<fft::Remap REMAP, typename T>
    inline float3_t subpixelRegistration3DParabola(const T* window, int3_t window_radius, int3_t shape, int3_t peak) {
        return {
                subpixelRegistration1DParabola<REMAP>(
                        window, window_radius[0], shape[0], peak[0]),
                subpixelRegistration1DParabola<REMAP>(
                        window + window_radius[0] * 2 + 1, window_radius[1], shape[1], peak[1]),
                subpixelRegistration1DParabola<REMAP>(
                        window + window_radius[0] * 2 + 1 + window_radius[1] * 2 + 1, window_radius[2], shape[2], peak[2])
        };
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename T, typename>
    void xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float[]>& peaks,
                 float max_radius, int64_t registration_radius, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(strides[3 - is_column] > 0);
        NOA_ASSERT(all(shape > 0) && dim3_t(shape.get(1)).ndim() == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_1d{u_shape[0], u_shape[3 - is_column]};
        const uint2_t u_strides_1d{u_strides[0], u_strides[3 - is_column]};

        // In-place mask on the phase-correlation map.
        if (max_radius > 0)
            enforceMaxRadiusInPlace1D_<REMAP>(xmap.get(), max_radius, u_shape_1d, u_strides_1d, stream);

        // Find the maximum value.
        memory::PtrPinned<uint32_t> max_offset(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, max_offset.share(), true, true, stream);

        // Copy to the host the window centered on the max value.
        const auto registration_size = safe_cast<size_t>(registration_radius * 2 + 1);
        memory::PtrPinned<T> window(registration_size);
        const uint32_t threads = noa::math::nextMultipleOf(registration_size, size_t{32});
        const uint32_t blocks = u_shape_1d[0];
        stream.enqueue("signal::fft::xpeak1D", copy1DWindowParabola_<REMAP, T>, {blocks, threads},
                       xmap.get(), u_strides_1d, u_shape_1d[1],
                       max_offset.get(), window.get(), registration_radius);

        // Optional buffer on the host if the output peaks are on the device...
        float* peaks_ptr = utils::hostPointer(peaks.get());
        shared_t<float[]> buffer;
        if (!peaks_ptr) { // on the device
            buffer = memory::PtrPinned<float>::alloc(u_shape_1d[0]);
            peaks_ptr = buffer.get();
        }

        // Subpixel registration on the host.
        stream.synchronize(); // TODO Add callback
        for (size_t i = 0; i < shape[0]; ++i) {
            peaks_ptr[i] = subpixelRegistration1DParabola<REMAP>(
                    window.get(), registration_radius, u_shape_1d[1], max_offset[i]);
        }

        if (buffer) // host -> device
            memory::copy(buffer, peaks, shape[0], stream);
    }

    template<Remap REMAP, typename T, typename>
    float xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                  float max_radius, int64_t registration_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float peak;
        const shared_t<float[]> peak_ptr(xmap, &peak);
        xpeak1D<REMAP>(xmap, strides, shape, peak_ptr, max_radius, registration_radius, stream);
        // TODO Add sync when callback is added. Until then it is not necessary since we pass a host pointer.
        return peak;
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float2_t[]>& peaks,
                 float2_t max_radius, long2_t registration_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const auto i_radius = static_cast<int2_t>(registration_radius);
        const auto i_shape_2d = int2_t{u_shape[2], u_shape[3]};

        // In-place mask on the phase-correlation map.
        if (any(max_radius > 0)) {
            // TODO If centered, select subregion within max_radius.

            const float2_t center(i_shape_2d / 2);
            const float edge_size = static_cast<float>(noa::math::max(i_shape_2d)) * 0.05f;
            ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                           center, max_radius, edge_size, float22_t{}, false, stream);
        }

        // Find the maximum value.
        memory::PtrPinned<uint32_t> max_offset(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, max_offset.share(), true, true, stream);

        // Copy to the host the window centered on the max value.
        const auto registration_size = safe_cast<dim2_t>(registration_radius) * 2 + 1;
        memory::PtrPinned<T> window(noa::math::sum(registration_size));
        const uint32_t threads = noa::math::nextMultipleOf(noa::math::max(registration_size), dim_t{32});
        const uint32_t blocks = u_shape[0];
        stream.enqueue("signal::fft::xpeak2D", copy2DWindowParabola_<REMAP, T>, {blocks, threads},
                       xmap.get(), uint3_t{u_strides[0], u_strides[2], u_strides[3]}, i_shape_2d,
                       max_offset.get(), window.get(), i_radius);

        // Optional buffer on the host if the output peaks are on the device...
        float2_t* peaks_ptr = utils::hostPointer(peaks.get());
        shared_t<float2_t[]> buffer;
        if (!peaks_ptr) { // on the device
            buffer = memory::PtrPinned<float2_t>::alloc(shape[0]);
            peaks_ptr = buffer.get();
        }

        // Subpixel registration on the host.
        stream.synchronize(); // TODO Add callback
        const auto u_strides_2d = uint2_t(u_strides.get(2));
        const auto u_shape_2d = uint2_t(i_shape_2d);
        for (size_t i = 0; i < shape[0]; ++i) {
            const auto peak = int2_t(indexing::indexes(max_offset[i], u_strides_2d, u_shape_2d));
            peaks_ptr[i] = subpixelRegistration2DParabola<REMAP>(
                    window.get(), i_radius, i_shape_2d, peak);
        }

        if (buffer) // host -> device
            memory::copy(buffer, peaks, shape[0], stream);
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                     float2_t max_radius, long2_t registration_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float2_t peak;
        const shared_t<float2_t[]> peak_ptr(xmap, &peak);
        xpeak2D<REMAP>(xmap, strides, shape, peak_ptr, max_radius, registration_radius, stream);
        // TODO Add sync when callback is added. Until then it is not necessary since we pass a host pointer.
        return peak;
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, const shared_t<float3_t[]>& peaks,
                 float3_t max_radius, long3_t registration_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const auto i_radius = static_cast<int3_t>(registration_radius);
        const auto i_shape_3d = int3_t{u_shape[1], u_shape[2], u_shape[3]};

        // In-place mask on the phase-correlation map.
        if (any(max_radius > 0)) {
            // TODO If centered, select subregion within max_radius.

            const float3_t center(i_shape_3d / 2);
            const float edge_size = static_cast<float>(noa::math::max(i_shape_3d)) * 0.05f;
            ellipse<REMAP>(xmap, strides, xmap, strides, shape,
                           center, max_radius, edge_size, float33_t{}, false, stream);
        }

        // Find the maximum value.
        memory::PtrPinned<uint32_t> max_offset(shape[0]);
        math::find(noa::math::first_max_t{}, xmap, strides, shape, max_offset.share(), true, true, stream);

        // Copy to the host the window centered on the max value.
        const auto registration_size = safe_cast<dim3_t>(registration_radius) * 2 + 1;
        memory::PtrPinned<T> window(noa::math::sum(registration_size));
        const uint32_t threads = noa::math::nextMultipleOf(noa::math::max(registration_size), dim_t{32});
        const uint32_t blocks = u_shape[0];
        stream.enqueue("signal::fft::xpeak3D", copy3DWindowParabola_<REMAP, T>, {blocks, threads},
                       xmap.get(), u_strides, i_shape_3d,
                       max_offset.get(), window.get(), i_radius);

        // Optional buffer on the host if the output peaks are on the device...
        float3_t* peaks_ptr = utils::hostPointer(peaks.get());
        shared_t<float3_t[]> buffer;
        if (!peaks_ptr) { // on the device
            buffer = memory::PtrPinned<float3_t>::alloc(shape[0]);
            peaks_ptr = buffer.get();
        }

        // Subpixel registration on the host.
        stream.synchronize(); // TODO Add callback
        const auto u_strides_3d = uint3_t(u_strides.get(1));
        const auto u_shape_3d = uint3_t(i_shape_3d);
        for (size_t i = 0; i < shape[0]; ++i) {
            const auto peak = int3_t(indexing::indexes(max_offset[i], u_strides_3d, u_shape_3d));
            peaks_ptr[i] = subpixelRegistration3DParabola<REMAP>(
                    window.get(), i_radius, i_shape_3d, peak);
        }

        if (buffer) // host -> device
            memory::copy(buffer, peaks, shape[0], stream);
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                     float3_t max_radius, long3_t registration_radius, Stream& stream) {
        if (shape[0] != 1) // throw instead of assert because this could result in segfault
            NOA_THROW("This overload does not supported batched arrays, but got {} batches", shape[0]);

        float3_t peak;
        const shared_t<float3_t[]> peak_ptr(xmap, &peak);
        xpeak3D<REMAP>(xmap, strides, shape, peak_ptr, max_radius, registration_radius, stream);
        // TODO Add sync when callback is added. Until then it is not necessary since we pass a host pointer.
        return peak;
    }

    #define INSTANTIATE_XPEAK(R, T) \
    template void xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float[]>&, float, int64_t, Stream&);         \
    template void xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float2_t, long2_t, Stream&);   \
    template void xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float3_t, long3_t, Stream&);   \
    template float xpeak1D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, int64_t, Stream&);                                  \
    template float2_t xpeak2D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, long2_t, Stream&);                            \
    template float3_t xpeak3D<R, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, long3_t, Stream&)

    #define INSTANTIATE_XPEAK_ALL(T)    \
    INSTANTIATE_XPEAK(Remap::F2F, T);   \
    INSTANTIATE_XPEAK(Remap::FC2FC, T)

    INSTANTIATE_XPEAK_ALL(float);
    INSTANTIATE_XPEAK_ALL(double);
}
