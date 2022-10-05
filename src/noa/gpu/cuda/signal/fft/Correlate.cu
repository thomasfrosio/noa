#include "noa/common/signal/Shape.h"

#include "noa/gpu/cuda/fft/Transforms.h"
#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/math/Find.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#include "noa/gpu/cuda/signal/fft/Shift.h"

#include "noa/gpu/cuda/util/EwiseBinary.cuh"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"
#include "noa/gpu/cuda/util/ReduceBinary.cuh"
#include "noa/gpu/cuda/util/Warp.cuh"

namespace {
    using namespace ::noa;
    constexpr dim3 BLOCK_SIZE_2D(32, 8);

    template<bool IS_CENTERED, typename T, typename line_t>
    __global__ void maskEllipse1D_(Accessor<T, 2, uint32_t> xmap, uint32_t size, line_t line) {
        const uint gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= size)
            return;

        const float coord = IS_CENTERED ? gid : math::FFTShift(gid, size);
        xmap(blockIdx.y, gid) *= line(coord);
    }

    template<bool IS_CENTERED, typename T, typename ellipse_t>
    __global__ __launch_bounds__(BLOCK_SIZE_2D.x * BLOCK_SIZE_2D.y)
    void maskEllipse2D_(Accessor<T, 3, uint32_t> xmap,
                        uint2_t shape, int32_t batches,
                        ellipse_t ellipse) {
        const uint2_t gid{blockIdx.y * BLOCK_SIZE_2D.y + threadIdx.y,
                          blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        if (gid[0] >= shape[0] || gid[1] >= shape[1])
            return;

        const float2_t coords{IS_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[0]),
                              IS_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[1])};

        const auto mask = ellipse(coords);
        for (int32_t batch = 0; batch < batches; ++batch)
            xmap(batch, gid[0], gid[1]) *= mask;
    }

    template<bool IS_CENTERED, typename T, typename ellipse_t>
    __global__ __launch_bounds__(BLOCK_SIZE_2D.x * BLOCK_SIZE_2D.y)
    void maskEllipse3D_(Accessor<T, 4, uint32_t> xmap,
                        uint3_t shape, int32_t batches,
                        ellipse_t ellipse) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * BLOCK_SIZE_2D.y + threadIdx.y,
                          blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        if (gid[1] >= shape[1] || gid[2] >= shape[2])
            return;

        const float3_t coords{IS_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[0]),
                              IS_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[1]),
                              IS_CENTERED ? gid[2] : math::FFTShift(gid[2], shape[2])};

        const auto mask = ellipse(coords);
        for (int32_t batch = 0; batch < batches; ++batch)
            xmap(batch, gid[0], gid[1], gid[2]) *= mask;
    }

    template<bool IS_CENTERED, typename T>
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
        stream.enqueue("signal::xpeak1D_mask", maskEllipse1D_<IS_CENTERED, T, line_t>, config,
                       accessor, shape_1d[1], line);
    }

    template<bool IS_CENTERED, typename T>
    void enforceMaxRadiusInPlace2D_(T* xmap, float2_t max_radius,
                                    uint4_t shape, uint2_t shape_2d, uint4_t strides,
                                    cuda::Stream& stream) {
        const float2_t center(shape_2d / 2);
        const float edge_size = static_cast<float>(noa::math::max(shape_2d)) * 0.05f;

        const uint3_t strides_2d{strides[0], strides[2], strides[3]};
        const Accessor<T, 3, uint32_t> accessor(xmap, strides_2d);

        using real_t = traits::value_type_t<T>;
        using ellipse_t = noa::signal::EllipseSmooth<2, real_t>;
        const ellipse_t ellipse(center, max_radius, edge_size);

        const dim3 blocks(noa::math::divideUp(shape[3], BLOCK_SIZE_2D.x),
                          noa::math::divideUp(shape[2], BLOCK_SIZE_2D.y),
                          shape[1]);
        const cuda::LaunchConfig config{blocks, BLOCK_SIZE_2D};
        stream.enqueue("signal::xpeak2D_mask", maskEllipse2D_<IS_CENTERED, T, ellipse_t>, config,
                       accessor, shape_2d, shape[0], ellipse);
    }

    template<bool IS_CENTERED, typename T>
    void enforceMaxRadiusInPlace3D_(T* xmap, float3_t max_radius,
                                    uint4_t shape, uint3_t shape_3d, uint4_t strides,
                                    cuda::Stream& stream) {
        const float3_t center(shape_3d / 2);
        const float edge_size = static_cast<float>(noa::math::max(shape_3d)) * 0.05f;

        const Accessor<T, 4, uint32_t> accessor(xmap, strides);
        using real_t = traits::value_type_t<T>;
        using ellipse_t = noa::signal::EllipseSmooth<3, real_t>;
        const ellipse_t ellipse(center, max_radius, edge_size);

        const dim3 blocks(noa::math::divideUp(shape[3], BLOCK_SIZE_2D.x),
                          noa::math::divideUp(shape[2], BLOCK_SIZE_2D.y),
                          shape[1]);
        const cuda::LaunchConfig config{blocks, BLOCK_SIZE_2D};
        stream.enqueue("signal::xpeak3D_mask", maskEllipse3D_<IS_CENTERED, T, ellipse_t>, config,
                       accessor, shape_3d, shape[0], ellipse);
    }
}

namespace {
    using namespace ::noa;
    constexpr uint32_t BLOCK_SIZE = cuda::Limits::WARP_SIZE;

    // From the DC-centered frequency to a valid index in the non-centered output.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
    constexpr NOA_FD int64_t getIndex_(int64_t frequency, int64_t volume_dim) {
        return frequency < 0 ? volume_dim + frequency : frequency;
    }

    // From a valid index to the DC-centered frequency.
    constexpr NOA_FD int3_t getFrequency_(int3_t index, int3_t shape) {
        return {index[0] < (shape[0] + 1) / 2 ? index[0] : index[0] - shape[0],
                index[1] < (shape[1] + 1) / 2 ? index[1] : index[1] - shape[1],
                index[2] < (shape[2] + 1) / 2 ? index[2] : index[2] - shape[2]};
    }

    // From a valid index to the DC-centered frequency.
    constexpr NOA_FD int2_t getFrequency_(int2_t index, int2_t shape) {
        return {index[0] < (shape[0] + 1) / 2 ? index[0] : index[0] - shape[0],
                index[1] < (shape[1] + 1) / 2 ? index[1] : index[1] - shape[1]};
    }

    constexpr NOA_FD int32_t getFrequency_(int32_t index, int32_t shape) {
        return index < (shape + 1) / 2 ? index : index - shape;
    }

    // Given values at three successive positions, y[0], y[1], y[2], where
    // y[1] is the peak value, this fits a parabola to the values and returns the
    // offset (from -0.5 to 0.5) from the center position.
    template<typename T>
    constexpr NOA_FD T getParabolicVertex_(T y0, T y1, T y2) noexcept {
        const T d = 2 * (y0 + y2 - 2 * y1);
        T x = 0;
        // From IMOD/libcfshr/filtxcorr.c::parabolicFitPosition
        if (math::abs(d) > math::abs(static_cast<T>(1e-2) * (y0 - y2)))
            x = (y0 - y2) / d;
        if (x > T{0.5})
            x = T{0.5};
        if (x < T{-0.5})
            x = T{-0.5};
        return x;
    }

    template<bool IS_CENTERED, typename T>
    constexpr NOA_FD T fetchPeack1D_(const T* input, uint32_t stride, int32_t shape, int32_t peak,
                                     int32_t tidx, int32_t offset) {
        T value = 0;
        if (tidx < 3) {
            if constexpr (!IS_CENTERED) {
                const int32_t tid = getFrequency_(peak, shape) + offset;
                if (-shape / 2 <= tid && tid <= (shape - 1) / 2) {
                    value = input[getIndex_(tid, shape) * stride];
                }
            } else {
                const int32_t tid = peak + offset;
                if (0 <= tid && tid < shape)
                    value = input[tid * stride];
            }
        }
        return value;
    }

    // Fetch the 3x3 window around the peak
    // No coalescing here I'm afraid.
    template<bool IS_CENTERED, typename T>
    constexpr NOA_FD T fetchPeack2D_(const T* input, uint2_t strides, int2_t shape, int2_t peak,
                                     int32_t tidx, int2_t offset) {
        T value = 0;
        if (tidx < 9) {
            if constexpr (!IS_CENTERED) {
                const int2_t tid = getFrequency_(peak, shape) + offset;
                if (all(-shape / 2 <= tid && tid <= (shape - 1) / 2)) {
                    value = input[indexing::at(getIndex_(tid[0], shape[0]),
                                               getIndex_(tid[1], shape[1]),
                                               strides)];
                }
            } else {
                const int2_t tid = peak + offset;
                if (all(0 <= tid && tid < shape))
                    value = input[indexing::at(tid, strides)];
            }
        }
        return value;
    }

    // Fetch the 3x3x3 window around the peak
    // No coalescing here I'm afraid.
    template<bool IS_CENTERED, typename T>
    constexpr NOA_FD T fetchPeack3D_(const T* input, uint3_t strides, int3_t shape, int3_t peak,
                                     int32_t tidx, int3_t offset) {
        T value = 0;
        if (tidx < 27) {
            if constexpr (!IS_CENTERED) {
                const int3_t tid = getFrequency_(peak, shape) + offset;
                if (all(-shape / 2 <= tid && tid <= (shape - 1) / 2)) {
                    value = input[indexing::at(getIndex_(tid[0], shape[0]),
                                               getIndex_(tid[1], shape[1]),
                                               getIndex_(tid[2], shape[2]),
                                               strides)];
                }
            } else {
                const int3_t tid = peak + offset;
                if (all(0 <= tid && tid < shape))
                    value = input[indexing::at(tid, strides)];
            }
        }
        return value;
    }


    template<bool IS_CENTERED, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak1D_(const T* __restrict__ input, uint32_t stride, int32_t shape,
                       int32_t peak, float* __restrict__ coordinates) {
        using namespace cuda::util;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int32_t offset = tidx - 1;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack1D_<IS_CENTERED>(input, stride, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float refined_peak = IS_CENTERED ? peak : math::FFTShift(peak, shape);
            refined_peak += getParabolicVertex_(square[0], square[1], square[2]);
            *coordinates = refined_peak;
        }
    }

    template<bool IS_CENTERED, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak1DBatched_(const T* __restrict__ input, uint32_t batch_stride,
                              uint32_t stride, int32_t shape,
                              const uint32_t* __restrict__ peaks,
                              float* __restrict__ coordinates) {
        using namespace cuda::util;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int32_t offset = tidx - 1;

        const auto peak = static_cast<int>(peaks[batch] / stride);
        input += batch_stride * batch;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack1D_<IS_CENTERED>(input, stride, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float refined_peak = IS_CENTERED ? peak : math::FFTShift(peak, shape);
            refined_peak += getParabolicVertex_(square[0], square[1], square[2]);
            coordinates[batch] = refined_peak;
        }
    }

    template<bool IS_CENTERED, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak2D_(const T* __restrict__ input, uint2_t strides, int2_t shape,
                       int2_t peak, float2_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int2_t offset = indexing::indexes(tidx, 3) - 1;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack2D_<IS_CENTERED>(input, strides, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float refined_peak0 = IS_CENTERED ? peak[0] : math::FFTShift(peak[0], shape[0]);
            float refined_peak1 = IS_CENTERED ? peak[1] : math::FFTShift(peak[1], shape[1]);
            const T peak_value = square[4];
            refined_peak0 += getParabolicVertex_(square[1], peak_value, square[7]);
            refined_peak1 += getParabolicVertex_(square[3], peak_value, square[5]);
            *coordinates = float2_t{refined_peak0, refined_peak1};
        }
    }

    template<bool IS_CENTERED, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak2DBatched_(const T* __restrict__ input, uint32_t batch_stride,
                              uint2_t strides, int2_t shape,
                              const uint32_t* __restrict__ peak_offsets,
                              float2_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int2_t offset = indexing::indexes(tidx, 3) - 1;

        const uint32_t peak_offset = peak_offsets[batch];
        const int2_t peak(indexing::indexes(peak_offset, strides, uint2_t(shape)));
        input += batch_stride * batch;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack2D_<IS_CENTERED>(input, strides, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float refined_peak0 = IS_CENTERED ? peak[0] : math::FFTShift(peak[0], shape[0]);
            float refined_peak1 = IS_CENTERED ? peak[1] : math::FFTShift(peak[1], shape[1]);
            const T peak_value = square[4];
            refined_peak0 += getParabolicVertex_(square[1], peak_value, square[7]);
            refined_peak1 += getParabolicVertex_(square[3], peak_value, square[5]);
            coordinates[batch] = float2_t{refined_peak0, refined_peak1};
        }
    }

    template<bool IS_CENTERED, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak3D_(const T* __restrict__ input, uint3_t strides, int3_t shape,
                       int3_t peak, float3_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int3_t offset = indexing::indexes(tidx, 3, 3) - 1;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack3D_<IS_CENTERED>(input, strides, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float refined_peak0 = IS_CENTERED ? peak[0] : math::FFTShift(peak[0], shape[0]);
            float refined_peak1 = IS_CENTERED ? peak[1] : math::FFTShift(peak[1], shape[1]);
            float refined_peak2 = IS_CENTERED ? peak[2] : math::FFTShift(peak[2], shape[2]);
            const T peak_value = square[13];
            refined_peak0 += getParabolicVertex_(square[4], peak_value, square[22]);
            refined_peak1 += getParabolicVertex_(square[10], peak_value, square[16]);
            refined_peak2 += getParabolicVertex_(square[12], peak_value, square[14]);
            *coordinates = float3_t{refined_peak0, refined_peak1, refined_peak2};
        }
    }

    template<bool IS_CENTERED, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak3DBatched_(const T* __restrict__ input, uint32_t batch_stride,
                              uint3_t strides, int3_t shape,
                              const uint32_t* __restrict__ peak_offsets,
                              float3_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const uint32_t batch = blockIdx.x;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int3_t offset = indexing::indexes(tidx, 3, 3) - 1;

        const uint32_t peak_offset = peak_offsets[batch];
        const int3_t peak(indexing::indexes(peak_offset, strides, uint3_t(shape)));
        input += batch_stride * batch;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack3D_<IS_CENTERED>(input, strides, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float refined_peak0 = IS_CENTERED ? peak[0] : math::FFTShift(peak[0], shape[0]);
            float refined_peak1 = IS_CENTERED ? peak[1] : math::FFTShift(peak[1], shape[1]);
            float refined_peak2 = IS_CENTERED ? peak[2] : math::FFTShift(peak[2], shape[2]);
            const T peak_value = square[13];
            refined_peak0 += getParabolicVertex_(square[4], peak_value, square[22]);
            refined_peak1 += getParabolicVertex_(square[10], peak_value, square[16]);
            refined_peak2 += getParabolicVertex_(square[12], peak_value, square[14]);
            coordinates[batch] = float3_t{refined_peak0, refined_peak1, refined_peak2};
        }
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename T, typename U>
    void xmap(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_strides,
              const shared_t<Complex<T>[]>& rhs, dim4_t rhs_strides,
              const shared_t<T[]>& output, dim4_t output_strides,
              dim4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp, dim4_t tmp_strides) {

        const shared_t<Complex<T>[]>& buffer = tmp ? tmp : rhs;
        const dim4_t& buffer_strides = tmp ? tmp_strides : rhs_strides;
        NOA_ASSERT(all(buffer_strides > 0));

        if (normalize) {
            cuda::util::ewise::binary(
                    "signal::fft::xmap",
                    lhs.get(), lhs_strides, rhs.get(), rhs_strides,
                    buffer.get(), buffer_strides,
                    shape.fft(), true, stream,
                    []__device__(Complex<T> l, Complex<T> r) {
                        const Complex<T> product = l * noa::math::conj(r);
                        const T magnitude = noa::math::abs(product);
                        return product / (magnitude + static_cast<T>(1e-13));
                        // The epsilon could be scaled by the max(abs(rhs)), but this seems to be useful only
                        // for input values close to zero (less than 1e-10). In most cases, this is fine.
                        // Note that the normalization can sharpen the peak considerably.
                    });
        } else {
            cuda::math::ewise(lhs, lhs_strides, rhs, rhs_strides, buffer, buffer_strides,
                              shape.fft(), noa::math::multiply_conj_t{}, stream);
        }

        if constexpr (REMAP == Remap::H2FC) {
            const dim3_t shape_3d(shape.get(1));
            if (shape_3d.ndim() == 3) {
                cuda::signal::fft::shift3D<Remap::H2H>(buffer, buffer_strides, buffer, buffer_strides, shape,
                                                       float3_t(shape_3d / 2), 1, stream);
            } else {
                cuda::signal::fft::shift2D<Remap::H2H>(buffer, buffer_strides, buffer, buffer_strides, shape,
                                                       float2_t{shape_3d[1] / 2, shape_3d[2] / 2}, 1, stream);
            }
        }

        cuda::fft::c2r(buffer, buffer_strides, output, output_strides, shape, norm, stream);
    }

    template<Remap REMAP, typename T, typename>
    void xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                 const shared_t<float[]>& coordinates, float max_radius, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(strides[3 - is_column] > 0);
        NOA_ASSERT(all(shape > 0) && dim3_t(shape.get(1)).ndim() == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_1d{u_shape[0], u_shape[3 - is_column]};
        const uint2_t u_strides_1d{u_strides[0], u_strides[3 - is_column]};

        // In-place mask on the phase-correlation map.
        if (max_radius > 0)
            enforceMaxRadiusInPlace1D_<IS_CENTERED>(xmap.get(), max_radius, u_shape_1d, u_strides_1d, stream);

        // Find the maximum value.
        cuda::memory::PtrDevice<uint32_t> offsets(shape[0], stream);
        cuda::math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

        // Compute the final subpixel position and center if not already.
        float* coordinates_ptr = util::devicePointer(coordinates.get(), stream.device());
        using unique_ptr_t = memory::PtrDevice<float>::alloc_unique_t;
        unique_ptr_t buffer{};
        if (!coordinates_ptr) {
            buffer = memory::PtrDevice<float>::alloc(shape[0], stream);
            coordinates_ptr = buffer.get();
        }

        NOA_ASSERT(strides[3 - is_column] > 0);
        stream.enqueue("signal::fft::xpeak1D", singlePeak1DBatched_<IS_CENTERED, T>, LaunchConfig{shape[0], BLOCK_SIZE},
                       xmap.get(), u_strides[0], u_strides_1d[1], u_shape_1d[1], offsets.get(), coordinates_ptr);

        if (buffer)
            memory::copy(coordinates_ptr, coordinates.get(), shape[0], stream);
        stream.attach(xmap, coordinates);
    }

    template<Remap REMAP, typename T, typename>
    float xpeak1D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float max_radius, Stream& stream) {
        const bool is_column = shape[3] == 1;
        NOA_ASSERT(strides[3 - is_column] > 0);
        NOA_ASSERT(all(shape > 0) && shape.ndim() == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_1d{u_shape[0], u_shape[3 - is_column]};
        const uint2_t u_strides_1d{u_strides[0], u_strides[3 - is_column]};

        // In-place mask on the phase-correlation map.
        if (max_radius > 0)
            enforceMaxRadiusInPlace1D_<IS_CENTERED>(xmap.get(), max_radius, u_shape_1d, u_strides_1d, stream);

        const auto peak_offset = cuda::math::find<uint32_t>(noa::math::first_max_t{}, xmap, strides, shape, true, stream);
        const uint32_t peak_index = peak_offset / u_strides_1d[1];

        cuda::memory::PtrPinned<float> coordinate(1);
        stream.enqueue("signal::fft::xpeak1D", singlePeak1D_<IS_CENTERED, T>, LaunchConfig{1, BLOCK_SIZE},
                       xmap.get(), u_strides_1d[1], u_shape_1d[1], peak_index, coordinate.get());
        stream.synchronize();
        return coordinate[0];
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                 const shared_t<float2_t[]>& coordinates, float2_t max_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && shape[1] == 1);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_2d(u_shape.get(2));
        const uint2_t u_strides_2d(u_strides.get(2));

        // In-place mask on the phase-correlation map.
        if (any(max_radius > 0))
            enforceMaxRadiusInPlace2D_<IS_CENTERED>(xmap.get(), max_radius, u_shape, u_shape_2d, u_strides, stream);

        // Find the maximum value.
        cuda::memory::PtrDevice<uint32_t> offsets(shape[0], stream);
        cuda::math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

        // Compute the final subpixel position and center if not already.
        float2_t* coordinates_ptr = util::devicePointer(coordinates.get(), stream.device());
        using unique_ptr_t = memory::PtrDevice<float2_t>::alloc_unique_t;
        unique_ptr_t buffer{};
        if (!coordinates_ptr) {
            buffer = memory::PtrDevice<float2_t>::alloc(shape[0], stream);
            coordinates_ptr = buffer.get();
        }
        stream.enqueue("signal::fft::xpeak2D", singlePeak2DBatched_<IS_CENTERED, T>, LaunchConfig{shape[0], BLOCK_SIZE},
                       xmap.get(), u_strides[0], u_strides_2d, int2_t(u_shape_2d), offsets.get(), coordinates_ptr);

        if (buffer)
            memory::copy(coordinates_ptr, coordinates.get(), shape[0], stream);
        stream.attach(xmap, coordinates);
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float2_t max_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && shape.ndim() == 2);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint2_t u_shape_2d(u_shape.get(2));
        const uint2_t u_strides_2d(u_strides.get(2));

        // In-place mask on the phase-correlation map.
        if (any(max_radius > 0))
            enforceMaxRadiusInPlace2D_<IS_CENTERED>(xmap.get(), max_radius, u_shape, u_shape_2d, u_strides, stream);

        // Find the maximum value.
        const auto peak_offset = cuda::math::find<uint32_t>(noa::math::first_max_t{}, xmap, strides, shape, true, stream);

        // Compute the final subpixel position and center if not already.
        const int2_t peak_index(indexing::indexes(peak_offset, u_strides_2d, u_shape_2d));

        cuda::memory::PtrPinned<float2_t> coordinate(1);
        stream.enqueue("signal::fft::xpeak2D", singlePeak2D_<IS_CENTERED, T>, LaunchConfig{1, BLOCK_SIZE},
                       xmap.get(), u_strides_2d, int2_t(u_shape_2d), peak_index, coordinate.get());
        stream.synchronize();
        return coordinate[0];
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape,
                 const shared_t<float3_t[]>& coordinates, float3_t max_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());
        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint3_t u_shape_3d(u_shape.get(1));
        const uint3_t u_strides_3d(u_strides.get(1));

        // In-place mask on the phase-correlation map.
        if (any(max_radius > 0))
            enforceMaxRadiusInPlace3D_<IS_CENTERED>(xmap.get(), max_radius, u_shape, u_shape_3d, u_strides, stream);

        cuda::memory::PtrPinned<uint32_t> offsets(shape[0]);
        cuda::math::find(noa::math::first_max_t{}, xmap, strides, shape, offsets.share(), true, true, stream);

        float3_t* coordinates_ptr = util::devicePointer(coordinates.get(), stream.device());
        using unique_ptr_t = memory::PtrDevice<float3_t>::alloc_unique_t;
        unique_ptr_t buffer{};
        if (!coordinates_ptr) {
            buffer = memory::PtrDevice<float3_t>::alloc(shape[0], stream);
            coordinates_ptr = buffer.get();
        }
        stream.enqueue("signal::fft::xpeak3D", singlePeak3DBatched_<IS_CENTERED, T>, LaunchConfig{shape[0], BLOCK_SIZE},
                       xmap.get(), u_strides[0], u_strides_3d, int3_t(u_shape_3d), offsets.get(), coordinates_ptr);

        if (buffer)
            memory::copy(coordinates_ptr, coordinates.get(), shape[0], stream);
        stream.attach(xmap, offsets.share(), coordinates);
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const shared_t<T[]>& xmap, dim4_t strides, dim4_t shape, float3_t max_radius, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && shape.ndim() == 3);
        NOA_ASSERT_DEVICE_PTR(xmap.get(), stream.device());

        constexpr bool IS_CENTERED = static_cast<std::underlying_type_t<Remap>>(REMAP) & noa::fft::Layout::DST_CENTERED;
        const auto u_shape = safe_cast<uint4_t>(shape);
        const auto u_strides = safe_cast<uint4_t>(strides);
        const uint3_t u_shape_3d(u_shape.get(1));
        const uint3_t u_strides_3d(u_strides.get(1));

        // In-place mask on the phase-correlation map.
        if (any(max_radius > 0))
            enforceMaxRadiusInPlace3D_<IS_CENTERED>(xmap.get(), max_radius, u_shape, u_shape_3d, u_strides, stream);

        const auto peak_offset = cuda::math::find<uint32_t>(noa::math::first_max_t{}, xmap, strides, shape, true, stream);
        const int3_t peak_index(indexing::indexes(peak_offset, u_strides_3d, u_shape_3d));

        cuda::memory::PtrPinned<float3_t> coordinate(1);
        stream.enqueue("signal::fft::xpeak3D", singlePeak3D_<IS_CENTERED, T>, LaunchConfig{1, BLOCK_SIZE},
                       xmap.get(), u_strides_3d, int3_t(u_shape_3d), peak_index, coordinate.get());
        stream.synchronize();
        return coordinate[0];
    }

    #define INSTANTIATE_XMAP(T) \
    template void xmap<Remap::H2F, T, void>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, bool, Norm, Stream&, const shared_t<Complex<T>[]>&, dim4_t);    \
    template void xmap<Remap::H2FC, T, void>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, bool, Norm, Stream&, const shared_t<Complex<T>[]>&, dim4_t);   \
    template void xpeak1D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float[]>&, float, Stream&);         \
    template void xpeak1D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float[]>&, float, Stream&);       \
    template void xpeak2D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float2_t, Stream&);   \
    template void xpeak2D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float2_t[]>&, float2_t, Stream&); \
    template void xpeak3D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float3_t, Stream&);   \
    template void xpeak3D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<float3_t[]>&, float3_t, Stream&); \
    template float xpeak1D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, Stream&);          \
    template float xpeak1D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float, Stream&);        \
    template float2_t xpeak2D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, Stream&);    \
    template float2_t xpeak2D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float2_t, Stream&);  \
    template float3_t xpeak3D<Remap::F2F, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, Stream&);    \
    template float3_t xpeak3D<Remap::FC2FC, T, void>(const shared_t<T[]>&, dim4_t, dim4_t, float3_t, Stream&)

    INSTANTIATE_XMAP(float);
    INSTANTIATE_XMAP(double);
}

namespace noa::cuda::signal::fft::details {
    template<typename T>
    void xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_stride,
               const shared_t<Complex<T>[]>& rhs, dim4_t rhs_stride,
               dim4_t shape, const shared_t<T[]>& coefficients,
               Stream& stream, bool is_half) {
        const dim_t batches = shape[0];
        const dim4_t shape_fft = is_half ? shape.fft() : shape;

        cuda::memory::PtrPinned<T> buffer(batches * 3);
        auto denominator_lhs = buffer.get() + batches;
        auto denominator_rhs = buffer.get() + batches * 2;

        T* null{};
        cuda::util::reduce(
                "signal::fft::xcorr", lhs.get(), lhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                denominator_lhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                false, true, stream);
        cuda::util::reduce(
                "signal::fft::xcorr", rhs.get(), rhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                denominator_rhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                false, true, stream);

        auto combine_op = []__device__(Complex<T> l, Complex<T> r) { return noa::math::real(l * r); };
        cuda::util::reduce<false>(
                "signal::fft::xcorr",
                lhs.get(), lhs_stride, rhs.get(), rhs_stride, shape_fft,
                noa::math::copy_t{}, noa::math::conj_t{}, combine_op, noa::math::plus_t{}, T{0},
                buffer.get(), 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, false, stream);

        stream.synchronize(); // FIXME Add callback
        for (dim_t batch = 0; batch < batches; ++batch) {
            coefficients.get()[batch] =
                    buffer[batch] / noa::math::sqrt(denominator_lhs[batch] * denominator_rhs[batch]);
        }
    }

    template<typename T>
    T xcorr(const shared_t<Complex<T>[]>& lhs, dim4_t lhs_stride,
            const shared_t<Complex<T>[]>& rhs, dim4_t rhs_stride,
            dim4_t shape, Stream& stream, bool is_half) {
        NOA_ASSERT(shape[0] == 1);
        const dim4_t shape_fft = is_half ? shape.fft() : shape;

        T numerator{}, denominator_lhs{}, denominator_rhs{};
        T* null{};
        cuda::util::reduce(
                "signal::fft::xcorr", lhs.get(), lhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                &denominator_lhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                true, true, stream);
        cuda::util::reduce(
                "signal::fft::xcorr", rhs.get(), rhs_stride, shape_fft,
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                &denominator_rhs, 1, noa::math::copy_t{},
                null, 0, noa::math::copy_t{},
                true, true, stream);

        auto combine_op = []__device__(Complex<T> l, Complex<T> r) { return noa::math::real(l * r); };
        cuda::util::reduce<false>(
                "signal::fft::xcorr",
                lhs.get(), lhs_stride, rhs.get(), rhs_stride, shape_fft,
                noa::math::copy_t{}, noa::math::conj_t{}, combine_op, noa::math::plus_t{}, T{0},
                &numerator, 1, noa::math::copy_t{}, null, 1, noa::math::copy_t{}, false, stream);

        stream.synchronize();
        const T denominator = noa::math::sqrt(denominator_lhs * denominator_rhs);
        return numerator / denominator;
    }

    #define INSTANTIATE_XCORR(T) \
    template void xcorr<T>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, dim4_t, const shared_t<T[]>&, Stream&, bool); \
    template T xcorr<T>(const shared_t<Complex<T>[]>&, dim4_t, const shared_t<Complex<T>[]>&, dim4_t, dim4_t, Stream&, bool)

    INSTANTIATE_XCORR(float);
    INSTANTIATE_XCORR(double);
}
