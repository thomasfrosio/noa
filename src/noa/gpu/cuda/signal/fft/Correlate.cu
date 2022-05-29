#include "noa/gpu/cuda/fft/Transforms.h"
#include "noa/gpu/cuda/math/Ewise.h"
#include "noa/gpu/cuda/math/Find.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/signal/fft/Correlate.h"
#include "noa/gpu/cuda/signal/fft/Shift.h"

#include "noa/gpu/cuda/util/EwiseBinary.cuh"
#include "noa/gpu/cuda/util/Reduce.cuh"
#include "noa/gpu/cuda/util/Warp.cuh"

namespace {
    using namespace ::noa;
    constexpr uint BLOCK_SIZE = cuda::Limits::WARP_SIZE;

    // From the DC-centered frequency to a valid index in the non-centered output.
    // The input frequency should be in-bound, i.e. -n/2 <= frequency <= (n-1)/2
    constexpr NOA_FD int64_t getIndex_(int64_t frequency, int64_t volume_dim) {
        return frequency + volume_dim / 2;
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

    // Fetch the 3x3 window around the peak
    // No coalescing here I'm afraid.
    template<fft::Remap REMAP, typename T>
    constexpr NOA_FD T fetchPeack2D_(const T* input, uint2_t stride, int2_t shape, int2_t peak,
                                     int tidx, int2_t offset) {
        T value = 0;
        if (tidx < 9) {
            if constexpr (REMAP == fft::F2F) {
                const int2_t tid = getFrequency_(peak, shape) - offset;
                if (all(-shape / 2 <= tid && tid <= (shape - 1) / 2)) {
                    value = input[indexing::at(getIndex_(tid[0], shape[0]),
                                               getIndex_(tid[1], shape[1]),
                                               stride)];
                }
            } else {
                const int2_t tid = peak - offset;
                if (all(0 <= tid && tid < shape))
                    value = input[indexing::at(tid, stride)];
            }
        }
        return value;
    }

    // Fetch the 3x3x3 window around the peak
    // No coalescing here I'm afraid.
    template<fft::Remap REMAP, typename T>
    constexpr NOA_FD T fetchPeack3D_(const T* input, uint3_t stride, int3_t shape, int3_t peak,
                                     int tidx, int3_t offset) {
        T value = 0;
        if (tidx < 27) {
            if constexpr (REMAP == fft::F2F) {
                const int3_t tid = getFrequency_(peak, shape) - offset;
                if (all(-shape / 2 <= tid && tid <= (shape - 1) / 2)) {
                    value = input[indexing::at(getIndex_(tid[0], shape[0]),
                                               getIndex_(tid[1], shape[1]),
                                               getIndex_(tid[2], shape[2]),
                                               stride)];
                }
            } else {
                const int3_t tid = peak - offset;
                if (all(0 <= tid && tid < shape))
                    value = input[indexing::at(tid, stride)];
            }
        }
        return value;
    }

    template<fft::Remap REMAP, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak2D_(const T* __restrict__ input, uint2_t stride, int2_t shape,
                       int2_t peak, float2_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int2_t offset = indexing::indexes(tidx, 3) - 1;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack2D_<REMAP>(input, stride, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float2_t output{peak};
            const T peak_value = square[4];
            output[0] += getParabolicVertex_(square[1], peak_value, square[7]);
            output[1] += getParabolicVertex_(square[3], peak_value, square[5]);
            *coordinates = output;
        }
    }

    template<fft::Remap REMAP, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak2DBatched_(const T* __restrict__ input, uint batch_stride,
                              uint2_t stride, int2_t shape, uint pitch_x,
                              const uint32_t* __restrict__ peak_offsets,
                              float2_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const uint batch = blockIdx.x;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int2_t offset = indexing::indexes(tidx, 3) - 1;

        const uint32_t peak_offset = peak_offsets[batch];
        const int2_t peak{indexing::indexes(peak_offset, pitch_x)};
        input += batch_stride * batch;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack2D_<REMAP>(input, stride, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float2_t output{peak};
            const T peak_value = square[4];
            output[0] += getParabolicVertex_(square[1], peak_value, square[7]);
            output[1] += getParabolicVertex_(square[3], peak_value, square[5]);
            coordinates[batch] = output;
        }
    }

    template<fft::Remap REMAP, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak3D_(const T* __restrict__ input, uint3_t stride, int3_t shape,
                       int3_t peak, float3_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int3_t offset = indexing::indexes(tidx, 3, 3) - 1;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack3D_<REMAP>(input, stride, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float3_t output{peak};
            const T peak_value = square[13];
            output[0] += getParabolicVertex_(square[4], peak_value, square[22]);
            output[1] += getParabolicVertex_(square[10], peak_value, square[16]);
            output[1] += getParabolicVertex_(square[12], peak_value, square[14]);
            *coordinates = output;
        }
    }

    template<fft::Remap REMAP, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void singlePeak3DBatched_(const T* __restrict__ input, uint batch_stride,
                              uint3_t stride, int3_t shape, uint pitch_y, uint pitch_x,
                              const uint32_t* __restrict__ peak_offsets,
                              float3_t* __restrict__ coordinates) {
        using namespace cuda::util;
        const uint batch = blockIdx.x;
        const auto tidx = static_cast<int>(threadIdx.x);
        const int3_t offset = indexing::indexes(tidx, 3, 3) - 1;

        const uint32_t peak_offset = peak_offsets[batch];
        const int3_t peak{indexing::indexes(peak_offset, pitch_y, pitch_x)};
        input += batch_stride * batch;

        __shared__ T square[BLOCK_SIZE];
        square[tidx] = fetchPeack3D_<REMAP>(input, stride, shape, peak, tidx, offset);
        block::synchronize();

        if (tidx == 0) {
            float3_t output{peak};
            const T peak_value = square[13];
            output[0] += getParabolicVertex_(square[4], peak_value, square[22]);
            output[1] += getParabolicVertex_(square[10], peak_value, square[16]);
            output[1] += getParabolicVertex_(square[12], peak_value, square[14]);
            coordinates[batch] = output;
        }
    }
}

namespace noa::cuda::signal::fft {
    template<Remap REMAP, typename T, typename U>
    void xmap(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
              const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
              const shared_t<T[]>& output, size4_t output_stride,
              size4_t shape, bool normalize, Norm norm, Stream& stream,
              const shared_t<Complex<T>[]>& tmp, size4_t tmp_stride) {

        const shared_t<Complex<T>[]>& buffer = tmp ? tmp : rhs;
        const size4_t& buffer_stride = tmp ? tmp_stride : rhs_stride;
        NOA_ASSERT(all(buffer_stride > 0));

        if (normalize) {
            cuda::util::ewise::binary(
                    "signal::fft::xmap",
                    lhs.get(), lhs_stride, rhs.get(), rhs_stride, buffer.get(), buffer_stride, shape.fft(), stream,
                    []__device__(Complex<T> l, Complex<T> r) {
                        const Complex<T> product = l * noa::math::conj(r);
                        const T magnitude = noa::math::abs(product);
                        return magnitude < static_cast<T>(1e-7) ? 0 : product / magnitude;
                    });
        } else {
            cuda::math::ewise(lhs, lhs_stride, rhs, rhs_stride, buffer, buffer_stride,
                              shape.fft(), noa::math::multiply_conj_t{}, stream);
        }

        if constexpr (REMAP == Remap::H2FC) {
            const size3_t shape_3d{shape.get() + 1};
            if (shape_3d.ndim() == 3) {
                cuda::signal::fft::shift3D<Remap::H2H>(buffer, buffer_stride, buffer, buffer_stride, shape,
                                                       float3_t{shape_3d / 2}, 1, stream);
            } else {
                cuda::signal::fft::shift2D<Remap::H2H>(buffer, buffer_stride, buffer, buffer_stride, shape,
                                                       float2_t{shape_3d[1] / 2, shape_3d[2] / 2}, 1, stream);
            }
        }

        cuda::fft::c2r(buffer, buffer_stride, output, output_stride, shape, norm, stream);
    }

    template<Remap REMAP, typename T, typename>
    void xpeak2D(const shared_t<T[]>& map, size4_t stride, size4_t shape,
                 const shared_t<float2_t[]>& coordinates, Stream& stream) {
        NOA_ASSERT(shape[1] == 0);
        NOA_ASSERT(stride[2] > 0);
        cuda::memory::PtrDevice<uint32_t> offsets{shape[0], stream};
        cuda::math::find(noa::math::max_t{}, map, stride, shape, offsets.share(), false, stream);

        float2_t* coordinates_ptr = util::devicePointer(coordinates.get(), stream.device());
        memory::PtrDevice<float2_t> buffer;
        if (!coordinates_ptr) {
            buffer = memory::PtrDevice<float2_t>{shape[0], stream};
            coordinates_ptr = buffer.get();
        }
        stream.enqueue("signal::fft::xpeak2D", singlePeak2DBatched_<REMAP, T>, {BLOCK_SIZE, shape[0]},
                       map.get(), stride[0], uint2_t{stride.get() + 2}, int2_t{shape.get() + 2},
                       stride[2], offsets.get(), coordinates_ptr);

        if (!buffer.empty())
            memory::copy(coordinates_ptr, coordinates.get(), shape[0], stream);
        stream.attach(map, coordinates);
    }

    template<Remap REMAP, typename T, typename>
    float2_t xpeak2D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(shape.ndim() == 2);
        NOA_ASSERT(stride[2] > 0);
        const auto peak_offset = cuda::math::find<uint32_t>(noa::math::max_t{}, xmap, stride, shape, false, stream);
        const int2_t peak_index = indexing::indexes(peak_offset, static_cast<uint32_t>(stride[2]));

        cuda::memory::PtrPinned<float2_t> coordinate{1};
        stream.enqueue("signal::fft::xpeak2D", singlePeak2D_<REMAP>, {BLOCK_SIZE, 1},
                       xmap.get(), uint2_t{stride.get() + 2}, uint2_t{shape.get() + 2},
                       peak_index, coordinate.get());
        stream.synchronize();
        return coordinate[0];
    }

    template<Remap REMAP, typename T, typename>
    void xpeak3D(const shared_t<T[]>& map, size4_t stride, size4_t shape,
                 const shared_t<float3_t[]>& coordinates, Stream& stream) {
        NOA_ASSERT(stride[1] > 0 && stride[2] > 0);
        cuda::memory::PtrPinned<uint32_t> offsets{shape[0]};
        cuda::math::find(noa::math::max_t{}, map, stride, shape, offsets.share(), false, stream);

        float3_t* coordinates_ptr = util::devicePointer(coordinates.get(), stream.device());
        memory::PtrDevice<float3_t> buffer;
        if (!coordinates_ptr) {
            buffer = memory::PtrDevice<float3_t>{shape[0], stream};
            coordinates_ptr = buffer.get();
        }
        stream.enqueue("signal::fft::xpeak3D", singlePeak3DBatched_<REMAP, T>, {BLOCK_SIZE, shape[0]},
                       map.get(), stride[0], uint3_t{stride.get() + 1}, int3_t{shape.get() + 1},
                       stride[1] / stride[2], stride[2], offsets.get(), coordinates_ptr);

        if (!buffer.empty())
            memory::copy(coordinates_ptr, coordinates.get(), shape[0], stream);
        stream.attach(map, offsets.share(), coordinates);
    }

    template<Remap REMAP, typename T, typename>
    float3_t xpeak3D(const shared_t<T[]>& xmap, size4_t stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(shape.ndim() == 3);
        NOA_ASSERT(stride[1] > 0 && stride[2] > 0);
        const auto peak_offset = cuda::math::find<uint32_t>(noa::math::max_t{}, xmap, stride, shape, false, stream);
        const int3_t peak_index = indexing::indexes(peak_offset,
                                                    static_cast<uint32_t>(stride[1] / stride[2]),
                                                    static_cast<uint32_t>(stride[2]));

        cuda::memory::PtrPinned<float3_t> coordinate{1};
        stream.enqueue("signal::fft::xpeak3D", singlePeak3D_<REMAP>, {BLOCK_SIZE, 1},
                       xmap.get(), uint3_t{stride.get() + 1}, uint3_t{shape.get() + 1},
                       peak_index, coordinate.get());
        stream.synchronize();
        return coordinate[0];
    }

    #define INSTANTIATE_XMAP(T) \
    template void xmap<Remap::H2F, T>(const shared_t<Complex<T>[]>&, size4_t, const shared_t<Complex<T>[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, bool, Norm, Stream&, const shared_t<Complex<T>[]>&, size4_t);   \
    template void xmap<Remap::H2FC, T>(const shared_t<Complex<T>[]>&, size4_t, const shared_t<Complex<T>[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, bool, Norm, Stream&, const shared_t<Complex<T>[]>&, size4_t);  \
    template void xpeak2D<Remap::F2F, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, Stream&);     \
    template void xpeak2D<Remap::FC2FC, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<float2_t[]>&, Stream&);   \
    template void xpeak3D<Remap::F2F, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, Stream&);     \
    template void xpeak3D<Remap::FC2FC, T>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<float3_t[]>&, Stream&)

    INSTANTIATE_XMAP(float);
    INSTANTIATE_XMAP(double);
}

namespace noa::cuda::signal::fft::details {
    template<typename T>
    void xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
               const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
               size4_t shape, const shared_t<T[]>& coefficients,
               Stream& stream, const shared_t<T[]>& tmp, bool is_half) {
        const size_t batches = shape[0];
        const size4_t shape_fft = is_half ? shape.fft() : shape;
        const size4_t stride_fft = shape_fft.stride();
        const size4_t reduced_shape{batches, 1, 1, 1};
        const size4_t reduced_stride = reduced_shape.stride();

        cuda::memory::PtrPinned<T> buffer{batches * 3};
        auto denominator_lhs = buffer.get() + batches;
        auto denominator_rhs = buffer.get() + batches * 2;

        cuda::util::reduce<false, Complex<T>, T>(
                "signal::fft::xcorr", lhs.get(), uint4_t{lhs_stride}, uint4_t{shape_fft},
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                denominator_lhs, 1, noa::math::copy_t{},
                nullptr, 0, noa::math::copy_t{},
                stream);
        cuda::util::reduce<false, Complex<T>, T>(
                "signal::fft::xcorr", rhs.get(), uint4_t{rhs_stride}, uint4_t{shape_fft},
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                denominator_rhs, 1, noa::math::copy_t{},
                nullptr, 0, noa::math::copy_t{},
                stream);

        const shared_t<T[]>& tmp_ = tmp ? cuda::memory::PtrDevice<T>::alloc(shape_fft.elements(), stream) : tmp;
        cuda::util::ewise::binary(
                "signal::fft::xcorr",
                lhs.get(), lhs_stride, rhs.get(), rhs_stride, tmp_.get(), stride_fft, shape_fft, stream,
                []__device__(Complex<T> l, Complex<T> r) { return noa::math::real(l * noa::math::conj(r)); });
        cuda::math::sum(tmp_, stride_fft, shape_fft, buffer.share(), reduced_stride, reduced_shape, stream);

        stream.synchronize();
        for (size_t batch = 0; batch < batches; ++batch) {
            coefficients.get()[batch] =
                    buffer[batch] / noa::math::sqrt(denominator_lhs[batch] * denominator_rhs[batch]);
        }
    }

    template<typename T>
    T xcorr(const shared_t<Complex<T>[]>& lhs, size4_t lhs_stride,
            const shared_t<Complex<T>[]>& rhs, size4_t rhs_stride,
            size4_t shape, Stream& stream, const shared_t<T[]>& tmp, bool is_half) {
        NOA_ASSERT(shape[0] == 1);
        const size4_t shape_fft = is_half ? shape.fft() : shape;
        const size4_t stride_fft = shape_fft.stride();

        T denominator_lhs, denominator_rhs;
        cuda::util::reduce<true, Complex<T>, T>(
                "signal::fft::xcorr", lhs.get(), uint4_t{lhs_stride}, uint4_t{shape_fft},
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                &denominator_lhs, 1, noa::math::copy_t{},
                nullptr, 0, noa::math::copy_t{},
                stream);
        cuda::util::reduce<true, Complex<T>, T>(
                "signal::fft::xcorr", rhs.get(), uint4_t{rhs_stride}, uint4_t{shape_fft},
                noa::math::abs_squared_t{}, noa::math::plus_t{}, T{0},
                &denominator_rhs, 1, noa::math::copy_t{},
                nullptr, 0, noa::math::copy_t{},
                stream);
        const T denominator = noa::math::sqrt(denominator_lhs * denominator_rhs);

        const shared_t<T[]>& tmp_ = tmp ? cuda::memory::PtrDevice<T>::alloc(shape_fft.elements(), stream) : tmp;
        cuda::util::ewise::binary<true>(
                "signal::fft::xcorr",
                lhs.get(), lhs_stride, rhs.get(), rhs_stride, tmp_.get(), stride_fft, shape_fft, stream,
                []__device__(Complex<T> l, Complex<T> r) { return noa::math::real(l * noa::math::conj(r)); });
        T numerator = cuda::math::sum(tmp_, stride_fft, shape.fft(), stream);
        return numerator / denominator;
    }

    #define INSTANTIATE_XCORR(T) \
    template void xcorr<T>(const shared_t<Complex<T>[]>&, size4_t, const shared_t<Complex<T>[]>&, size4_t, size4_t, const shared_t<T[]>&, Stream&, const shared_t<T[]>&, bool); \
    template T xcorr<T>(const shared_t<Complex<T>[]>&, size4_t, const shared_t<Complex<T>[]>&, size4_t, size4_t, Stream&, const shared_t<T[]>&, bool)

    INSTANTIATE_XCORR(float);
    INSTANTIATE_XCORR(double);
}
