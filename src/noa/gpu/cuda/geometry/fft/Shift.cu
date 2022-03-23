#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/geometry/fft/Shift.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<bool IS_CENTERED>
    __forceinline__ __device__ int getFrequency_(int idx, int dim) {
        if constexpr(IS_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
        return 0;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    __forceinline__ __device__ int getOutputIndex_(int i_idx, [[maybe_unused]] int dim) {
        (void) dim;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return i_idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(i_idx, dim);
        else
            return noa::math::FFTShift(i_idx, dim);
        return 0;
    }

    template<typename C, typename T>
    __forceinline__ __device__ C getPhaseShift_(T shift, T freq) {
        using real_t = traits::value_type_t<C>;
        const float factor = -math::dot(shift, freq);
        C phase_shift;
        math::sincos(static_cast<real_t>(factor), &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift2D_(const T* input, uint3_t input_stride, T* output, uint3_t output_stride, int2_t shape,
                  const float2_t* shifts, float cutoff_sqd, float2_t f_shape) {
        const int3_t gid(blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x);
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        float2_t shift = shifts[gid[0]];
        shift *= math::Constants<float>::PI2 / float2_t{shape};

        const float2_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<false>(gid[2], shape[1])};

        T phase_shift{1, 0};
        const float2_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        output[at(gid[0], o_y, gid[2], output_stride)] =
                input ? input[at(gid, input_stride)] * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift2D_single_(const T* input, uint3_t input_stride, T* output, uint3_t output_stride, int2_t shape,
                         float2_t shift, float cutoff_sqd, float2_t f_shape) {
        const int3_t gid(blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x);
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        const float2_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<false>(gid[2], shape[1])};

        T phase_shift{1, 0};
        const float2_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        output[at(gid[0], o_y, gid[2], output_stride)] =
                input ? input[at(gid, input_stride)] * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift3D_(const T* input, uint4_t input_stride, T* output, uint4_t output_stride, int3_t shape,
                  const float3_t* shifts, float cutoff_sqd, float3_t f_shape, uint blocks_x) {
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         index[0] * THREADS.y + threadIdx.y,
                         index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[1] || gid[3] >= shape[2] / 2 + 1)
            return;

        float3_t shift = shifts[gid[0]];
        shift *= math::Constants<float>::PI2 / float3_t{shape};

        const float3_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<IS_SRC_CENTERED>(gid[2], shape[1]),
                            getFrequency_<false>(gid[3], shape[2])};

        T phase_shift{1, 0};
        const float3_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[2], shape[1]);
        output[at(gid[0], o_z, o_y, gid[3], output_stride)] =
                input ? input[at(gid, input_stride)] * phase_shift : phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift3D_single_(const T* input, uint4_t input_stride, T* output, uint4_t output_stride, int3_t shape,
                         float3_t shift, float cutoff_sqd, float3_t f_shape, uint blocks_x) {
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         index[0] * THREADS.y + threadIdx.y,
                         index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[1] || gid[3] >= shape[2] / 2 + 1)
            return;

        const float3_t freq{getFrequency_<IS_SRC_CENTERED>(gid[1], shape[0]),
                            getFrequency_<IS_SRC_CENTERED>(gid[2], shape[1]),
                            getFrequency_<false>(gid[3], shape[2])};

        T phase_shift{1, 0};
        const float3_t norm_freq = freq / f_shape;
        if (math::dot(norm_freq, norm_freq) <= cutoff_sqd)
            phase_shift = getPhaseShift_<T>(shift, freq);

        const uint o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[1], shape[0]);
        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid[2], shape[1]);
        output[at(gid[0], o_z, o_y, gid[3], output_stride)] =
                input ? input[at(gid, input_stride)] * phase_shift : phase_shift;
    }
}

namespace noa::cuda::geometry::fft {
    using Layout = noa::fft::Layout;

    template<Remap REMAP, typename T>
    void shift2D(const T* input, size4_t input_stride,
                 T* output, size4_t output_stride, size4_t shape,
                 const float2_t* shifts, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(shape[1] == 1);

        cuda::memory::PtrDevice<float2_t> buffer;
        shifts = util::ensureDeviceAccess(shifts, stream, buffer, output_stride[0]);

        const int2_t s_shape{shape.get() + 2};
        const float2_t f_shape{s_shape / 2 * 2 + int2_t{s_shape == 1}}; // if odd, n-1
        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int>(THREADS.y)),
                          shape[0]);
        const LaunchConfig config{blocks, THREADS};
        stream.enqueue("geometry::fft::shift2D", shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       input, uint3_t{input_stride[0], input_stride[2], input_stride[3]},
                       output, uint3_t{output_stride[0], output_stride[2], output_stride[3]},
                       s_shape, shifts, cutoff * cutoff, f_shape);
    }

    template<Remap REMAP, typename T>
    void shift2D(const T* input, size4_t input_stride,
                 T* output, size4_t output_stride, size4_t shape,
                 float2_t shift, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);
        NOA_ASSERT(shape[1] == 1);

        const int2_t s_shape{shape.get() + 2};
        const float2_t f_shape{s_shape / 2 * 2 + int2_t{s_shape == 1}}; // if odd, n-1
        shift *= math::Constants<float>::PI2 / float2_t{s_shape};

        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int>(THREADS.y)),
                          shape[0]);
        const LaunchConfig config{blocks, THREADS};
        stream.enqueue("geometry::fft::shift2D", shift2D_single_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       input, uint3_t{input_stride[0], input_stride[2], input_stride[3]},
                       output, uint3_t{output_stride[0], output_stride[2], output_stride[3]},
                       s_shape, shift, cutoff * cutoff, f_shape);
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* input, size4_t input_stride,
                 T* output, size4_t output_stride, size4_t shape,
                 const float3_t* shifts, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);

        cuda::memory::PtrDevice<float3_t> buffer;
        shifts = util::ensureDeviceAccess(shifts, stream, buffer, output_stride[0]);

        const int3_t s_shape{shape.get() + 1};
        const float3_t f_shape{s_shape / 2 * 2 + int3_t{s_shape == 1}}; // if odd, n-1
        const uint blocks_x = math::divideUp(s_shape[2] / 2 + 1, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape[1], static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const LaunchConfig config{blocks, THREADS};
        stream.enqueue("geometry::fft::shift3D", shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       input, uint4_t{input_stride}, output, uint4_t{output_stride}, s_shape,
                       shifts, cutoff * cutoff, f_shape, blocks_x);
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* input, size4_t input_stride,
                 T* output, size4_t output_stride, size4_t shape,
                 float3_t shift, float cutoff, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output || IS_SRC_CENTERED == IS_DST_CENTERED);

        const int3_t s_shape{shape.get() + 1};
        const float3_t f_shape{s_shape / 2 * 2 + int3_t{s_shape == 1}}; // if odd, n-1
        shift *= math::Constants<float>::PI2 / float3_t{s_shape};

        const uint blocks_x = math::divideUp(s_shape[2] / 2 + 1, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape[1], static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
        const LaunchConfig config{blocks, THREADS};
        stream.enqueue("geometry::fft::shift3D", shift3D_single_<IS_SRC_CENTERED, IS_DST_CENTERED, T>, config,
                       input, uint4_t{input_stride}, output, uint4_t{output_stride}, s_shape,
                       shift, cutoff * cutoff, f_shape, blocks_x);
    }

    #define NOA_INSTANTIATE_SHIFT(T)                                                                                \
    template void shift2D<Remap::H2H>(const T*, size4_t, T*, size4_t, size4_t, const float2_t*, float, Stream&);    \
    template void shift2D<Remap::H2H>(const T*, size4_t, T*, size4_t, size4_t, float2_t, float, Stream&);           \
    template void shift2D<Remap::H2HC>(const T*, size4_t, T*, size4_t, size4_t, const float2_t*, float, Stream&);   \
    template void shift2D<Remap::H2HC>(const T*, size4_t, T*, size4_t, size4_t, float2_t, float, Stream&);          \
    template void shift2D<Remap::HC2H>(const T*, size4_t, T*, size4_t, size4_t, const float2_t*, float, Stream&);   \
    template void shift2D<Remap::HC2H>(const T*, size4_t, T*, size4_t, size4_t, float2_t, float, Stream&);          \
    template void shift2D<Remap::HC2HC>(const T*, size4_t, T*, size4_t, size4_t, const float2_t*, float, Stream&);  \
    template void shift2D<Remap::HC2HC>(const T*, size4_t, T*, size4_t, size4_t, float2_t, float, Stream&);         \
    template void shift3D<Remap::H2H>(const T*, size4_t, T*, size4_t, size4_t, const float3_t*, float, Stream&);    \
    template void shift3D<Remap::H2H>(const T*, size4_t, T*, size4_t, size4_t, float3_t, float, Stream&);           \
    template void shift3D<Remap::H2HC>(const T*, size4_t, T*, size4_t, size4_t, const float3_t*, float, Stream&);   \
    template void shift3D<Remap::H2HC>(const T*, size4_t, T*, size4_t, size4_t, float3_t, float, Stream&);          \
    template void shift3D<Remap::HC2H>(const T*, size4_t, T*, size4_t, size4_t, const float3_t*, float, Stream&);   \
    template void shift3D<Remap::HC2H>(const T*, size4_t, T*, size4_t, size4_t, float3_t, float, Stream&);          \
    template void shift3D<Remap::HC2HC>(const T*, size4_t, T*, size4_t, size4_t, const float3_t*, float, Stream&);  \
    template void shift3D<Remap::HC2HC>(const T*, size4_t, T*, size4_t, size4_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_SHIFT(cfloat_t);
    NOA_INSTANTIATE_SHIFT(cdouble_t);
}