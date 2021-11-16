#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/transform/fft/Shift.h"

// TODO(TF) Save shifts to constant memory?

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<bool IS_DST_CENTERED>
    __forceinline__ __device__ int getFrequency_(int idx, int dim) {
        if constexpr(IS_DST_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
        return 0; // false warning: missing return statement at end of non-void function
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED>
    __forceinline__ __device__ int getOutputIndex_(int i_idx, [[maybe_unused]] int dim) {
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED)
            return i_idx;
        else if constexpr (IS_SRC_CENTERED)
            return noa::math::iFFTShift(i_idx, dim);
        else
            return noa::math::FFTShift(i_idx, dim);
        return 0; // false warning: missing return statement at end of non-void function
    }

    template<typename T>
    __forceinline__ __device__ cfloat_t getPhaseShift_(T shift, T freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift2D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch, int2_t shape,
                  const float2_t* shifts) {
        const int2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                         blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        float2_t shift = shifts[blockIdx.z];
        shift *= math::Constants<float>::PI2 / float2_t(shape);

        const int v = getFrequency_<IS_SRC_CENTERED>(gid.y, shape.y);
        const T phase_shift(getPhaseShift_(shift, float2_t(gid.x, v))); // u == gid.x

        outputs += blockIdx.z * shape.y * output_pitch;
        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.y, shape.y);
        outputs[o_y * output_pitch + gid.x] *= phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift2D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch, int2_t shape, float2_t shift) {
        const int2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                         blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        const int v = getFrequency_<IS_SRC_CENTERED>(gid.y, shape.y);
        const T phase_shift(getPhaseShift_(shift, float2_t(gid.x, v))); // u == gid.x

        outputs += blockIdx.z * shape.y * output_pitch;
        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.y, shape.y);
        outputs[o_y * output_pitch + gid.x] *= phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift3D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch, int3_t shape,
                  const float3_t* shifts, uint blocks_x) {
        const uint2_t block_idx(blockIdx.x, blocks_x);
        const int3_t gid(block_idx.x * THREADS.x + threadIdx.x,
                         block_idx.y * THREADS.y + threadIdx.y,
                         blockIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        float3_t shift = shifts[blockIdx.z];
        shift *= math::Constants<float>::PI2 / float3_t(shape);

        const int v = getFrequency_<IS_SRC_CENTERED>(gid.y, shape.y);
        const int w = getFrequency_<IS_SRC_CENTERED>(gid.z, shape.z);
        const T phase_shift(getPhaseShift_(shift, float3_t(gid.x, v, w))); // u == gid.x

        outputs += blockIdx.z * rows(shape) * output_pitch;
        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.y, shape.y);
        const uint o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.z, shape.z);
        outputs[(o_z * shape.y + o_y) * output_pitch + gid.x] *= phase_shift;
    }

    template<bool IS_SRC_CENTERED, bool IS_DST_CENTERED, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void shift3D_(const T* inputs, uint input_pitch, T* outputs, uint output_pitch, int3_t shape,
                  float3_t shift, uint blocks_x) {
        const uint2_t block_idx(blockIdx.x, blocks_x);
        const int3_t gid(block_idx.x * THREADS.x + threadIdx.x,
                         block_idx.y * THREADS.y + threadIdx.y,
                         blockIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        const int v = getFrequency_<IS_SRC_CENTERED>(gid.y, shape.y);
        const int w = getFrequency_<IS_SRC_CENTERED>(gid.z, shape.z);
        const T phase_shift(getPhaseShift_(shift, float3_t(gid.x, v, w))); // u == gid.x

        outputs += blockIdx.z * rows(shape) * output_pitch;
        const uint o_y = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.y, shape.y);
        const uint o_z = getOutputIndex_<IS_SRC_CENTERED, IS_DST_CENTERED>(gid.z, shape.z);
        outputs[(o_z * shape.y + o_y) * output_pitch + gid.x] *= phase_shift;
    }
}

namespace noa::cuda::transform::fft {
    using Layout = noa::fft::Layout;

    template<Remap REMAP, typename T>
    void shift2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                 const float2_t* shifts, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        int2_t s_shape(shape);
        const dim3 blocks(math::divideUp(s_shape.x, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape.y, static_cast<int>(THREADS.y)),
                          batches);
        shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED><<<blocks, THREADS, 0, stream.get()>>>(
                inputs, input_pitch, outputs, output_pitch, s_shape, shifts);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void shift2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                 float2_t shift, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        int2_t s_shape(shape);
        shift *= math::Constants<float>::PI2 / float2_t(s_shape);
        const dim3 blocks(math::divideUp(s_shape.x, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape.y, static_cast<int>(THREADS.y)),
                          batches);
        shift2D_<IS_SRC_CENTERED, IS_DST_CENTERED><<<blocks, THREADS, 0, stream.get()>>>(
                inputs, input_pitch, outputs, output_pitch, s_shape, shift);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                 const float3_t* shifts, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        int3_t s_shape(shape);
        const uint blocks_x = math::divideUp(s_shape.x, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape.y, static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, s_shape.z, batches);
        shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED><<<blocks, THREADS, 0, stream.get()>>>(
                inputs, input_pitch, outputs, output_pitch, s_shape, shifts, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void shift3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                 float3_t shift, size_t batches, Stream& stream) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs || IS_SRC_CENTERED == IS_DST_CENTERED);

        int3_t s_shape(shape);
        shift *= math::Constants<float>::PI2 / float3_t(s_shape);
        const uint blocks_x = math::divideUp(s_shape.x, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape.y, static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, s_shape.z, batches);
        shift3D_<IS_SRC_CENTERED, IS_DST_CENTERED><<<blocks, THREADS, 0, stream.get()>>>(
                inputs, input_pitch, outputs, output_pitch, s_shape, shift, blocks_x);
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_SHIFT(T)                                                                               \
    template void shift2D<Remap::H2H>(const T*, size_t, T*, size_t, size2_t, const float2_t*, size_t, Stream&);    \
    template void shift2D<Remap::H2H>(const T*, size_t, T*, size_t, size2_t, float2_t, size_t, Stream&);           \
    template void shift2D<Remap::H2HC>(const T*, size_t, T*, size_t, size2_t, const float2_t*, size_t, Stream&);   \
    template void shift2D<Remap::H2HC>(const T*, size_t, T*, size_t, size2_t, float2_t, size_t, Stream&);          \
    template void shift2D<Remap::HC2H>(const T*, size_t, T*, size_t, size2_t, const float2_t*, size_t, Stream&);   \
    template void shift2D<Remap::HC2H>(const T*, size_t, T*, size_t, size2_t, float2_t, size_t, Stream&);          \
    template void shift2D<Remap::HC2HC>(const T*, size_t, T*, size_t, size2_t, const float2_t*, size_t, Stream&);  \
    template void shift2D<Remap::HC2HC>(const T*, size_t, T*, size_t, size2_t, float2_t, size_t, Stream&);         \
    template void shift3D<Remap::H2H>(const T*, size_t, T*, size_t, size3_t, const float3_t*, size_t, Stream&);    \
    template void shift3D<Remap::H2H>(const T*, size_t, T*, size_t, size3_t, float3_t, size_t, Stream&);           \
    template void shift3D<Remap::H2HC>(const T*, size_t, T*, size_t, size3_t, const float3_t*, size_t, Stream&);   \
    template void shift3D<Remap::H2HC>(const T*, size_t, T*, size_t, size3_t, float3_t, size_t, Stream&);          \
    template void shift3D<Remap::HC2H>(const T*, size_t, T*, size_t, size3_t, const float3_t*, size_t, Stream&);   \
    template void shift3D<Remap::HC2H>(const T*, size_t, T*, size_t, size3_t, float3_t, size_t, Stream&);          \
    template void shift3D<Remap::HC2HC>(const T*, size_t, T*, size_t, size3_t, const float3_t*, size_t, Stream&);  \
    template void shift3D<Remap::HC2HC>(const T*, size_t, T*, size_t, size3_t, float3_t, size_t, Stream&)

    NOA_INSTANTIATE_SHIFT(cfloat_t);
    NOA_INSTANTIATE_SHIFT(cdouble_t);
}
