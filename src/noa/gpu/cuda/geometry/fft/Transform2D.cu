#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/fft/Transform.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(32, 8);

    template<bool IS_DST_CENTERED>
    __forceinline__ __device__ int getFrequency_(int idx, int dim) {
        if constexpr(IS_DST_CENTERED)
            return idx - dim / 2;
        else
            return idx < (dim + 1) / 2 ? idx : idx - dim;
        return 0;
    }

    __forceinline__ __device__ cfloat_t getPhaseShift_(float2_t shift, float2_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP,
             typename T, typename MAT, typename VEC>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transform2D_(cudaTextureObject_t tex, T* output, uint3_t output_stride,
                      int2_t shape, float2_t norm_shape,
                      MAT matrices, // const float22_t* or float22_t
                      [[maybe_unused]] VEC shifts, // const float2_t* or float2_t
                      float cutoff_sqd) {

        const int3_t gid(blockIdx.z,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.x * THREADS.x + threadIdx.x);
        if (gid[1] >= shape[0] || gid[2] >= shape[1] / 2 + 1)
            return;

        // Compute the frequency corresponding to the gid and inverse transform.
        const int v = getFrequency_<IS_DST_CENTERED>(gid[1], shape[0]);
        float2_t freq{v, gid[2]};
        freq /= norm_shape; // [-0.5, 0.5]
        if (math::dot(freq, freq) > cutoff_sqd) {
            output[at(gid, output_stride)] = 0;
            return;
        }

        if constexpr (std::is_pointer_v<MAT>)
            freq = matrices[gid[0]] * freq;
        else
            freq = matrices * freq;

        // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (freq[1] < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }

        // Convert back to index and fetch value from input texture.
        freq[0] += 0.5f; // [0, 1]
        freq *= norm_shape; // [0, N-1]
        T value = cuda::geometry::tex2D<T, INTERP>(tex, freq + 0.5f);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        // Phase shift the interpolated value.
        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT) {
            if constexpr (std::is_pointer_v<VEC>) {
                const float2_t shift = shifts[gid[0]] * math::Constants<float>::PI2 / float2_t{shape};
                value *= getPhaseShift_(shift, float2_t{v, gid[2]});
            } else {
                value *= getPhaseShift_(shifts, float2_t{v, gid[2]});
            }
        } else {
            (void) shifts;
        }

        output[at(gid, output_stride)] = value;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT,
             typename T, typename MAT, typename VEC>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 MAT matrices, VEC shifts,
                 float cutoff, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const uint3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const int2_t s_shape{output_shape.get() + 2};
        const float2_t f_shape{s_shape / 2 * 2 + int2_t{s_shape == 1}}; // if odd, n-1

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        if constexpr (!std::is_pointer_v<VEC>)
            shifts *= math::Constants<float>::PI2 / float2_t{s_shape};

        const dim3 blocks(math::divideUp(s_shape[1] / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape[0], static_cast<int>(THREADS.y)),
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        NOA_ASSERT(!cuda::memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
        switch (texture_interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_NEAREST, T, MAT, VEC>, config,
                        texture, output, o_stride, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_LINEAR:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_LINEAR, T, MAT, VEC>, config,
                        texture, output, o_stride, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_COSINE:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_COSINE, T, MAT, VEC>, config,
                        texture, output, o_stride, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_LINEAR_FAST:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_LINEAR_FAST, T, MAT, VEC>, config,
                        texture, output, o_stride, s_shape, f_shape, matrices, shifts, cutoff);
            case INTERP_COSINE_FAST:
                return stream.enqueue(
                        "geometry::fft::transform2D",
                        transform2D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_COSINE_FAST, T, MAT, VEC>, config,
                        texture, output, o_stride, s_shape, f_shape, matrices, shifts, cutoff);
            default:
                NOA_THROW_FUNC("transform2D", "{} is not supported", texture_interp_mode);
        }
    }

    template<fft::Remap REMAP, typename T = void>
    constexpr bool parseRemap_() noexcept {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (!IS_SRC_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL)
            static_assert(traits::always_false_v<T>);

        return IS_DST_CENTERED;
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T>
    void transform2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t shape,
                     const float22_t* matrices, const float2_t* shifts, float cutoff, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();

        cuda::memory::PtrDevice<float22_t> m_buffer;
        cuda::memory::PtrDevice<float2_t> s_buffer;
        matrices = util::ensureDeviceAccess(matrices, stream, m_buffer, shape[0]);
        if (shifts) {
            shifts = util::ensureDeviceAccess(shifts, stream, s_buffer, shape[0]);
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, output, output_stride, shape,
                    matrices, shifts, cutoff, stream);
        } else {
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, output, output_stride, shape,
                    matrices, shifts, cutoff, stream);
        }
    }

    template<Remap REMAP, typename T>
    void transform2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t shape,
                     float22_t matrix, float2_t shift, float cutoff, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        if (any(shift != 0.f))
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, output, output_stride, shape,
                    matrix, shift, cutoff, stream);
        else
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, output, output_stride, shape,
                    matrix, shift, cutoff, stream);
    }
}

namespace noa::cuda::geometry::fft {
    template<Remap REMAP, typename T>
    void transform2D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     const float22_t* matrices, const float2_t* shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(isContiguous(input_stride, shape.fft())[3]);
        NOA_ASSERT(shape[1] == 1);

        const size3_t shape_3d{1, shape[2], shape[3] / 2 + 1};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);

        size_t iter;
        size4_t o_shape;
        if (input_stride[0] == 0) {
            iter = 1;
            o_shape = {shape[0], 1, shape[2], shape[3]};
        } else {
            iter = shape[0];
            o_shape = {1, 1, shape[2], shape[3]};
        }
        for (size_t i = 0; i < iter; ++i) {
            cuda::memory::copy(input + i * input_stride[0], input_stride[2], array.get(), shape_3d, stream);
            transform2D<REMAP>(texture.get(), interp_mode, output + i * output_stride[0], output_stride,
                               o_shape, matrices + i, shifts + i, cutoff, stream);
        }
        stream.synchronize();
    }

    template<Remap REMAP, typename T>
    void transform2D(const T* input, size4_t input_stride,
                     T* output, size4_t output_stride, size4_t shape,
                     float22_t matrix, float2_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(isContiguous(input_stride, shape.fft())[3]);
        NOA_ASSERT(shape[1] == 1);

        const size3_t shape_3d{1, shape[2], shape[3] / 2 + 1};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);

        size_t iter;
        size4_t o_shape;
        if (input_stride[0] == 0) {
            iter = 1;
            o_shape = {shape[0], 1, shape[2], shape[3]};
        } else {
            iter = shape[0];
            o_shape = {1, 1, shape[2], shape[3]};
        }
        for (size_t i = 0; i < iter; ++i) {
            cuda::memory::copy(input + i * input_stride[0], input_stride[2], array.get(), shape_3d, stream);
            transform2D<REMAP>(texture.get(), interp_mode, output + i * output_stride[0], output_stride,
                               o_shape, matrix, shift, cutoff, stream);
        }
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_TRANSFORM2D_(T)                                                                                                             \
    template void transform2D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, const float22_t*, const float2_t*, float, InterpMode, Stream&);  \
    template void transform2D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, const float22_t*, const float2_t*, float, InterpMode, Stream&); \
    template void transform2D<Remap::HC2H, T>(const T*, size4_t, T*, size4_t, size4_t, float22_t, float2_t, float, InterpMode, Stream&);                \
    template void transform2D<Remap::HC2HC, T>(const T*, size4_t, T*, size4_t, size4_t, float22_t, float2_t, float, InterpMode, Stream&)

    NOA_INSTANTIATE_TRANSFORM2D_(float);
    NOA_INSTANTIATE_TRANSFORM2D_(cfloat_t);
}
