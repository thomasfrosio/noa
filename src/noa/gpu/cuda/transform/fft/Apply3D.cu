#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/fft/Apply.h"

// TODO(TF) Add kernel for square/cube shapes using unnormalized coordinates?
// TODO(TF) Save transforms/shifts to constant memory?

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

    __forceinline__ __device__ cfloat_t getPhaseShift_(float3_t shift, float3_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP, typename T, typename TRANSFORM, typename SHIFT>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void apply3DNormalized_(cudaTextureObject_t tex, T* outputs, uint output_pitch,
                            int3_t shape, float3_t length,
                            TRANSFORM rotm, // const float33_t* or float33_t
                            [[maybe_unused]] SHIFT shifts, // const float3_t* or float3_t
                            float max_frequency_sqd, uint blocks_x) {

        const uint2_t block_idx(coordinates(blockIdx.x, blocks_x));
        const int3_t gid(block_idx.x * THREADS.x + threadIdx.x,
                         block_idx.y * THREADS.y + threadIdx.y,
                         blockIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        outputs += blockIdx.z * rows(shape) * output_pitch;
        outputs += (gid.z * shape.y + gid.y) * output_pitch + gid.x;

        const int v = getFrequency_<IS_DST_CENTERED>(gid.y, shape.y);
        const int w = getFrequency_<IS_DST_CENTERED>(gid.z, shape.z);
        float3_t freq(gid.x, v, w);
        freq /= length; // [-0.5, 0.5]
        if (math::dot(freq, freq) > max_frequency_sqd) {
            *outputs = 0;
            return;
        }

        if constexpr (std::is_pointer_v<TRANSFORM>)
            freq = rotm[blockIdx.z] * freq;
        else
            freq = rotm * freq;

        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (freq.x < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }

        freq.y += 0.5f;
        freq.z += 0.5f;
        freq *= length;
        T value = cuda::transform::tex3D<T, INTERP>(tex, freq + 0.5f);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;

        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT) {
            if constexpr (std::is_pointer_v<SHIFT>) {
                const float3_t shift = shifts[blockIdx.z] * math::Constants<float>::PI2 / float3_t(shape);
                value *= getPhaseShift_(shift, float3_t(gid.x, v, w));
            } else {
                value *= getPhaseShift_(shifts, float3_t(gid.x, v, w));
            }
        }

        *outputs = value;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT,
            typename T, typename TRANSFORM, typename SHIFT>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size3_t shape,
                 TRANSFORM transforms, SHIFT shifts, size_t nb_transforms,
                 float max_frequency, cuda::Stream& stream) {
        int3_t s_shape(shape);
        float3_t f_shape(s_shape.x > 1 ? s_shape.x / 2 * 2 : 1,
                         s_shape.y > 1 ? s_shape.y / 2 * 2 : 1,
                         s_shape.z > 1 ? s_shape.z / 2 * 2 : 1); // if odd, shape-1
        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        if constexpr (!std::is_pointer_v<SHIFT>)
            shifts *= math::Constants<float>::PI2 / float3_t(s_shape);

        const uint blocks_x = math::divideUp(s_shape.x, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape.y, static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, s_shape.z, nb_transforms);

        NOA_ASSERT(!cuda::memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
        switch (texture_interp_mode) {
            case InterpMode::INTERP_NEAREST:
                apply3DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_NEAREST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency, blocks_x);
                break;
            case InterpMode::INTERP_LINEAR:
                apply3DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_LINEAR>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency, blocks_x);
                break;
            case InterpMode::INTERP_COSINE:
                apply3DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_COSINE>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency, blocks_x);
                break;
            case InterpMode::INTERP_LINEAR_FAST:
                apply3DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_LINEAR_FAST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency, blocks_x);
                break;
            case InterpMode::INTERP_COSINE_FAST:
                apply3DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_COSINE_FAST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency, blocks_x);
                break;
            default:
                NOA_THROW_FUNC("apply3D", "{} is not supported", texture_interp_mode);
        }
    }

    // Atm, input FFTs should be centered. The only flexibility is whether the output should be centered or not.
    template<fft::Remap REMAP, typename T = void>
    constexpr bool parseRemap_() noexcept {
        using Layout = ::noa::fft::Layout;
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        if constexpr (!IS_SRC_CENTERED || REMAP_ & Layout::SRC_FULL || REMAP_ & Layout::DST_FULL) {
            static_assert(traits::always_false_v<T>);
        }
        return IS_DST_CENTERED;
    }
}

namespace noa::cuda::transform::fft {
    template<Remap REMAP, typename T>
    void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size3_t shape,
                 const float33_t* transforms, const float3_t* shifts, size_t nb_transforms,
                 float max_frequency, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        if (shifts)
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, outputs, output_pitch, shape,
                    transforms, shifts, nb_transforms, max_frequency, stream);
        else
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, outputs, output_pitch, shape,
                    transforms, shifts, nb_transforms, max_frequency, stream);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size3_t shape,
                 const float33_t* transforms, float3_t shift, size_t nb_transforms,
                 float max_frequency, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        if (any(shift != 0.f))
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, outputs, output_pitch, shape,
                    transforms, shift, nb_transforms, max_frequency, stream);
        else
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, outputs, output_pitch, shape,
                    transforms, shift, nb_transforms, max_frequency, stream);
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, size_t output_pitch, size3_t shape,
                 float33_t transform, float3_t shift,
                 float max_frequency, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        if (any(shift != 0.f))
            launch_<IS_DST_CENTERED, true>(
                    texture, texture_interp_mode, output, output_pitch, shape,
                    transform, shift, 1, max_frequency, stream);
        else
            launch_<IS_DST_CENTERED, false>(
                    texture, texture_interp_mode, output, output_pitch, shape,
                    transform, shift, 1, max_frequency, stream);
        NOA_THROW_IF(cudaGetLastError());
    }
}

namespace noa::cuda::transform::fft {
    template<Remap REMAP, typename T>
    void apply3D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                 const float33_t* transforms, const float3_t* shifts, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != outputs);
        memory::PtrArray<T> array(shapeFFT(shape));
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply3D<REMAP>(texture.get(), interp_mode, outputs, output_pitch, shape,
                       transforms, shifts, nb_transforms, max_frequency, stream);
        stream.synchronize();
    }

    template<Remap REMAP, typename T>
    void apply3D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size3_t shape,
                 const float33_t* transforms, float3_t shift, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != outputs);
        memory::PtrArray<T> array(shapeFFT(shape));
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply3D<REMAP>(texture.get(), interp_mode, outputs, output_pitch, shape,
                       transforms, shift, nb_transforms, max_frequency, stream);
        stream.synchronize();
    }

    template<Remap REMAP, typename T>
    void apply3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                 float33_t transform, float3_t shift,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input != output);
        memory::PtrArray<T> array(shapeFFT(shape));
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply3D<REMAP>(texture.get(), interp_mode, output, output_pitch, shape,
                       transform, shift, max_frequency, stream);
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_APPLY3D_(T)                                                                                                                     \
    template void apply3D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size3_t, const float33_t*, const float3_t*, size_t, float, InterpMode, Stream&);    \
    template void apply3D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size3_t, const float33_t*, const float3_t*, size_t, float, InterpMode, Stream&);   \
    template void apply3D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size3_t, const float33_t*, float3_t, size_t, float, InterpMode, Stream&);           \
    template void apply3D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size3_t, const float33_t*, float3_t, size_t, float, InterpMode, Stream&);          \
    template void apply3D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size3_t, float33_t, float3_t, float, InterpMode, Stream&);                          \
    template void apply3D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size3_t, float33_t, float3_t, float, InterpMode, Stream&)

    NOA_INSTANTIATE_APPLY3D_(float);
    NOA_INSTANTIATE_APPLY3D_(cfloat_t);
}
