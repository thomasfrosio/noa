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

    __forceinline__ __device__ cfloat_t getPhaseShift_(float2_t shift, float2_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    #pragma nv_diag_suppress 177
    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP, typename T, typename TRANSFORM, typename SHIFT>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void apply2DNormalized_(cudaTextureObject_t tex, T* outputs, uint output_pitch,
                            int2_t shape, float2_t length,
                            TRANSFORM rotm, // const float22_t* or float22_t
                            [[maybe_unused]] SHIFT shifts, // const float2_t* or float2_t
                            float max_frequency_sqd) {

        // Get the current global output index.
        const int2_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                         blockIdx.y * THREADS.y + threadIdx.y);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        // Offset to current element in output
        outputs += blockIdx.z * shape.y * output_pitch;
        outputs += gid.y * output_pitch + gid.x;

        // Compute the frequency corresponding to the gid and inverse transform.
        const int v = getFrequency_<IS_DST_CENTERED>(gid.y, shape.y);
        float2_t freq(gid.x, v);
        freq /= length; // [-0.5, 0.5]
        if (math::dot(freq, freq) > max_frequency_sqd) {
            *outputs = 0;
            return;
        }

        if constexpr (std::is_pointer_v<TRANSFORM>)
            freq = rotm[blockIdx.z] * freq;
        else
            freq = rotm * freq;

        // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (freq.x < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }

        // Convert back to index and fetch value from input texture.
        freq.y += 0.5f; // [0, 1]
        freq *= length; // [0, N-1]
        T value = cuda::transform::tex2D<T, INTERP>(tex, freq + 0.5f);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        // Phase shift the interpolated value.
        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT) {
            if constexpr (std::is_pointer_v<SHIFT>) {
                const float2_t shift = shifts[blockIdx.z] * math::Constants<float>::PI2 / float2_t(shape);
                value *= getPhaseShift_(shift, float2_t(gid.x, v));
            } else {
                value *= getPhaseShift_(shifts, float2_t(gid.x, v));
            }
        } else {
            (void) shifts;
        }

        *outputs = value;
    }
    #pragma nv_diagnostic pop

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT,
            typename T, typename TRANSFORM, typename SHIFT>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size2_t shape,
                 TRANSFORM transforms, SHIFT shifts, size_t nb_transforms,
                 float max_frequency, cuda::Stream& stream) {
        int2_t s_shape(shape);
        float2_t f_shape(s_shape.x > 1 ? s_shape.x / 2 * 2 : 1,
                         s_shape.y > 1 ? s_shape.y / 2 * 2 : 1); // if odd, shape-1
        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        if constexpr (!std::is_pointer_v<SHIFT>)
            shifts *= math::Constants<float>::PI2 / float2_t(s_shape);

        const dim3 blocks(math::divideUp(s_shape.x / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape.y, static_cast<int>(THREADS.y)),
                          nb_transforms);

        NOA_ASSERT(!cuda::memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
        switch (texture_interp_mode) {
            case InterpMode::INTERP_NEAREST:
                apply2DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_NEAREST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency);
                break;
            case InterpMode::INTERP_LINEAR:
                apply2DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_LINEAR>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency);
                break;
            case InterpMode::INTERP_COSINE:
                apply2DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_COSINE>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency);
                break;
            case InterpMode::INTERP_LINEAR_FAST:
                apply2DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_LINEAR_FAST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency);
                break;
            case InterpMode::INTERP_COSINE_FAST:
                apply2DNormalized_<IS_DST_CENTERED, APPLY_SHIFT, InterpMode::INTERP_COSINE_FAST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape, transforms, shifts, max_frequency);
                break;
            default:
                NOA_THROW_FUNC("apply2D", "{} is not supported", texture_interp_mode);
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
    void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size2_t shape,
                 const float22_t* transforms, const float2_t* shifts, size_t nb_transforms,
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
    void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size2_t shape,
                 const float22_t* transforms, float2_t shift, size_t nb_transforms,
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
    void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, size_t output_pitch, size2_t shape,
                 float22_t transform, float2_t shift,
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
    void apply2D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                 const float22_t* transforms, const float2_t* shifts, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrArray<T> array({shape.x / 2 + 1, shape.y, 1});
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply2D<REMAP>(texture.get(), interp_mode, outputs, output_pitch, shape,
                       transforms, shifts, nb_transforms, max_frequency, stream);
        stream.synchronize();
    }

    template<Remap REMAP, typename T>
    void apply2D(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size2_t shape,
                 const float22_t* transforms, float2_t shift, size_t nb_transforms,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrArray<T> array({shape.x / 2 + 1, shape.y, 1});
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply2D<REMAP>(texture.get(), interp_mode, outputs, output_pitch, shape,
                       transforms, shift, nb_transforms, max_frequency, stream);
        stream.synchronize();
    }

    template<Remap REMAP, typename T>
    void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                 float22_t transform, float2_t shift,
                 float max_frequency, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrArray<T> array({shape.x / 2 + 1, shape.y, 1});
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply2D<REMAP>(texture.get(), interp_mode, output, output_pitch, shape,
                       transform, shift, max_frequency, stream);
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_APPLY2D_(T)                                                                                                                     \
    template void apply2D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size2_t, const float22_t*, const float2_t*, size_t, float, InterpMode, Stream&);    \
    template void apply2D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size2_t, const float22_t*, const float2_t*, size_t, float, InterpMode, Stream&);   \
    template void apply2D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size2_t, const float22_t*, float2_t, size_t, float, InterpMode, Stream&);           \
    template void apply2D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size2_t, const float22_t*, float2_t, size_t, float, InterpMode, Stream&);          \
    template void apply2D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size2_t, float22_t, float2_t, float, InterpMode, Stream&);                          \
    template void apply2D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size2_t, float22_t, float2_t, float, InterpMode, Stream&)

    NOA_INSTANTIATE_APPLY2D_(float);
    NOA_INSTANTIATE_APPLY2D_(cfloat_t);
}
