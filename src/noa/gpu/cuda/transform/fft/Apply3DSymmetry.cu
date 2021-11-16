#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/fft/Apply.h"
#include "noa/gpu/cuda/transform/fft/Symmetry.h"

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

    // Interpolates the (complex) value at the normalized frequency "freq".
    template<InterpMode INTERP, typename T>
    inline __device__ T getValue_(cudaTextureObject_t tex, float3_t freq, float3_t length) {
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
        return value;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, bool IS_IDENTITY, InterpMode INTERP, typename T, typename R>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void applySymNormalized3D_(cudaTextureObject_t tex, T* outputs, uint output_pitch,
                               int3_t shape, float3_t length,
                               [[maybe_unused]] float33_t rotm, const float33_t* sym_matrices, uint sym_count,
                               [[maybe_unused]] float3_t shift, R scalar, float max_frequency_sqd) {

        const int3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                         blockIdx.y * THREADS.y + threadIdx.y,
                         blockIdx.z);
        if (gid.x >= shape.x / 2 + 1 || gid.y >= shape.y)
            return;

        const int v = getFrequency_<IS_DST_CENTERED>(gid.y, shape.y);
        const int w = getFrequency_<IS_DST_CENTERED>(gid.z, shape.z);
        float3_t coordinates(gid.x, v, w);
        coordinates /= length; // [-0.5, 0.5]
        if (math::dot(coordinates, coordinates) > max_frequency_sqd)
            return;

        if constexpr (!IS_IDENTITY)
            coordinates = rotm * coordinates;

        T value = getValue_<INTERP, T>(tex, coordinates, length);
        for (uint i = 0; i < sym_count; ++i) {
            float3_t i_freq(sym_matrices[i] * coordinates);
            value += getValue_<INTERP, T>(tex, i_freq, length);
        }

        value *= scalar;
        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT)
            value *= getPhaseShift_(shift, float3_t(gid.x, v, w));

        outputs[(gid.z * shape.y + gid.y) * output_pitch + gid.x] = value;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, bool IS_IDENTITY, typename T, typename R>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* outputs, size_t output_pitch, size3_t shape,
                 float33_t rotm, const float33_t* symmetry_matrices, size_t symmetry_count, float3_t shift,
                 float max_frequency, R scalar, cuda::Stream& stream) {
        int3_t s_shape(shape);
        float3_t f_shape(s_shape.x > 1 ? s_shape.x / 2 * 2 : 1,
                         s_shape.y > 1 ? s_shape.y / 2 * 2 : 1,
                         s_shape.z > 1 ? s_shape.z / 2 * 2 : 1);

        max_frequency = noa::math::clamp(max_frequency, 0.f, 0.5f);
        max_frequency *= max_frequency;

        shift *= math::Constants<float>::PI2 / float3_t(s_shape);

        const dim3 blocks(math::divideUp(s_shape.x / 2 + 1, static_cast<int>(THREADS.x)),
                          math::divideUp(s_shape.y, static_cast<int>(THREADS.y)),
                          s_shape.z);

        NOA_ASSERT(!cuda::memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
        switch (texture_interp_mode) {
            case InterpMode::INTERP_NEAREST:
                applySymNormalized3D_<IS_DST_CENTERED, APPLY_SHIFT, IS_IDENTITY, InterpMode::INTERP_NEAREST>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape,
                        rotm, symmetry_matrices, symmetry_count, shift, scalar, max_frequency);
                break;
            case InterpMode::INTERP_LINEAR:
                applySymNormalized3D_<IS_DST_CENTERED, APPLY_SHIFT, IS_IDENTITY, InterpMode::INTERP_LINEAR>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape,
                        rotm, symmetry_matrices, symmetry_count, shift, scalar, max_frequency);
                break;
            case InterpMode::INTERP_COSINE:
                applySymNormalized3D_<IS_DST_CENTERED, APPLY_SHIFT, IS_IDENTITY, InterpMode::INTERP_COSINE>
                <<<blocks, THREADS, 0, stream.id()>>>(
                        texture, outputs, output_pitch, s_shape, f_shape,
                        rotm, symmetry_matrices, symmetry_count, shift, scalar, max_frequency);
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
                 T* output, size_t output_pitch, size3_t shape,
                 float33_t transform, const float33_t* symmetry_matrices, size_t symmetry_count, float3_t shift,
                 float max_frequency, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(symmetry_count + 1) : 1;
        if (any(shift != 0.f)) {
            launch_<IS_DST_CENTERED, true, false>(texture, texture_interp_mode,
                                                  output, output_pitch, shape,
                                                  transform, symmetry_matrices, symmetry_count, shift,
                                                  max_frequency, scaling, stream);
        } else {
            launch_<IS_DST_CENTERED, false, false>(texture, texture_interp_mode,
                                                   output, output_pitch, shape,
                                                   transform, symmetry_matrices, symmetry_count, {},
                                                   max_frequency, scaling, stream);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void apply3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                 float33_t transform, const Symmetry& symmetry, float3_t shift,
                 float max_frequency, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t count = symmetry.count();
        if (!count)
            return apply3D<REMAP>(input, input_pitch, output, output_pitch, shape,
                                  transform, shift, max_frequency, interp_mode, stream);

        memory::PtrDevice<float33_t> d_matrices(count);
        memory::copy(symmetry.matrices(), d_matrices.get(), count, stream);

        memory::PtrArray<T> array(shapeFFT(shape));
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        apply3D<REMAP>(texture.get(), interp_mode, output, output_pitch, shape,
                       transform, d_matrices.get(), count, shift, max_frequency, normalize, stream);
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_APPLY3D_(T)                                                                                                                     \
    template void apply3D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size3_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);  \
    template void apply3D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size3_t, float33_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_APPLY3D_(float);
    NOA_INSTANTIATE_APPLY3D_(cfloat_t);
}

namespace noa::cuda::transform::fft {
    template<Remap REMAP, typename T>
    void symmetrize3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      T* output, size_t output_pitch, size3_t shape,
                      const float33_t* symmetry_matrices, size_t symmetry_count, float3_t shift,
                      float max_frequency, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();
        const auto scaling = normalize ? 1 / static_cast<traits::value_type_t<T>>(symmetry_count + 1) : 1;
        if (any(shift != 0.f)) {
            launch_<IS_DST_CENTERED, true, true>(texture, texture_interp_mode,
                                                 output, output_pitch, shape,
                                                 {}, symmetry_matrices, symmetry_count, shift,
                                                 max_frequency, scaling, stream);
        } else {
            launch_<IS_DST_CENTERED, false, true>(texture, texture_interp_mode,
                                                  output, output_pitch, shape,
                                                  {}, symmetry_matrices, symmetry_count, {},
                                                  max_frequency, scaling, stream);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    template<Remap REMAP, typename T>
    void symmetrize3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                      const Symmetry& symmetry, float3_t shift,
                      float max_frequency, InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrDevice<float33_t> d_matrices;
        const size_t count = symmetry.count();
        if (count) {
            d_matrices.reset(count);
            memory::copy(symmetry.matrices(), d_matrices.get(), count, stream);
        }
        memory::PtrArray<T> array(shapeFFT(shape));
        memory::PtrTexture<T> texture(array.get(), interp_mode, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), array.shape(), stream);
        symmetrize3D<REMAP>(texture.get(), interp_mode, output, output_pitch, shape,
                            d_matrices.get(), count, shift, max_frequency, normalize, stream);
        stream.synchronize();
    }

    #define NOA_INSTANTIATE_SYMMETRIZE_3D_(T)                                                                                                           \
    template void symmetrize3D<Remap::HC2HC, T>(const T*, size_t, T*, size_t, size3_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&);    \
    template void symmetrize3D<Remap::HC2H, T>(const T*, size_t, T*, size_t, size3_t, const Symmetry&, float3_t, float, InterpMode, bool, Stream&)

    NOA_INSTANTIATE_SYMMETRIZE_3D_(float);
    NOA_INSTANTIATE_SYMMETRIZE_3D_(cfloat_t);
}
