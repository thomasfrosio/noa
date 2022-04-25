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

    __forceinline__ __device__ cfloat_t getPhaseShift_(float3_t shift, float3_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT, InterpMode INTERP,
             typename T, typename MAT, typename VEC>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void transform3D_(cudaTextureObject_t tex, T* output, uint4_t output_stride,
                      int3_t shape, float3_t norm_shape,
                      MAT matrices, // const float33_t* or float33_t
                      [[maybe_unused]] VEC shifts, // const float3_t* or float3_t
                      float cutoff_sqd, uint blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[1] || gid[3] >= shape[2] / 2 + 1)
            return;

        const int w = getFrequency_<IS_DST_CENTERED>(gid[1], shape[0]);
        const int v = getFrequency_<IS_DST_CENTERED>(gid[2], shape[1]);
        float3_t freq{w, v, gid[3]};
        freq /= norm_shape; // [-0.5, 0.5]
        if (math::dot(freq, freq) > cutoff_sqd) {
            output[indexing::at(gid, output_stride)] = 0;
            return;
        }

        if constexpr (std::is_pointer_v<MAT>)
            freq = matrices[gid[0]] * freq;
        else
            freq = matrices * freq;

        using real_t = traits::value_type_t<T>;
        [[maybe_unused]] real_t conj = 1;
        if (freq[2] < 0.f) {
            freq = -freq;
            if constexpr (traits::is_complex_v<T>)
                conj = -1;
        }

        freq[0] += 0.5f;
        freq[1] += 0.5f;
        freq *= norm_shape;
        T value = cuda::geometry::tex3D<T, INTERP>(tex, freq + 0.5f);
        if constexpr (traits::is_complex_v<T>)
            value.imag *= conj;
        else
            (void) conj;

        if constexpr (traits::is_complex_v<T> && APPLY_SHIFT) {
            if constexpr (std::is_pointer_v<VEC>) {
                const float3_t shift = shifts[gid[0]] * math::Constants<float>::PI2 / float3_t{shape};
                value *= getPhaseShift_(shift, float3_t{w, v, gid[3]});
            } else {
                value *= getPhaseShift_(shifts, float3_t{w, v, gid[3]});
            }
        } else {
            (void) shifts;
        }

        output[indexing::at(gid, output_stride)] = value;
    }

    template<bool IS_DST_CENTERED, bool APPLY_SHIFT,
            typename T, typename MAT, typename VEC>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 MAT matrices, VEC shifts,
                 float cutoff, cuda::Stream& stream) {
        const int3_t s_shape{output_shape.get() + 1};
        const float3_t f_shape{s_shape / 2 * 2 + int3_t{s_shape == 1}}; // if odd, n-1

        cutoff = noa::math::clamp(cutoff, 0.f, 0.5f);
        cutoff *= cutoff;

        if constexpr (!std::is_pointer_v<VEC>)
            shifts *= math::Constants<float>::PI2 / float3_t(s_shape);

        const uint blocks_x = math::divideUp(s_shape[2] / 2 + 1, static_cast<int>(THREADS.x));
        const uint blocks_y = math::divideUp(s_shape[1], static_cast<int>(THREADS.y));
        const dim3 blocks(blocks_x * blocks_y, output_shape[1], output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
        switch (texture_interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_NEAREST, T, MAT, VEC>, config,
                        texture, output, uint4_t{output_stride}, s_shape, f_shape, matrices, shifts, cutoff, blocks_x);
            case INTERP_LINEAR:
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_LINEAR, T, MAT, VEC>, config,
                        texture, output, uint4_t{output_stride}, s_shape, f_shape, matrices, shifts, cutoff, blocks_x);
            case INTERP_COSINE:
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_COSINE, T, MAT, VEC>, config,
                        texture, output, uint4_t{output_stride}, s_shape, f_shape, matrices, shifts, cutoff, blocks_x);
            case INTERP_LINEAR_FAST:
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_LINEAR_FAST, T, MAT, VEC>, config,
                        texture, output, uint4_t{output_stride}, s_shape, f_shape, matrices, shifts, cutoff, blocks_x);
            case INTERP_COSINE_FAST:
                return stream.enqueue(
                        "geometry::fft::transform3D",
                        transform3D_<IS_DST_CENTERED, APPLY_SHIFT, INTERP_COSINE_FAST, T, MAT, VEC>, config,
                        texture, output, uint4_t{output_stride}, s_shape, f_shape, matrices, shifts, cutoff, blocks_x);
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", texture_interp_mode);
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
    void transform3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t shape,
                     const float33_t* matrices, const float3_t* shifts, float cutoff, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        constexpr bool IS_DST_CENTERED = parseRemap_<REMAP>();

        cuda::memory::PtrDevice<float33_t> m_buffer;
        cuda::memory::PtrDevice<float3_t> s_buffer;
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
    void transform3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                     T* output, size4_t output_stride, size4_t shape,
                     float33_t matrix, float3_t shift, float cutoff, Stream& stream) {
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
    void transform3D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     const shared_t<float33_t[]>& matrices,
                     const shared_t<float3_t[]>& shifts,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(indexing::isContiguous(input_stride, shape.fft())[3]);
        NOA_ASSERT(indexing::isContiguous(input_stride, shape.fft())[1]);

        const size3_t shape_3d{shape[1], shape[2], shape[3] / 2 + 1};
        memory::PtrArray<T> array{shape_3d};
        memory::PtrTexture texture{array.get(), interp_mode, BORDER_ZERO};

        size_t iter;
        size4_t o_shape;
        if (input_stride[0] == 0) {
            iter = 1;
            o_shape = shape;
        } else {
            iter = shape[0];
            o_shape = {1, shape[1], shape[2], shape[3]};
        }
        for (size_t i = 0; i < iter; ++i) {
            cuda::memory::copy(input.get() + i * input_stride[0], input_stride[2], array.get(), shape_3d, stream);
            transform3D<REMAP>(texture.get(), interp_mode, output.get() + i * output_stride[0], output_stride,
                               o_shape, matrices.get() + i, shifts.get() + i, cutoff, stream);
        }
        stream.attach(input, output, matrices, shifts, array.share(), texture.share());
    }

    template<Remap REMAP, typename T>
    void transform3D(const shared_t<T[]>& input, size4_t input_stride,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t shape,
                     float33_t matrix, float3_t shift,
                     float cutoff, InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(indexing::isContiguous(input_stride, shape.fft())[3]);
        NOA_ASSERT(indexing::isContiguous(input_stride, shape.fft())[1]);

        const size3_t shape_3d{shape[1], shape[2], shape[3] / 2 + 1};
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);

        size_t iter;
        size4_t o_shape;
        if (input_stride[0] == 0) {
            iter = 1;
            o_shape = shape;
        } else {
            iter = shape[0];
            o_shape = {1, shape[1], shape[2], shape[3]};
        }
        for (size_t i = 0; i < iter; ++i) {
            cuda::memory::copy(input.get() + i * input_stride[0], input_stride[2], array.get(), shape_3d, stream);
            transform3D<REMAP>(texture.get(), interp_mode, output.get() + i * output_stride[0], output_stride,
                               o_shape, matrix, shift, cutoff, stream);
        }
        stream.attach(input, output, array.share(), texture.share());
    }

    #define NOA_INSTANTIATE_TRANSFORM2D_(T)                                                                                                                                                                     \
    template void transform3D<Remap::HC2H, T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float33_t[]>&, const shared_t<float3_t[]>&, float, InterpMode, Stream&);    \
    template void transform3D<Remap::HC2HC, T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float33_t[]>&, const shared_t<float3_t[]>&, float, InterpMode, Stream&);   \
    template void transform3D<Remap::HC2H, T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float33_t, float3_t, float, InterpMode, Stream&);                                          \
    template void transform3D<Remap::HC2HC, T>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, float33_t, float3_t, float, InterpMode, Stream&)

    NOA_INSTANTIATE_TRANSFORM2D_(float);
    NOA_INSTANTIATE_TRANSFORM2D_(cfloat_t);
}
