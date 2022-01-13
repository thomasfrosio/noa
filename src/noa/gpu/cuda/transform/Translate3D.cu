#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Translate.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    translate3D_(cudaTextureObject_t texture, [[maybe_unused]] float3_t texture_shape,
                 T* outputs, uint output_pitch, uint3_t output_shape,
                 const float3_t* translations, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t gid(idx.x * THREADS.x + threadIdx.x,
                          idx.y * THREADS.y + threadIdx.y,
                          blockIdx.y);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y) // z cannot be out of bounds
            return;

        float3_t pos(gid);
        pos -= translations[blockIdx.z];
        if constexpr (TEXTURE_OFFSET)
            pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        outputs += blockIdx.z * output_shape.y + output_pitch;
        outputs[(gid.z * output_shape.y + gid.y) * output_pitch + gid.x] =
                cuda::transform::tex3D<T, MODE>(texture, pos);
    }

    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    translate3D_(cudaTextureObject_t texture, [[maybe_unused]] float3_t texture_shape,
                 T* output, uint output_pitch, uint3_t output_shape,
                 float3_t translation) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y)
            return;

        float3_t pos(gid);
        pos -= translation;
        if constexpr (TEXTURE_OFFSET)
            pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output[(gid.z * output_shape.y + gid.y) * output_pitch + gid.x] =
                cuda::transform::tex3D<T, MODE>(texture, pos);
    }
}

// -- Using textures -- //
namespace noa::cuda::transform {
    template<bool TEXTURE_OFFSET, typename T>
    void translate3D(cudaTextureObject_t texture, size3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* outputs, size_t output_pitch, size3_t output_shape,
                     const float3_t* translations, size_t nb_translations,
                     Stream& stream) {
        const float3_t i_shape(texture_shape);
        const uint3_t o_shape(output_shape);
        const uint blocks_x = math::divideUp(o_shape.x, THREADS.x);
        const uint blocks_y = math::divideUp(o_shape.y, THREADS.y);
        const dim3 blocks(blocks_x * blocks_y, o_shape.z, nb_translations);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                translate3D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                translate3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<float>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    translate3D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                case INTERP_LINEAR:
                    translate3D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                case INTERP_COSINE:
                    translate3D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    translate3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                case INTERP_LINEAR_FAST:
                    translate3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                case INTERP_COSINE_FAST:
                    translate3D_<TEXTURE_OFFSET, INTERP_COSINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                case INTERP_CUBIC_BSPLINE_FAST:
                    translate3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations, blocks_x);
                    break;
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    template<bool TEXTURE_OFFSET, typename T>
    void translate3D(cudaTextureObject_t texture, size3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size_t output_pitch, size3_t output_shape,
                     float3_t translation, Stream& stream) {
        const float3_t i_shape(texture_shape);
        const uint3_t o_shape(output_shape);
        const dim3 blocks(math::divideUp(o_shape.x, THREADS.x),
                          math::divideUp(o_shape.y, THREADS.y),
                          o_shape.z);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<float>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                translate3D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, translation);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                translate3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, translation);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<float>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    translate3D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_LINEAR:
                    translate3D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_COSINE:
                    translate3D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    translate3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_LINEAR_FAST:
                    translate3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_COSINE_FAST:
                    translate3D_<TEXTURE_OFFSET, INTERP_COSINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_CUBIC_BSPLINE_FAST:
                    translate3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
        NOA_THROW_IF(cudaGetLastError());
    }
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T>
    void translate3D(const T* input, size_t input_pitch, size3_t input_shape,
                     T* outputs, size_t output_pitch, size3_t output_shape,
                     const float3_t* translations, size_t nb_translations,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrArray<T> i_array(input_shape);
        memory::PtrTexture<T> i_texture;

        memory::PtrDevicePadded<T> tmp; // or PtrDevice?
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (any(input_shape != output_shape)) {
                tmp.reset(input_shape);
                bspline::prefilter3D(input, input_pitch, tmp.get(), tmp.pitch(), input_shape, 1, stream);
                memory::copy(tmp.get(), tmp.pitch(), i_array.get(), input_shape, stream);
            } else {
                bspline::prefilter3D(input, input_pitch, outputs, output_pitch, input_shape, 1, stream);
                memory::copy(outputs, output_pitch, i_array.get(), input_shape, stream);
            }
        } else {
            memory::copy(input, input_pitch, i_array.get(), input_shape, stream);
        }
        i_texture.reset(i_array.get(), interp_mode, border_mode);

        translate3D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                    outputs, output_pitch, output_shape, translations, nb_translations, stream);
        tmp.dispose();
        stream.synchronize();
    }

    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T>
    void translate3D(const T* input, size_t input_pitch, size3_t input_shape,
                     T* output, size_t output_pitch, size3_t output_shape,
                     float3_t translation,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrArray<T> i_array(input_shape);
        memory::PtrTexture<T> i_texture;

        memory::PtrDevicePadded<T> tmp;
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (any(input_shape != output_shape)) {
                tmp.reset(input_shape);
                bspline::prefilter3D(input, input_pitch, tmp.get(), tmp.pitch(), input_shape, 1, stream);
                memory::copy(tmp.get(), tmp.pitch(), i_array.get(), input_shape, stream);
            } else {
                bspline::prefilter3D(input, input_pitch, output, output_pitch, input_shape, 1, stream);
                memory::copy(output, output_pitch, i_array.get(), input_shape, stream);
            }
        } else {
            memory::copy(input, input_pitch, i_array.get(), input_shape, stream);
        }
        i_texture.reset(i_array.get(), interp_mode, border_mode);

        translate3D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                    output, output_pitch, output_shape, translation, stream);
        tmp.dispose();
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_TRANSLATE_3D_(T)                                                                                                                \
    template void translate3D<false, false, T>(const T*, size_t, size3_t, T*, size_t, size3_t, const float3_t*, size_t, InterpMode, BorderMode, Stream&);   \
    template void translate3D<false, true, T>(const T*, size_t, size3_t, T*, size_t, size3_t, const float3_t*, size_t, InterpMode, BorderMode, Stream&);    \
    template void translate3D<true, false, T>(const T*, size_t, size3_t, T*, size_t, size3_t, const float3_t*, size_t, InterpMode, BorderMode, Stream&);    \
    template void translate3D<true, true, T>(const T*, size_t, size3_t, T*, size_t, size3_t, const float3_t*, size_t, InterpMode, BorderMode, Stream&);     \
    template void translate3D<false, false, T>(const T*, size_t, size3_t, T*, size_t, size3_t, float3_t, InterpMode, BorderMode, Stream&);                  \
    template void translate3D<false, true, T>(const T*, size_t, size3_t, T*, size_t, size3_t, float3_t, InterpMode, BorderMode, Stream&);                   \
    template void translate3D<true, false, T>(const T*, size_t, size3_t, T*, size_t, size3_t, float3_t, InterpMode, BorderMode, Stream&);                   \
    template void translate3D<true, true, T>(const T*, size_t, size3_t, T*, size_t, size3_t, float3_t, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_TRANSLATE_3D_(float);
    NOA_INSTANTIATE_TRANSLATE_3D_(cfloat_t);
}
