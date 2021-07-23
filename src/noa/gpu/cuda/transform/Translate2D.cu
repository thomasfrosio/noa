#include "noa/common/Assert.h"
#include "noa/common/Math.h"

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
    __global__ void translate2D_(cudaTextureObject_t texture, float2_t texture_shape,
                                 T* outputs, uint output_pitch, uint2_t output_shape,
                                 const float2_t* translations) {
        const uint translation_id = blockIdx.z;
        const uint2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y)
            return;

        float2_t pos(gid);
        pos += translations[translation_id];
        if constexpr (TEXTURE_OFFSET)
            pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;

        outputs[(translation_id * output_shape.y + gid.y) * output_pitch + gid.x] =
                cuda::transform::tex2D<T, MODE>(texture, pos.x, pos.y);
    }

    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void translate2D_(cudaTextureObject_t texture, float2_t texture_shape,
                                 T* output, uint output_pitch, uint2_t output_shape,
                                 float2_t translation) {
        const uint2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y)
            return;

        float2_t pos(gid);
        pos += translation;
        if constexpr (TEXTURE_OFFSET)
            pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;

        output[gid.y * output_pitch + gid.x] = cuda::transform::tex2D<T, MODE>(texture, pos.x, pos.y);
    }
}

// -- Using textures -- //
namespace noa::cuda::transform {
    template<bool TEXTURE_OFFSET, typename T>
    void translate2D(cudaTextureObject_t texture, size2_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* outputs, size_t output_pitch, size2_t output_shape,
                     const float2_t* translations, uint nb_translations, Stream& stream) {
        const float2_t i_shape(texture_shape);
        const uint2_t o_shape(output_shape);
        const dim3 blocks(math::divideUp(o_shape.x, THREADS.x),
                          math::divideUp(o_shape.y, THREADS.y),
                          nb_translations);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                translate2D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, translations);
            } else if (texture_interp_mode == INTERP_LINEAR) {
                translate2D_<TEXTURE_OFFSET, INTERP_LINEAR, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, translations);
            } else {
                NOA_THROW("{} is not supported with {} or {}",
                          texture_interp_mode, BORDER_PERIODIC, BORDER_MIRROR);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    translate2D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations);
                    break;
                case INTERP_LINEAR:
                    translate2D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations);
                    break;
                case INTERP_COSINE:
                    translate2D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    translate2D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, translations);
                    break;
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool TEXTURE_OFFSET, typename T>
    void translate2D(cudaTextureObject_t texture, size2_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size_t output_pitch, size2_t output_shape,
                     float2_t translation, Stream& stream) {
        const float2_t i_shape(texture_shape);
        const uint2_t o_shape(output_shape);
        const dim3 blocks(math::divideUp(o_shape.x, THREADS.x),
                          math::divideUp(o_shape.y, THREADS.y));

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                translate2D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, translation);
            } else if (texture_interp_mode == INTERP_LINEAR) {
                translate2D_<TEXTURE_OFFSET, INTERP_LINEAR, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, translation);
            } else {
                NOA_THROW("{} is not supported with {} or {}",
                          texture_interp_mode, BORDER_PERIODIC, BORDER_MIRROR);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    translate2D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_LINEAR:
                    translate2D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_COSINE:
                    translate2D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    translate2D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, translation);
                    break;
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T>
    void translate2D(const T* input, size_t input_pitch, size2_t input_shape,
                     T* outputs, size_t output_pitch, size2_t output_shape,
                     const float2_t* translations, uint nb_translations,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        size3_t shape_3d(input_shape.x, input_shape.y, 1);
        memory::PtrDevicePadded<T> tmp;
        memory::PtrArray<T> i_array(shape_3d);
        memory::PtrTexture<T> i_texture;

        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            if (any(input_shape != output_shape)) {
                tmp.reset(shape_3d);
                bspline::prefilter2D(input, input_pitch, tmp.get(), tmp.pitch(), input_shape, 1, stream);
                memory::copy(tmp.get(), tmp.pitch(), i_array.get(), shape_3d, stream);
            } else {
                bspline::prefilter2D(input, input_pitch, outputs, output_pitch, input_shape, 1, stream);
                memory::copy(outputs, output_pitch, i_array.get(), shape_3d, stream);
            }
        } else {
            memory::copy(input, input_pitch, i_array.get(), shape_3d, stream);
        }
        i_texture.reset(i_array.get(), interp_mode, border_mode);

        translate2D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                    outputs, output_pitch, output_shape, translations, nb_translations, stream);
        tmp.dispose();
        stream.synchronize(); // don't free the CUDA array before the kernel is done
    }

    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T>
    void translate2D(const T* input, size_t input_pitch, size2_t input_shape,
                     T* output, size_t output_pitch, size2_t output_shape,
                     float2_t translation,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        size3_t shape_3d(input_shape.x, input_shape.y, 1);
        memory::PtrDevicePadded<T> tmp; // or PtrDevice?
        memory::PtrArray<T> i_array(shape_3d);
        memory::PtrTexture<T> i_texture;

        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            if (any(input_shape != output_shape)) {
                tmp.reset(shape_3d);
                bspline::prefilter2D(input, input_pitch, tmp.get(), tmp.pitch(), input_shape, 1, stream);
                memory::copy(tmp.get(), tmp.pitch(), i_array.get(), shape_3d, stream);
            } else {
                bspline::prefilter2D(input, input_pitch, output, output_pitch, input_shape, 1, stream);
                memory::copy(output, output_pitch, i_array.get(), shape_3d, stream);
            }
        } else {
            memory::copy(input, input_pitch, i_array.get(), shape_3d, stream);
        }
        i_texture.reset(i_array.get(), interp_mode, border_mode);

        translate2D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                    output, output_pitch, output_shape, translation, stream);
        tmp.dispose();
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_TRANSLATE_2D_(T)                                                                                                            \
    template void translate2D<false, false, T>(const T*, size_t, size2_t, T*, size_t, size2_t, const float2_t*, uint, InterpMode, BorderMode, Stream&); \
    template void translate2D<false, true, T>(const T*, size_t, size2_t, T*, size_t, size2_t, const float2_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void translate2D<true, false, T>(const T*, size_t, size2_t, T*, size_t, size2_t, const float2_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void translate2D<true, true, T>(const T*, size_t, size2_t, T*, size_t, size2_t, const float2_t*, uint, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_TRANSLATE_2D_(float);
    NOA_INSTANTIATE_TRANSLATE_2D_(cfloat_t);
}
