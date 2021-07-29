#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevicePadded.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Apply.h"

namespace {
    using namespace ::noa;

    constexpr dim3 THREADS(16, 16);

    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T, typename MATRIX>
    __global__ void apply2D_(cudaTextureObject_t texture, float2_t texture_shape,
                             T* outputs, uint output_pitch, uint2_t output_shape,
                             const MATRIX* affine) {
        const uint rotation_id = blockIdx.z;
        const uint2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y) // any(gid >= output_shape)
            return;

        float3_t pos(gid.x, gid.y, 1.f);
        float2_t coordinates(math::dot(affine[rotation_id][0], pos),
                             math::dot(affine[rotation_id][1], pos)); // 2x3 * 3x1 matrix-vector multiplication
        if constexpr (TEXTURE_OFFSET) {
            coordinates.x += 0.5f;
            coordinates.y += 0.5f;
        }
        if constexpr (NORMALIZED) {
            coordinates.x /= texture_shape.x;
            coordinates.y /= texture_shape.y;
        }
        outputs[(rotation_id * output_shape.y + gid.y) * output_pitch + gid.x] =
                cuda::transform::tex2D<T, MODE>(texture, coordinates.x, coordinates.y);
    }

    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void apply2D_(cudaTextureObject_t texture, float2_t texture_shape,
                             T* output, uint output_pitch, uint2_t output_shape,
                             float23_t rotm) {
        const uint2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y)
            return;

        float3_t pos(gid.x, gid.y, 1.f);
        float2_t coordinates(rotm * pos);
        if constexpr (TEXTURE_OFFSET) {
            coordinates.x += 0.5f;
            coordinates.y += 0.5f;
        }
        if constexpr (NORMALIZED) {
            coordinates.x /= texture_shape.x;
            coordinates.y /= texture_shape.y;
        }
        output[gid.y * output_pitch + gid.x] =
                cuda::transform::tex2D<T, MODE>(texture, coordinates.x, coordinates.y);
    }
}

// -- Using textures -- //
namespace noa::cuda::transform {
    template<bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply2D(cudaTextureObject_t texture, size2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* outputs, size_t output_pitch, size2_t output_shape,
                 const MATRIX* transforms, uint nb_transforms, Stream& stream) {
        const float2_t i_shape(texture_shape);
        const uint2_t o_shape(output_shape);
        const dim3 blocks(math::divideUp(o_shape.x, THREADS.x),
                          math::divideUp(o_shape.y, THREADS.y),
                          nb_transforms);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                apply2D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, transforms);
            } else if (texture_interp_mode == INTERP_LINEAR) {
                apply2D_<TEXTURE_OFFSET, INTERP_LINEAR, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, transforms);
            } else {
                NOA_THROW("{} is not supported with {} or {}",
                          texture_interp_mode, BORDER_PERIODIC, BORDER_MIRROR);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    apply2D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms);
                    break;
                case INTERP_LINEAR:
                    apply2D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms);
                    break;
                case INTERP_COSINE:
                    apply2D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    apply2D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms);
                    break;
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply2D(cudaTextureObject_t texture, size2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size_t output_pitch, size2_t output_shape,
                 MATRIX transform, Stream& stream) {
        const float23_t affine(transform);
        const float2_t i_shape(texture_shape);
        const uint2_t o_shape(output_shape);
        const dim3 blocks(math::divideUp(o_shape.x, THREADS.x),
                          math::divideUp(o_shape.y, THREADS.y));
        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                apply2D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, affine);
            } else if (texture_interp_mode == INTERP_LINEAR) {
                apply2D_<TEXTURE_OFFSET, INTERP_LINEAR, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, affine);
            } else {
                NOA_THROW("{} is not supported with {} or {}",
                          texture_interp_mode, BORDER_PERIODIC, BORDER_MIRROR);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    apply2D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_LINEAR:
                    apply2D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_COSINE:
                    apply2D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    apply2D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
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
    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply2D(const T* input, size_t input_pitch, size2_t input_shape,
                 T* outputs, size_t output_pitch, size2_t output_shape,
                 const MATRIX* transforms, uint nb_transforms,
                 InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        size3_t shape_3d(input_shape.x, input_shape.y, 1);
        memory::PtrDevicePadded<T> tmp; // or PtrDevice?
        memory::PtrArray<T> i_array(shape_3d);
        memory::PtrTexture<T> i_texture;

        if (PREFILTER && interp_mode == INTERP_CUBIC_BSPLINE) {
            // If input == output, the prefilter is in-place, otherwise it's out-of-place.
            // If they don't have the same shape, in-place is not possible.
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
        i_texture.reset(i_array.get(), interp_mode, border_mode); // no need to wait for the copy to finish

        apply2D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                outputs, output_pitch, output_shape, transforms, nb_transforms, stream);
        tmp.dispose();
        stream.synchronize(); // don't free the CUDA array before the kernel is done
    }

    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply2D(const T* input, size_t input_pitch, size2_t input_shape,
                 T* output, size_t output_pitch, size2_t output_shape,
                 MATRIX transform, InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
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

        apply2D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                output, output_pitch, output_shape, transform, stream);
        tmp.dispose();
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_APPLY_2D_(T)                                                                                                                        \
    template void apply2D<true, false, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float23_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void apply2D<true, true, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float23_t*, uint, InterpMode, BorderMode, Stream&);   \
    template void apply2D<true, false, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float33_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void apply2D<true, true, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float33_t*, uint, InterpMode, BorderMode, Stream&);   \
    template void apply2D<false, false, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float23_t*, uint, InterpMode, BorderMode, Stream&); \
    template void apply2D<false, true, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float23_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void apply2D<false, false, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float33_t*, uint, InterpMode, BorderMode, Stream&); \
    template void apply2D<false, true, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, const float33_t*, uint, InterpMode, BorderMode, Stream&);  \
                                                                                                                                                                \
    template void apply2D<true, false, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float23_t, InterpMode, BorderMode, Stream&);  \
    template void apply2D<true, true, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float23_t, InterpMode, BorderMode, Stream&);   \
    template void apply2D<true, false, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float33_t, InterpMode, BorderMode, Stream&);  \
    template void apply2D<true, true, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float33_t, InterpMode, BorderMode, Stream&);   \
    template void apply2D<false, false, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float23_t, InterpMode, BorderMode, Stream&); \
    template void apply2D<false, true, T, float23_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float23_t, InterpMode, BorderMode, Stream&);  \
    template void apply2D<false, false, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float33_t, InterpMode, BorderMode, Stream&); \
    template void apply2D<false, true, T, float33_t>(const T*, size_t, size2_t, T*, size_t, size2_t, float33_t, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_APPLY_2D_(float);
    NOA_INSTANTIATE_APPLY_2D_(cfloat_t);
}
