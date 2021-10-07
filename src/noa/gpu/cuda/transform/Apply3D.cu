#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

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

    // 3D, batched
    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T, typename MATRIX>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    apply3D_(cudaTextureObject_t texture, float3_t texture_shape,
             T* outputs, uint output_pitch, uint3_t output_shape,
             const MATRIX* affine, uint blocks_x) { // 3x4 or 4x4 matrices
        const uint rotation_id = blockIdx.z;
        const uint block_y = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_y * blocks_x;
        const uint3_t gid(block_x * THREADS.x + threadIdx.x,
                          block_y * THREADS.y + threadIdx.y,
                          blockIdx.y);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y) // cannot be out of bound in z
            return;

        float4_t pos(gid.x, gid.y, gid.z, 1.f);
        float3_t coordinates(math::dot(affine[rotation_id][0], pos),
                             math::dot(affine[rotation_id][1], pos),
                             math::dot(affine[rotation_id][2], pos)); // 3x4 * 4x1 matrix-vector multiplication
        if constexpr (TEXTURE_OFFSET)
            coordinates += 0.5f;
        if constexpr (NORMALIZED)
            coordinates /= texture_shape;

        outputs += rotation_id * output_shape.y * output_pitch;
        outputs[(gid.z * output_shape.y + gid.y) * output_pitch + gid.x] =
                cuda::transform::tex3D<T, MODE>(texture, coordinates);
    }

    // 3D, single
    template<bool TEXTURE_OFFSET, InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    apply3D_(cudaTextureObject_t texture, float3_t texture_shape,
             T* output, uint output_pitch, uint3_t output_shape,
             float34_t affine) {
        const uint3_t gid(blockIdx.x * THREADS.x + threadIdx.x,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= output_shape.x || gid.y >= output_shape.y)
            return;

        float4_t pos(gid.x, gid.y, gid.z, 1.f);
        float3_t coordinates(affine * pos);
        if constexpr (TEXTURE_OFFSET)
            coordinates += 0.5f;
        if constexpr (NORMALIZED)
            coordinates /= texture_shape;

        output[(gid.z * output_shape.y + gid.y) * output_pitch + gid.x] =
                cuda::transform::tex3D<T, MODE>(texture, coordinates);
    }
}

// -- Using textures -- //
namespace noa::cuda::transform {
    template<bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply3D(cudaTextureObject_t texture, size3_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* outputs, size_t output_pitch, size3_t output_shape,
                 const MATRIX* transforms, uint nb_transforms, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const float3_t i_shape(texture_shape);
        const uint3_t o_shape(output_shape);
        const uint blocks_x = math::divideUp(o_shape.x, THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape.y, THREADS.y),
                          o_shape.z,
                          nb_transforms);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                apply3D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                apply3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    apply3D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_LINEAR:
                    apply3D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_COSINE:
                    apply3D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_CUBIC:
                    apply3D_<TEXTURE_OFFSET, INTERP_CUBIC, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    apply3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_LINEAR_FAST:
                    apply3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_COSINE_FAST:
                    apply3D_<TEXTURE_OFFSET, INTERP_COSINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                case INTERP_CUBIC_BSPLINE_FAST:
                    apply3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, outputs, output_pitch, o_shape, transforms, blocks_x);
                    break;
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply3D(cudaTextureObject_t texture, size3_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size_t output_pitch, size3_t output_shape,
                 MATRIX transform, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const float34_t affine(transform);
        const float3_t i_shape(texture_shape);
        const uint3_t o_shape(output_shape);
        const dim3 blocks(math::divideUp(o_shape.x, THREADS.x),
                          math::divideUp(o_shape.y, THREADS.y),
                          o_shape.z);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                apply3D_<TEXTURE_OFFSET, INTERP_NEAREST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, affine);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                apply3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, true><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, i_shape, output, output_pitch, o_shape, affine);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    apply3D_<TEXTURE_OFFSET, INTERP_NEAREST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_LINEAR:
                    apply3D_<TEXTURE_OFFSET, INTERP_LINEAR, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_COSINE:
                    apply3D_<TEXTURE_OFFSET, INTERP_COSINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_CUBIC:
                    apply3D_<TEXTURE_OFFSET, INTERP_CUBIC, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_CUBIC_BSPLINE:
                    apply3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_LINEAR_FAST:
                    apply3D_<TEXTURE_OFFSET, INTERP_LINEAR_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_COSINE_FAST:
                    apply3D_<TEXTURE_OFFSET, INTERP_COSINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
                            texture, i_shape, output, output_pitch, o_shape, affine);
                    break;
                case INTERP_CUBIC_BSPLINE_FAST:
                    apply3D_<TEXTURE_OFFSET, INTERP_CUBIC_BSPLINE_FAST, false><<<blocks, THREADS, 0, stream.id()>>>(
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
    void apply3D(const T* input, size_t input_pitch, size3_t input_shape,
                 T* outputs, size_t output_pitch, size3_t output_shape,
                 const MATRIX* transforms, uint nb_transforms,
                 InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrDevicePadded<T> tmp; // or PtrDevice?
        memory::PtrArray<T> i_array(input_shape);
        memory::PtrTexture<T> i_texture;

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

        apply3D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                outputs, output_pitch, output_shape, transforms, nb_transforms, stream);
        tmp.dispose();
        stream.synchronize();
    }

    template<bool PREFILTER, bool TEXTURE_OFFSET, typename T, typename MATRIX>
    void apply3D(const T* input, size_t input_pitch, size3_t input_shape,
                 T* output, size_t output_pitch, size3_t output_shape,
                 MATRIX transform, InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        memory::PtrDevicePadded<T> tmp;
        memory::PtrArray<T> i_array(input_shape);
        memory::PtrTexture<T> i_texture;

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

        apply3D<TEXTURE_OFFSET>(i_texture.get(), input_shape, interp_mode, border_mode,
                                output, output_pitch, output_shape, transform, stream);
        tmp.dispose();
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_APPLY_3D_(T)                                                                                                                        \
    template void apply3D<true, false, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float34_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void apply3D<true, true, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float34_t*, uint, InterpMode, BorderMode, Stream&);   \
    template void apply3D<true, false, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float44_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void apply3D<true, true, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float44_t*, uint, InterpMode, BorderMode, Stream&);   \
    template void apply3D<false, false, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float34_t*, uint, InterpMode, BorderMode, Stream&); \
    template void apply3D<false, true, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float34_t*, uint, InterpMode, BorderMode, Stream&);  \
    template void apply3D<false, false, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float44_t*, uint, InterpMode, BorderMode, Stream&); \
    template void apply3D<false, true, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, const float44_t*, uint, InterpMode, BorderMode, Stream&);  \
                                                                                                                                                                \
    template void apply3D<true, false, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float34_t, InterpMode, BorderMode, Stream&);  \
    template void apply3D<true, true, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float34_t, InterpMode, BorderMode, Stream&);   \
    template void apply3D<true, false, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float44_t, InterpMode, BorderMode, Stream&);  \
    template void apply3D<true, true, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float44_t, InterpMode, BorderMode, Stream&);   \
    template void apply3D<false, false, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float34_t, InterpMode, BorderMode, Stream&); \
    template void apply3D<false, true, T, float34_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float34_t, InterpMode, BorderMode, Stream&);  \
    template void apply3D<false, false, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float44_t, InterpMode, BorderMode, Stream&); \
    template void apply3D<false, true, T, float44_t>(const T*, size_t, size3_t, T*, size_t, size3_t, float44_t, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_APPLY_3D_(float);
    NOA_INSTANTIATE_APPLY_3D_(cfloat_t);
}
