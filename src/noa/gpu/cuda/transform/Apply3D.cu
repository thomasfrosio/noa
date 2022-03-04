#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Pointers.h"

#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include "noa/gpu/cuda/transform/Apply.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Prefilter.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    // 3D, batched
    template<InterpMode MODE, bool NORMALIZED, typename T, typename MATRIX>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    transform3D_(cudaTextureObject_t texture, float3_t texture_shape,
                 T* output, uint4_t output_stride, uint2_t output_shape,
                 const MATRIX* matrices, uint blocks_x) { // 3x4 or 4x4 matrices
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= output_shape[0] || gid[3] >= output_shape[1])
            return;

        const float4_t pos(gid[1], gid[2], gid[3], 1.f);
        const float34_t matrix{matrices[gid[0]]};
        float3_t coordinates = matrix * pos;
        coordinates += 0.5f;
        if constexpr (NORMALIZED)
            coordinates /= texture_shape;
        else
            (void) texture_shape;

        output[at(gid, output_stride)] = cuda::geometry::tex3D<T, MODE>(texture, coordinates);
    }

    // 3D, single
    template<InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    transform3D_single_(cudaTextureObject_t texture, [[maybe_unused]] float3_t texture_shape,
                        T* output, uint4_t output_stride, uint2_t output_shape,
                        float34_t matrix, uint blocks_x) {
        const uint2_t index = indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= output_shape[0] || gid[3] >= output_shape[1])
            return;

        const float4_t pos(gid[1], gid[2], gid[3], 1.f);
        float3_t coordinates = matrix * pos;
        coordinates += 0.5f;
        if constexpr (NORMALIZED)
            coordinates /= texture_shape;
        else
            (void) texture_shape;

        output[at(gid, output_stride)] = cuda::geometry::tex3D<T, MODE>(texture, coordinates);
    }

    template<bool PREFILTER, typename T, typename MAT, typename = void>
    void launchTransform3D_(const T* input, size4_t input_stride, size4_t input_shape,
                            T* output, size4_t output_stride, size4_t output_shape,
                            MAT matrices, InterpMode interp_mode, BorderMode border_mode,
                            cuda::Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] > 1);

        if (input_stride[0] == 0)
            input_shape[0] = 1;

        // Prepare the input array:
        cuda::memory::PtrDevice<T> buffer;
        const T* buffer_ptr;
        size_t buffer_pitch;
        size_t buffer_offset;
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[1] != output_shape[1] ||
                input_shape[2] != output_shape[2] ||
                input_shape[3] != output_shape[3]) {
                buffer.reset(input_shape.elements(), stream);
                const size4_t contiguous_stride = input_shape.strides();
                cuda::geometry::bspline::prefilter(input, input_stride,
                                                   buffer.get(), contiguous_stride, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_stride[2];
                buffer_offset = contiguous_stride[0];
            } else {
                NOA_ASSERT(isContiguous(output_stride, output_shape)[3]);
                NOA_ASSERT(isContiguous(output_stride, output_shape)[1]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                cuda::geometry::bspline::prefilter(input, input_stride, output, output_stride, input_shape, stream);
                buffer_ptr = output;
                buffer_pitch = output_stride[2];
                buffer_offset = output_stride[0];
            }
        } else {
            NOA_ASSERT(isContiguous(input_stride, input_shape)[3]);
            NOA_ASSERT(isContiguous(input_stride, input_shape)[1]);
            buffer_ptr = input;
            buffer_pitch = input_stride[2];
            buffer_offset = input_stride[0];
        }

        // Broadcast input if it is not batched:
        size4_t o_shape{input_shape[0] > 1 ? 1 : output_shape[0],
                        output_shape[1], output_shape[2],output_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{input_shape.get() + 1};
        cuda::memory::PtrArray<T> i_array(shape_3d);
        cuda::memory::PtrTexture<T> i_texture;
        for (size_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, i_array.get(), shape_3d, stream);
            i_texture.reset(i_array.get(), interp_mode, border_mode); // no need to wait here
            cuda::geometry::transform3D(
                    i_texture.get(), shape_3d, interp_mode, border_mode,
                    output + i * output_stride[0], output_stride, o_shape, matrices, stream);
            stream.synchronize();
        }
    }
}

namespace noa::cuda::geometry {
    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform3D(const T* input, size4_t input_stride, size4_t input_shape,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const MAT* matrices, InterpMode interp_mode, BorderMode border_mode,
                     Stream& stream) {
        launchTransform3D_<PREFILTER>(
                input, input_stride, input_shape, output, output_stride, output_shape,
                matrices, interp_mode, border_mode, stream);
    }

    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform3D(const T* input, size4_t input_stride, size4_t input_shape,
                     T* output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, InterpMode interp_mode, BorderMode border_mode,
                     Stream& stream) {
        launchTransform3D_<PREFILTER>(
                input, input_stride, input_shape, output, output_stride, output_shape,
                matrix, interp_mode, border_mode, stream);
    }

    template<typename T, typename MAT, typename>
    void transform3D(cudaTextureObject_t texture, size3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const MAT* matrices, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const float3_t i_shape{texture_shape};
        const uint2_t o_shape{output_shape.get() + 2};
        const uint4_t o_stride{output_stride};
        const uint blocks_x = math::divideUp(o_shape[1], THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape[0], THREADS.y),
                          output_shape[1],
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        memory::PtrDevice<MAT> buffer;
        matrices = util::ensureDeviceAccess(matrices, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::transform3D",
                               transform3D_<INTERP_NEAREST, true, T, MAT>,
                               config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::transform3D",
                               transform3D_<INTERP_LINEAR_FAST, true, T, MAT>,
                               config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_NEAREST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_LINEAR, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_COSINE, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_CUBIC, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_CUBIC_BSPLINE, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_LINEAR_FAST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_COSINE_FAST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_<INTERP_CUBIC_BSPLINE_FAST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices, blocks_x);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    template<typename T, typename MAT, typename>
    void transform3D(cudaTextureObject_t texture, size3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const float3_t i_shape{texture_shape};
        const uint2_t o_shape{output_shape.get() + 2};
        const uint4_t o_stride{output_stride};
        const uint blocks_x = math::divideUp(o_shape[1], THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape[0], THREADS.y),
                          output_shape[1],
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const float34_t matrix_{matrix};

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::transform3D",
                               transform3D_single_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::transform3D",
                               transform3D_single_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::transform3D",
                                          transform3D_single_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_, blocks_x);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM_3D_(T)                                                                                                                    \
    template void transform3D<true, T, float34_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const float34_t*, InterpMode, BorderMode, Stream&);   \
    template void transform3D<true, T, float44_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const float44_t*, InterpMode, BorderMode, Stream&);   \
    template void transform3D<false, T, float34_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const float34_t*, InterpMode, BorderMode, Stream&);  \
    template void transform3D<false, T, float44_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const float44_t*, InterpMode, BorderMode, Stream&);  \
    template void transform3D<true, T, float34_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, float34_t, InterpMode, BorderMode, Stream&);          \
    template void transform3D<true, T, float44_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, float44_t, InterpMode, BorderMode, Stream&);          \
    template void transform3D<false, T, float34_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, float34_t, InterpMode, BorderMode, Stream&);         \
    template void transform3D<false, T, float44_t, void>(const T*, size4_t, size4_t, T*, size4_t, size4_t, float44_t, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_TRANSFORM_3D_(float);
    NOA_INSTANTIATE_TRANSFORM_3D_(cfloat_t);
}
