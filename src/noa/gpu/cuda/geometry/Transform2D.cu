#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Pointers.h"

#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include "noa/gpu/cuda/geometry/Transform.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    // 2D, batched
    template<InterpMode MODE, bool NORMALIZED, typename T, typename MATRIX>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    transform2D_(cudaTextureObject_t texture, float2_t texture_shape,
                 T* output, uint3_t output_stride, uint2_t output_shape,
                 const MATRIX* matrices) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        const float3_t pos{gid[1], gid[2], 1.f};
        const float23_t matrix{matrices[gid[0]]};
        float2_t coordinates = matrix * pos;
        coordinates += 0.5f;
        if constexpr (NORMALIZED)
            coordinates /= texture_shape;
        else
            (void) texture_shape;

        output[indexing::at(gid, output_stride)] = cuda::geometry::tex2D<T, MODE>(texture, coordinates);
    }

    // 2D, single
    template<InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    transform2D_single_(cudaTextureObject_t texture, float2_t texture_shape,
                        T* output, uint3_t output_stride, uint2_t output_shape,
                        float23_t matrix) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        const float3_t pos{gid[1], gid[2], 1.f};
        float2_t coordinates = matrix * pos;
        coordinates += 0.5f;
        if constexpr (NORMALIZED)
            coordinates /= texture_shape;
        else
            (void) texture_shape;

        output[indexing::at(gid, output_stride)] = cuda::geometry::tex2D<T, MODE>(texture, coordinates);
    }

    template<bool PREFILTER, typename T, typename U, typename = void>
    void launchTransform2D_(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                            const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                            U matrices, InterpMode interp_mode, BorderMode border_mode,
                            cuda::Stream& stream) {
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        if (input_stride[0] == 0)
            input_shape[0] = 1;

        // Prepare the input array:
        cuda::memory::PtrDevice<T> buffer;
        const T* buffer_ptr;
        size_t buffer_pitch;
        size_t buffer_offset;
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[2] != output_shape[2] || input_shape[3] != output_shape[3]) {
                buffer = cuda::memory::PtrDevice<T>{input_shape.elements(), stream};
                const size4_t contiguous_stride = input_shape.strides();
                cuda::geometry::bspline::prefilter(input, input_stride,
                                                   buffer.share(), contiguous_stride, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_stride[2];
                buffer_offset = contiguous_stride[0];
            } else {
                NOA_ASSERT(indexing::isContiguous(output_stride, output_shape)[3]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                cuda::geometry::bspline::prefilter(input, input_stride, output, output_stride, input_shape, stream);
                buffer_ptr = output.get();
                buffer_pitch = output_stride[2];
                buffer_offset = output_stride[0];
            }
        } else {
            NOA_ASSERT(indexing::isContiguous(input_stride, input_shape)[3]);
            buffer_ptr = input.get();
            buffer_pitch = input_stride[2];
            buffer_offset = input_stride[0];
        }

        // Broadcast input if it is not batched:
        const size4_t o_shape{input_shape[0] > 1 ? 1 : output_shape[0],
                              output_shape[1], output_shape[2], output_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{1, input_shape[2], input_shape[3]};
        cuda::memory::PtrArray<T> array{shape_3d};
        cuda::memory::PtrTexture texture{array.get(), interp_mode, border_mode};
        for (size_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            if constexpr (noa::traits::is_floatXX_v<U>) {
                cuda::geometry::transform2D(
                        texture.get(), size2_t{input_shape.get() + 2}, interp_mode, border_mode,
                        output.get() + i * output_stride[0], output_stride, o_shape,
                        matrices, stream);
            } else {
                cuda::geometry::transform2D(
                        texture.get(), size2_t{input_shape.get() + 2}, interp_mode, border_mode,
                        output.get() + i * output_stride[0], output_stride, o_shape,
                        matrices.get() + i, stream);
            }
        }
        stream.attach(input, output, array.share(), texture.share());
        if (!buffer.empty())
            stream.attach(buffer.share());
        if constexpr (!noa::traits::is_floatXX_v<U>)
            stream.attach(matrices);
    }
}

namespace noa::cuda::geometry {
    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     const shared_t<MAT[]>& matrices,
                     InterpMode interp_mode, BorderMode border_mode, Stream& stream) {
        launchTransform2D_<PREFILTER>(
                input, input_stride, input_shape, output, output_stride, output_shape,
                matrices, interp_mode, border_mode, stream);
    }

    template<bool PREFILTER, typename T, typename MAT, typename>
    void transform2D(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, InterpMode interp_mode, BorderMode border_mode,
                     Stream& stream) {
        launchTransform2D_<PREFILTER>(
                input, input_stride, input_shape, output, output_stride, output_shape,
                matrix, interp_mode, border_mode, stream);
    }

    template<typename T, typename MAT, typename>
    void transform2D(cudaTextureObject_t texture, size2_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     const MAT* matrices, Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const float2_t i_shape{texture_shape};
        const uint2_t o_shape{output_shape.get() + 2};
        const uint3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        memory::PtrDevice<MAT> buffer;
        matrices = util::ensureDeviceAccess(matrices, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::transform2D",
                               transform2D_<INTERP_NEAREST, true, T, MAT>,
                               config, texture, i_shape, output, o_stride, o_shape, matrices);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::transform2D",
                               transform2D_<INTERP_LINEAR_FAST, true, T, MAT>,
                               config, texture, i_shape, output, o_stride, o_shape, matrices);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_NEAREST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_LINEAR, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_COSINE, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_CUBIC, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_CUBIC_BSPLINE, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_LINEAR_FAST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_COSINE_FAST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_<INTERP_CUBIC_BSPLINE_FAST, false, T, MAT>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrices);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    template<typename T, typename MAT, typename>
    void transform2D(cudaTextureObject_t texture, size2_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     T* output, size4_t output_stride, size4_t output_shape,
                     MAT matrix, Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const float2_t i_shape{texture_shape};
        const uint2_t o_shape{output_shape.get() + 2};
        const uint3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        const float23_t matrix_{matrix};

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::transform2D",
                               transform2D_single_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, matrix_);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::transform2D",
                               transform2D_single_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, matrix_);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::transform2D",
                                          transform2D_single_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, matrix_);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM_2D_(T)                                                                                                                    \
    template void transform2D<true, T, float23_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float23_t[]>&, InterpMode, BorderMode, Stream&);   \
    template void transform2D<true, T, float33_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float33_t[]>&, InterpMode, BorderMode, Stream&);   \
    template void transform2D<false, T, float23_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float23_t[]>&, InterpMode, BorderMode, Stream&);  \
    template void transform2D<false, T, float33_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<float33_t[]>&, InterpMode, BorderMode, Stream&);  \
    template void transform2D<true, T, float23_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float23_t, InterpMode, BorderMode, Stream&);    \
    template void transform2D<true, T, float33_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float33_t, InterpMode, BorderMode, Stream&);    \
    template void transform2D<false, T, float23_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float23_t, InterpMode, BorderMode, Stream&);   \
    template void transform2D<false, T, float33_t, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float33_t, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_TRANSFORM_2D_(float);
    NOA_INSTANTIATE_TRANSFORM_2D_(cfloat_t);
}
