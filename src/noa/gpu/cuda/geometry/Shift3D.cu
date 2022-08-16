#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/geometry/Shift.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    shift3D_(cudaTextureObject_t texture, float3_t texture_shape,
             T* output, uint4_t output_strides, uint2_t output_shape,
             const float3_t* shifts, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= output_shape[0] || gid[3] >= output_shape[1])
            return;

        float3_t pos(gid[1], gid[2], gid[3]);
        pos -= shifts[gid[0]];
        pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output[indexing::at(gid, output_strides)] = cuda::geometry::tex3D<T, MODE>(texture, pos);
    }

    template<InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    shift3D_single_(cudaTextureObject_t texture, float3_t texture_shape,
                    T* output, uint4_t output_strides, uint2_t output_shape,
                    float3_t shift, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= output_shape[0] || gid[3] >= output_shape[1])
            return;

        float3_t pos(gid[1], gid[2], gid[3]);
        pos -= shift;
        pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output[indexing::at(gid, output_strides)] = cuda::geometry::tex3D<T, MODE>(texture, pos);
    }

    template<typename T>
    void launchShiftTexture3D_(cudaTextureObject_t texture, size3_t texture_shape,
                               InterpMode texture_interp_mode, BorderMode texture_border_mode,
                               T* output, size4_t output_strides, size4_t output_shape,
                               const float3_t* shifts, cuda::Stream& stream) {
        const float3_t i_shape(texture_shape);
        const uint2_t o_shape(output_shape.get(2));
        const uint4_t o_strides(output_strides);
        const uint blocks_x = math::divideUp(o_shape[1], THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape[0], THREADS.y),
                          output_shape[1],
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        cuda::memory::PtrDevice<float3_t> buffer;
        shifts = cuda::util::ensureDeviceAccess(shifts, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::shift3D",
                               shift3D_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::shift3D",
                               shift3D_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shifts, blocks_x);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    template<typename T>
    void launchShiftTextureSingle3D_(cudaTextureObject_t texture, size3_t texture_shape,
                                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                     T* output, size4_t output_strides, size4_t output_shape,
                                     float3_t shift, cuda::Stream& stream) {
        const float3_t i_shape(texture_shape);
        const uint2_t o_shape(output_shape.get(2));
        const uint4_t o_strides(output_strides);
        const uint blocks_x = math::divideUp(o_shape[1], THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape[0], THREADS.y),
                          output_shape[1],
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::shift3D",
                               shift3D_single_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::shift3D",
                               shift3D_single_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::shift3D",
                                          shift3D_single_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_strides, o_shape, shift, blocks_x);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    // NOTE: almost identical to launchTransform3D_
    template<typename T, typename U>
    void launchShift3D_(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                        const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                        U shifts, InterpMode interp_mode, BorderMode border_mode, bool prefilter,
                        cuda::Stream& stream) {
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] > 1);

        if (input_strides[0] == 0)
            input_shape[0] = 1;

        // Prepare the input array:
        cuda::memory::PtrDevice<T> buffer;
        const T* buffer_ptr;
        size_t buffer_pitch;
        size_t buffer_offset;
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[1] != output_shape[1] ||
                input_shape[2] != output_shape[2] ||
                input_shape[3] != output_shape[3]) {
                buffer = cuda::memory::PtrDevice<T>(input_shape.elements(), stream);
                const size4_t contiguous_strides = input_shape.strides();
                cuda::geometry::bspline::prefilter(input, input_strides,
                                                   buffer.share(), contiguous_strides, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_strides[2];
                buffer_offset = contiguous_strides[0];
            } else {
                NOA_ASSERT(indexing::isContiguous(output_strides, output_shape)[3]);
                NOA_ASSERT(indexing::isContiguous(output_strides, output_shape)[1]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                cuda::geometry::bspline::prefilter(input, input_strides, output, output_strides, input_shape, stream);
                buffer_ptr = output.get();
                buffer_pitch = output_strides[2];
                buffer_offset = output_strides[0];
            }
        } else {
            NOA_ASSERT(indexing::isContiguous(input_strides, input_shape)[3]);
            NOA_ASSERT(indexing::isContiguous(input_strides, input_shape)[1]);
            buffer_ptr = input.get();
            buffer_pitch = input_strides[2];
            buffer_offset = input_strides[0];
        }

        // Broadcast input if it is not batched:
        const size4_t o_shape{input_shape[0] > 1 ? 1 : output_shape[0],
                              output_shape[1], output_shape[2], output_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d(input_shape.get(1));
        cuda::memory::PtrArray<T> array(shape_3d);
        cuda::memory::PtrTexture texture(array.get(), interp_mode, border_mode);
        for (size_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            if constexpr (traits::is_floatX_v<U>) {
                launchShiftTextureSingle3D_(
                        texture.get(), shape_3d, interp_mode, border_mode,
                        output.get() + i * output_strides[0], output_strides, o_shape, shifts, stream);
            } else {
                launchShiftTexture3D_(
                        texture.get(), shape_3d, interp_mode, border_mode,
                        output.get() + i * output_strides[0], output_strides, o_shape, shifts.get() + i, stream);
            }
        }
        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_floatX_v<U>)
            stream.attach(shifts);
    }
}

namespace noa::cuda::geometry {
    template<typename T, typename S, typename>
    void shift3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                 const S& shifts, InterpMode interp_mode, BorderMode border_mode, bool prefilter,
                 Stream& stream) {
        launchShift3D_(input, input_strides, input_shape, output, output_strides, output_shape,
                       shifts, interp_mode, border_mode, prefilter, stream);
    }

    template<typename T, typename S, typename>
    void shift3D(const shared_t<cudaTextureObject_t>& texture, size3_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                 const S& shifts, Stream& stream) {
        if constexpr (traits::is_floatX_v<S>) {
            launchShiftTextureSingle3D_(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output.get(), output_strides, output_shape, shifts, stream);
            stream.attach(texture, output);
        } else {
            launchShiftTexture3D_(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output.get(), output_strides, output_shape, shifts.get(), stream);
            stream.attach(texture, output, shifts);
        }
    }

    #define NOA_INSTANTIATE_SHIFT_3D_VECTOR_(T, S)                                                                                                                                              \
    template void shift3D<T, shared_t<S[]>, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const shared_t<S[]>&, InterpMode, BorderMode, bool, Stream&); \
    template void shift3D<T, S, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, const S&, InterpMode, BorderMode, bool, Stream&);                         \
    template void shift3D<T, shared_t<S[]>, void>(const shared_t<cudaTextureObject_t>&, size3_t, InterpMode, BorderMode, const shared_t<T[]>&, size4_t, size4_t, const shared_t<S[]>&, Stream&);\
    template void shift3D<T, S, void>(const shared_t<cudaTextureObject_t>&, size3_t, InterpMode, BorderMode, const shared_t<T[]>&, size4_t, size4_t, const S&, Stream&)

    #define NOA_INSTANTIATE_SHIFT_3D_(T)           \
    NOA_INSTANTIATE_SHIFT_3D_VECTOR_(T, float3_t); \

    NOA_INSTANTIATE_SHIFT_3D_(float);
    NOA_INSTANTIATE_SHIFT_3D_(cfloat_t);
}
