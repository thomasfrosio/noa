#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/Profiler.h"

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
    shift2D_(cudaTextureObject_t texture, float2_t texture_shape,
             T* output, uint3_t output_stride, uint2_t output_shape,
             const float2_t* shifts) {
        const uint3_t gid(blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x);
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        float2_t pos{gid[1], gid[2]};
        pos -= shifts[gid[0]];
        pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output[at(gid, output_stride)] = cuda::geometry::tex2D<T, MODE>(texture, pos);
    }

    template<InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    shift2D_single_(cudaTextureObject_t texture, float2_t texture_shape,
                    T* output, uint3_t output_stride, uint2_t output_shape,
                    float2_t shift) {
        const uint3_t gid(blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x);
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        float2_t pos{gid[1], gid[2]};
        pos -= shift;
        pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output[at(gid, output_stride)] = cuda::geometry::tex2D<T, MODE>(texture, pos);
    }

    // NOTE: almost identical to launchTransform2D_
    template<bool PREFILTER, typename T, typename VEC>
    void launchShift2D_(const T* input, size4_t input_stride, size4_t input_shape,
                        T* output, size4_t output_stride, size4_t output_shape,
                        VEC shifts, InterpMode interp_mode, BorderMode border_mode,
                        cuda::Stream& stream) {
        NOA_PROFILE_FUNCTION();
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
                buffer.reset(input_shape.elements(), stream);
                const size4_t contiguous_stride = input_shape.strides();
                cuda::geometry::bspline::prefilter(input, input_stride,
                                                   buffer.get(), contiguous_stride, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_stride[2];
                buffer_offset = contiguous_stride[0];
            } else {
                NOA_ASSERT(isContiguous(output_stride, output_shape)[3]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                cuda::geometry::bspline::prefilter(input, input_stride, output, output_stride, input_shape, stream);
                buffer_ptr = output;
                buffer_pitch = output_stride[2];
                buffer_offset = output_stride[0];
            }
        } else {
            NOA_ASSERT(isContiguous(input_stride, input_shape)[3]);
            buffer_ptr = input;
            buffer_pitch = input_stride[2];
            buffer_offset = input_stride[0];
        }

        // Broadcast input if it is not batched:
        size4_t o_shape{input_shape[0] > 1 ? 1 : output_shape[0],
                        output_shape[1], output_shape[2],output_shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{1, input_shape[2], input_shape[3]};
        cuda::memory::PtrArray<T> i_array(shape_3d);
        cuda::memory::PtrTexture<T> i_texture;
        for (size_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, i_array.get(), shape_3d, stream);
            i_texture.reset(i_array.get(), interp_mode, border_mode); // no need to wait here
            cuda::geometry::shift2D(
                    i_texture.get(), size2_t{input_shape.get() + 2}, interp_mode, border_mode,
                    output + i * output_stride[0], output_stride, o_shape, shifts, stream);
            stream.synchronize();
        }
    }
}

namespace noa::cuda::geometry {
    template<typename T>
    void shift2D(cudaTextureObject_t texture, size2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 const float2_t* shifts, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(output_shape[1] == 1);
        const float2_t i_shape{texture_shape};
        const uint2_t o_shape{output_shape.get() + 2};
        const uint3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        memory::PtrDevice<float2_t> buffer;
        shifts = util::ensureDeviceAccess(shifts, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, shifts);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, shifts);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shifts);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    template<typename T>
    void shift2D(cudaTextureObject_t texture, size2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 T* output, size4_t output_stride, size4_t output_shape,
                 float2_t shift, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(output_shape[1] == 1);
        const float2_t i_shape{texture_shape};
        const uint2_t o_shape{output_shape.get() + 2};
        const uint3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_single_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, shift);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_single_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output, o_stride, o_shape, shift);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!memory::PtrTexture<T>::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output, o_stride, o_shape, shift);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    template<bool PREFILTER, typename T>
    void shift2D(const T* input, size4_t input_stride, size4_t input_shape,
                 T* output, size4_t output_stride, size4_t output_shape,
                 const float2_t* shifts, InterpMode interp_mode, BorderMode border_mode,
                 Stream& stream) {
        launchShift2D_<PREFILTER>(
                input, input_stride, input_shape, output, output_stride, output_shape,
                shifts, interp_mode, border_mode, stream);
    }

    template<bool PREFILTER, typename T>
    void shift2D(const T* input, size4_t input_stride, size4_t input_shape,
                 T* output, size4_t output_stride, size4_t output_shape,
                 float2_t shift, InterpMode interp_mode, BorderMode border_mode,
                 Stream& stream) {
        launchShift2D_<PREFILTER>(
                input, input_stride, input_shape, output, output_stride, output_shape,
                shift, interp_mode, border_mode, stream);
    }

    #define NOA_INSTANTIATE_SHIFT_2D_(T)                                                                                                 \
    template void shift2D<false, T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const float2_t*, InterpMode, BorderMode, Stream&); \
    template void shift2D<true, T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, const float2_t*, InterpMode, BorderMode, Stream&);  \
    template void shift2D<false, T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, float2_t, InterpMode, BorderMode, Stream&);        \
    template void shift2D<true, T>(const T*, size4_t, size4_t, T*, size4_t, size4_t, float2_t, InterpMode, BorderMode, Stream&)

    NOA_INSTANTIATE_SHIFT_2D_(float);
    NOA_INSTANTIATE_SHIFT_2D_(cfloat_t);
}
