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
    shift2D_(cudaTextureObject_t texture, float2_t texture_shape,
             Accessor<T, 3, uint32_t> output, uint2_t output_shape,
             const float2_t* shifts) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        float2_t pos{gid[1], gid[2]};
        pos -= shifts[gid[0]];
        pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output(gid) = cuda::geometry::tex2D<T, MODE>(texture, pos);
    }

    template<InterpMode MODE, bool NORMALIZED, typename T>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    shift2D_single_(cudaTextureObject_t texture, float2_t texture_shape,
                    Accessor<T, 3, uint32_t> output, uint2_t output_shape,
                    float2_t shift) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        float2_t pos{gid[1], gid[2]};
        pos -= shift;
        pos += 0.5f;
        if constexpr (NORMALIZED)
            pos /= texture_shape;
        else
            (void) texture_shape;

        output(gid) = cuda::geometry::tex2D<T, MODE>(texture, pos);
    }

    template<typename T>
    void launchShiftTexture2D_(cudaTextureObject_t texture, dim2_t texture_shape,
                               InterpMode texture_interp_mode, BorderMode texture_border_mode,
                               T* output, dim4_t output_strides, dim4_t output_shape,
                               const float2_t* shifts, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const float2_t i_shape(texture_shape);
        const auto o_shape = safe_cast<uint2_t>(dim2_t(output_shape.get(2)));
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<T, 3, uint32_t> output_accessor(output, o_strides);

        cuda::memory::PtrDevice<float2_t> buffer;
        shifts = cuda::util::ensureDeviceAccess(shifts, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output_accessor, o_shape, shifts);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output_accessor, o_shape, shifts);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shifts);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    template<typename T>
    void launchShiftTextureSingle2D_(cudaTextureObject_t texture, dim2_t texture_shape,
                                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                     T* output, dim4_t output_strides, dim4_t output_shape,
                                     float2_t shift, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const float2_t i_shape(texture_shape);
        const auto o_shape = safe_cast<uint2_t>(dim2_t(output_shape.get(2)));
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<T, 3, uint32_t> output_accessor(output, o_strides);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            NOA_ASSERT(cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            if (texture_interp_mode == INTERP_NEAREST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_single_<INTERP_NEAREST, true, T>,
                               config, texture, i_shape, output_accessor, o_shape, shift);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                stream.enqueue("geometry::shift2D",
                               shift2D_single_<INTERP_LINEAR_FAST, true, T>,
                               config, texture, i_shape, output_accessor, o_shape, shift);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));
            switch (texture_interp_mode) {
                case INTERP_NEAREST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_NEAREST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_LINEAR:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_LINEAR, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_COSINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_COSINE, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_CUBIC:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_CUBIC, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_CUBIC_BSPLINE:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_CUBIC_BSPLINE, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_LINEAR_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_LINEAR_FAST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_COSINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_COSINE_FAST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                case INTERP_CUBIC_BSPLINE_FAST:
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_single_<INTERP_CUBIC_BSPLINE_FAST, false, T>,
                                          config, texture, i_shape, output_accessor, o_shape, shift);
                default:
                    NOA_THROW("{} is not supported", texture_interp_mode);
            }
        }
    }

    // NOTE: almost identical to launchTransform2D_
    template<typename T, typename U>
    void launchShift2D_(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                        const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                        U shifts, InterpMode interp_mode, BorderMode border_mode, bool prefilter,
                        cuda::Stream& stream) {
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        if (input_strides[0] == 0)
            input_shape[0] = 1;

        // Prepare the input array:
        cuda::memory::PtrDevice<T> buffer;
        const T* buffer_ptr;
        dim_t buffer_pitch;
        dim_t buffer_offset;
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[2] != output_shape[2] || input_shape[3] != output_shape[3]) {
                buffer = cuda::memory::PtrDevice<T>(input_shape.elements(), stream);
                const dim4_t contiguous_strides = input_shape.strides();
                cuda::geometry::bspline::prefilter(input, input_strides,
                                                   buffer.share(), contiguous_strides, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_strides[2];
                buffer_offset = contiguous_strides[0];
            } else {
                NOA_ASSERT(indexing::isContiguous(output_strides, output_shape)[3]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                cuda::geometry::bspline::prefilter(input, input_strides, output, output_strides, input_shape, stream);
                buffer_ptr = output.get();
                buffer_pitch = output_strides[2];
                buffer_offset = output_strides[0];
            }
        } else {
            NOA_ASSERT(indexing::isContiguous(input_strides, input_shape)[3]);
            buffer_ptr = input.get();
            buffer_pitch = input_strides[2];
            buffer_offset = input_strides[0];
        }

        // Broadcast input if it is not batched:
        const dim4_t o_shape{input_shape[0] > 1 ? 1 : output_shape[0],
                             output_shape[1], output_shape[2], output_shape[3]};

        // Copy to texture and launch (per input batch):
        const dim3_t shape_3d{1, input_shape[2], input_shape[3]};
        cuda::memory::PtrArray<T> array(shape_3d);
        cuda::memory::PtrTexture texture(array.get(), interp_mode, border_mode);
        for (dim_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            if constexpr (traits::is_floatX_v<U>) {
                launchShiftTextureSingle2D_(
                        texture.get(), dim2_t(input_shape.get(2)), interp_mode, border_mode,
                        output.get() + i * output_strides[0], output_strides, o_shape, shifts, stream);
            } else {
                NOA_ASSERT(shifts);
                launchShiftTexture2D_(
                        texture.get(), dim2_t(input_shape.get(2)), interp_mode, border_mode,
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
    void shift2D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, InterpMode interp_mode, BorderMode border_mode, bool prefilter,
                 Stream& stream) {
        NOA_ASSERT(all(input_shape > 0) && all(output_shape > 0) && input);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        launchShift2D_(input, input_strides, input_shape, output, output_strides, output_shape,
                       shifts, interp_mode, border_mode, prefilter, stream);
    }

    template<typename T, typename S, typename>
    void shift2D(const shared_t<cudaArray>& array,
                 const shared_t<cudaTextureObject_t>& texture, dim2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, Stream& stream) {
        NOA_ASSERT(all(texture_shape > 0) && all(output_shape > 0) && array && texture);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        if constexpr (traits::is_floatX_v<S>) {
            launchShiftTextureSingle2D_(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output.get(), output_strides, output_shape, shifts, stream);
            stream.attach(array, texture, output);
        } else {
            NOA_ASSERT(shifts);
            launchShiftTexture2D_(
                    *texture, texture_shape, texture_interp_mode, texture_border_mode,
                    output.get(), output_strides, output_shape, shifts.get(), stream);
            stream.attach(array, texture, output, shifts);
        }
    }

    #define NOA_INSTANTIATE_SHIFT_3D_VECTOR_(T, S)                                                                                                                                                                          \
    template void shift2D<T, shared_t<S[]>, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<S[]>&, InterpMode, BorderMode, bool, Stream&);                                 \
    template void shift2D<T, S, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const S&, InterpMode, BorderMode, bool, Stream&);                                                         \
    template void shift2D<T, shared_t<S[]>, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim2_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<S[]>&, Stream&);   \
    template void shift2D<T, S, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim2_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const S&, Stream&)

    #define NOA_INSTANTIATE_SHIFT_3D_(T)           \
    NOA_INSTANTIATE_SHIFT_3D_VECTOR_(T, float2_t); \

    NOA_INSTANTIATE_SHIFT_3D_(float);
    NOA_INSTANTIATE_SHIFT_3D_(cfloat_t);
}
