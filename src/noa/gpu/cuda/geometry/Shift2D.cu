#include "noa/common/Assert.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"

#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/geometry/Shift.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename data_t, typename interpolator_t, typename shift_t>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    shift2D_(interpolator_t interpolator,
             Accessor<data_t, 3, uint32_t> output, uint2_t output_shape,
             shift_t shifts) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x};
        if (gid[1] >= output_shape[0] || gid[2] >= output_shape[1])
            return;

        float2_t pos{gid[1], gid[2]};

        if constexpr (std::is_same_v<shift_t, const float2_t*>) {
            pos -= shifts[gid[0]];
            output(gid) = interpolator(pos);
        } else if constexpr (std::is_same_v<shift_t, float2_t>) {
            pos -= shifts;
            output(gid) = interpolator(pos);
        } else {
            static_assert(traits::always_false_v<data_t>);
        }
    }

    template<typename data_t, typename shift_t>
    void launchShiftTexture2D_(cudaTextureObject_t texture, dim2_t texture_shape,
                               InterpMode texture_interp_mode, BorderMode texture_border_mode,
                               data_t* output, dim4_t output_strides, dim4_t output_shape,
                               shift_t shifts, cuda::Stream& stream) {
        NOA_ASSERT(output_shape[1] == 1);
        const auto o_shape = safe_cast<uint2_t>(dim2_t(output_shape.get(2)));
        const auto o_strides = safe_cast<uint3_t>(dim3_t{output_strides[0], output_strides[2], output_strides[3]});
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<data_t, 3, uint32_t> output_accessor(output, o_strides);

        // Copy matrices to device if not available yet.
        cuda::memory::PtrDevice<float2_t> buffer;
        if constexpr (std::is_pointer_v<shift_t>)
            shifts = cuda::util::ensureDeviceAccess(shifts, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            const float2_t i_shape(texture_shape);

            if (texture_interp_mode == INTERP_NEAREST) {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t, true>;
                stream.enqueue("geometry::shift2D",
                               shift2D_<data_t, interpolator_t, shift_t>, config,
                               interpolator_t(texture, i_shape), output_accessor, o_shape, shifts);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t, true>;
                stream.enqueue("geometry::shift2D",
                               shift2D_<data_t, interpolator_t, shift_t>, config,
                               interpolator_t(texture, i_shape), output_accessor, o_shape, shifts);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case INTERP_NEAREST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_NEAREST, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_LINEAR: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_COSINE: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_CUBIC: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_CUBIC_BSPLINE: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_LINEAR_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_LINEAR_FAST, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_COSINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_COSINE_FAST, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
                case INTERP_CUBIC_BSPLINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator2D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                    return stream.enqueue("geometry::shift2D",
                                          shift2D_<data_t, interpolator_t, shift_t>, config,
                                          interpolator_t(texture), output_accessor, o_shape, shifts);
                }
            }
        }
    }

    template<typename T>
    auto shiftOrRawConstPtr(const T& v, size_t index = 0) {
        if constexpr (traits::is_float2_v<T>) {
            return float2_t(v);
        } else {
            NOA_ASSERT(v != nullptr);
            using clean_t = traits::remove_ref_cv_t<T>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(v.get() + index);
        }
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
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);
        NOA_ASSERT(input_shape[1] == 1 && output_shape[1] == 1);

        if (input_strides[0] == 0)
            input_shape[0] = 1;
        else if (input_shape[0] == 1)
            input_strides[0] = 0;

        // Prepare the input array:
        shared_t<T[]> buffer;
        const T* buffer_ptr;
        dim_t buffer_pitch;
        dim_t buffer_offset;
        if (prefilter && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            if (input_shape[2] != output_shape[2] || input_shape[3] != output_shape[3]) {
                buffer = memory::PtrDevice<T>::alloc(input_shape.elements(), stream);
                const dim4_t contiguous_strides = input_shape.strides();
                bspline::prefilter(input, input_strides, buffer, contiguous_strides, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_strides[2];
                buffer_offset = contiguous_strides[0];
            } else {
                NOA_ASSERT(indexing::isContiguous(output_strides, output_shape)[3]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                bspline::prefilter(input, input_strides, output, output_strides, input_shape, stream);
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
            launchShiftTexture2D_(
                    texture.get(), dim2_t(input_shape.get(2)), interp_mode, border_mode,
                    output.get() + i * output_strides[0], output_strides, o_shape,
                    shiftOrRawConstPtr(shifts, i), stream);
        }
        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_floatX_v<S>)
            stream.attach(shifts);
    }

    template<typename T, typename S, typename>
    void shift2D(const shared_t<cudaArray>& array,
                 const shared_t<cudaTextureObject_t>& texture, dim2_t texture_shape,
                 InterpMode texture_interp_mode, BorderMode texture_border_mode,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                 const S& shifts, Stream& stream) {
        NOA_ASSERT(all(texture_shape > 0) && all(output_shape > 0) && array && texture);
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        launchShiftTexture2D_(
                *texture, texture_shape, texture_interp_mode, texture_border_mode,
                output.get(), output_strides, output_shape, shiftOrRawConstPtr(shifts, 0), stream);

        if constexpr (traits::is_floatX_v<S>)
            stream.attach(array, texture, output);
        else
            stream.attach(array, texture, output, shifts);
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
