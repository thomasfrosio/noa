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
#include "noa/gpu/cuda/geometry/Interpolator.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename T, typename interpolator_t, typename matrix_t>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    transform3D_(interpolator_t interpolator,
                 Accessor<T, 4, uint32_t> output, uint2_t output_shape,
                 matrix_t matrices, uint32_t blocks_x) { // 3x4 or 4x4 matrices
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= output_shape[0] || gid[3] >= output_shape[1])
            return;

        const float4_t pos{gid[1], gid[2], gid[3], 1.f};

        if constexpr (traits::is_any_v<matrix_t, const float34_t*, const float44_t*>) {
            const float34_t matrix(matrices[gid[0]]);
            const float3_t coordinates = matrix * pos;
            output(gid) = interpolator(coordinates);
        } else if constexpr (std::is_same_v<matrix_t, float34_t>) {
            const float3_t coordinates = matrices * pos;
            output(gid) = interpolator(coordinates);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    template<typename data_t, typename matrix_t>
    void launchTransformTexture3D_(cudaTextureObject_t texture, dim3_t texture_shape,
                                   InterpMode texture_interp_mode, BorderMode texture_border_mode,
                                   data_t* output, dim4_t output_strides, dim4_t output_shape,
                                   matrix_t matrices, cuda::Stream& stream) {
        const auto o_shape = safe_cast<uint2_t>(dim2_t(output_shape.get(2)));
        const auto o_strides = safe_cast<uint4_t>(output_strides);
        const uint32_t blocks_x = math::divideUp(o_shape[1], THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape[0], THREADS.y),
                          output_shape[1],
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};
        const Accessor<data_t, 4, uint32_t> output_accessor(output, o_strides);

        // Copy matrices to device if not available yet.
        using value_t = std::remove_cv_t<std::remove_pointer_t<matrix_t>>;
        cuda::memory::PtrDevice<value_t> buffer;
        if constexpr (std::is_pointer_v<matrix_t>)
            matrices = cuda::util::ensureDeviceAccess(matrices, stream, buffer, output_shape[0]);

        if (texture_border_mode == BORDER_PERIODIC || texture_border_mode == BORDER_MIRROR) {
            const float3_t i_shape(texture_shape);
            if (texture_interp_mode == INTERP_NEAREST) {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t, true>;
                stream.enqueue("geometry::transform3D",
                               transform3D_<data_t, interpolator_t, matrix_t>, config,
                               interpolator_t(texture, i_shape), output_accessor, o_shape, matrices, blocks_x);
            } else if (texture_interp_mode == INTERP_LINEAR_FAST) {
                using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t, true>;
                stream.enqueue("geometry::transform3D",
                               transform3D_<data_t, interpolator_t, matrix_t>, config,
                               interpolator_t(texture, i_shape), output_accessor, o_shape, matrices, blocks_x);
            } else {
                NOA_THROW("{} is not supported with {}", texture_interp_mode, texture_border_mode);
            }
        } else {
            switch (texture_interp_mode) {
                case INTERP_NEAREST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_NEAREST, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_LINEAR: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_COSINE: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_CUBIC: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_CUBIC_BSPLINE: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_LINEAR_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_LINEAR_FAST, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_COSINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_COSINE_FAST, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
                case INTERP_CUBIC_BSPLINE_FAST: {
                    using interpolator_t = cuda::geometry::Interpolator3D<INTERP_CUBIC_BSPLINE_FAST, data_t>;
                    return stream.enqueue(
                            "geometry::transform3D",
                            transform3D_<data_t, interpolator_t, matrix_t>, config,
                            interpolator_t(texture), output_accessor, o_shape, matrices, blocks_x);
                }
            }
        }
    }

    template<typename T>
    auto matrixOrRawConstPtr(const T& v, size_t index = 0) {
        if constexpr (traits::is_float34_v<T> || traits::is_float44_v<T>) {
            return float34_t(v);
        } else {
            NOA_ASSERT(v != nullptr);
            using clean_t = traits::remove_ref_cv_t<T>;
            using raw_const_ptr_t = const typename clean_t::element_type*;
            return static_cast<raw_const_ptr_t>(v.get() + index);
        }
    }
}

namespace noa::cuda::geometry {
    template<typename T, typename MAT, typename>
    void transform3D(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, InterpMode interp_mode, BorderMode border_mode,
                     bool prefilter, Stream& stream) {
        NOA_ASSERT(input && all(input_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

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
            if (input_shape[1] != output_shape[1] ||
                input_shape[2] != output_shape[2] ||
                input_shape[3] != output_shape[3]) {
                buffer = memory::PtrDevice<T>::alloc(input_shape.elements(), stream);
                const dim4_t contiguous_strides = input_shape.strides();
                bspline::prefilter(input, input_strides, buffer, contiguous_strides, input_shape, stream);
                buffer_ptr = buffer.get();
                buffer_pitch = contiguous_strides[2];
                buffer_offset = contiguous_strides[0];
            } else {
                NOA_ASSERT(indexing::isContiguous(output_strides, output_shape)[3]);
                NOA_ASSERT(indexing::isContiguous(output_strides, output_shape)[1]);
                // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
                bspline::prefilter(input, input_strides, output, output_strides, input_shape, stream);
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
        const dim4_t o_shape{input_shape[0] > 1 ? 1 : output_shape[0],
                             output_shape[1], output_shape[2], output_shape[3]};

        // Copy to texture and launch (per input batch):
        const dim3_t shape_3d(input_shape.get(1));
        cuda::memory::PtrArray<T> array(shape_3d);
        cuda::memory::PtrTexture texture(array.get(), interp_mode, border_mode);
        for (dim_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            launchTransformTexture3D_(
                    texture.get(), shape_3d, interp_mode, border_mode,
                    output.get() + i * output_strides[0], output_strides, o_shape,
                    matrixOrRawConstPtr(matrices, i), stream);
        }
        stream.attach(input, output, array.share(), texture.share());
        if constexpr (!traits::is_floatXX_v<MAT>)
            stream.attach(matrices);
    }

    template<typename T, typename MAT, typename>
    void transform3D(const shared_t<cudaArray>& array,
                     const shared_t<cudaTextureObject_t>& texture, dim3_t texture_shape,
                     InterpMode texture_interp_mode, BorderMode texture_border_mode,
                     const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape,
                     const MAT& matrices, Stream& stream) {
        NOA_ASSERT(array && texture && all(texture_shape > 0) && all(output_shape > 0));
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        launchTransformTexture3D_(*texture, texture_shape, texture_interp_mode, texture_border_mode,
                                  output.get(), output_strides, output_shape,
                                  matrixOrRawConstPtr(matrices), stream);

        if constexpr (traits::is_floatXX_v<MAT>)
            stream.attach(array, texture, output);
        else
            stream.attach(array, texture, output, matrices);
    }

    #define NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, M)                                                                                                                                                                           \
    template void transform3D<T, shared_t<M[]>, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<M[]>&, InterpMode, BorderMode, bool, Stream&);                                 \
    template void transform3D<T, M, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, const M&, InterpMode, BorderMode, bool, Stream&);                                                         \
    template void transform3D<T, shared_t<M[]>, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim3_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<M[]>&, Stream&);   \
    template void transform3D<T, M, void>(const shared_t<cudaArray>&, const shared_t<cudaTextureObject_t>&, dim3_t, InterpMode, BorderMode, const shared_t<T[]>&, dim4_t, dim4_t, const M&, Stream&)

    #define NOA_INSTANTIATE_TRANSFORM_3D_(T)            \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, float34_t);  \
    NOA_INSTANTIATE_TRANSFORM_3D_MATRIX(T, float44_t)

    NOA_INSTANTIATE_TRANSFORM_3D_(float);
    NOA_INSTANTIATE_TRANSFORM_3D_(cfloat_t);
}
