#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"

#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

#include "noa/gpu/cuda/geometry/Transform.h"
#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename T, InterpMode INTERP>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    transformWithSymmetry3D_(cudaTextureObject_t texture, T* output, uint4_t output_strides, uint2_t shape,
                             float3_t shift, float33_t matrix, float3_t center,
                             const float33_t* symmetry_matrices, uint symmetry_count, float scaling, uint blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          index[0] * THREADS.y + threadIdx.y,
                          index[1] * THREADS.x + threadIdx.x};
        if (gid[2] >= shape[0] || gid[3] >= shape[1])
            return;

        float3_t coordinates{gid[1], gid[2], gid[3]};
        coordinates -= center;
        coordinates = matrix * coordinates;
        T value = cuda::geometry::tex3D<T, INTERP>(texture, coordinates + center + shift + 0.5f);
        for (uint i = 0; i < symmetry_count; ++i) {
            const float3_t i_coordinates(symmetry_matrices[i] * coordinates);
            value += cuda::geometry::tex3D<T, INTERP>(texture, i_coordinates + center + shift + 0.5f);
        }

        output[indexing::at(gid, output_strides)] = value * scaling;
    }

    template<typename T>
    void launchTransformSymmetryTexture3D_(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                                           T* output, size4_t output_strides, size4_t output_shape,
                                           float3_t shift, float33_t matrix, const geometry::Symmetry& symmetry,
                                           float3_t center, bool normalize, cuda::Stream& stream) {
        NOA_ASSERT(!cuda::memory::PtrTexture::hasNormalizedCoordinates(texture));

        // TODO Move symmetry matrices to constant memory?
        const size_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.matrices();
        cuda::memory::PtrDevice<float33_t> d_matrices(count, stream);
        cuda::memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const uint2_t o_shape(output_shape.get(2));
        const uint4_t o_strides(output_strides);
        const uint blocks_x = math::divideUp(o_shape[1], THREADS.x);
        const dim3 blocks(blocks_x * math::divideUp(o_shape[0], THREADS.y),
                          output_shape[1],
                          output_shape[0]);
        const cuda::LaunchConfig config{blocks, THREADS};

        switch (texture_interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_NEAREST>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_LINEAR:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_LINEAR>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_COSINE:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_COSINE>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_CUBIC:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_CUBIC>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_CUBIC_BSPLINE>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_LINEAR_FAST:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_LINEAR_FAST>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_COSINE_FAST:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_COSINE_FAST>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            case INTERP_CUBIC_BSPLINE_FAST:
                return stream.enqueue("geometry::transform3D", transformWithSymmetry3D_<T, INTERP_CUBIC_BSPLINE_FAST>,
                                      config, texture, output, o_strides, o_shape, shift, matrix, center,
                                      d_matrices.get(), count, scaling, blocks_x);
            default:
                NOA_THROW_FUNC("transform3D", "{} is not supported", texture_interp_mode);
        }
    }
}

namespace noa::cuda::geometry {
    template<typename T, typename>
    void transform3D(const shared_t<T[]>& input, size4_t input_strides, size4_t input_shape,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     InterpMode interp_mode, bool prefilter, bool normalize, Stream& stream) {
        NOA_ASSERT(input_shape[0] == 1 || input_shape[0] == output_shape[0]);

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
        cuda::memory::PtrTexture texture(array.get(), interp_mode, BORDER_ZERO);
        for (size_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            launchTransformSymmetryTexture3D_(
                    texture.get(), interp_mode, output.get() + i * output_strides[0], output_strides, o_shape,
                    shift, matrix, symmetry, center, normalize, stream);
        }
        stream.attach(input, output, array.share(), texture.share());
        if (!buffer.empty())
            stream.attach(buffer.share());
    }

    template<typename T, typename>
    void transform3D(const shared_t<cudaTextureObject_t>& texture, InterpMode texture_interp_mode,
                     const shared_t<T[]>& output, size4_t output_strides, size4_t output_shape,
                     float3_t shift, float33_t matrix, const Symmetry& symmetry, float3_t center,
                     bool normalize, Stream& stream) {
        launchTransformSymmetryTexture3D_(*texture, texture_interp_mode,
                                          output.get(), output_strides, output_shape,
                                          shift, matrix, symmetry, center, normalize, stream);
        stream.attach(texture, output);
    }

    #define NOA_INSTANTIATE_TRANSFORM_SYM_(T)                                                                                                                                                               \
    template void transform3D<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, float3_t, float33_t, const Symmetry&, float3_t, InterpMode, bool, bool, Stream&);    \
    template void transform3D<T, void>(const shared_t<cudaTextureObject_t>&, InterpMode, const shared_t<T[]>&, size4_t, size4_t, float3_t, float33_t, const Symmetry&, float3_t, bool, Stream&)

    NOA_INSTANTIATE_TRANSFORM_SYM_(float);
    NOA_INSTANTIATE_TRANSFORM_SYM_(cfloat_t);
}
