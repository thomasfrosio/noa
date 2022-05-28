#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"

#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"

#include "noa/gpu/cuda/geometry/Interpolate.h"
#include "noa/gpu/cuda/geometry/Prefilter.h"
#include "noa/gpu/cuda/geometry/Symmetry.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename T, InterpMode INTERP>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    symmetrize2D_(cudaTextureObject_t texture, T* output, uint3_t output_stride, uint2_t shape,
                  const float33_t* symmetry_matrices, uint symmetry_count, float2_t center, float scaling) {
        const uint3_t gid{blockIdx.z,
                          blockIdx.y * THREADS.y + threadIdx.y,
                          blockIdx.x * THREADS.x + threadIdx.x};
        if (gid[1] >= shape[0] || gid[2] >= shape[1])
            return;

        float2_t coordinates{gid[1], gid[2]};
        T value = cuda::geometry::tex2D<T, INTERP>(texture, coordinates + 0.5f);
        coordinates -= center;
        for (uint i = 0; i < symmetry_count; ++i) {
            const float33_t& m = symmetry_matrices[i];
            float22_t sym_matrix{m[1][1], m[1][2],
                                 m[2][1], m[2][2]};
            float2_t i_coordinates{sym_matrix * coordinates};
            value += cuda::geometry::tex2D<T, INTERP>(texture, i_coordinates + center + 0.5f);
        }

        output[indexing::at(gid, output_stride)] = value * scaling;
    }
}

namespace noa::cuda::geometry {
    template<bool PREFILTER, typename T, typename>
    void symmetrize2D(const shared_t<T[]>& input, size4_t input_stride,
                      const shared_t<T[]>& output, size4_t output_stride,
                      size4_t shape, const Symmetry& symmetry, float2_t center,
                      InterpMode interp_mode, bool normalize, Stream& stream) {
        NOA_ASSERT(shape[1] == 1);

        if (!symmetry.count()) {
            if (input != output)
                memory::copy(input, input_stride, output, output_stride, shape, stream);
            return;
        }

        size4_t input_shape{shape};
        if (input_stride[0] == 0)
            input_shape[0] = 1;

        // Prepare the input array:
        const T* buffer_ptr;
        size_t buffer_pitch;
        size_t buffer_offset;
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            NOA_ASSERT(indexing::isContiguous(output_stride, shape)[3]);
            // Whether input is batched or not, since we copy to the CUDA array, we can use the output as buffer.
            cuda::geometry::bspline::prefilter(input, input_stride, output, output_stride, input_shape, stream);
            buffer_ptr = output.get();
            buffer_pitch = output_stride[2];
            buffer_offset = output_stride[0];
        } else {
            NOA_ASSERT(indexing::isContiguous(input_stride, input_shape)[3]);
            buffer_ptr = input.get();
            buffer_pitch = input_stride[2];
            buffer_offset = input_stride[0];
        }

        // Broadcast input if it is not batched:
        const size4_t o_shape{input_shape[0] > 1 ? 1 : shape[0],
                              shape[1], shape[2], shape[3]};

        // Copy to texture and launch (per input batch):
        const size3_t shape_3d{1, input_shape[2], input_shape[3]};
        cuda::memory::PtrArray<T> array{shape_3d};
        cuda::memory::PtrTexture texture{array.get(), interp_mode, BORDER_ZERO};
        for (size_t i = 0; i < input_shape[0]; ++i) {
            cuda::memory::copy(buffer_ptr + i * buffer_offset, buffer_pitch, array.get(), shape_3d, stream);
            cuda::geometry::symmetrize2D(
                    texture.get(), interp_mode, output.get() + i * output_stride[0], output_stride, o_shape,
                    symmetry, center, normalize, stream);
        }
        stream.attach(input, output, array.share(), texture.share());
    }

    template<typename T, typename>
    void symmetrize2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      T* output, size4_t output_stride, size4_t output_shape,
                      const Symmetry& symmetry, float2_t center, bool normalize, Stream& stream) {
        NOA_ASSERT(!memory::PtrTexture::hasNormalizedCoordinates(texture));

        // TODO Move symmetry matrices to constant memory?
        const size_t count = symmetry.count();
        const float33_t* symmetry_matrices = symmetry.matrices();
        memory::PtrDevice<float33_t> d_matrices{count, stream};
        memory::copy(symmetry_matrices, d_matrices.get(), count, stream);
        const float scaling = normalize ? 1 / static_cast<float>(count + 1) : 1;

        const uint2_t o_shape{output_shape.get() + 2};
        const uint3_t o_stride{output_stride[0], output_stride[2], output_stride[3]};
        const dim3 blocks(math::divideUp(o_shape[1], THREADS.x),
                          math::divideUp(o_shape[0], THREADS.y),
                          output_shape[0]);
        const LaunchConfig config{blocks, THREADS};

        switch (texture_interp_mode) {
            case INTERP_NEAREST:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_NEAREST>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_LINEAR:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_LINEAR>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_COSINE:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_COSINE>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_CUBIC:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_CUBIC>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_CUBIC_BSPLINE:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_CUBIC_BSPLINE>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_LINEAR_FAST:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_LINEAR_FAST>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_COSINE_FAST:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_COSINE_FAST>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            case INTERP_CUBIC_BSPLINE_FAST:
                return stream.enqueue("geometry::symmetrize2D", symmetrize2D_<T, INTERP_CUBIC_BSPLINE_FAST>,
                                      config, texture, output, o_stride, o_shape,
                                      d_matrices.get(), count, center, scaling);
            default:
                NOA_THROW("{} is not supported", texture_interp_mode);
        }
    }

    #define NOA_INSTANTIATE_TRANSFORM_SYM_(T)                                                                                                                               \
    template void symmetrize2D<true, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const Symmetry&, float2_t, InterpMode, bool, Stream&); \
    template void symmetrize2D<false, T, void>(const shared_t<T[]>&, size4_t, const shared_t<T[]>&, size4_t, size4_t, const Symmetry&, float2_t, InterpMode, bool, Stream&);

    NOA_INSTANTIATE_TRANSFORM_SYM_(float);
    NOA_INSTANTIATE_TRANSFORM_SYM_(cfloat_t);
}
