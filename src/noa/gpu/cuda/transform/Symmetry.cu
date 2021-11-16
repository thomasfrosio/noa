#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Symmetry.h"

// TODO(TF) Is it faster to replace the first copy by a texture fetching? Or, take the input array directly and
//          perform the copy in the kernel...

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename T, InterpMode INTERP>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    symmetrize_(cudaTextureObject_t texture, T* output, size_t output_pitch, size2_t shape,
                const float33_t* matrix, uint count, float scaling, float2_t center) {

        const uint2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float2_t coordinates(gid.x, gid.y);
        coordinates -= center;

        T value = 0;
        for (uint i = 0; i < count; ++i) {
            float2_t i_coordinates(float22_t(matrix[i]) * coordinates);
            value += cuda::transform::tex2D<T, INTERP>(texture, i_coordinates + center + 0.5f);
        }

        output += gid.y * output_pitch + gid.x;
        *output += value;
        *output *= scaling;
    }

    template<typename T, InterpMode INTERP>
    __global__ void __launch_bounds__(THREADS.x * THREADS.y)
    symmetrize_(cudaTextureObject_t texture, T* output, size_t output_pitch, size3_t shape,
                const float33_t* matrix, uint count, float scaling, float3_t center) {

        const uint3_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y) // z cannot be out
            return;

        float3_t coordinates(gid.x, gid.y, gid.z);
        coordinates -= center;

        T value = 0;
        for (uint i = 0; i < count; ++i) {
            float3_t i_coordinates(matrix[i] * coordinates);
            value += cuda::transform::tex3D<T, INTERP>(texture, i_coordinates + center + 0.5f);
        }

        output += (gid.z * shape.y + gid.y) * output_pitch + gid.x;
        *output += value;
        *output *= scaling;
    }

    template<typename T, typename SHAPE, typename CENTER>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp,
                 T* output, size_t output_pitch, SHAPE shape,
                 const float33_t* matrix, uint count, float scaling, CENTER center,
                 dim3 blocks, cuda::Stream& stream) {
        switch (texture_interp) {
            case INTERP_NEAREST:
                symmetrize_<T, INTERP_NEAREST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_LINEAR:
                symmetrize_<T, INTERP_LINEAR><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_COSINE:
                symmetrize_<T, INTERP_COSINE><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_CUBIC:
                symmetrize_<T, INTERP_CUBIC><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_CUBIC_BSPLINE:
                symmetrize_<T, INTERP_CUBIC_BSPLINE><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_LINEAR_FAST:
                symmetrize_<T, INTERP_LINEAR_FAST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_COSINE_FAST:
                symmetrize_<T, INTERP_COSINE_FAST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            case INTERP_CUBIC_BSPLINE_FAST:
                symmetrize_<T, INTERP_CUBIC_BSPLINE_FAST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, matrix, count, scaling, center);
                break;
            default:
                NOA_THROW_FUNC("symmetrize(2|3)D", "{} is not supported", texture_interp);
        }
        NOA_THROW_IF(cudaGetLastError());
    }
}

// -- Using textures -- //
namespace noa::cuda::transform {
    template<typename T>
    void symmetrize2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      T* output, size_t output_pitch, size2_t shape, const float33_t* symmetry_matrices,
                      size_t symmetry_count, float2_t symmetry_center, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        size3_t shape_3d(shape.x, shape.y, 1);
        cudaResourceDesc resource = memory::PtrTexture<T>::getResource(texture);
        memory::copy(resource.res.array.array, output, output_pitch, shape_3d, stream);

        if (symmetry_count == 0)
            return;

        uint2_t tmp(shape.x, shape.y);
        const dim3 blocks(noa::math::divideUp(tmp.x, THREADS.x),
                          noa::math::divideUp(tmp.y, THREADS.y));

        float scaling = 1 / static_cast<float>(symmetry_count + 1);
        launch_(texture, texture_interp_mode, output, output_pitch, shape,
                symmetry_matrices, symmetry_count, scaling, symmetry_center, blocks, stream);
    }

    template<typename T>
    void symmetrize3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                      T* output, size_t output_pitch, size3_t shape, const float33_t* symmetry_matrices,
                      size_t symmetry_count, float3_t symmetry_center, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        cudaResourceDesc resource = memory::PtrTexture<T>::getResource(texture);
        memory::copy(resource.res.array.array, output, output_pitch, shape, stream);

        if (symmetry_count == 0)
            return;

        uint3_t tmp(shape);
        const dim3 blocks(noa::math::divideUp(tmp.x, THREADS.x),
                          noa::math::divideUp(tmp.y, THREADS.y),
                          tmp.z);

        float scaling = 1 / static_cast<float>(symmetry_count + 1);
        launch_(texture, texture_interp_mode, output, output_pitch, shape,
                symmetry_matrices, symmetry_count, scaling, symmetry_center, blocks, stream);
    }
}

// -- Using arrays -- //
namespace noa::cuda::transform {
    template<bool PREFILTER, typename T>
    void symmetrize2D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                      size2_t shape, size_t batches, const Symmetry& symmetry, float2_t symmetry_center,
                      InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size3_t shape_3d(shape.x, shape.y, 1);
        const size_t count = symmetry.count();
        if (!count) {
            memory::copy(inputs, input_pitch, outputs, output_pitch, shape_3d, batches, stream);
            stream.synchronize(); // be consistent
            return;
        }

        const float33_t* matrices = symmetry.matrices();
        memory::PtrDevice<float33_t> d_matrices(count);
        memory::copy(matrices, d_matrices.get(), count, stream);

        const T* tmp;
        size_t pitch;
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            transform::bspline::prefilter2D(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);
            tmp = outputs;
            pitch = output_pitch;
        } else {
            tmp = inputs;
            pitch = input_pitch;
        }

        memory::PtrArray<T> buffer(shape_3d);
        memory::PtrTexture<T> texture(buffer.get(), interp_mode, BORDER_ZERO);
        for (uint batch = 0; batch < batches; ++batch) {
            size_t offset = batch * pitch * shape.y;
            T* output = outputs + batch * output_pitch * shape.y;
            memory::copy(tmp + offset, pitch, buffer.get(), shape_3d, stream);
            symmetrize2D(texture.get(), interp_mode, output, output_pitch, shape,
                         d_matrices.get(), count, symmetry_center, stream);
        }
        stream.synchronize();
    }

    template<bool PREFILTER, typename T>
    void symmetrize3D(const T* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                      size3_t shape, size_t batches, const Symmetry& symmetry, float3_t symmetry_center,
                      InterpMode interp_mode, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t count = symmetry.count();
        if (!count) {
            memory::copy(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);
            stream.synchronize();
            return;
        }

        const float33_t* matrices = symmetry.matrices();
        memory::PtrDevice<float33_t> d_matrices(count);
        memory::copy(matrices, d_matrices.get(), count, stream);

        const T* tmp;
        size_t pitch;
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            transform::bspline::prefilter3D(inputs, input_pitch, outputs, output_pitch, shape, batches, stream);
            tmp = outputs;
            pitch = output_pitch;
        } else {
            tmp = inputs;
            pitch = input_pitch;
        }

        memory::PtrArray<T> buffer(shape);
        memory::PtrTexture<T> texture(buffer.get(), interp_mode, BORDER_ZERO);
        for (size_t batch = 0; batch < batches; ++batch) {
            size_t offset = batch * pitch * shape.y * shape.z;
            T* output = outputs + batch * output_pitch * shape.y * shape.z;
            memory::copy(tmp + offset, pitch, buffer.get(), shape, stream);
            symmetrize3D(texture.get(), interp_mode, output, output_pitch, shape,
                         d_matrices.get(), count, symmetry_center, stream);
        }
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_SYMMETRIZE_(T)                                                                                                  \
    template void symmetrize2D<true, T>(const T*, size_t, T*, size_t, size2_t, size_t, const Symmetry&, float2_t, InterpMode, Stream&);     \
    template void symmetrize3D<true, T>(const T*, size_t, T*, size_t, size3_t, size_t, const Symmetry&, float3_t, InterpMode, Stream&);     \
    template void symmetrize2D<false, T>(const T*, size_t, T*, size_t, size2_t, size_t, const Symmetry&, float2_t, InterpMode, Stream&);    \
    template void symmetrize3D<false, T>(const T*, size_t, T*, size_t, size3_t, size_t, const Symmetry&, float3_t, InterpMode, Stream&)

    NOA_INSTANTIATE_SYMMETRIZE_(float);
    NOA_INSTANTIATE_SYMMETRIZE_(cfloat_t);
}
