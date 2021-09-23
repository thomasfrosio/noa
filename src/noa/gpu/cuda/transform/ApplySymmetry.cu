#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Symmetry.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<typename T, InterpMode INTERP>
    __global__ void applyWithSymmetry_(cudaTextureObject_t texture, T* output, size_t output_pitch, size2_t shape,
                                       float2_t center, float2_t shifts, float22_t matrix,
                                       const float33_t* symmetry_matrices, uint symmetry_count, float scaling) {

        const uint2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x >= shape.x || gid.y >= shape.y)
            return;

        float2_t coordinates(gid.x, gid.y);
        coordinates -= center + shifts;
        coordinates += 0.5f;
        coordinates = matrix * coordinates;

        T value = cuda::transform::tex2D<T, INTERP>(texture, coordinates.x + center.x, coordinates.y + center.y);
        for (uint i = 0; i < symmetry_count; ++i) {
            float2_t i_coordinates(float22_t(symmetry_matrices[i]) * coordinates);
            i_coordinates += center;
            value += cuda::transform::tex2D<T, INTERP>(texture, i_coordinates.x, i_coordinates.y);
        }

        output += gid.y * output_pitch + gid.x;
        *output = value * scaling;
    }

    template<typename T, InterpMode INTERP>
    __global__ void applyWithSymmetry_(cudaTextureObject_t texture, T* output, size_t output_pitch, size3_t shape,
                                       float3_t center, float3_t shifts, float33_t matrix,
                                       const float33_t* symmetry_matrices, uint symmetry_count, float scaling) {

        const uint3_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                          blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.z);
        if (gid.x >= shape.x || gid.y >= shape.y) // z cannot be out
            return;

        float3_t coordinates(gid.x, gid.y, gid.z);
        coordinates -= center + shifts;
        coordinates += 0.5f;
        coordinates = matrix * coordinates;

        T value = cuda::transform::tex3D<T, INTERP>(texture,
                                                    coordinates.x + center.x,
                                                    coordinates.y + center.y,
                                                    coordinates.z + center.z);
        for (uint i = 0; i < symmetry_count; ++i) {
            float3_t i_coordinates(symmetry_matrices[i] * coordinates);
            i_coordinates += center;
            value += cuda::transform::tex3D<T, INTERP>(texture, i_coordinates.x, i_coordinates.y, i_coordinates.z);
        }

        output += gid.y * output_pitch + gid.x;
        *output = value * scaling;
    }

    template<typename T, typename SHAPE, typename CENTER, typename MATRIX>
    void launch_(cudaTextureObject_t texture, InterpMode texture_interp,
                 T* output, size_t output_pitch, SHAPE shape,
                 CENTER center, CENTER shifts, MATRIX matrix,
                 const float33_t* symmetry_matrices, uint symmetry_count, float scaling,
                 dim3 blocks, cuda::Stream& stream) {
        switch (texture_interp) {
            case INTERP_NEAREST:
                applyWithSymmetry_<T, INTERP_NEAREST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_LINEAR:
                applyWithSymmetry_<T, INTERP_LINEAR><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_COSINE:
                applyWithSymmetry_<T, INTERP_COSINE><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_CUBIC:
                applyWithSymmetry_<T, INTERP_CUBIC><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_CUBIC_BSPLINE:
                applyWithSymmetry_<T, INTERP_CUBIC_BSPLINE><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_LINEAR_FAST:
                applyWithSymmetry_<T, INTERP_LINEAR_FAST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_COSINE_FAST:
                applyWithSymmetry_<T, INTERP_COSINE_FAST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            case INTERP_CUBIC_BSPLINE_FAST:
                applyWithSymmetry_<T, INTERP_CUBIC_BSPLINE_FAST><<<blocks, THREADS, 0, stream.id()>>>(
                        texture, output, output_pitch, shape, center, shifts, matrix,
                        symmetry_matrices, symmetry_count, scaling);
                break;
            default:
                NOA_THROW_FUNC("apply(2|3)D", "{} is not supported", texture_interp);
        }
    }
}

namespace noa::cuda::transform {
    template<typename T>
    void apply2D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, size_t output_pitch, size2_t shape,
                 float2_t center, float2_t shifts, float22_t matrix,
                 const float33_t* symmetry_matrices, uint symmetry_count, Stream& stream) {
        uint2_t tmp(shape);
        const dim3 blocks(noa::math::divideUp(tmp.x, THREADS.x),
                          noa::math::divideUp(tmp.y, THREADS.y));

        float scaling = 1 / static_cast<float>(symmetry_count + 1);
        launch_(texture, texture_interp_mode, output, output_pitch, shape, center, shifts, matrix, symmetry_matrices,
                symmetry_count, scaling, blocks, stream);
    }

    template<typename T>
    void apply3D(cudaTextureObject_t texture, InterpMode texture_interp_mode,
                 T* output, size_t output_pitch, size3_t shape,
                 float3_t center, float3_t shifts, float33_t matrix,
                 const float33_t* symmetry_matrices, uint symmetry_count, Stream& stream) {
        uint3_t tmp(shape);
        const dim3 blocks(noa::math::divideUp(tmp.x, THREADS.x),
                          noa::math::divideUp(tmp.y, THREADS.y),
                          tmp.z);

        float scaling = 1 / static_cast<float>(symmetry_count + 1);
        launch_(texture, texture_interp_mode, output, output_pitch, shape, center, shifts, matrix, symmetry_matrices,
                symmetry_count, scaling, blocks, stream);
    }
}

namespace noa::cuda::transform {
    template<bool PREFILTER, typename T>
    void apply2D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size2_t shape,
                 float2_t center, float2_t shifts, float22_t matrix, Symmetry symmetry,
                 InterpMode interp_mode, Stream& stream) {
        uint count = symmetry.getCount();
        const float33_t* matrices = symmetry.getMatrices();
        memory::PtrDevice<float33_t> d_matrices(count);
        memory::copy(matrices, d_matrices.get(), count, stream);

        size3_t shape_3d(shape.x, shape.y, 1);
        memory::PtrArray<T> buffer(shape_3d);
        memory::PtrTexture<T> texture(buffer.get(), interp_mode, BORDER_ZERO);
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            transform::bspline::prefilter2D(input, input_pitch, output, output_pitch, shape, 1, stream);
            memory::copy(output, output_pitch, buffer.get(), shape_3d, stream);
        } else {
            memory::copy(input, input_pitch, buffer.get(), shape_3d, stream);
        }

        apply2D(texture.get(), interp_mode, output, output_pitch, shape,
                center, shifts, matrix, d_matrices.get(), count, stream);
        stream.synchronize();
    }

    template<bool PREFILTER, typename T>
    void apply3D(const T* input, size_t input_pitch, T* output, size_t output_pitch, size3_t shape,
                 float3_t center, float3_t shifts, float33_t matrix, Symmetry symmetry,
                 InterpMode interp_mode, Stream& stream) {
        uint count = symmetry.getCount();
        const float33_t* matrices = symmetry.getMatrices();
        memory::PtrDevice<float33_t> d_matrices(count);
        memory::copy(matrices, d_matrices.get(), count, stream);

        memory::PtrArray<T> buffer(shape);
        memory::PtrTexture<T> texture(buffer.get(), interp_mode, BORDER_ZERO);
        if (PREFILTER && (interp_mode == INTERP_CUBIC_BSPLINE || interp_mode == INTERP_CUBIC_BSPLINE_FAST)) {
            transform::bspline::prefilter3D(input, input_pitch, output, output_pitch, shape, 1, stream);
            memory::copy(output, output_pitch, buffer.get(), shape, stream);
        } else {
            memory::copy(input, input_pitch, buffer.get(), shape, stream);
        }

        apply3D(texture.get(), interp_mode, output, output_pitch, shape,
                center, shifts, matrix, d_matrices.get(), count, stream);
        stream.synchronize();
    }

    #define NOA_APPLY_SYMMETRY_(T)                                                                                                          \
    template void apply2D<true, T>(const T*, size_t, T*, size_t, size2_t, float2_t, float2_t, float22_t, Symmetry, InterpMode, Stream&);    \
    template void apply3D<true, T>(const T*, size_t, T*, size_t, size3_t, float3_t, float3_t, float33_t, Symmetry, InterpMode, Stream&)

    NOA_APPLY_SYMMETRY_(float);
    NOA_APPLY_SYMMETRY_(cfloat_t);
}