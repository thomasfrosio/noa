#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/transform/Euler.h"

#include "noa/cpu/memory/PtrHost.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Rotate.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<bool REMAP, typename T>
    __global__ void rotate3DFT_(cudaTextureObject_t texture,
                                T* outputs, uint output_pitch, int shape, const float33_t* rotm,
                                float freq_cutoff, uint blocks_x) {
        const int HALF = shape / 2;
        const uint rotation_id = blockIdx.z;
        const uint block_y = blockIdx.x / blocks_x;
        const uint block_x = blockIdx.x - block_y * blocks_x;
        const int3_t gid(block_x * THREADS.x + threadIdx.x,
                         block_y * THREADS.y + threadIdx.y,
                         blockIdx.y);
        if (gid.x > HALF || gid.y >= shape) // z cannot be out of bounds
            return;

        int o_y, o_z;
        if constexpr (REMAP) {
            o_y = math::iFFTShift(gid.y, shape);
            o_z = math::iFFTShift(gid.z, shape);
        } else {
            o_y = gid.y;
            o_z = gid.z;
        }
        float3_t pos = rotm[rotation_id] * float3_t(gid.x, gid.y - HALF, gid.z - HALF);
        if (math::length(pos / float3_t(shape)) > freq_cutoff) {
            outputs += rotation_id * shape * output_pitch;
            outputs[(o_z * shape + o_y) * output_pitch + gid.x] = 0.f;
            return;
        }

        T value;
        if constexpr (std::is_same_v<T, float>) {
            if (pos.x < -0.0001f)
                pos = -pos;
            pos += float3_t(0.5f, static_cast<float>(HALF) + 0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex3D<float, INTERP_LINEAR>(texture, pos.x, pos.y, pos.z);
        } else { // cfloat_t
            bool negate;
            if (pos.x < -0.0001f) {
                pos = -pos;
                negate = true;
            } else {
                negate = false;
            }
            pos += float3_t(0.5f, static_cast<float>(HALF) + 0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex3D<float, INTERP_LINEAR>(texture, pos.x, pos.y, pos.z);
            if (negate)
                value.imag(-value.imag());
        }
        outputs[(o_z * shape + o_y) * output_pitch + gid.x] = value;
    }

    template<bool REMAP, typename T>
    __global__ void rotate3DFT_(cudaTextureObject_t texture,
                                T* output, uint output_pitch, int shape, float33_t rotm, float freq_cutoff) {
        const int HALF = shape / 2;
        const int3_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y,
                         blockIdx.z);
        if (gid.x > HALF || gid.y >= shape)
            return;

        int o_y, o_z;
        if constexpr (REMAP) {
            o_y = math::iFFTShift(gid.y, shape);
            o_z = math::iFFTShift(gid.z, shape);
        } else {
            o_y = gid.y;
            o_z = gid.z;
        }
        float3_t pos = rotm * float3_t(gid.x, gid.y - HALF, gid.z - HALF);
        if (math::length(pos / float3_t(shape)) > freq_cutoff) {
            output[(o_z * shape + o_y) * output_pitch + gid.x] = 0.f;
            return;
        }
        T value;
        if constexpr (std::is_same_v<T, float>) {
            if (pos.x < -0.0001f)
                pos = -pos;
            pos += float3_t(0.5f, static_cast<float>(HALF) + 0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex3D<float, INTERP_LINEAR>(texture, pos.x, pos.y, pos.z);
        } else { // cfloat_t
            bool negate;
            if (pos.x < -0.0001f) {
                pos = -pos;
                negate = true;
            } else {
                negate = false;
            }
            pos += float3_t(0.5f, static_cast<float>(HALF) + 0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex3D<float, INTERP_LINEAR>(texture, pos.x, pos.y, pos.z);
            if (negate)
                value.imag(-value.imag());
        }
        output[(o_z * shape + o_y) * output_pitch + gid.x] = value;
    }
}

namespace noa::cuda::transform {
    template<bool REMAP, typename T>
    void rotate3DFT(cudaTextureObject_t texture, T* outputs, size_t output_pitch, size_t shape,
                    const float3_t* rotations, uint nb_rotations, float freq_cutoff, Stream& stream) {
        constexpr bool INVERT = true;
        noa::memory::PtrHost<float33_t> h_rotm(nb_rotations);
        for (uint i = 0; i < nb_rotations; ++i)
            h_rotm[i] = noa::transform::toMatrix<INVERT>(rotations[i]);
        memory::PtrDevice<float33_t> d_rotm(nb_rotations);
        memory::copy(h_rotm.get(), d_rotm.get(), nb_rotations, stream);

        const uint2_t half_shape(shape / 2 + 1, shape);
        const uint blocks_x = math::divideUp(half_shape.x, THREADS.x);
        const dim3 BLOCKS(blocks_x * math::divideUp(half_shape.y, THREADS.y),
                          shape, nb_rotations);
        rotate3DFT_<REMAP><<<BLOCKS, THREADS, 0, stream.id()>>>(
                texture, outputs, output_pitch, shape, d_rotm.get(), freq_cutoff, blocks_x);
        NOA_THROW_IF(cudaPeekAtLastError());
        stream.synchronize();
    }

    template<bool REMAP, typename T>
    void rotate3DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                    float3_t rotation, float freq_cutoff, Stream& stream) {
        constexpr bool INVERT = true;
        float33_t rotm(noa::transform::toMatrix<INVERT>(rotation));

        const uint2_t half_shape(shape / 2 + 1, shape);
        const dim3 BLOCKS(math::divideUp(half_shape.x, THREADS.x),
                          math::divideUp(half_shape.y, THREADS.y),
                          shape);
        rotate3DFT_<REMAP><<<BLOCKS, THREADS, 0, stream.id()>>>(
                texture, output, output_pitch, shape, rotm, freq_cutoff);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool REMAP, typename T>
    void rotate3DFT(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size_t shape,
                    const float3_t* rotations, uint nb_rotations, float freq_cutoff, Stream& stream) {
        size3_t shape_3d(shape);
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture<T> texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), shape_3d, stream);

        // This function sync the stream.
        rotate3DFT<REMAP>(texture.get(), outputs, output_pitch, shape, rotations, nb_rotations, freq_cutoff, stream);
    }

    template<bool REMAP, typename T>
    void rotate3DFT(const T* input, size_t input_pitch, T* output, size_t output_pitch, size_t shape,
                    float3_t rotation, float freq_cutoff, Stream& stream) {
        size3_t shape_3d(shape);
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture<T> texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), shape_3d, stream);

        rotate3DFT<REMAP>(texture.get(), output, output_pitch, shape, rotation, freq_cutoff, stream);
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_ROTATE_FT_(T) \
    template void rotate3DFT<false, T>(const T*, size_t, T*, size_t, size_t, const float3_t*, uint, float, Stream&);\
    template void rotate3DFT<true, T>(const T*, size_t, T*, size_t, size_t, const float3_t*, uint, float, Stream&);\
    template void rotate3DFT<false, T>(const T*, size_t, T*, size_t, size_t, float3_t, float, Stream&);\
    template void rotate3DFT<true, T>(const T*, size_t, T*, size_t, size_t, float3_t, float, Stream&)

    NOA_INSTANTIATE_ROTATE_FT_(float);
    NOA_INSTANTIATE_ROTATE_FT_(cfloat_t);
}
