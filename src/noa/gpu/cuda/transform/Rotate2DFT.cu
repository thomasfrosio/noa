#include "noa/common/Assert.h"
#include "noa/common/Math.h"
#include "noa/common/transform/Euler.h"
#include "noa/common/transform/Geometry.h"

#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/PtrArray.h"
#include "noa/gpu/cuda/memory/PtrTexture.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/transform/Interpolate.h"
#include "noa/gpu/cuda/transform/Rotate.h"

namespace {
    using namespace ::noa;
    constexpr dim3 THREADS(16, 16);

    template<bool REMAP, typename T>
    __global__ void rotate2DFT_(cudaTextureObject_t texture,
                                T* outputs, uint output_pitch, int shape, const float* rotations,
                                float freq_cutoff) {
        const int HALF = shape / 2;
        const uint rotation_id = blockIdx.z;
        const int2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x > HALF || gid.y >= shape)
            return;

        int o_y;
        if constexpr (REMAP)
            o_y = math::iFFTShift(gid.y, shape);
        else
            o_y = gid.y;

        float2_t pos = transform::rotate(-rotations[rotation_id]) * float2_t(gid.x, gid.y - HALF);
        if (math::length(pos / float2_t(shape)) > freq_cutoff) {
            outputs[(rotation_id * shape + o_y) * output_pitch + gid.x] = 0.f;
            return;
        }

        T value;
        if constexpr (std::is_same_v<T, float>) {
            if (pos.x < -0.0001f)
                pos = -pos;
            pos += float2_t(0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex2D<float, INTERP_LINEAR>(texture, pos.x, pos.y);
        } else { // cfloat_t
            bool negate;
            if (pos.x < -0.0001f) {
                pos = -pos;
                negate = true;
            } else {
                negate = false;
            }
            pos += float2_t(0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex2D<cfloat_t, INTERP_LINEAR>(texture, pos.x, pos.y);
            if (negate)
                value.imag(-value.imag());
        }
        outputs[(rotation_id * shape + o_y) * output_pitch + gid.x] = value;
    }

    template<bool REMAP, typename T>
    __global__ void rotate2DFT_(cudaTextureObject_t texture,
                                T* output, uint output_pitch, int shape, float22_t rotm, float freq_cutoff) {
        const int HALF = shape / 2;
        const int2_t gid(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y);
        if (gid.x > HALF || gid.y >= shape)
            return;

        int o_y;
        if constexpr (REMAP)
            o_y = math::iFFTShift(gid.y, shape);
        else
            o_y = gid.y;

        float2_t pos = rotm * float2_t(gid.x, gid.y - HALF);
        if (math::length(pos / float2_t(shape)) > freq_cutoff) {
            output[o_y * output_pitch + gid.x] = 0.f;
            return;
        }

        T value;
        if constexpr (std::is_same_v<T, float>) {
            if (pos.x < -0.0001f)
                pos = -pos;
            pos += float2_t(0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex2D<float, INTERP_LINEAR>(texture, pos.x, pos.y);
        } else { // cfloat_t
            bool negate;
            if (pos.x < -0.0001f) {
                pos = -pos;
                negate = true;
            } else {
                negate = false;
            }
            pos += float2_t(0.5f, static_cast<float>(HALF) + 0.5f);
            value = cuda::transform::tex2D<cfloat_t, INTERP_LINEAR>(texture, pos.x, pos.y);
            if (negate)
                value.imag(-value.imag());
        }
        output[o_y * output_pitch + gid.x] = value;
    }
}

namespace noa::cuda::transform {
    template<bool REMAP, typename T>
    void rotate2DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                    const float* rotations, uint nb_rotations, float freq_cutoff, Stream& stream) {
        const uint2_t half_shape(shape / 2 + 1, shape);
        const dim3 BLOCKS(math::divideUp(half_shape.x, THREADS.x),
                          math::divideUp(half_shape.y, THREADS.y),
                          nb_rotations);
        rotate2DFT_<REMAP><<<BLOCKS, THREADS, 0, stream.id()>>>(
                texture, output, output_pitch, shape, rotations, freq_cutoff); // compute the rotm in the kernels
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool REMAP, typename T>
    void rotate2DFT(cudaTextureObject_t texture, T* output, size_t output_pitch, size_t shape,
                    float rotation, float freq_cutoff, Stream& stream) {
        const uint2_t half_shape(shape / 2 + 1, shape);
        const dim3 BLOCKS(math::divideUp(half_shape.x, THREADS.x),
                          math::divideUp(half_shape.y, THREADS.y));
        rotate2DFT_<REMAP><<<BLOCKS, THREADS, 0, stream.id()>>>(
                texture, output, output_pitch, shape, noa::transform::rotate(-rotation), freq_cutoff);
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    template<bool REMAP, typename T>
    void rotate2DFT(const T* input, size_t input_pitch, T* outputs, size_t output_pitch, size_t shape,
                    const float* rotations, uint nb_rotations, float freq_cutoff, Stream& stream) {
        size3_t shape_3d(shape, shape, 1);
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture<T> texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), shape_3d, stream);

        rotate2DFT<REMAP>(texture.get(), outputs, output_pitch, shape, rotations, nb_rotations, freq_cutoff, stream);
        stream.synchronize();
    }

    template<bool REMAP, typename T>
    void rotate2DFT(const T* input, size_t input_pitch, T* output, size_t output_pitch, size_t shape,
                    float rotation, float freq_cutoff, Stream& stream) {
        size3_t shape_3d(shape, shape, 1);
        memory::PtrArray<T> array(shape_3d);
        memory::PtrTexture<T> texture(array.get(), INTERP_LINEAR, BORDER_ZERO);
        memory::copy(input, input_pitch, array.get(), shape_3d, stream);

        rotate2DFT<REMAP>(texture.get(), output, output_pitch, shape, rotation, freq_cutoff, stream);
        stream.synchronize();
    }
}

namespace noa::cuda::transform {
    #define NOA_INSTANTIATE_ROTATE_FT_(T) \
    template void rotate2DFT<false, T>(const T*, size_t, T*, size_t, size_t, const float*, uint, float, Stream&);\
    template void rotate2DFT<true, T>(const T*, size_t, T*, size_t, size_t, const float*, uint, float, Stream&);\
    template void rotate2DFT<false, T>(const T*, size_t, T*, size_t, size_t, float, float, Stream&);\
    template void rotate2DFT<true, T>(const T*, size_t, T*, size_t, size_t, float, float, Stream&)

    NOA_INSTANTIATE_ROTATE_FT_(float);
    NOA_INSTANTIATE_ROTATE_FT_(cfloat_t);
}
