#include "noa/common/Assert.h"
#include "noa/common/Profiler.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Resize.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace {
    using namespace noa;
    constexpr dim3 THREADS(32, 8);

    // Computes two elements per thread.
    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void resizeWithNothing_(const T* __restrict__ inputs, uint input_pitch, uint3_t input_shape,
                            T* __restrict__ outputs, uint output_pitch, uint3_t output_shape,
                            int3_t crop_left, int3_t pad_left, int3_t pad_right, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);

        // If within the padding, stop.
        if (gid.z < pad_left.z || gid.z >= static_cast<int>(output_shape.z) - pad_right.z ||
            gid.y < pad_left.y || gid.y >= static_cast<int>(output_shape.y) - pad_right.y)
            return;

        uint i_y = static_cast<uint>(gid.y - pad_left.y + crop_left.y); // cannot be negative
        uint i_z = static_cast<uint>(gid.z - pad_left.z + crop_left.z);

        inputs += rows(input_shape) * input_pitch * blockIdx.z;
        outputs += rows(output_shape) * output_pitch * blockIdx.z;
        inputs += (i_z * input_shape.y + i_y) * input_pitch;
        outputs += (gid.z * output_shape.y + gid.y) * output_pitch;

        for (int i = 0; i < 2; ++i) {
            int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            if (o_x >= pad_left.x && o_x < static_cast<int>(output_shape.x) - pad_right.x) {
                int i_x = o_x - pad_left.x + crop_left.x; // cannot be negative
                outputs[o_x] = inputs[i_x];
            }
        }
    }

    template<typename T>
    void launchResizeWithNothing_(const T* inputs, uint input_pitch, uint3_t input_shape,
                                  T* outputs, uint output_pitch, uint3_t output_shape,
                                  int3_t border_left, int3_t border_right, uint batches,
                                  cuda::Stream& stream) {
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));
        int3_t pad_right(math::max(border_right, 0));

        uint blocks_x = math::divideUp(output_shape.x, 2 * THREADS.x);
        uint blocks_y = math::divideUp(output_shape.y, THREADS.y);
        dim3 blocks{blocks_x * blocks_y, output_shape.z, batches};
        resizeWithNothing_<<<blocks, THREADS, 0, stream.id()>>>(
                inputs, input_pitch, input_shape,
                outputs, output_pitch, output_shape,
                crop_left, pad_left, pad_right, blocks_x);
    }

    template<typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void resizeWithValue_(const T* __restrict__ inputs, uint input_pitch, uint3_t input_shape,
                          T* __restrict__ outputs, uint output_pitch, uint3_t output_shape,
                          int3_t crop_left, int3_t pad_left, int3_t pad_right, T value, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const int3_t gid(THREADS.x * idx.x * 2 + threadIdx.x,
                         THREADS.y * idx.y + threadIdx.y,
                         blockIdx.y);
        if (gid.y >= output_shape.y)
            return;

        const bool is_valid = gid.z >= pad_left.z && gid.z < static_cast<int>(output_shape.z) - pad_right.z &&
                              gid.y >= pad_left.y && gid.y < static_cast<int>(output_shape.y) - pad_right.y;

        inputs += rows(input_shape) * input_pitch * blockIdx.z;
        outputs += rows(output_shape) * output_pitch * blockIdx.z;
        outputs += (gid.z * output_shape.y + gid.y) * output_pitch;

        const int i_y = gid.y - pad_left.y + crop_left.y;
        const int i_z = gid.z - pad_left.z + crop_left.z;
        for (int i = 0; i < 2; ++i) {
            const int o_x = gid.x + static_cast<int>(THREADS.x) * i;
            if (o_x >= output_shape.x)
                return;

            if (is_valid && o_x >= pad_left.x && o_x < static_cast<int>(output_shape.x) - pad_right.x) {
                auto i_x = static_cast<uint>(o_x - pad_left.x + crop_left.x); // cannot be negative
                outputs[o_x] = inputs[(i_z * input_shape.y + i_y) * input_pitch + i_x];
            } else {
                outputs[o_x] = value;
            }
        }
    }

    template<typename T>
    void launchResizeWithValue_(const T* inputs, uint input_pitch, uint3_t input_shape,
                                T* outputs, uint output_pitch, uint3_t output_shape,
                                int3_t border_left, int3_t border_right, T value,
                                uint batches, cuda::Stream& stream) {
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));
        int3_t pad_right(math::max(border_right, 0));

        uint blocks_x = math::divideUp(output_shape.x, 2 * THREADS.x);
        uint blocks_y = math::divideUp(output_shape.y, THREADS.y);
        dim3 blocks{blocks_x * blocks_y, output_shape.z, batches};
        resizeWithValue_<<<blocks, THREADS, 0, stream.id()>>>(
                inputs, input_pitch, input_shape,
                outputs, output_pitch, output_shape,
                crop_left, pad_left, pad_right, value, blocks_x);
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(THREADS.x * THREADS.y)
    void resizeWith_(const T* __restrict__ inputs, uint input_pitch, uint3_t input_shape,
                     T* __restrict__ outputs, uint output_pitch, uint3_t output_shape,
                     int3_t crop_left, int3_t pad_left, uint blocks_x) {
        const uint2_t idx = coordinates(blockIdx.x, blocks_x);
        const uint3_t o_gid(THREADS.x * idx.x + threadIdx.x,
                            THREADS.y * idx.y + threadIdx.y,
                            blockIdx.y);
        if (o_gid.x >= output_shape.x || o_gid.y >= output_shape.y)
            return;

        int3_t i_gid(o_gid);
        i_gid -= pad_left;
        i_gid += crop_left;
        i_gid.x = getBorderIndex<MODE>(i_gid.x, static_cast<int>(input_shape.x));
        i_gid.y = getBorderIndex<MODE>(i_gid.y, static_cast<int>(input_shape.y));
        i_gid.z = getBorderIndex<MODE>(i_gid.z, static_cast<int>(input_shape.z));

        inputs += rows(input_shape) * input_pitch * blockIdx.z;
        outputs += rows(output_shape) * output_pitch * blockIdx.z;
        outputs[(o_gid.z * output_shape.y + o_gid.y) * output_pitch + o_gid.x] =
                inputs[(i_gid.z * input_shape.y + i_gid.y) * input_pitch + i_gid.x];
    }

    template<BorderMode MODE, typename T>
    void launchResizeWith_(const T* inputs, uint input_pitch, uint3_t input_shape,
                           T* outputs, uint output_pitch, uint3_t output_shape,
                           int3_t border_left, uint batches, cuda::Stream& stream) {
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));

        uint blocks_x = math::divideUp(output_shape.x, THREADS.x);
        uint blocks_y = math::divideUp(output_shape.y, THREADS.y);
        dim3 blocks{blocks_x * blocks_y, output_shape.z, batches};
        resizeWith_<MODE><<<blocks, THREADS, 0, stream.id()>>>(
                inputs, input_pitch, input_shape,
                outputs, output_pitch, output_shape,
                crop_left, pad_left, blocks_x);
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void resize(const T* inputs, size_t input_pitch, size3_t input_shape, int3_t border_left, int3_t border_right,
                T* outputs, size_t output_pitch, BorderMode border_mode, T border_value,
                size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        NOA_ASSERT(inputs != outputs);

        if (all(border_left == 0) && all(border_right == 0))
            return copy(inputs, input_pitch, outputs, output_pitch, input_shape, batches, stream);

        uint3_t output_shape(int3_t(input_shape) + border_left + border_right); // assumed to be > 0
        switch (border_mode) {
            case BORDER_NOTHING:
                launchResizeWithNothing_(inputs, input_pitch, uint3_t(input_shape),
                                         outputs, output_pitch, output_shape,
                                         border_left, border_right, batches, stream);
                break;
            case BORDER_ZERO:
                launchResizeWithValue_(inputs, input_pitch, uint3_t(input_shape),
                                       outputs, output_pitch, output_shape,
                                       border_left, border_right, T{0}, batches, stream);
                break;
            case BORDER_VALUE:
                launchResizeWithValue_(inputs, input_pitch, uint3_t(input_shape),
                                       outputs, output_pitch, output_shape,
                                       border_left, border_right, border_value, batches, stream);
                break;
            case BORDER_CLAMP:
                launchResizeWith_<BORDER_CLAMP>(inputs, input_pitch, uint3_t(input_shape),
                                                outputs, output_pitch, output_shape,
                                                border_left, batches, stream);
                break;
            case BORDER_PERIODIC:
                launchResizeWith_<BORDER_PERIODIC>(inputs, input_pitch, uint3_t(input_shape),
                                                   outputs, output_pitch, output_shape,
                                                   border_left, batches, stream);
                break;
            case BORDER_REFLECT:
                launchResizeWith_<BORDER_REFLECT>(inputs, input_pitch, uint3_t(input_shape),
                                                  outputs, output_pitch, output_shape,
                                                  border_left, batches, stream);
                break;
            case BORDER_MIRROR:
                launchResizeWith_<BORDER_MIRROR>(inputs, input_pitch, uint3_t(input_shape),
                                                 outputs, output_pitch, output_shape,
                                                 border_left, batches, stream);
                break;
            default:
                NOA_THROW("BorderMode not recognized. Got: {}", border_mode);
        }
        NOA_THROW_IF(cudaGetLastError());
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T>(const T*, size_t, size3_t, int3_t, int3_t, T*, size_t, BorderMode, T, size_t, Stream&)

    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(bool);
    NOA_INSTANTIATE_RESIZE_(char);
    NOA_INSTANTIATE_RESIZE_(short);
    NOA_INSTANTIATE_RESIZE_(int);
    NOA_INSTANTIATE_RESIZE_(long);
    NOA_INSTANTIATE_RESIZE_(long long);
    NOA_INSTANTIATE_RESIZE_(unsigned char);
    NOA_INSTANTIATE_RESIZE_(unsigned short);
    NOA_INSTANTIATE_RESIZE_(unsigned int);
    NOA_INSTANTIATE_RESIZE_(unsigned long);
    NOA_INSTANTIATE_RESIZE_(unsigned long long);
}
