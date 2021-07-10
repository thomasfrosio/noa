#include "noa/Session.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Resize.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace {
    using namespace noa;

    template<typename T>
    __global__ void resizeWithNothing_(const T* inputs, uint input_pitch, uint3_t input_shape, uint input_elements,
                                       T* outputs, uint output_pitch, uint3_t output_shape, uint output_elements,
                                       int3_t crop_left, int3_t pad_left, int3_t pad_right, uint batches) {
        uint o_y = blockIdx.y;
        uint o_z = blockIdx.z;
        int i_y = static_cast<int>(o_y) - pad_left.y + crop_left.y; // negative if withing padding
        int i_z = static_cast<int>(o_z) - pad_left.z + crop_left.z;

        if (o_z < pad_left.z || o_z >= output_shape.z - pad_right.z ||
            o_y < pad_left.y || o_y >= output_shape.y - pad_right.y)
            return;

        outputs += (o_z * output_shape.y + o_y) * output_pitch;
        for (uint o_x = blockIdx.x * blockDim.x + threadIdx.x; o_x < output_shape.x; o_x += blockDim.x * gridDim.x) {
            if (o_x < pad_left.x || o_x >= output_shape.x - pad_right.x)
                break;

            uint i_x = o_x - pad_left.x + crop_left.x; // cannot be negative
            inputs += (i_z * input_shape.y + i_y) * input_pitch + i_x;
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * output_elements + o_x] = inputs[batch * input_elements];
        }
    }

    template<typename T>
    void launchResizeWithNothing_(const T* inputs, uint input_pitch, uint3_t input_shape,
                                  T* outputs, uint output_pitch, uint3_t output_shape,
                                  int3_t border_left, int3_t border_right, uint batches,
                                  cuda::Stream& stream) {
        uint input_elements = getRows(input_shape) * input_pitch;
        uint output_elements = getRows(output_shape) * output_pitch;
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));
        int3_t pad_right(math::max(border_right, 0));

        uint threads = math::min(256U, math::nextMultipleOf(output_shape.x, 32U));
        dim3 blocks{(output_shape.x + threads - 1) / threads, output_shape.y, output_shape.z};
        resizeWithNothing_<<<blocks, threads, 0, stream.id()>>>(
                inputs, input_pitch, input_shape, input_elements,
                outputs, output_pitch, output_shape, output_elements,
                crop_left, pad_left, pad_right, batches);
    }

    template<typename T>
    __global__ void resizeWithValue_(const T* inputs, uint input_pitch, uint3_t input_shape, uint input_elements,
                                     T* outputs, uint output_pitch, uint3_t output_shape, uint output_elements,
                                     int3_t crop_left, int3_t pad_left, int3_t pad_right, T value, uint batches) {
        uint o_y = blockIdx.y;
        uint o_z = blockIdx.z;
        int i_y = static_cast<int>(o_y) - pad_left.y + crop_left.y;
        int i_z = static_cast<int>(o_z) - pad_left.z + crop_left.z;

        bool is_padding = o_z < pad_left.z || o_z >= output_shape.z - pad_right.z ||
                          o_y < pad_left.y || o_y >= output_shape.y - pad_right.y;

        outputs += (o_z * output_shape.y + o_y) * output_pitch;
        for (uint o_x = blockIdx.x * blockDim.x + threadIdx.x; o_x < output_shape.x; o_x += blockDim.x * gridDim.x) {
            if (is_padding || o_x < pad_left.x || o_x >= output_shape.x - pad_right.x) {
                for (uint batch = 0; batch < batches; ++batch)
                    outputs[batch * output_elements + o_x] = value;
            } else {
                uint i_x = o_x - pad_left.x + crop_left.x;
                inputs += (i_z * input_shape.y + i_y) * input_pitch + i_x;
                for (uint batch = 0; batch < batches; ++batch)
                    outputs[batch * output_elements + o_x] = inputs[batch * input_elements];
            }
        }
    }

    template<typename T>
    void launchResizeWithValue_(const T* inputs, uint input_pitch, uint3_t input_shape,
                                T* outputs, uint output_pitch, uint3_t output_shape,
                                int3_t border_left, int3_t border_right, T value,
                                uint batches, cuda::Stream& stream) {
        uint input_elements = getRows(input_shape) * input_pitch;
        uint output_elements = getRows(output_shape) * output_pitch;
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));
        int3_t pad_right(math::max(border_right, 0));

        uint threads = math::min(256U, math::nextMultipleOf(output_shape.x, 32U));
        dim3 blocks{(output_shape.x + threads - 1) / threads, output_shape.y, output_shape.z};
        resizeWithValue_<<<blocks, threads, 0, stream.id()>>>(
                inputs, input_pitch, input_shape, input_elements,
                outputs, output_pitch, output_shape, output_elements,
                crop_left, pad_left, pad_right, value, batches);
    }

    template<int MODE>
    inline __device__ int getBorderIndex_(int idx, int pad_left, int crop_left, int len) {
        static_assert(MODE == BORDER_CLAMP || MODE == BORDER_PERIODIC ||
                      MODE == BORDER_MIRROR || MODE == BORDER_REFLECT);
        int out_idx;
        if constexpr (MODE == BORDER_CLAMP) {
            out_idx = math::max(0, math::min(idx - pad_left + crop_left, len - 1));
        } else if constexpr (MODE == BORDER_PERIODIC) {
            int rem = (idx - pad_left + crop_left) % len;
            out_idx = rem < 0 ? rem + len : rem;
        } else if constexpr (MODE == BORDER_MIRROR) {
            out_idx = idx - pad_left + crop_left;
            if (out_idx < 0) {
                int offset = (-out_idx - 1) % len;
                out_idx = offset;
            } else if (out_idx >= len) {
                int offset = out_idx % len;
                out_idx = len - offset - 1;
            }
        } else if constexpr (MODE == BORDER_REFLECT) {
            out_idx = idx - pad_left + crop_left;
            if (out_idx < 0) {
                int offset = -out_idx % len;
                out_idx = offset;
            } else if (out_idx >= len) {
                int offset = (out_idx + 1) % len;
                out_idx = len - offset - 1;
            }
        }
        return out_idx;
    }

    template<int MODE, typename T>
    __global__ void resizeWith_(const T* inputs, uint input_pitch, uint3_t input_shape, uint input_elements,
                                T* outputs, uint output_pitch, uint3_t output_shape, uint output_elements,
                                int3_t crop_left, int3_t pad_left, uint batches) {
        uint o_y = blockIdx.y;
        uint o_z = blockIdx.z;
        int3_t int_input_shape(input_shape);

        uint i_z = getBorderIndex_<MODE>(o_z, pad_left.z, crop_left.z, int_input_shape.z);
        uint i_y = getBorderIndex_<MODE>(o_y, pad_left.y, crop_left.y, int_input_shape.y);

        outputs += (o_z * output_shape.y + o_y) * output_pitch;
        inputs += (i_z * input_shape.y + i_y) * input_pitch;
        for (uint o_x = blockIdx.x * blockDim.x + threadIdx.x; o_x < output_shape.x; o_x += blockDim.x * gridDim.x) {
            uint i_x = getBorderIndex_<MODE>(o_x, pad_left.x, crop_left.x, int_input_shape.x);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * output_elements + o_x] = inputs[batch * input_elements + i_x];
        }
    }

    template<int MODE, typename T>
    void launchResizeWith_(const T* inputs, uint input_pitch, uint3_t input_shape,
                           T* outputs, uint output_pitch, uint3_t output_shape,
                           int3_t border_left, int3_t border_right,
                           uint batches, cuda::Stream& stream) {
        uint input_elements = getRows(input_shape) * input_pitch;
        uint output_elements = getRows(output_shape) * output_pitch;
        int3_t crop_left(math::min(border_left, 0) * -1);
        int3_t pad_left(math::max(border_left, 0));
        int3_t pad_right(math::max(border_right, 0));

        if constexpr (MODE == BORDER_MIRROR) {
            int3_t int_input_shape(input_shape);
            if (any(pad_left > int_input_shape) || any(pad_right > int_input_shape))
                Session::logger.warn("Edge case: {} used with padding larger than the original shape. "
                                     "This might not produce the expect result. "
                                     "Got: pad_left={}, pad_right={}, input_shape={}",
                                     BORDER_MIRROR, pad_left, pad_right, int_input_shape);
        } else if constexpr (MODE == BORDER_REFLECT) {
            int3_t int_input_shape(input_shape);
            if (any(pad_left >= int_input_shape) || any(pad_right >= int_input_shape))
                Session::logger.warn("Edge case: {} used with padding larger or equal than the original "
                                     "shape. This might not produce the expect result. "
                                     "Got: pad_left={}, pad_right={}, input_shape={}",
                                     BORDER_REFLECT, pad_left, pad_right, int_input_shape);
        }

        uint threads = math::min(256U, math::nextMultipleOf(output_shape.x, 32U));
        dim3 blocks{(output_shape.x + threads - 1) / threads, output_shape.y, output_shape.z};
        resizeWith_<MODE><<<blocks, threads, 0, stream.id()>>>(
                inputs, input_pitch, input_shape, input_elements,
                outputs, output_pitch, output_shape, output_elements,
                crop_left, pad_left, batches);
    }
}

namespace noa::cuda::memory {
    template<typename T>
    void resize(const T* inputs, size_t input_pitch, size3_t input_shape, int3_t border_left, int3_t border_right,
                T* outputs, size_t output_pitch, BorderMode border_mode, T border_value,
                uint batches, Stream& stream) {
        if (all(border_left == 0) && all(border_right == 0)) {
            copy(inputs, input_pitch, outputs, output_pitch, {input_shape.x, getRows(input_shape), batches});
            return;
        }

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
                                                border_left, border_right, batches, stream);
                break;
            case BORDER_PERIODIC:
                launchResizeWith_<BORDER_PERIODIC>(inputs, input_pitch, uint3_t(input_shape),
                                                   outputs, output_pitch, output_shape,
                                                   border_left, border_right, batches, stream);
                break;
            case BORDER_REFLECT:
                launchResizeWith_<BORDER_REFLECT>(inputs, input_pitch, uint3_t(input_shape),
                                                  outputs, output_pitch, output_shape,
                                                  border_left, border_right, batches, stream);
                break;
            case BORDER_MIRROR:
                launchResizeWith_<BORDER_MIRROR>(inputs, input_pitch, uint3_t(input_shape),
                                                 outputs, output_pitch, output_shape,
                                                 border_left, border_right, batches, stream);
                break;
            default:
                NOA_THROW("BorderMode not recognized. Got: {}", border_mode);
        }
        NOA_THROW_IF(cudaPeekAtLastError());
    }

    #define INSTANTIATE_RESIZE(T) \
    template void resize<T>(const T*, size_t, size3_t, int3_t, int3_t, T*, size_t, BorderMode, T, uint, Stream&)

    INSTANTIATE_RESIZE(float);
    INSTANTIATE_RESIZE(double);
    INSTANTIATE_RESIZE(bool);
    INSTANTIATE_RESIZE(char);
    INSTANTIATE_RESIZE(short);
    INSTANTIATE_RESIZE(int);
    INSTANTIATE_RESIZE(long);
    INSTANTIATE_RESIZE(long long);
    INSTANTIATE_RESIZE(unsigned char);
    INSTANTIATE_RESIZE(unsigned short);
    INSTANTIATE_RESIZE(unsigned int);
    INSTANTIATE_RESIZE(unsigned long);
    INSTANTIATE_RESIZE(unsigned long long);
}