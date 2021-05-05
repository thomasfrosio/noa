#include "noa/gpu/cuda/memory/Resize.h"
#include "noa/Session.h"
#include "Copy.h"

namespace {
    using namespace Noa;

    template<typename T>
    __global__ void resizeWithNothing(const T* inputs, uint input_pitch, uint3_t input_shape, uint input_elements,
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
    NOA_HOST void launchResizeWithNothing(const T* inputs, uint input_pitch, uint3_t input_shape,
                                          T* outputs, uint output_pitch, uint3_t output_shape,
                                          int3_t border_left, int3_t border_right, uint batches, CUDA::Stream& stream) {
        uint input_elements = getRows(input_shape) * input_pitch;
        uint output_elements = getRows(output_shape) * output_pitch;
        int3_t crop_left(Math::min(border_left, 0) * -1);
        int3_t pad_left(Math::max(border_left, 0));
        int3_t pad_right(Math::max(border_right, 0));

        uint threads = Math::min(256U, Math::nextMultipleOf(output_shape.x, 32U));
        dim3 blocks{(output_shape.x + threads - 1) / threads, output_shape.y, output_shape.z};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        resizeWithNothing,
                        inputs, input_pitch, input_shape, input_elements,
                        outputs, output_pitch, output_shape, output_elements,
                        crop_left, pad_left, pad_right, batches);
    }

    template<typename T>
    __global__ void resizeWithValue(const T* inputs, uint input_pitch, uint3_t input_shape, uint input_elements,
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
    NOA_HOST void launchResizeWithValue(const T* inputs, uint input_pitch, uint3_t input_shape,
                                        T* outputs, uint output_pitch, uint3_t output_shape,
                                        int3_t border_left, int3_t border_right, T value,
                                        uint batches, CUDA::Stream& stream) {
        uint input_elements = getRows(input_shape) * input_pitch;
        uint output_elements = getRows(output_shape) * output_pitch;
        int3_t crop_left(Math::min(border_left, 0) * -1);
        int3_t pad_left(Math::max(border_left, 0));
        int3_t pad_right(Math::max(border_right, 0));

        uint threads = Math::min(256U, Math::nextMultipleOf(output_shape.x, 32U));
        dim3 blocks{(output_shape.x + threads - 1) / threads, output_shape.y, output_shape.z};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        resizeWithValue,
                        inputs, input_pitch, input_shape, input_elements,
                        outputs, output_pitch, output_shape, output_elements,
                        crop_left, pad_left, pad_right, value, batches);
    }

    template<int MODE>
    NOA_ID int getBorderIndex(int idx, int pad_left, int crop_left, int len) {
        static_assert(MODE == BORDER_CLAMP || MODE == BORDER_PERIODIC || MODE == BORDER_MIRROR);
        int out_idx;
        if constexpr (MODE == BORDER_CLAMP) {
            out_idx = Math::max(0, Math::min(idx - pad_left + crop_left, len - 1));
        } else if constexpr (MODE == BORDER_PERIODIC) {
            int rem = (idx - pad_left + crop_left) % len;
            out_idx = rem < 0 ? rem + len : rem;
        } else if constexpr (MODE == BORDER_MIRROR) {
            out_idx = idx - pad_left + crop_left;
            if (out_idx < 0) {
                int offset = (Math::abs(out_idx) - 1) % len;
                out_idx = offset;
            } else if (out_idx >= len) {
                int offset = Math::abs(out_idx) % len;
                out_idx = len - offset - 1;
            }
        }
        return out_idx;
    }

    template<int MODE, typename T>
    __global__ void resizeWith(const T* inputs, uint input_pitch, uint3_t input_shape, uint input_elements,
                               T* outputs, uint output_pitch, uint3_t output_shape, uint output_elements,
                               int3_t crop_left, int3_t pad_left, uint batches) {
        uint o_y = blockIdx.y;
        uint o_z = blockIdx.z;
        int3_t int_input_shape(input_shape);

        uint i_z = getBorderIndex<MODE>(o_z, pad_left.z, crop_left.z, int_input_shape.z);
        uint i_y = getBorderIndex<MODE>(o_y, pad_left.y, crop_left.y, int_input_shape.y);

        outputs += (o_z * output_shape.y + o_y) * output_pitch;
        inputs += (i_z * input_shape.y + i_y) * input_pitch;
        for (uint o_x = blockIdx.x * blockDim.x + threadIdx.x; o_x < output_shape.x; o_x += blockDim.x * gridDim.x) {
            uint i_x = getBorderIndex<MODE>(o_x, pad_left.x, crop_left.x, int_input_shape.x);
            for (uint batch = 0; batch < batches; ++batch)
                outputs[batch * output_elements + o_x] = inputs[batch * input_elements + i_x];
        }
    }

    template<int MODE, typename T>
    NOA_HOST void launchResizeWith(const T* inputs, uint input_pitch, uint3_t input_shape,
                                   T* outputs, uint output_pitch, uint3_t output_shape,
                                   int3_t border_left, int3_t border_right,
                                   uint batches, CUDA::Stream& stream) {
        uint input_elements = getRows(input_shape) * input_pitch;
        uint output_elements = getRows(output_shape) * output_pitch;
        int3_t crop_left(Math::min(border_left, 0) * -1);
        int3_t pad_left(Math::max(border_left, 0));
        int3_t pad_right(Math::max(border_right, 0));

        if constexpr (MODE == BORDER_MIRROR) {
            int3_t int_input_shape(input_shape);
            if (pad_left > int_input_shape || pad_right > int_input_shape)
                Session::logger.warn("Edge case: BORDER_MIRROR used with padding larger than the original shape. "
                                     "This might not produce the expect result. "
                                     "Got: pad_left={}, pad_right={}, input_shape={}",
                                     pad_left, pad_right, int_input_shape);
        }

        uint threads = Math::min(256U, Math::nextMultipleOf(output_shape.x, 32U));
        dim3 blocks{(output_shape.x + threads - 1) / threads, output_shape.y, output_shape.z};
        NOA_CUDA_LAUNCH(blocks, threads, 0, stream.id(),
                        resizeWith<MODE>,
                        inputs, input_pitch, input_shape, input_elements,
                        outputs, output_pitch, output_shape, output_elements,
                        crop_left, pad_left, batches);
    }
}

namespace Noa::CUDA::Memory {
    template<typename T>
    void resize(const T* inputs, size_t input_pitch, size3_t input_shape,
                T* outputs, size_t output_pitch, size3_t output_shape,
                int3_t border_left, int3_t border_right, BorderMode border_mode, T border_value,
                uint batches, Stream& stream) {
        if (int3_t(input_shape) + border_left + border_right != int3_t(output_shape)) {
            NOA_THROW("Cannot resize an array with shape {} to {} given the borders left:{}, right:{}",
                      input_shape, output_shape, border_left, border_right);
        } else if (inputs == outputs) {
            NOA_THROW("In-place resizing is not allowed");
        } else if (border_left == 0 && border_right == 0) {
            copy(inputs, input_pitch, outputs, output_pitch, {input_shape.x, input_shape.y * input_shape.z, batches});
            return;
        }

        if (border_mode == BORDER_NOTHING)
            launchResizeWithNothing(inputs, input_pitch, uint3_t(input_shape),
                                    outputs, output_pitch, uint3_t(output_shape),
                                    border_left, border_right, batches, stream);
        else if (border_mode == BORDER_ZERO)
            launchResizeWithValue(inputs, input_pitch, uint3_t(input_shape),
                                  outputs, output_pitch, uint3_t(output_shape),
                                  border_left, border_right, T{0}, batches, stream);
        else if (border_mode == BORDER_VALUE)
            launchResizeWithValue(inputs, input_pitch, uint3_t(input_shape),
                                  outputs, output_pitch, uint3_t(output_shape),
                                  border_left, border_right, border_value, batches, stream);
        else if (border_mode == BORDER_CLAMP)
            launchResizeWith<BORDER_CLAMP>(inputs, input_pitch, uint3_t(input_shape),
                                           outputs, output_pitch, uint3_t(output_shape),
                                           border_left, border_right, batches, stream);
        else if (border_mode == BORDER_PERIODIC)
            launchResizeWith<BORDER_PERIODIC>(inputs, input_pitch, uint3_t(input_shape),
                                              outputs, output_pitch, uint3_t(output_shape),
                                              border_left, border_right, batches, stream);
        else if (border_mode == BORDER_MIRROR)
            launchResizeWith<BORDER_MIRROR>(inputs, input_pitch, uint3_t(input_shape),
                                            outputs, output_pitch, uint3_t(output_shape),
                                            border_left, border_right, batches, stream);
        else
            NOA_THROW("BorderMode not recognized. Got: {}", border_mode);
    }

    #define INSTANTIATE_RESIZE(T) \
    template void resize<T>(const T*, size_t, size3_t, T*, size_t, size3_t, int3_t, int3_t, BorderMode, T, uint, Stream&)

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
