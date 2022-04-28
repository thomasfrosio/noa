#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Resize.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace {
    using namespace noa;
    constexpr uint BLOCK_SIZE = 512;
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr uint ELEMENT_PER_THREAD = 4;
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENT_PER_THREAD, BLOCK_SIZE_2D.y);

    // Computes two elements per thread.
    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void resizeWithNothing_(const T* __restrict__ input, uint4_t input_stride,
                            T* __restrict__ output, uint4_t output_stride, uint2_t output_shape /* YX */,
                            int4_t crop_left, int4_t pad_left, int4_t pad_right, uint blocks_x) {
        const uint2_t idx = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * idx[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * idx[1] + threadIdx.x};

        // If within the padding, stop.
        if (gid[0] < pad_left[0] || gid[0] >= static_cast<int>(gridDim.z) - pad_right[0] ||
            gid[1] < pad_left[1] || gid[1] >= static_cast<int>(gridDim.y) - pad_right[1] ||
            gid[2] < pad_left[2] || gid[2] >= static_cast<int>(output_shape[0]) - pad_right[2])
            return;

        const int ii = gid[0] - pad_left[0] + crop_left[0]; // cannot be negative
        const int ij = gid[1] - pad_left[1] + crop_left[1];
        const int ik = gid[2] - pad_left[2] + crop_left[2];

        input += indexing::at(ii, ij, ik, input_stride);
        output += indexing::at(gid[0], gid[1], gid[2], output_stride);

        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            const int ol = gid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            if (ol >= pad_left[3] && ol < static_cast<int>(output_shape[1]) - pad_right[3]) {
                const int il = ol - pad_left[3] + crop_left[3]; // cannot be negative
                output[ol * output_stride[3]] = input[il * input_stride[3]];
            }
        }
    }

    template<typename T>
    void launchResizeWithNothing_(const shared_t<T[]>& input, uint4_t input_stride,
                                  const shared_t<T[]>& output, uint4_t output_stride, uint4_t output_shape,
                                  int4_t border_left, int4_t border_right, cuda::Stream& stream) {
        const int4_t crop_left{math::min(border_left, 0) * -1};
        const int4_t pad_left{math::max(border_left, 0)};
        const int4_t pad_right{math::max(border_right, 0)};

        const uint2_t uint_shape{output_shape.get() + 2};
        const uint blocks_x = math::divideUp(uint_shape[1], BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(uint_shape[0], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks{blocks_x * blocks_y, output_shape[1], output_shape[0]};
        stream.enqueue("memory::resizeWithNothing", resizeWithNothing_<T>, {blocks, BLOCK_SIZE_2D},
                       input.get(), input_stride, output.get(), output_stride, uint_shape,
                       crop_left, pad_left, pad_right, blocks_x);
        stream.attach(input, output);
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void resizeWithValue_(const T* __restrict__ input, uint4_t input_stride,
                          T* __restrict__ output, uint4_t output_stride, uint2_t output_shape /* YX */,
                          int4_t crop_left, int4_t pad_left, int4_t pad_right, T value, uint blocks_x) {
        const uint2_t idx = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t ogid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE_2D.y * idx[0] + threadIdx.y,
                          BLOCK_WORK_SIZE_2D.x * idx[1] + threadIdx.x};
        if (ogid[2] >= output_shape[0])
            return;

        const bool is_valid = ogid[0] >= pad_left[0] && ogid[0] < static_cast<int>(gridDim.z) - pad_right[0] &&
                              ogid[1] >= pad_left[1] && ogid[1] < static_cast<int>(gridDim.y) - pad_right[1] &&
                              ogid[2] >= pad_left[2] && ogid[2] < static_cast<int>(output_shape[0]) - pad_right[2];

        const int ii = ogid[0] - pad_left[0] + crop_left[0]; // cannot be negative
        const int ij = ogid[1] - pad_left[1] + crop_left[1];
        const int ik = ogid[2] - pad_left[2] + crop_left[2];
        input += indexing::at(ii, ij, ik, input_stride);
        output += indexing::at(ogid[0], ogid[1], ogid[2], output_stride);

        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            const int ol = ogid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            if (ol >= output_shape[1])
                return;

            if (is_valid && ol >= pad_left[3] && ol < static_cast<int>(output_shape[1]) - pad_right[3]) {
                const auto il = static_cast<uint>(ol - pad_left[3] + crop_left[3]); // cannot be negative
                output[ol * output_stride[3]] = input[il * input_stride[3]];
            } else {
                output[ol * output_stride[3]] = value;
            }
        }
    }

    template<typename T>
    void launchResizeWithValue_(const shared_t<T[]>& input, uint4_t input_stride,
                                const shared_t<T[]>& output, uint4_t output_stride, uint4_t output_shape,
                                int4_t border_left, int4_t border_right, T value, cuda::Stream& stream) {
        const int4_t crop_left{math::min(border_left, 0) * -1};
        const int4_t pad_left{math::max(border_left, 0)};
        const int4_t pad_right{math::max(border_right, 0)};

        const uint2_t uint_shape{output_shape.get() + 2};
        const uint blocks_x = math::divideUp(uint_shape[1], BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(uint_shape[0], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks{blocks_x * blocks_y, output_shape[1], output_shape[0]};
        stream.enqueue("memory::resizeWithValue", resizeWithValue_<T>, {blocks, BLOCK_SIZE_2D},
                       input.get(), input_stride, output.get(), output_stride, uint_shape,
                       crop_left, pad_left, pad_right, value, blocks_x);
        stream.attach(input, output);
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void resizeWith_(const T* __restrict__ input, uint4_t input_stride, uint4_t input_shape,
                     T* __restrict__ output, uint4_t output_stride, uint2_t output_shape /* YX */,
                     int4_t crop_left, int4_t pad_left, uint blocks_x) {
        const uint2_t idx = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t ogid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE_2D.y * idx[0] + threadIdx.y,
                          BLOCK_WORK_SIZE_2D.x * idx[1] + threadIdx.x};
        if (ogid[2] >= output_shape[0])
            return;

        int3_t igid{ogid[0] - pad_left[0] + crop_left[0],
                    ogid[1] - pad_left[1] + crop_left[1],
                    ogid[2] - pad_left[2] + crop_left[2]};
        igid[0] = getBorderIndex<MODE>(igid[0], static_cast<int>(input_shape[0]));
        igid[1] = getBorderIndex<MODE>(igid[1], static_cast<int>(input_shape[1]));
        igid[2] = getBorderIndex<MODE>(igid[2], static_cast<int>(input_shape[2]));
        input += indexing::at(igid[0], igid[1], igid[2], input_stride);
        output += indexing::at(ogid[0], ogid[1], ogid[2], output_stride);

        for (int i = 0; i < ELEMENT_PER_THREAD; ++i) {
            const int ol = ogid[3] + static_cast<int>(BLOCK_SIZE_2D.x) * i;
            if (ol >= output_shape[1])
                return;
            int il = ol - pad_left[3] + crop_left[3];
            il = getBorderIndex<MODE>(il, static_cast<int>(input_shape[3]));
            output[ol * output_stride[3]] = input[il * input_stride[3]];
        }
    }

    template<BorderMode MODE, typename T>
    void launchResizeWith_(const shared_t<T[]>& input, uint4_t input_stride, uint4_t input_shape,
                           const shared_t<T[]>& output, uint4_t output_stride, uint4_t output_shape,
                           int4_t border_left, cuda::Stream& stream) {
        const int4_t crop_left{math::min(border_left, 0) * -1};
        const int4_t pad_left{math::max(border_left, 0)};

        const uint2_t uint_shape{output_shape.get() + 2};
        const uint blocks_x = math::divideUp(uint_shape[1], BLOCK_WORK_SIZE_2D.x);
        const uint blocks_y = math::divideUp(uint_shape[0], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks{blocks_x * blocks_y, output_shape[1], output_shape[0]};
        stream.enqueue("memory::resizeWith", resizeWith_<MODE, T>, {blocks, BLOCK_SIZE_2D},
                       input.get(), input_stride, input_shape, output.get(), output_stride, uint_shape,
                       crop_left, pad_left, blocks_x);
        stream.attach(input, output);
    }
}

namespace noa::cuda::memory {
    template<typename T, typename>
    void resize(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
                int4_t border_left, int4_t border_right,
                const shared_t<T[]>& output, size4_t output_stride,
                BorderMode border_mode, T border_value, Stream& stream) {
        NOA_ASSERT(input != output);

        if (all(border_left == 0) && all(border_right == 0))
            return copy(input, input_stride, output, output_stride, input_shape, stream);

        const uint4_t output_shape{int4_t{input_shape} + border_left + border_right}; // assumed to be > 0
        switch (border_mode) {
            case BORDER_NOTHING:
                return launchResizeWithNothing_(input, uint4_t{input_stride},
                                                output, uint4_t{output_stride}, output_shape,
                                                border_left, border_right, stream);
            case BORDER_ZERO:
                return launchResizeWithValue_(input, uint4_t{input_stride},
                                              output, uint4_t{output_stride}, output_shape,
                                              border_left, border_right, T{0}, stream);
            case BORDER_VALUE:
                return launchResizeWithValue_(input, uint4_t{input_stride},
                                              output, uint4_t{output_stride}, output_shape,
                                              border_left, border_right, border_value, stream);
            case BORDER_CLAMP:
                return launchResizeWith_<BORDER_CLAMP>(input, uint4_t{input_stride}, uint4_t(input_shape),
                                                       output, uint4_t{output_stride}, output_shape,
                                                       border_left, stream);
            case BORDER_PERIODIC:
                return launchResizeWith_<BORDER_PERIODIC>(input, uint4_t{input_stride}, uint4_t(input_shape),
                                                          output, uint4_t{output_stride}, output_shape,
                                                          border_left, stream);
            case BORDER_REFLECT:
                return launchResizeWith_<BORDER_REFLECT>(input, uint4_t{input_stride}, uint4_t(input_shape),
                                                         output, uint4_t{output_stride}, output_shape,
                                                         border_left, stream);
            case BORDER_MIRROR:
                return launchResizeWith_<BORDER_MIRROR>(input, uint4_t{input_stride}, uint4_t(input_shape),
                                                        output, uint4_t{output_stride}, output_shape,
                                                        border_left, stream);
            default:
                NOA_THROW("BorderMode not supported. Got: {}", border_mode);
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T, void>(const shared_t<T[]>&, size4_t, size4_t, int4_t, int4_t, const shared_t<T[]>&, size4_t, BorderMode, T, Stream&)

    NOA_INSTANTIATE_RESIZE_(bool);
    NOA_INSTANTIATE_RESIZE_(int8_t);
    NOA_INSTANTIATE_RESIZE_(int16_t);
    NOA_INSTANTIATE_RESIZE_(int32_t);
    NOA_INSTANTIATE_RESIZE_(int64_t);
    NOA_INSTANTIATE_RESIZE_(uint8_t);
    NOA_INSTANTIATE_RESIZE_(uint16_t);
    NOA_INSTANTIATE_RESIZE_(uint32_t);
    NOA_INSTANTIATE_RESIZE_(uint64_t);
    NOA_INSTANTIATE_RESIZE_(half_t);
    NOA_INSTANTIATE_RESIZE_(float);
    NOA_INSTANTIATE_RESIZE_(double);
    NOA_INSTANTIATE_RESIZE_(chalf_t);
    NOA_INSTANTIATE_RESIZE_(cfloat_t);
    NOA_INSTANTIATE_RESIZE_(cdouble_t);
}
