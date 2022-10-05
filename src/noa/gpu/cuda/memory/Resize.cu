#include "noa/common/Assert.h"
#include "noa/gpu/cuda/Exception.h"
#include "noa/gpu/cuda/memory/Resize.h"
#include "noa/gpu/cuda/memory/Copy.h"

namespace {
    using namespace noa;
    constexpr uint32_t BLOCK_SIZE = 512;
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr uint32_t ELEMENT_PER_THREAD = 4;
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x * ELEMENT_PER_THREAD, BLOCK_SIZE_2D.y);

    // Computes two elements per thread.
    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void resizeWithNothing_(AccessorRestrict<const T, 4, uint32_t> input,
                            AccessorRestrict<T, 4, uint32_t> output, uint2_t output_shape /* YX */,
                            int4_t crop_left, int4_t pad_left, int4_t pad_right, uint32_t blocks_x) {
        const uint2_t idx = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         BLOCK_WORK_SIZE_2D.y * idx[0] + threadIdx.y,
                         BLOCK_WORK_SIZE_2D.x * idx[1] + threadIdx.x};

        // If within the padding, stop.
        if (gid[0] < pad_left[0] || gid[0] >= static_cast<int32_t>(gridDim.z) - pad_right[0] ||
            gid[1] < pad_left[1] || gid[1] >= static_cast<int32_t>(gridDim.y) - pad_right[1] ||
            gid[2] < pad_left[2] || gid[2] >= static_cast<int32_t>(output_shape[0]) - pad_right[2])
            return;

        const int32_t ii = gid[0] - pad_left[0] + crop_left[0]; // cannot be negative
        const int32_t ij = gid[1] - pad_left[1] + crop_left[1];
        const int32_t ik = gid[2] - pad_left[2] + crop_left[2];

        const auto input_row = input[ii][ij][ik];
        const auto output_row = output[gid[0]][gid[1]][gid[2]];

        for (int32_t i = 0; i < ELEMENT_PER_THREAD; ++i) {
            const int32_t ol = gid[3] + static_cast<int32_t>(BLOCK_SIZE_2D.x) * i;
            if (ol >= pad_left[3] && ol < static_cast<int32_t>(output_shape[1]) - pad_right[3]) {
                const int32_t il = ol - pad_left[3] + crop_left[3]; // cannot be negative
                output_row[ol] = input_row[il];
            }
        }
    }

    template<typename T>
    void launchResizeWithNothing_(const shared_t<T[]>& input, uint4_t input_strides,
                                  const shared_t<T[]>& output, uint4_t output_strides, uint4_t output_shape,
                                  int4_t border_left, int4_t border_right, cuda::Stream& stream) {
        const int4_t crop_left(math::min(border_left, 0) * -1);
        const int4_t pad_left(math::max(border_left, 0));
        const int4_t pad_right(math::max(border_right, 0));

        const uint2_t uint_shape(output_shape.get(2));
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_WORK_SIZE_2D.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks{blocks_x * blocks_y, output_shape[1], output_shape[0]};
        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), input_strides);
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), output_strides);

        stream.enqueue("memory::resizeWithNothing", resizeWithNothing_<T>, {blocks, BLOCK_SIZE_2D},
                       input_accessor, output_accessor, uint_shape, crop_left, pad_left, pad_right, blocks_x);
        stream.attach(input, output);
    }

    template<typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void resizeWithValue_(AccessorRestrict<const T, 4, uint32_t> input,
                          AccessorRestrict<T, 4, uint32_t> output, uint2_t output_shape /* YX */,
                          int4_t crop_left, int4_t pad_left, int4_t pad_right, T value, uint32_t blocks_x) {
        const uint2_t idx = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t ogid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE_2D.y * idx[0] + threadIdx.y,
                          BLOCK_WORK_SIZE_2D.x * idx[1] + threadIdx.x};
        if (ogid[2] >= output_shape[0])
            return;

        const bool is_valid = ogid[0] >= pad_left[0] && ogid[0] < static_cast<int32_t>(gridDim.z) - pad_right[0] &&
                              ogid[1] >= pad_left[1] && ogid[1] < static_cast<int32_t>(gridDim.y) - pad_right[1] &&
                              ogid[2] >= pad_left[2] && ogid[2] < static_cast<int32_t>(output_shape[0]) - pad_right[2];

        const int32_t ii = ogid[0] - pad_left[0] + crop_left[0]; // can be negative, but is_valid protects against it-
        const int32_t ij = ogid[1] - pad_left[1] + crop_left[1];
        const int32_t ik = ogid[2] - pad_left[2] + crop_left[2];
        // Cast the indexes here, since the accessor asserts against negative indexes and loss of range.
        // In this case, we allow it since we precompute the offset but only use it when the index is valid.
        const auto input_row = input[static_cast<uint32_t>(ii)][static_cast<uint32_t>(ij)][static_cast<uint32_t>(ik)];
        const auto output_row = output[ogid[0]][ogid[1]][ogid[2]];

        for (int32_t i = 0; i < ELEMENT_PER_THREAD; ++i) {
            const int32_t ol = ogid[3] + static_cast<int32_t>(BLOCK_SIZE_2D.x) * i;
            if (ol >= output_shape[1])
                return;

            if (is_valid && ol >= pad_left[3] && ol < static_cast<int32_t>(output_shape[1]) - pad_right[3]) {
                const auto il = static_cast<uint32_t>(ol - pad_left[3] + crop_left[3]); // cannot be negative
                output_row[ol] = input_row[il];
            } else {
                output_row[ol] = value;
            }
        }
    }

    template<typename T>
    void launchResizeWithValue_(const shared_t<T[]>& input, uint4_t input_strides,
                                const shared_t<T[]>& output, uint4_t output_strides, uint4_t output_shape,
                                int4_t border_left, int4_t border_right, T value, cuda::Stream& stream) {
        const int4_t crop_left(math::min(border_left, 0) * -1);
        const int4_t pad_left(math::max(border_left, 0));
        const int4_t pad_right(math::max(border_right, 0));

        const uint2_t uint_shape(output_shape.get(2));
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_WORK_SIZE_2D.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks{blocks_x * blocks_y, output_shape[1], output_shape[0]};
        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), input_strides);
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), output_strides);

        stream.enqueue("memory::resizeWithValue", resizeWithValue_<T>, {blocks, BLOCK_SIZE_2D},
                       input_accessor, output_accessor, uint_shape, crop_left, pad_left, pad_right, value, blocks_x);
        stream.attach(input, output);
    }

    template<BorderMode MODE, typename T>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void resizeWith_(AccessorRestrict<const T, 4, uint32_t> input, uint4_t input_shape,
                     AccessorRestrict<T, 4, uint32_t> output, uint2_t output_shape /* YX */,
                     int4_t crop_left, int4_t pad_left, uint32_t blocks_x) {
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
        igid[0] = indexing::at<MODE>(igid[0], static_cast<int32_t>(input_shape[0]));
        igid[1] = indexing::at<MODE>(igid[1], static_cast<int32_t>(input_shape[1]));
        igid[2] = indexing::at<MODE>(igid[2], static_cast<int32_t>(input_shape[2]));
        const auto input_row = input[igid[0]][igid[1]][igid[2]];
        const auto output_row = output[ogid[0]][ogid[1]][ogid[2]];

        for (int32_t i = 0; i < ELEMENT_PER_THREAD; ++i) {
            const int32_t ol = ogid[3] + static_cast<int32_t>(BLOCK_SIZE_2D.x) * i;
            if (ol >= output_shape[1])
                return;
            int32_t il = ol - pad_left[3] + crop_left[3];
            il = indexing::at<MODE>(il, static_cast<int32_t>(input_shape[3]));
            output_row[ol] = input_row[il];
        }
    }

    template<BorderMode MODE, typename T>
    void launchResizeWith_(const shared_t<T[]>& input, uint4_t input_strides, uint4_t input_shape,
                           const shared_t<T[]>& output, uint4_t output_strides, uint4_t output_shape,
                           int4_t border_left, cuda::Stream& stream) {
        const int4_t crop_left(math::min(border_left, 0) * -1);
        const int4_t pad_left(math::max(border_left, 0));

        const uint2_t uint_shape(output_shape.get(2));
        const uint32_t blocks_x = math::divideUp(uint_shape[1], BLOCK_WORK_SIZE_2D.x);
        const uint32_t blocks_y = math::divideUp(uint_shape[0], BLOCK_WORK_SIZE_2D.y);
        const dim3 blocks{blocks_x * blocks_y, output_shape[1], output_shape[0]};
        const AccessorRestrict<const T, 4, uint32_t> input_accessor(input.get(), input_strides);
        const AccessorRestrict<T, 4, uint32_t> output_accessor(output.get(), output_strides);

        stream.enqueue("memory::resizeWith", resizeWith_<MODE, T>, {blocks, BLOCK_SIZE_2D},
                       input_accessor, input_shape, output_accessor, uint_shape,
                       crop_left, pad_left, blocks_x);
        stream.attach(input, output);
    }
}

namespace noa::cuda::memory {
    template<typename T, typename>
    void resize(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                int4_t border_left, int4_t border_right,
                const shared_t<T[]>& output, dim4_t output_strides,
                BorderMode border_mode, T border_value, Stream& stream) {
        if (all(border_left == 0) && all(border_right == 0))
            return copy(input, input_strides, output, output_strides, input_shape, stream);

        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());
        const auto tmp = safe_cast<int4_t>(input_shape) + border_left + border_right;
        NOA_ASSERT(all(tmp >= 1));
        uint4_t output_shape(tmp);

        // Optimize reads/writes for output:
        auto input_strides_ = safe_cast<uint4_t>(input_strides);
        auto output_strides_ = safe_cast<uint4_t>(output_strides);
        auto input_shape_ = safe_cast<uint4_t>(input_shape);
        const auto order = indexing::order(output_strides_, output_shape);
        if (!all(order == uint4_t{0, 1, 2, 3})) {
            input_strides_ = indexing::reorder(input_strides_, order);
            input_shape_ = indexing::reorder(input_shape_, order);
            border_left = indexing::reorder(border_left, order);
            border_right = indexing::reorder(border_right, order);
            output_strides_ = indexing::reorder(output_strides_, order);
            output_shape = indexing::reorder(output_shape, order);
        }

        switch (border_mode) {
            case BORDER_NOTHING:
                return launchResizeWithNothing_(input, input_strides_,
                                                output, output_strides_, output_shape,
                                                border_left, border_right, stream);
            case BORDER_ZERO:
                return launchResizeWithValue_(input, input_strides_,
                                              output, output_strides_, output_shape,
                                              border_left, border_right, T{0}, stream);
            case BORDER_VALUE:
                return launchResizeWithValue_(input, input_strides_,
                                              output, output_strides_, output_shape,
                                              border_left, border_right, border_value, stream);
            case BORDER_CLAMP:
                return launchResizeWith_<BORDER_CLAMP>(input, input_strides_, input_shape_,
                                                       output, output_strides_, output_shape,
                                                       border_left, stream);
            case BORDER_PERIODIC:
                return launchResizeWith_<BORDER_PERIODIC>(input, input_strides_, input_shape_,
                                                          output, output_strides_, output_shape,
                                                          border_left, stream);
            case BORDER_REFLECT:
                return launchResizeWith_<BORDER_REFLECT>(input, input_strides_, input_shape_,
                                                         output, output_strides_, output_shape,
                                                         border_left, stream);
            case BORDER_MIRROR:
                return launchResizeWith_<BORDER_MIRROR>(input, input_strides_, input_shape_,
                                                        output, output_strides_, output_shape,
                                                        border_left, stream);
            default:
                NOA_THROW("BorderMode not supported. Got: {}", border_mode);
        }
    }

    #define NOA_INSTANTIATE_RESIZE_(T) \
    template void resize<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, int4_t, int4_t, const shared_t<T[]>&, dim4_t, BorderMode, T, Stream&)

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
