#include "noa/common/Math.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/Set.h"
#include "noa/gpu/cuda/fft/Exception.h"
#include "noa/gpu/cuda/fft/Resize.h"
#include "noa/gpu/cuda/util/Pointers.h"

namespace {
    using namespace noa;
    constexpr uint32_t MAX_THREADS = 512;
    constexpr uint32_t ELEMENTS_PER_THREAD_X = 2;
    constexpr dim3 BLOCK_SIZE(32, MAX_THREADS / 32);
    constexpr dim3 BLOCK_WORK_SIZE(BLOCK_SIZE.x * ELEMENTS_PER_THREAD_X, BLOCK_SIZE.y);

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void cropH2H_(AccessorRestrict<const T, 4, uint32_t> input, uint3_t input_shape,
                  AccessorRestrict<T, 4, uint32_t> output, uint3_t output_shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x};
        if (gid[2] >= output_shape[1])
            return;

        const uint32_t iz = gid[1] < (output_shape[0] + 1) / 2 ? gid[1] : gid[1] + input_shape[0] - output_shape[0];
        const uint32_t iy = gid[2] < (output_shape[1] + 1) / 2 ? gid[2] : gid[2] + input_shape[1] - output_shape[1];
        const auto input_row = input[gid[0]][iz][iy];
        const auto output_row = output[gid[0]][gid[1]][gid[2]];

        for (int32_t i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint32_t x = gid[3] + BLOCK_SIZE.x * i;
            if (x < output_shape[2] / 2 + 1)
                output_row[x] = input_row[x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void cropF2F_(AccessorRestrict<const T, 4, uint32_t> input, uint3_t input_shape,
                  AccessorRestrict<T, 4, uint32_t> output, uint3_t output_shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x};
        if (gid[2] >= output_shape[1])
            return;

        const uint32_t iz = gid[1] < (output_shape[0] + 1) / 2 ? gid[1] : gid[1] + input_shape[0] - output_shape[0];
        const uint32_t iy = gid[2] < (output_shape[1] + 1) / 2 ? gid[2] : gid[2] + input_shape[1] - output_shape[1];
        const auto input_row = input[gid[0]][iz][iy];
        const auto output_row = output[gid[0]][gid[1]][gid[2]];

        for (int32_t i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint32_t ox = gid[3] + BLOCK_SIZE.x * i;
            const uint32_t ix = ox < (output_shape[2] + 1) / 2 ? ox : ox + input_shape[2] - output_shape[2];
            if (ox < output_shape[2])
                output_row[ox] = input_row[ix];
        }
    }

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void padH2H_(AccessorRestrict<const T, 4, uint32_t> input, uint3_t input_shape,
                 AccessorRestrict<T, 4, uint32_t> output, uint3_t output_shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x};
        if (gid[2] >= input_shape[1])
            return;

        const uint32_t oz = gid[1] < (input_shape[0] + 1) / 2 ? gid[1] : gid[1] + output_shape[0] - input_shape[0];
        const uint32_t oy = gid[2] < (input_shape[1] + 1) / 2 ? gid[2] : gid[2] + output_shape[1] - input_shape[1];
        const auto input_row = input[gid[0]][gid[1]][gid[2]];
        const auto output_row = output[gid[0]][oz][oy];

        for (int32_t i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint32_t x = gid[3] + BLOCK_SIZE.x * i;
            if (x < input_shape[2] / 2 + 1)
                output_row[x] = input_row[x];
        }
    }

    template<class T>
    __global__ __launch_bounds__(BLOCK_SIZE.x * BLOCK_SIZE.y)
    void padF2F_(AccessorRestrict<const T, 4, uint32_t> input, uint3_t input_shape,
                 AccessorRestrict<T, 4, uint32_t> output, uint3_t output_shape, uint32_t blocks_x) {
        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          BLOCK_WORK_SIZE.y * index[0] + threadIdx.y,
                          BLOCK_WORK_SIZE.x * index[1] + threadIdx.x};
        if (gid[2] >= input_shape[1])
            return;

        const uint32_t oz = gid[1] < (input_shape[0] + 1) / 2 ? gid[1] : gid[1] + output_shape[0] - input_shape[0];
        const uint32_t oy = gid[2] < (input_shape[1] + 1) / 2 ? gid[2] : gid[2] + output_shape[1] - input_shape[1];
        const auto input_row = input[gid[0]][gid[1]][gid[2]];
        const auto output_row = output[gid[0]][oz][oy];

        for (int32_t i = 0; i < ELEMENTS_PER_THREAD_X; ++i) {
            const uint32_t ix = gid[3] + BLOCK_SIZE.x * i;
            if (ix < input_shape[2]) {
                const uint32_t ox = ix < (input_shape[2] + 1) / 2 ? ix : ix + output_shape[2] - input_shape[2];
                output_row[ox] = input_row[ix];
            }
        }
    }
}

namespace noa::cuda::fft::details {
    template<typename T>
    void cropH2H(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        if (all(input_shape == output_shape))
            return memory::copy(input, input_strides, output, output_strides, input_shape.fft(), stream);

        const auto old_shape = safe_cast<uint3_t>(dim3_t(input_shape.get(1)));
        const auto new_shape = safe_cast<uint3_t>(dim3_t(output_shape.get(1)));
        const uint32_t blocks_x = math::divideUp(new_shape[2] / 2 + 1, BLOCK_WORK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(new_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape[0], output_shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("cropH2H_", cropH2H_<T>, {blocks, BLOCK_SIZE},
                       input_, old_shape, output_, new_shape, blocks_x);
        stream.attach(input, output);
    }

    template<typename T>
    void cropF2F(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                 const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        if (all(input_shape == output_shape))
            return memory::copy(input, input_strides, output, output_strides, input_shape, stream);

        const auto old_shape = safe_cast<uint3_t>(dim3_t(input_shape.get(1)));
        const auto new_shape = safe_cast<uint3_t>(dim3_t(output_shape.get(1)));
        const uint32_t blocks_x = math::divideUp(new_shape[2], BLOCK_WORK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(new_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, new_shape[0], output_shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("cropF2F_", cropF2F_<T>, {blocks, BLOCK_SIZE},
                       input_, old_shape, output_, new_shape, blocks_x);
        stream.attach(input, output);
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padH2H(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        if (all(input_shape == output_shape))
            return memory::copy(input, input_strides, output, output_strides, input_shape.fft(), stream);

        memory::set(output, output_strides, output_shape.fft(), T{0}, stream);
        const auto old_shape = safe_cast<uint3_t>(dim3_t(input_shape.get(1)));
        const auto new_shape = safe_cast<uint3_t>(dim3_t(output_shape.get(1)));
        const uint32_t blocks_x = math::divideUp(old_shape[2] / 2 + 1, BLOCK_WORK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(old_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape[0], output_shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("padH2H_", padH2H_<T>, {blocks, BLOCK_SIZE},
                       input_, old_shape, output_, new_shape, blocks_x);
        stream.attach(input, output);
    }

    // TODO(TF) Replace memset with a single kernel that loops through padded regions as well.
    template<typename T>
    void padF2F(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
                const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input.get(), stream.device());
        NOA_ASSERT_DEVICE_PTR(output.get(), stream.device());

        if (all(input_shape == output_shape))
            return memory::copy(input, input_strides, output, output_strides, input_shape, stream);

        memory::set(output, output_strides, output_shape, T{0}, stream);
        const auto old_shape = safe_cast<uint3_t>(dim3_t(input_shape.get(1)));
        const auto new_shape = safe_cast<uint3_t>(dim3_t(output_shape.get(1)));
        const uint32_t blocks_x = math::divideUp(old_shape[2], BLOCK_WORK_SIZE.x);
        const uint32_t blocks_y = math::divideUp(old_shape[1], BLOCK_WORK_SIZE.y);
        const dim3 blocks(blocks_x * blocks_y, old_shape[0], output_shape[0]);

        const AccessorRestrict<const T, 4, uint32_t> input_(input.get(), safe_cast<uint4_t>(input_strides));
        const AccessorRestrict<T, 4, uint32_t> output_(output.get(), safe_cast<uint4_t>(output_strides));

        stream.enqueue("padF2F_", padF2F_<T>, {blocks, BLOCK_SIZE},
                       input_, old_shape, output_, new_shape, blocks_x);
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_CROP_(T)                                                                                \
    template void cropH2H<T>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
    template void cropF2F<T>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);  \
    template void padH2H<T>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);   \
    template void padF2F<T>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_CROP_(half_t);
    NOA_INSTANTIATE_CROP_(float);
    NOA_INSTANTIATE_CROP_(double);
    NOA_INSTANTIATE_CROP_(chalf_t);
    NOA_INSTANTIATE_CROP_(cfloat_t);
    NOA_INSTANTIATE_CROP_(cdouble_t);
}
