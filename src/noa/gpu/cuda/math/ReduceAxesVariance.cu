#include "noa/gpu/cuda/Ewise.hpp"
#include "noa/gpu/cuda/math/Reduce.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr u32 ELEMENTS_PER_THREAD = 4;
    constexpr u32 BLOCK_SIZE = 512;

    // For the variance, it is more difficult since we need to first reduce to get the mean, and then reduce
    // once again using that same mean to get the variance (one mean per element in the reduced axis). This kernel
    // does exactly that...
    template<typename Input, typename Output, bool STDDEV, int32_t BLOCK_DIM_X>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduce_variance_rows_(
            AccessorRestrict<const Input, 4, u32> input, Shape2<u32> shape_hw,
            AccessorRestrict<Output, 3, u32> output, Output inv_count) {
        const u32 tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            blockIdx.x * blockDim.y + threadIdx.y,
                            threadIdx.x};
        const bool is_valid_row = gid[2] < shape_hw[0];

        const auto input_row = input[gid[0]][gid[1]][gid[2]];

        // Get the mean:
        Input reduced{0};
        for (u32 tidx = gid[3]; tidx < shape_hw[1] && is_valid_row; tidx += BLOCK_DIM_X) // compute entire row
            reduced += input_row[tidx]; // TODO Save input values to shared memory?

        // Each row in shared memory is reduced to one value.
        Input* s_data = utils::block_dynamic_shared_resource<Input>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        utils::block_synchronize();
        Input mean = utils::block_reduce_shared<BLOCK_DIM_X>(
                s_data + BLOCK_DIM_X * threadIdx.y, gid[3], noa::plus_t{});

        // Share the mean of the row to all threads within that row:
        if (gid[3] == 0)
            s_data[threadIdx.y] = mean * inv_count;
        utils::block_synchronize();
        mean = s_data[threadIdx.y]; // bank-conflict...

        // Now get the variance:
        Output reduced_dist2{0};
        for (u32 tidx = gid[3]; tidx < shape_hw[1] && is_valid_row; tidx += BLOCK_DIM_X) {  // compute entire row
            Input tmp = input_row[tidx];
            if constexpr (noa::traits::is_complex_v<Input>) {
                Output distance = noa::math::abs(tmp - mean);
                reduced_dist2 += distance * distance;
            } else {
                Output distance = tmp - mean;
                reduced_dist2 += distance * distance;
            }
        }
        auto* s_data_real = reinterpret_cast<Output*>(s_data);
        s_data_real[tid] = reduced_dist2;
        utils::block_synchronize();
        Output var = utils::block_reduce_shared<BLOCK_DIM_X>(
                s_data_real + BLOCK_DIM_X * threadIdx.y, gid[3], noa::plus_t{});
        if (gid[3] == 0 && is_valid_row) {
            var *= inv_count;
            if constexpr (STDDEV)
                var = noa::math::sqrt(var);
            output(gid[0], gid[1], gid[2]) = var;
        }
    }

    // Keep X to one warp to have memory coalescing, even though a half-warp should be OK as well.
    // The Y dimension of the block is where the reduction happens.
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x, BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD);

    template<typename Input, typename Output, bool STDDEV>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduce_variance_dim_(
            const Input* __restrict__ input, Strides4<u32> input_strides, Shape2<u32> shape,
            Output* __restrict__ output, Strides4<u32> output_strides, Output inv_count) {
        const u32 tid = threadIdx.y * blockDim.x + threadIdx.x;
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            threadIdx.y, // one block in the dimension to reduce
                            blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        const bool is_valid_column = gid[3] < shape[1];

        input += indexing::at(gid[0], gid[1], input_strides) + gid[3] * input_strides[3];

        // Get the sum:
        Input reduced_sum{0};
        for (u32 tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) // compute entire row
            reduced_sum += input[tidy * input_strides[2]]; // TODO Save input values to shared memory?

        Input* s_data = utils::block_dynamic_shared_resource<Input>(); // BLOCK_SIZE elements.
        Input* s_data_tid = s_data + tid;
        *s_data_tid = reduced_sum;
        utils::block_synchronize();

        #pragma unroll
        for (u32 SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_tid += s_data_tid[BLOCK_SIZE_2D.x * SIZE / 2];
            utils::block_synchronize();
        }
        Input mean = s_data[threadIdx.x];
        mean *= inv_count;

        // Get the variance:
        Output reduced_dist2{0};
        for (u32 tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) {
            Input tmp = input[tidy * input_strides[2]];
            if constexpr (noa::traits::is_complex_v<Input>) {
                Output distance = noa::math::abs(tmp - mean);
                reduced_dist2 += distance * distance;
            } else {
                Output distance = tmp - mean;
                reduced_dist2 += distance * distance;
            }
        }
        auto* s_data_real = reinterpret_cast<Output*>(s_data);
        Output* s_data_real_tid = s_data_real + tid;
        *s_data_real_tid = reduced_dist2;
        utils::block_synchronize();

        #pragma unroll
        for (u32 SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_real_tid += s_data_real_tid[BLOCK_SIZE_2D.x * SIZE / 2];
            utils::block_synchronize();
        }

        if (gid[2] == 0 && is_valid_column) {
            Output var = *s_data_real_tid; // s_data[threadIdx.x]
            var *= inv_count;
            if constexpr (STDDEV)
                var = noa::math::sqrt(var);
            output[indexing::at(gid[0], gid[1], output_strides) + gid[3] * output_strides[3]] = var;
        }
    }

    template<bool STDDEV, typename Input, typename Output>
    void reduce_variance_axis_(const char* name,
                               const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                               Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                               const Vec4<bool>& mask, int32_t ddof, Stream& stream) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));

        if (noa::math::sum(mask.as<i32>()) > 1) {
            NOA_THROW_FUNC(name,
                           "Reducing more than one axis at a time is only supported if the reduction results in "
                           "one value per batch, i.e. the 3 innermost dimensions are shape=1 after reduction. "
                           "Got input:{}, output:{}, reduce:{}", input_shape, output_shape, mask);
        }

        const auto u_input_strides = input_strides.as_safe<u32>();
        const auto u_input_shape = input_shape.as_safe<u32>();
        const auto u_output_strides = output_strides.as_safe<u32>();

        if (mask[3]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[3] - ddof);
            const u32 block_dim_x = u_input_shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
            const u32 blocks_y = noa::math::divide_up(u_input_shape[2], threads.y);
            const dim3 blocks(blocks_y, u_input_shape[1], u_input_shape[0]);
            const LaunchConfig config{blocks, threads, BLOCK_SIZE * sizeof(Input)};

            const AccessorRestrict<const Input, 4, u32> input_accessor(input, u_input_strides);
            const AccessorRestrict<Output, 3, u32> output_accessor(output, u_output_strides.pop_front());
            if (threads.x == 256) {
                stream.enqueue(name, reduce_variance_rows_<Input, Output, STDDEV, 256>, config,
                               input_accessor, u_input_shape.pop_front<2>(),
                               output_accessor, inv_count);
            } else {
                stream.enqueue(name, reduce_variance_rows_<Input, Output, STDDEV, 64>, config,
                               input_accessor, u_input_shape.pop_front<2>(),
                               output_accessor, inv_count);
            }

        } else if (mask[2]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[2] - ddof);
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[1], u_input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Input)};
            stream.enqueue(name, reduce_variance_dim_<Input, Output, STDDEV>, config,
                           input, u_input_strides, u_input_shape.filter(2, 3),
                           output, u_output_strides, inv_count);

        } else if (mask[1]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[1] - ddof);
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[2], u_input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Input)};
            stream.enqueue(name, reduce_variance_dim_<Input, Output, STDDEV>, config,
                           input, u_input_strides.filter(0, 2, 1, 3), u_input_shape.filter(1, 3),
                           output, u_output_strides.filter(0, 2, 1, 3), inv_count);

        } else if (mask[0]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[0] - ddof);
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[2], u_input_shape[1]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Input)};
            stream.enqueue(name, reduce_variance_dim_<Input, Output, STDDEV>, config,
                           input, u_input_strides.filter(1, 2, 0, 3), u_input_shape.filter(0, 3),
                           output, u_output_strides.filter(1, 2, 0, 3), inv_count);
        }
    }

    Vec4<bool> get_mask_(const char* func, const Shape4<i64>& input_shape, const Shape4<i64>& output_shape) {
        const Vec4<bool> mask{input_shape != output_shape};
        if (noa::any(mask && (output_shape != 1))) {
            NOA_THROW_FUNC(func,
                           "Dimensions should match the input shape, or be 1, indicating the dimension should be "
                           "reduced to one element. Got input:{}, output:{}", input_shape, output_shape);
        }
        return mask;
    }
}

namespace noa::cuda::math {
    template<typename Input, typename Output, typename>
    void var(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, Stream& stream) {
        const char* name = "math::var";
        const auto mask = get_mask_(name, input_shape, output_shape);
        const auto is_or_should_reduce(output_shape == 1 || mask);

        if (!noa::any(mask)) {
            if constexpr (noa::traits::is_complex_v<Input>)
                ewise_unary(input, input_strides, output, output_strides, output_shape, noa::abs_t{}, stream);
            else
                memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            utils::reduce_variance<false>(
                    name, input, input_strides, input_shape,
                    output, output_strides.filter(0),
                    ddof, is_or_should_reduce[0], true, stream);
        } else {
            reduce_variance_axis_<false>(
                    name,
                    input, input_strides, input_shape,
                    output, output_strides, output_shape,
                    mask, ddof, stream);
        }
    }

    template<typename Input, typename Output, typename>
    void std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             i64 ddof, Stream& stream) {
        const char* name = "math::std";
        const auto mask = get_mask_(name, input_shape, output_shape);
        const auto is_or_should_reduce(output_shape == 1 || mask);

        if (!any(mask)) {
            if constexpr (noa::traits::is_complex_v<Input>)
                ewise_unary(input, input_strides, output, output_strides, output_shape, noa::abs_t{}, stream);
            else
                memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            utils::reduce_variance<true>(
                    name, input, input_strides, input_shape,
                    output, output_strides.filter(0),
                    ddof, is_or_should_reduce[0], true, stream);
        } else {
            reduce_variance_axis_<true>(
                    name,
                    input, input_strides, input_shape,
                    output, output_strides, output_shape,
                    mask, ddof, stream);
        }
    }

    #define NOA_INSTANTIATE_VAR_(T,U)                       \
    template void var<T,U,void>(                            \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        U*, const Strides4<i64>&, const Shape4<i64>&,       \
        i64, Stream&);                                      \
    template void std<T,U,void>(                            \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        U*, const Strides4<i64>&, const Shape4<i64>&,       \
        i64, Stream&)

    NOA_INSTANTIATE_VAR_(f32, f32);
    NOA_INSTANTIATE_VAR_(f64, f64);
    NOA_INSTANTIATE_VAR_(c32, f32);
    NOA_INSTANTIATE_VAR_(c64, f64);
}
