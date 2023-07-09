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
    void reduce_variance_width_(
            AccessorRestrict<const Input, 4, u32> input, Shape2<u32> shape_hw,
            AccessorRestrict<Input, 3, u32> output_mean,
            AccessorRestrict<Output, 3, u32> output,
            Output inv_count, Output inv_count_ddof
    ) {
        const u32 tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            blockIdx.x * blockDim.y + threadIdx.y,
                            threadIdx.x};
        const bool is_valid_row = gid[2] < shape_hw[0];

        const auto input_row = input[gid[0]][gid[1]][gid[2]];

        // Get the mean:
        Input reduced{0};
        for (u32 tidx = gid[3]; tidx < shape_hw[1] && is_valid_row; tidx += BLOCK_DIM_X) // compute entire width
            reduced += input_row[tidx]; // TODO Save input values to shared memory?

        // Each row in shared memory is reduced to one value.
        auto* s_data = noa::cuda::utils::block_dynamic_shared_resource<Input>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        noa::cuda::utils::block_synchronize();
        Input sum = noa::cuda::utils::block_reduce_shared<BLOCK_DIM_X>(
                s_data + BLOCK_DIM_X * threadIdx.y, gid[3], noa::plus_t{});

        // Share the mean of the row to all threads within that row.
        // But first, make sure the entire block is done with the reduction before saving
        // to the shared buffer.
        noa::cuda::utils::block_synchronize();
        if (gid[3] == 0)
            s_data[threadIdx.y] = sum * inv_count_ddof;
        noa::cuda::utils::block_synchronize();
        const Input mean = s_data[threadIdx.y]; // bank-conflict...

        // Now get the variance:
        Output reduced_dist2{0};
        for (u32 tidx = gid[3]; tidx < shape_hw[1] && is_valid_row; tidx += BLOCK_DIM_X) { // compute entire width
            Input tmp = input_row[tidx];
            if constexpr (noa::traits::is_complex_v<Input>) {
                Output distance = noa::math::abs(tmp - mean);
                reduced_dist2 += distance * distance;
            } else {
                Output distance = tmp - mean;
                reduced_dist2 += distance * distance;
            }
        }

        // Make sure all threads in the block have read their mean before writing again.
        noa::cuda::utils::block_synchronize();
        auto* s_data_real = reinterpret_cast<Output*>(s_data);
        s_data_real[tid] = reduced_dist2;
        noa::cuda::utils::block_synchronize();
        Output var = noa::cuda::utils::block_reduce_shared<BLOCK_DIM_X>(
                s_data_real + BLOCK_DIM_X * threadIdx.y, gid[3], noa::plus_t{});

        if (gid[3] == 0 && is_valid_row) {
            var *= inv_count_ddof;
            if constexpr (STDDEV)
                var = noa::math::sqrt(var);
            output(gid[0], gid[1], gid[2]) = var;
            if (!output_mean.is_empty())
                output_mean(gid[0], gid[1], gid[2]) = sum * inv_count;
        }
    }

    // Keep X to one warp to have memory coalescing, even though a half-warp should be OK as well.
    // The Y dimension of the block is where the reduction happens.
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);
    constexpr dim3 BLOCK_WORK_SIZE_2D(BLOCK_SIZE_2D.x, BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD);

    template<typename Input, typename Output, bool STDDEV>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduce_variance_height_(
            const Input* __restrict__ input, Strides4<u32> input_strides, Shape2<u32> shape,
            Input* __restrict__ output_mean, Strides4<u32> output_mean_strides,
            Output* __restrict__ output, Strides4<u32> output_strides,
            Output inv_count, Output inv_count_ddof
    ) {
        const u32 tid = threadIdx.y * blockDim.x + threadIdx.x;
        const Vec4<u32> gid{blockIdx.z,
                            blockIdx.y,
                            threadIdx.y, // one block in the dimension to reduce
                            blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        const bool is_valid_column = gid[3] < shape[1];

        input += noa::indexing::at(gid[0], gid[1], 0, gid[3], input_strides);

        // Get the sum:
        Input reduced_sum{0};
        for (u32 tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) // compute entire height
            reduced_sum += input[tidy * input_strides[2]]; // TODO Save input values to shared memory?

        auto* s_data = noa::cuda::utils::block_dynamic_shared_resource<Input>(); // BLOCK_SIZE elements.
        Input* s_data_tid = s_data + tid;
        *s_data_tid = reduced_sum;
        noa::cuda::utils::block_synchronize();

        #pragma unroll
        for (u32 SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_tid += s_data_tid[BLOCK_SIZE_2D.x * (SIZE / 2)];
            noa::cuda::utils::block_synchronize();
        }
        const Input sum = s_data[threadIdx.x]; // bank-conflict
        const Input mean = sum * inv_count_ddof;

        // Get the variance:
        Output reduced_dist2{0};
        for (u32 tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) { // compute entire height
            Input tmp = input[tidy * input_strides[2]];
            if constexpr (noa::traits::is_complex_v<Input>) {
                Output distance = noa::math::abs(tmp - mean);
                reduced_dist2 += distance * distance;
            } else {
                Output distance = tmp - mean;
                reduced_dist2 += distance * distance;
            }
        }

        // Make sure all threads in the block have read their mean before writing again.
        noa::cuda::utils::block_synchronize();
        auto* s_data_real = reinterpret_cast<Output*>(s_data);
        Output* s_data_real_tid = s_data_real + tid;
        *s_data_real_tid = reduced_dist2;
        noa::cuda::utils::block_synchronize();

        #pragma unroll
        for (u32 SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_real_tid += s_data_real_tid[BLOCK_SIZE_2D.x * (SIZE / 2)];
            noa::cuda::utils::block_synchronize();
        }

        if (gid[2] == 0 && is_valid_column) {
            Output var = s_data_real[threadIdx.x];
            var *= inv_count_ddof;
            if constexpr (STDDEV)
                var = noa::math::sqrt(var);
            output[noa::indexing::at(gid[0], gid[1], 0, gid[3], output_strides)] = var;
            if (output_mean)
                output_mean[noa::indexing::at(gid[0], gid[1], 0, gid[3], output_mean_strides)] = sum * inv_count;
        }
    }

    template<bool STDDEV, typename Input, typename Output>
    void reduce_variance_axis_(
            const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            Input* output_mean, const Strides4<i64>& output_mean_strides,
            Output* output, const Strides4<i64>& output_strides,
            const Shape4<i64>& output_shape,
            const Vec4<bool>& mask, int32_t ddof, Stream& stream
    ) {
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());
        NOA_ASSERT_DEVICE_OR_NULL_PTR(output_mean, stream.device());
        NOA_ASSERT(noa::all(input_shape > 0) && noa::all(output_shape > 0));

        if (noa::math::sum(mask.as<i32>()) > 1) {
            NOA_THROW("Reducing more than one axis at a time is only supported if the reduction results in "
                      "one value per batch, i.e. the 3 innermost dimensions are shape=1 after reduction. "
                      "Got input:{}, output:{}, reduce:{}", input_shape, output_shape, mask);
        }

        const auto u_input_strides = input_strides.as_safe<u32>();
        const auto u_input_shape = input_shape.as_safe<u32>();
        const auto u_output_mean_strides = output_mean_strides.as_safe<u32>();
        const auto u_output_strides = output_strides.as_safe<u32>();

        if (mask[3]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[3]);
            const Output inv_count_ddof = Output{1} / static_cast<Output>(u_input_shape[3] - ddof);
            const u32 block_dim_x = u_input_shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
            const u32 blocks_y = noa::math::divide_up(u_input_shape[2], threads.y);
            const dim3 blocks(blocks_y, u_input_shape[1], u_input_shape[0]);
            const LaunchConfig config{blocks, threads, BLOCK_SIZE * sizeof(Input)};

            const AccessorRestrict<const Input, 4, u32> input_accessor(input, u_input_strides);
            const AccessorRestrict<Input, 3, u32> output_mean_accessor(output_mean, u_output_mean_strides.pop_back());
            const AccessorRestrict<Output, 3, u32> output_accessor(output, u_output_strides.pop_back());
            if (threads.x == 256) {
                stream.enqueue(reduce_variance_width_<Input, Output, STDDEV, 256>, config,
                               input_accessor, u_input_shape.pop_front<2>(),
                               output_mean_accessor, output_accessor, inv_count, inv_count_ddof);
            } else {
                stream.enqueue(reduce_variance_width_<Input, Output, STDDEV, 64>, config,
                               input_accessor, u_input_shape.pop_front<2>(),
                               output_mean_accessor, output_accessor, inv_count, inv_count_ddof);
            }

        } else if (mask[2]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[2]);
            const Output inv_count_ddof = Output{1} / static_cast<Output>(u_input_shape[2] - ddof);
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[1], u_input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Input)};
            stream.enqueue(reduce_variance_height_<Input, Output, STDDEV>, config,
                           input, u_input_strides, u_input_shape.filter(2, 3),
                           output_mean, u_output_mean_strides,
                           output, u_output_strides,
                           inv_count, inv_count_ddof);

        } else if (mask[1]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[1]);
            const Output inv_count_ddof = Output{1} / static_cast<Output>(u_input_shape[1] - ddof);
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[2], u_input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Input)};
            stream.enqueue(reduce_variance_height_<Input, Output, STDDEV>, config,
                           input, u_input_strides.filter(0, 2, 1, 3), u_input_shape.filter(1, 3),
                           output_mean, u_output_mean_strides.filter(0, 2, 1, 3),
                           output, u_output_strides.filter(0, 2, 1, 3),
                           inv_count, inv_count_ddof);

        } else if (mask[0]) {
            const Output inv_count = Output{1} / static_cast<Output>(u_input_shape[0]);
            const Output inv_count_ddof = Output{1} / static_cast<Output>(u_input_shape[0] - ddof);
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_WORK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[2], u_input_shape[1]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Input)};
            stream.enqueue(reduce_variance_height_<Input, Output, STDDEV>, config,
                           input, u_input_strides.filter(1, 2, 0, 3), u_input_shape.filter(0, 3),
                           output_mean, u_output_mean_strides.filter(1, 2, 0, 3),
                           output, u_output_strides.filter(1, 2, 0, 3),
                           inv_count, inv_count_ddof);
        }
    }

    Vec4<bool> get_mask_(const Shape4<i64>& input_shape, const Shape4<i64>& output_shape) {
        const Vec4<bool> mask{input_shape != output_shape};
        if (noa::any(mask && (output_shape != 1))) {
            NOA_THROW("Dimensions should match the input shape, or be 1, indicating the dimension should be "
                      "reduced to one element. Got input:{}, output:{}", input_shape, output_shape);
        }
        return mask;
    }
}

namespace noa::cuda::math {
    template<typename Input, typename Output, typename>
    void var(
            const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            i64 ddof, Stream& stream
    ) {
        const auto mask = get_mask_(input_shape, output_shape);
        const auto is_or_should_reduce(output_shape == 1 || mask);

        if (!noa::any(mask)) {
            if constexpr (noa::traits::is_complex_v<Input>)
                noa::cuda::ewise_unary(input, input_strides, output, output_strides, output_shape, noa::abs_t{}, stream);
            else
                noa::cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            Input* null{};
            noa::cuda::utils::reduce_variance<false>(
                    input, input_strides, input_shape,
                    null, output_strides.filter(0),
                    output, output_strides.filter(0),
                    ddof, is_or_should_reduce[0], true, stream);
        } else {
            Input* null{};
            reduce_variance_axis_<false>(
                    input, input_strides, input_shape,
                    null, {}, output, output_strides,
                    output_shape, mask, ddof, stream);
        }
    }

    template<typename Input, typename Output, typename>
    void std(
            const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
            i64 ddof, Stream& stream
    ) {
        const auto mask = get_mask_(input_shape, output_shape);
        const auto is_or_should_reduce(output_shape == 1 || mask);

        if (!any(mask)) {
            if constexpr (noa::traits::is_complex_v<Input>)
                noa::cuda::ewise_unary(input, input_strides, output, output_strides, output_shape, noa::abs_t{}, stream);
            else
                noa::cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            Input* null{};
            noa::cuda::utils::reduce_variance<true>(
                    input, input_strides, input_shape,
                    null, output_strides.filter(0),
                    output, output_strides.filter(0),
                    ddof, is_or_should_reduce[0], true, stream);
        } else {
            Input* null{};
            reduce_variance_axis_<true>(
                    input, input_strides, input_shape,
                    null, {}, output, output_strides,
                    output_shape, mask, ddof, stream);
        }
    }

    template<typename Input, typename Output, typename>
    void mean_var(
            const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
            Input* mean, const Strides4<i64>& mean_strides,
            Output* variance, const Strides4<i64>& variance_strides,
            const Shape4<i64>& output_shape, i64 ddof, Stream& stream
    ) {
        const auto mask = get_mask_(input_shape, output_shape);
        const auto is_or_should_reduce(output_shape == 1 || mask);

        if (!noa::any(mask)) {
            if constexpr (noa::traits::is_complex_v<Input>) {
                noa::cuda::memory::copy(input, input_strides, mean, mean_strides, output_shape, stream);
                noa::cuda::ewise_unary(input, input_strides, variance, variance_strides, output_shape, noa::abs_t{}, stream);
            } else {
                noa::cuda::memory::copy(input, input_strides, mean, mean_strides, output_shape, stream);
                noa::cuda::memory::copy(input, input_strides, variance, variance_strides, output_shape, stream);
            }

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            noa::cuda::utils::reduce_variance<false>(
                    input, input_strides, input_shape,
                    mean, mean_strides.filter(0),
                    variance, variance_strides.filter(0),
                    ddof, is_or_should_reduce[0], true, stream);
        } else {
            reduce_variance_axis_<false>(
                    input, input_strides, input_shape,
                    mean, mean_strides,
                    variance, variance_strides,
                    output_shape, mask, ddof, stream);
        }
    }

    template<typename Input, typename Output, typename>
    void mean_std(const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                  Input* mean, const Strides4<i64>& mean_strides,
                  Output* stddev, const Strides4<i64>& stddev_strides,
                  const Shape4<i64>& output_shape, i64 ddof, Stream& stream) {
        const auto mask = get_mask_(input_shape, output_shape);
        const auto is_or_should_reduce(output_shape == 1 || mask);

        if (!noa::any(mask)) {
            if constexpr (noa::traits::is_complex_v<Input>) {
                noa::cuda::memory::copy(input, input_strides, mean, mean_strides, output_shape, stream);
                noa::cuda::ewise_unary(input, input_strides, stddev, stddev_strides, output_shape, noa::abs_t{}, stream);
            } else {
                noa::cuda::memory::copy(input, input_strides, mean, mean_strides, output_shape, stream);
                noa::cuda::memory::copy(input, input_strides, stddev, stddev_strides, output_shape, stream);
            }

        } else if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            noa::cuda::utils::reduce_variance<true>(
                    input, input_strides, input_shape,
                    mean, mean_strides.filter(0),
                    stddev, stddev_strides.filter(0),
                    ddof, is_or_should_reduce[0], true, stream);
        } else {
            reduce_variance_axis_<true>(
                    input, input_strides, input_shape,
                    mean, mean_strides,
                    stddev, stddev_strides,
                    output_shape, mask, ddof, stream);
        }
    }

    #define NOA_INSTANTIATE_VAR_(T,U) \
    template void var<T,U,void>(                            \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        U*, const Strides4<i64>&, const Shape4<i64>&,       \
        i64, Stream&);                                      \
    template void std<T,U,void>(                            \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        U*, const Strides4<i64>&, const Shape4<i64>&,       \
        i64, Stream&);                                      \
    template void mean_var<T,U,void>(                       \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T*, const Strides4<i64>&, U*, const Strides4<i64>&, \
        const Shape4<i64>&, i64, Stream&);                  \
    template void mean_std<T,U,void>(                       \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T*, const Strides4<i64>&, U*, const Strides4<i64>&, \
        const Shape4<i64>&, i64, Stream&)

    NOA_INSTANTIATE_VAR_(f32, f32);
    NOA_INSTANTIATE_VAR_(f64, f64);
    NOA_INSTANTIATE_VAR_(c32, f32);
    NOA_INSTANTIATE_VAR_(c64, f64);
}
