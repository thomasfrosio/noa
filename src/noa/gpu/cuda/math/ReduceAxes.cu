#include "noa/gpu/cuda/Exception.hpp"
#include "noa/gpu/cuda/math/Reduce.hpp"
#include "noa/gpu/cuda/memory/Copy.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr u32 ELEMENTS_PER_THREAD = 4;
    constexpr u32 BLOCK_SIZE = 512;

    // Reduce rows, one Block.X per row.
    // Since we need the entire block for the reduction, do not return prematurely.
    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessOp, typename ReduceOp, typename PostProcess,
             u32 BLOCK_DIM_X, u32 VECTOR_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduce_width_(AccessorRestrict<const Input, 4, Index> input,
                       Shape2<Index> shape_hw, Reduced initial_reduce,
                       AccessorRestrict<Output, 3, Index> output,
                       PreProcessOp preprocess_op, ReduceOp reduce_op,
                       PostProcess post_process_op) {

        NOA_ASSERT(BLOCK_DIM_X == blockDim.x);
        constexpr Index EPT = noa::math::max(ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_WORK_SIZE = EPT * BLOCK_DIM_X;

        const Vec2<Index> thread_index{threadIdx.y, threadIdx.x};
        const Vec4<Index> gid{blockIdx.z,
                              blockIdx.y,
                              blockIdx.x * blockDim.y + thread_index[0],
                              thread_index[1]};
        const bool is_valid_row = gid[2] < shape_hw[0];
        const auto input_row = input[gid[0]][gid[1]][gid[2]];

        // Initial reduction. Loop until the end of the row is reached.
        Reduced reduced = initial_reduce;
        for (Index cid = 0; cid < shape_hw[1] && is_valid_row; cid += BLOCK_WORK_SIZE) {
            const Index remaining = shape_hw[1] - cid;
            const Index stride = input_row.template stride<0>();
            const Index offset = cid * stride;
            utils::block_reduce_global_unary<BLOCK_DIM_X, EPT, VECTOR_SIZE>(
                    input_row.get() + offset, stride, remaining,
                    preprocess_op, reduce_op, &reduced, thread_index[1], offset);
        }

        // Share the threads' initial reduction with the rest of the block.
        const Index tid = thread_index[0] * BLOCK_DIM_X + thread_index[1];
        Reduced* s_data = utils::block_dynamic_shared_resource<Reduced>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        utils::block_synchronize();

        // Reduce shared data to one element.
        const Reduced final = utils::block_reduce_shared<BLOCK_DIM_X>(
                s_data + BLOCK_DIM_X * thread_index[0], thread_index[1], reduce_op);
        if (gid[3] == 0 && is_valid_row)
            output(gid[0], gid[1], gid[2]) = post_process_op(final);
    }

    // Keep X to one warp to have memory coalescing, even though a half-warp should be OK as well.
    // The Y dimension of the block is where the reduction happens.
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);

    // The second-most dimension is reduced, i.e. shape[0] and strides[2].
    // Grid.X Blocks per row.
    // Grid.Z/Y: blocks to reduce the two outermost
    template<typename Input, typename Reduced, typename Output, typename Index,
             typename PreProcessor, typename ReduceOp, typename PostProcess>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduce_height_(const Input* __restrict__ input, Strides4<Index> input_strides, Shape2<Index> shape,
                        Reduced initial_reduce, Output* __restrict__ output, Strides4<Index> output_strides,
                        PreProcessor pre_process_op, ReduceOp reduce_op, PostProcess post_process_op) {

        const Vec4<Index> gid{blockIdx.z,
                              blockIdx.y,
                              threadIdx.y, // one block in the dimension to reduce
                              blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        const bool is_valid_column = gid[3] < shape[1];
        input += noa::indexing::at(gid[0], gid[1], input_strides) + gid[3] * input_strides[3];

        // Initial reduction. Loop until the end of Y is reached.
        Reduced reduced = initial_reduce;
        for (Index tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) { // compute entire row
            const auto offset = tidy * input_strides[2];
            if constexpr (noa::traits::is_detected_v<noa::traits::has_binary_operator, PreProcessor, Input, Index>)
                reduced = reduce_op(reduced, pre_process_op(input[offset], offset));
            else
                reduced = reduce_op(reduced, pre_process_op(input[offset]));
        }

        // Share the threads' initial reduction with the rest of the block.
        const Index tid = gid[2] * blockDim.x + threadIdx.x;
        Reduced* s_data = utils::block_dynamic_shared_resource<Reduced>(); // BLOCK_SIZE elements.
        Reduced* s_data_tid = s_data + tid;
        *s_data_tid = reduced;
        utils::block_synchronize();

        // Reduce along Y:
        #pragma unroll
        for (u32 SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_tid = reduce_op(*s_data_tid, s_data_tid[BLOCK_SIZE_2D.x * SIZE / 2]);
            utils::block_synchronize();
        }

        if (gid[2] == 0 && is_valid_column) {
            const auto offset = noa::indexing::at(gid[0], gid[1], output_strides) + gid[3] * output_strides[3];
            output[offset] = post_process_op(*s_data_tid);
        }
    }

    template<typename Input, typename Reduced, typename Output,
             typename ReduceOp, typename PreProcessOp, typename PostProcess>
    inline void reduce_axis_(const char* name,
                            const Input* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                            Output* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                            const Vec4<bool>& mask, Reduced initial_reduce,
                            PreProcessOp pre_process_op, ReduceOp reduce_op,  PostProcess post_process_op,
                            Stream& stream) {
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
            const u32 block_dim_x = u_input_shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
            const u32 blocks_y = noa::math::divide_up(u_input_shape[2], threads.y);
            const dim3 blocks(blocks_y, u_input_shape[1], u_input_shape[0]);
            const LaunchConfig config{blocks, threads, BLOCK_SIZE * sizeof(Reduced)};

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            u32 vector_size = u_input_strides[3] == 1 ? std::min(utils::max_vector_count(input), i64{4}) : 1;
            for (; vector_size >= 2; vector_size /= 2) {
                if ((!(u_input_strides[2] % vector_size) || u_input_shape[2] == 1) &&
                    (!(u_input_strides[1] % vector_size) || u_input_shape[1] == 1) &&
                    (!(u_input_strides[0] % vector_size) || u_input_shape[0] == 1))
                    break;
            }

            const auto input_accessor = AccessorRestrict<const Input, 4, u32>(input, u_input_strides);
            const auto output_accessor = AccessorRestrict<Output, 3, u32>(output, u_output_strides.pop_back());
            if (threads.x == 256) {
                stream.enqueue(name,
                               vector_size == 4 ? reduce_width_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess, 256, 4> :
                               vector_size == 2 ? reduce_width_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess, 256, 2> :
                                                  reduce_width_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess, 256, 1>,
                               config, input_accessor, u_input_shape.pop_front<2>(), initial_reduce,
                               output_accessor, pre_process_op, reduce_op, post_process_op);
            } else {
                stream.enqueue(name,
                               vector_size == 4 ? reduce_width_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess, 64, 4> :
                               vector_size == 2 ? reduce_width_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess, 64, 2> :
                                                  reduce_width_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess, 64, 1>,
                               config, input_accessor, u_input_shape.pop_front<2>(), initial_reduce,
                               output_accessor, pre_process_op, reduce_op, post_process_op);
            }
        } else if (mask[2]) {
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[1], u_input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Reduced)};
            stream.enqueue(name, reduce_height_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess>, config,
                           input, u_input_strides, u_input_shape.pop_front<2>(), initial_reduce,
                           output, u_output_strides, pre_process_op, reduce_op, post_process_op);

        } else if (mask[1]) {
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[2], u_input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Reduced)};
            stream.enqueue(name, reduce_height_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess>, config,
                           input, u_input_strides.filter(0, 2, 1, 3), u_input_shape.filter(1, 3), initial_reduce,
                           output, u_output_strides.filter(0, 2, 1, 3), pre_process_op, reduce_op, post_process_op);

        } else if (mask[0]) {
            const u32 blocks_x = noa::math::divide_up(u_input_shape[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, u_input_shape[2], u_input_shape[1]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(Reduced)};
            stream.enqueue(name, reduce_height_<Input, Reduced, Output, u32, PreProcessOp, ReduceOp, PostProcess>, config,
                           input, u_input_strides.filter(1, 2, 0, 3), u_input_shape.filter(0, 3), initial_reduce,
                           output, u_output_strides.filter(1, 2, 0, 3), pre_process_op, reduce_op, post_process_op);
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
    template<typename Value, typename>
    void min(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             Stream& stream) {
        const char* name = "math::min";
        const auto mask = get_mask_(name, input_shape, output_shape);
        const auto is_or_should_reduce = output_shape == 1 || mask;

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            utils::reduce_unary(name, input, input_strides, input_shape,
                                output, output_strides.filter(0), noa::math::Limits<Value>::max(),
                                noa::copy_t{}, noa::min_t{}, noa::copy_t{},
                                is_or_should_reduce[0], true, stream);
        } else {
            reduce_axis_(name,
                         input, input_strides, input_shape,
                         output, output_strides, output_shape,
                         mask, noa::math::Limits<Value>::max(),
                         noa::copy_t{}, noa::min_t{}, noa::copy_t{}, stream);
        }
    }

    template<typename Value, typename>
    void max(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             Stream& stream) {
        const char* name = "math::max";
        const auto mask = get_mask_(name, input_shape, output_shape);
        const auto is_or_should_reduce = output_shape == 1 || mask;

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            utils::reduce_unary(name, input, input_strides, input_shape,
                                output, output_strides.filter(0), noa::math::Limits<Value>::lowest(),
                                noa::copy_t{}, noa::max_t{}, noa::copy_t{},
                                is_or_should_reduce[0], true, stream);
        } else {
            reduce_axis_(name,
                         input, input_strides, input_shape,
                         output, output_strides, output_shape,
                         mask, noa::math::Limits<Value>::lowest(),
                         noa::copy_t{}, noa::max_t{}, noa::copy_t{}, stream);
        }
    }

    template<typename Value, typename _>
    void sum(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
             Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
             Stream& stream) {
        const char* name = "math::sum";
        const auto mask = get_mask_(name, input_shape, output_shape);
        const auto is_or_should_reduce = output_shape == 1 || mask;

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            utils::reduce_unary(name, input, input_strides, input_shape,
                                output, output_strides.filter(0), Value{0},
                                noa::copy_t{}, noa::plus_t{}, noa::copy_t{},
                                is_or_should_reduce[0], true, stream);
        } else {
            using reduce_t = std::conditional_t<noa::traits::is_complex_v<Value>, c64,
                             std::conditional_t<noa::traits::is_real_v<Value>, f64, Value>>;
            const auto pre_process_op = []__device__(const Value& value) { return static_cast<reduce_t>(value); };
            const auto post_process_op = []__device__(const reduce_t& value) { return static_cast<Value>(value); };
            reduce_axis_(name,
                         input, input_strides, input_shape,
                         output, output_strides, output_shape,
                         mask, reduce_t{0},
                         pre_process_op, noa::plus_t{}, post_process_op, stream);
        }
    }

    template<typename Value, typename _>
    void mean(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
              Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
              Stream& stream) {
        const char* name = "math::mean";
        const auto mask = get_mask_(name, input_shape, output_shape);
        const auto is_or_should_reduce = output_shape == 1 || mask;

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            const auto element_per_batch =
                    input_shape[1] * input_shape[2] * input_shape[3] *
                    (is_or_should_reduce[0] ? input_shape[0] : 1);

            using real_t = noa::traits::value_type_t<Value>;
            const auto count = static_cast<real_t>(element_per_batch);
            auto sum_to_mean_op = [count]__device__(Value v) -> Value {
                if constexpr (noa::traits::is_int_v<real_t>) {
                    return static_cast<real_t>(noa::math::round(static_cast<f64>(v) / static_cast<f64>(count)));
                } else {
                    return v / count;
                }
            };

            utils::reduce_unary(name, input, input_strides, input_shape,
                                output, output_strides.filter(0), Value{0},
                                noa::copy_t{}, noa::plus_t{}, sum_to_mean_op,
                                is_or_should_reduce[0], true, stream);
        } else {
            // Since there's parallelism here, use double precision to preserve good accuracy.
            using reduce_t = std::conditional_t<noa::traits::is_complex_v<Value>, c64,
                             std::conditional_t<noa::traits::is_real_v<Value>, f64, Value>>;
            using reduce_real_t = noa::traits::value_type_t<reduce_t>;
            const auto count = static_cast<reduce_real_t>(noa::math::sum(input_shape * Shape4<i64>(mask)));

            const auto pre_process_op = []__device__(const Value& value) { return static_cast<reduce_t>(value); };
            auto sum_to_mean_op = [count]__device__(reduce_t value) -> Value {
                if constexpr (noa::traits::is_int_v<Value>) {
                    return static_cast<Value>(noa::math::round(static_cast<f64>(value) / static_cast<f64>(count)));
                } else {
                    return static_cast<Value>(value / count);
                }
            };
            reduce_axis_(name,
                         input, input_strides, input_shape,
                         output, output_strides, output_shape,
                         mask, reduce_t{0},
                         pre_process_op, noa::plus_t{}, sum_to_mean_op, stream);
        }
    }

    #define NOA_INSTANTIATE_REDUCE_(T)                          \
    template void min<T, void>(                                 \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&, Stream&); \
    template void max<T, void>(                                 \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&, Stream&); \
    template void sum<T, void>(                                 \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&, Stream&); \
    template void mean<T, void>(                                \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_REDUCE_(f32);
    NOA_INSTANTIATE_REDUCE_(f64);
    NOA_INSTANTIATE_REDUCE_(u32);
    NOA_INSTANTIATE_REDUCE_(u64);
    NOA_INSTANTIATE_REDUCE_(i32);
    NOA_INSTANTIATE_REDUCE_(i64);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX_(T)                  \
    template void sum<T, void>(                                 \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&, Stream&); \
    template void mean<T, void>(                                \
        const T*, const Strides4<i64>&, const Shape4<i64>&,     \
        T*, const Strides4<i64>&, const Shape4<i64>&, Stream&)

    NOA_INSTANTIATE_REDUCE_COMPLEX_(c32);
    NOA_INSTANTIATE_REDUCE_COMPLEX_(c64);
}
