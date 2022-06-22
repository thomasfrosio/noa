#include "noa/common/Exception.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr uint ELEMENTS_PER_THREAD = 4;
    constexpr uint BLOCK_SIZE = 512;

    // Reduce rows, one Block.X per row.
    // Since we need the entire block for the reduction, do not return prematurely.
    template<typename T, typename U, typename TransformOp, typename ReduceOp, typename PostProcess,
             int BLOCK_DIM_X, int VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduceRows_(const T* __restrict__ input, uint4_t input_stride, uint2_t shape /* YX */,
                     U* __restrict__ output, uint4_t output_stride,
                     TransformOp transform_op, ReduceOp reduce_op, U init, PostProcess post_process_op) {
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         blockIdx.x * blockDim.y + threadIdx.y,
                         threadIdx.x);
        const bool is_valid_row = gid[2] < shape[0];
        input += indexing::at(gid[0], gid[1], gid[2], input_stride);

        // Initial reduction. Loop until the end of the row is reached.
        U reduced = init;
        for (uint cid = 0; cid < shape[1] && is_valid_row; cid += BLOCK_DIM_X * ELEMENTS_PER_THREAD) {
            const uint remaining = shape[1] - cid;
            util::block::reduceUnaryGlobal1D<BLOCK_DIM_X, ELEMENTS_PER_THREAD, VEC_SIZE>(
                    input + cid, input_stride[3], remaining,
                    transform_op, reduce_op, &reduced, threadIdx.x);
        }

        // Share the threads' initial reduction with the rest of the block.
        const uint tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        util::block::synchronize();

        // Reduce shared data to one element.
        U final = util::block::reduceShared1D<BLOCK_DIM_X>(s_data + BLOCK_DIM_X * threadIdx.y, gid[3], reduce_op);
        if (gid[3] == 0 && is_valid_row)
            output[indexing::at(gid[0], gid[1], gid[2], output_stride)] = post_process_op(final);
    }

    // Keep X to one warp to have memory coalescing, even though a half-warp should be OK as well.
    // The Y dimension of the block is where the reduction happens.
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);

    // The second-most dimension is reduced, i.e. shape[0] and stride[2].
    // Grid.X Blocks per row.
    // Grid.Z/Y: blocks to reduce the two outermost
    template<typename T, typename U, typename TransformOp, typename ReduceOp, typename PostProcess>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduceDim_(const T* __restrict__ input, uint4_t input_stride, uint2_t shape,
                    U* __restrict__ output, uint4_t output_stride,
                    TransformOp transform_op, ReduceOp reduce_op, U init, PostProcess post_process_op) {
        const int4_t gid(blockIdx.z,
                         blockIdx.y,
                         threadIdx.y, // one block in the dimension to reduce
                         blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x);
        const bool is_valid_column = gid[3] < shape[1];
        input += indexing::at(gid[0], gid[1], input_stride) + gid[3] * input_stride[3];

        // Initial reduction. Loop until the end of Y is reached.
        U reduced = init;
        for (uint tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) { // compute entire row
            U transformed = transform_op(input[tidy * input_stride[2]]);
            reduced = reduce_op(reduced, transformed);
        }

        // Share the threads' initial reduction with the rest of the block.
        const uint tid = gid[2] * blockDim.x + threadIdx.x;
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        U* s_data_tid = s_data + tid;
        *s_data_tid = reduced;
        util::block::synchronize();

        // Reduce along Y:
        #pragma unroll
        for (uint SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_tid = reduce_op(*s_data_tid, s_data_tid[BLOCK_SIZE_2D.x * SIZE / 2]);
            util::block::synchronize();
        }

        if (gid[2] == 0 && is_valid_column)
            output[indexing::at(gid[0], gid[1], output_stride) + gid[3] * output_stride[3]] =
                    post_process_op(*s_data_tid);
    }

    template<typename T, typename U, typename ReduceOp, typename TransformOp, typename PostProcess>
    inline void reduceAxis_(const char* name,
                            const T* input, uint4_t input_stride, uint4_t input_shape,
                            U* output, uint4_t output_stride, uint4_t output_shape, bool4_t mask,
                            TransformOp transform_op, ReduceOp reduce_op, U init, PostProcess post_process_op,
                            Stream& stream) {
        if (noa::math::sum(int4_t{mask}) > 1) {
            NOA_THROW_FUNC(name, "Reducing more than one axis at a time is only supported if the reduction results in "
                                 "one value per batch, i.e. the 3 innermost dimensions are shape=1 after reduction. "
                                 "Got input:{}, output:{}, reduce:{}", input_shape, output_shape, mask);
        }

        if (mask[3]) {
            const uint block_dim_x = input_shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
            const uint blocks_y = noa::math::divideUp(input_shape[2], threads.y);
            const dim3 blocks(blocks_y, input_shape[1], input_shape[0]);
            const LaunchConfig config{blocks, threads, BLOCK_SIZE * sizeof(U)};

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            uint vec_size = input_stride[3] == 1 ? util::maxVectorCount(input) : 1;
            if ((input_stride[2] % vec_size && input_shape[2] > 1) ||
                (input_stride[1] % vec_size && input_shape[1] > 1) ||
                (input_stride[0] % vec_size && input_shape[0] > 1))
                vec_size = 1; // TODO If not multiple of 4, try 2 before turning off vectorization?

            if (threads.x == 256) {
                stream.enqueue(name,
                               vec_size == 4 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 256, 4> :
                               vec_size == 2 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 256, 2> :
                                               reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 256, 1>,
                               config, input, input_stride, uint2_t{input_shape[2], input_shape[3]},
                               output, output_stride, transform_op, reduce_op, init, post_process_op);
            } else {
                stream.enqueue(name,
                               vec_size == 4 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 64, 4> :
                               vec_size == 2 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 64, 2> :
                                               reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 64, 1>,
                               config, input, input_stride, uint2_t{input_shape[2], input_shape[3]},
                               output, output_stride, transform_op, reduce_op, init, post_process_op);
            }

        } else if (mask[2]) {
            const uint blocks_x = noa::math::divideUp(input_shape[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape[1], input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(U)};
            const uint2_t shape{input_shape[2], input_shape[3]};
            stream.enqueue(name, reduceDim_<T, U, TransformOp, ReduceOp, PostProcess>, config,
                           input, input_stride, shape, output, output_stride,
                           transform_op, reduce_op, init, post_process_op);

        } else if (mask[1]) {
            const uint blocks_x = noa::math::divideUp(input_shape[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape[2], input_shape[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(U)};
            const uint4_t i_stride{input_stride[0], input_stride[2], input_stride[1], input_stride[3]};
            const uint4_t o_stride{output_stride[0], output_stride[2], output_stride[1], output_stride[3]};
            const uint2_t shape{input_shape[1], input_shape[3]};
            stream.enqueue(name, reduceDim_<T, U, TransformOp, ReduceOp, PostProcess>, config,
                           input, i_stride, shape, output, o_stride,
                           transform_op, reduce_op, init, post_process_op);

        } else if (mask[0]) {
            const uint blocks_x = noa::math::divideUp(input_shape[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape[2], input_shape[1]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(U)};
            const uint4_t i_stride{input_stride[1], input_stride[2], input_stride[0], input_stride[3]};
            const uint4_t o_stride{output_stride[1], output_stride[2], output_stride[0], output_stride[3]};
            const uint2_t shape{input_shape[0], input_shape[3]};
            stream.enqueue(name, reduceDim_<T, U, TransformOp, ReduceOp, PostProcess>, config,
                           input, i_stride, shape, output, o_stride,
                           transform_op, reduce_op, init, post_process_op);
        }
    }

    bool4_t getMask_(const char* func, size4_t input_shape, size4_t output_shape) {
        const bool4_t mask{input_shape != output_shape};
        if (any(mask && (output_shape != 1))) {
            NOA_THROW_FUNC(func, "Dimensions should match the input shape, or be 1, indicating the dimension should be "
                                 "reduced to one element. Got input:{}, output:{}", input_shape, output_shape);
        }
        return mask;
    }
}

namespace noa::cuda::math {
    template<typename T, typename>
    void min(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const char* name = "math::min";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask))
            return cuda::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            T* null{};
            if (is_or_should_reduce[0]) {
                util::reduce<true>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                   noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                                   output.get(), output_stride[0], noa::math::copy_t{},
                                   null, 0, noa::math::copy_t{}, stream);
            } else {
                util::reduce<false>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                    noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                                    output.get(), output_stride[0], noa::math::copy_t{},
                                    null, 0, noa::math::copy_t{}, stream);
            }
        } else {
            reduceAxis_(name,
                        input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                        output.get(), uint4_t{output_stride}, uint4_t{output_shape}, mask,
                        noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                        noa::math::copy_t{}, stream);
        }
        stream.attach(input, output);
    }

    template<typename T, typename>
    void max(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const char* name = "math::max";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask))
            return cuda::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            T* null{};
            if (is_or_should_reduce[0]) {
                util::reduce<true>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                   noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                                   output.get(), output_stride[0], noa::math::copy_t{},
                                   null, 0, noa::math::copy_t{}, stream);
            } else {
                util::reduce<false>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                    noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                                    output.get(), output_stride[0], noa::math::copy_t{},
                                    null, 0, noa::math::copy_t{}, stream);
            }
        } else {
            reduceAxis_(name,
                        input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                        output.get(), uint4_t{output_stride}, uint4_t{output_shape}, mask,
                        noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                        noa::math::copy_t{}, stream);
        }
        stream.attach(input, output);
    }

    template<typename T, typename>
    void sum(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const char* name = "math::sum";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};

        if (!any(mask))
            return cuda::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            T* null{};
            if (is_or_should_reduce[0]) {
                util::reduce<true>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                   noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                   output.get(), output_stride[0], noa::math::copy_t{},
                                   null, 0, noa::math::copy_t{}, stream);
            } else {
                util::reduce<false>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                    noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                    output.get(), output_stride[0], noa::math::copy_t{},
                                    null, 0, noa::math::copy_t{}, stream);
            }
        } else {
            reduceAxis_(name,
                        input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                        output.get(), uint4_t{output_stride}, uint4_t{output_shape}, mask,
                        noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                        noa::math::copy_t{}, stream);
        }
        stream.attach(input, output);
    }

    template<typename T, typename U>
    void mean(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
              const shared_t<T[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        const char* name = "math::mean";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce{output_shape == 1 || mask};
        using real_t = noa::traits::value_type_t<T>;

        if (!any(mask))
            return cuda::memory::copy(input, input_stride, output, output_stride, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            const uint element_per_batch = input_shape[1] * input_shape[2] * input_shape[3] *
                                           (is_or_should_reduce[0] ? input_shape[0] : 1);
            const real_t inv_count = real_t(1) / static_cast<real_t>(element_per_batch);
            auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };

            T* null{};
            if (is_or_should_reduce[0]) {
                util::reduce<true>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                   noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                   output.get(), output_stride[0], sum_to_mean_op,
                                   null, 0, noa::math::copy_t{}, stream);
            } else {
                util::reduce<false>(name, input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                                    noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                    output.get(), output_stride[0], sum_to_mean_op,
                                    null, 0, noa::math::copy_t{}, stream);
            }
        } else {
            const real_t inv_count = real_t(1) / static_cast<real_t>(noa::math::sum(input_shape * size4_t{mask}));
            auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
            reduceAxis_(name,
                        input.get(), uint4_t{input_stride}, uint4_t{input_shape},
                        output.get(), uint4_t{output_stride}, uint4_t{output_shape}, mask,
                        noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                        sum_to_mean_op, stream);
        }
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_REDUCE_(T)                                                                                      \
    template void min<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template void max<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template void sum<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template void mean<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_REDUCE_(float);
    NOA_INSTANTIATE_REDUCE_(double);
    NOA_INSTANTIATE_REDUCE_(uint32_t);
    NOA_INSTANTIATE_REDUCE_(uint64_t);
    NOA_INSTANTIATE_REDUCE_(int32_t);
    NOA_INSTANTIATE_REDUCE_(int64_t);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX_(T)                                                                              \
    template void sum<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&);    \
    template void mean<T, void>(const shared_t<T[]>&, size4_t, size4_t, const shared_t<T[]>&, size4_t, size4_t, Stream&)

    NOA_INSTANTIATE_REDUCE_COMPLEX_(cfloat_t);
    NOA_INSTANTIATE_REDUCE_COMPLEX_(cdouble_t);
}
