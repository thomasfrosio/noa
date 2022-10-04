#include "noa/common/Exception.h"
#include "noa/common/Math.h"

#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/ReduceUnary.cuh"

namespace {
    using namespace ::noa;
    using namespace ::noa::cuda;

    constexpr uint32_t ELEMENTS_PER_THREAD = 4;
    constexpr uint32_t BLOCK_SIZE = 512;

    // Reduce rows, one Block.X per row.
    // Since we need the entire block for the reduction, do not return prematurely.
    template<typename T, typename U, typename TransformOp, typename ReduceOp, typename PostProcess,
             int32_t BLOCK_DIM_X, int32_t VEC_SIZE>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduceRows_(AccessorRestrict<const T, 4, uint32_t> input, uint2_t shape /* YX */,
                     AccessorRestrict<U, 3, uint32_t> output,
                     TransformOp transform_op, ReduceOp reduce_op, U init, PostProcess post_process_op) {
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         blockIdx.x * blockDim.y + threadIdx.y,
                         threadIdx.x};
        const bool is_valid_row = gid[2] < shape[0];
        const auto input_row = input[gid[0]][gid[1]][gid[2]];

        // Initial reduction. Loop until the end of the row is reached.
        U reduced = init;
        for (uint32_t cid = 0; cid < shape[1] && is_valid_row; cid += BLOCK_DIM_X * ELEMENTS_PER_THREAD) {
            const uint32_t remaining = shape[1] - cid;
            util::block::reduceUnaryGlobal1D<BLOCK_DIM_X, ELEMENTS_PER_THREAD, VEC_SIZE>(
                    input_row.offset(cid).get(), input_row.stride(0), remaining,
                    transform_op, reduce_op, &reduced, threadIdx.x);
        }

        // Share the threads' initial reduction with the rest of the block.
        const uint32_t tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        util::block::synchronize();

        // Reduce shared data to one element.
        U final = util::block::reduceShared1D<BLOCK_DIM_X>(s_data + BLOCK_DIM_X * threadIdx.y, gid[3], reduce_op);
        if (gid[3] == 0 && is_valid_row)
            output(gid[0], gid[1], gid[2]) = post_process_op(final);
    }

    // Keep X to one warp to have memory coalescing, even though a half-warp should be OK as well.
    // The Y dimension of the block is where the reduction happens.
    constexpr dim3 BLOCK_SIZE_2D(32, BLOCK_SIZE / 32);

    // The second-most dimension is reduced, i.e. shape[0] and strides[2].
    // Grid.X Blocks per row.
    // Grid.Z/Y: blocks to reduce the two outermost
    template<typename T, typename U, typename TransformOp, typename ReduceOp, typename PostProcess>
    __global__ __launch_bounds__(BLOCK_SIZE)
    void reduceDim_(const T* __restrict__ input, uint4_t input_strides, uint2_t shape,
                    U* __restrict__ output, uint4_t output_strides,
                    TransformOp transform_op, ReduceOp reduce_op, U init, PostProcess post_process_op) {
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         threadIdx.y, // one block in the dimension to reduce
                         blockIdx.x * BLOCK_SIZE_2D.x + threadIdx.x};
        const bool is_valid_column = gid[3] < shape[1];
        input += indexing::at(gid[0], gid[1], input_strides) + gid[3] * input_strides[3];

        // Initial reduction. Loop until the end of Y is reached.
        U reduced = init;
        for (uint32_t tidy = gid[2]; tidy < shape[0] && is_valid_column; tidy += BLOCK_SIZE_2D.y) { // compute entire row
            U transformed = transform_op(input[tidy * input_strides[2]]);
            reduced = reduce_op(reduced, transformed);
        }

        // Share the threads' initial reduction with the rest of the block.
        const uint32_t tid = gid[2] * blockDim.x + threadIdx.x;
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        U* s_data_tid = s_data + tid;
        *s_data_tid = reduced;
        util::block::synchronize();

        // Reduce along Y:
        #pragma unroll
        for (uint32_t SIZE = BLOCK_SIZE_2D.y; SIZE >= 2; SIZE /= 2) {
            if (gid[2] < SIZE / 2)
                *s_data_tid = reduce_op(*s_data_tid, s_data_tid[BLOCK_SIZE_2D.x * SIZE / 2]);
            util::block::synchronize();
        }

        if (gid[2] == 0 && is_valid_column)
            output[indexing::at(gid[0], gid[1], output_strides) + gid[3] * output_strides[3]] =
                    post_process_op(*s_data_tid);
    }

    template<typename T, typename U, typename ReduceOp, typename TransformOp, typename PostProcess>
    inline void reduceAxis_(const char* name,
                            const T* input, dim4_t input_strides, dim4_t input_shape,
                            U* output, dim4_t output_strides, dim4_t output_shape, bool4_t mask,
                            TransformOp transform_op, ReduceOp reduce_op, U init, PostProcess post_process_op,
                            Stream& stream) {
        if (noa::math::sum(int4_t(mask)) > 1) {
            NOA_THROW_FUNC(name, "Reducing more than one axis at a time is only supported if the reduction results in "
                                 "one value per batch, i.e. the 3 innermost dimensions are shape=1 after reduction. "
                                 "Got input:{}, output:{}, reduce:{}", input_shape, output_shape, mask);
        }

        const auto input_strides_ = safe_cast<uint4_t>(input_strides);
        const auto input_shape_ = safe_cast<uint4_t>(input_shape);
        const auto output_strides_ = safe_cast<uint4_t>(output_strides);

        if (mask[3]) {
            const uint32_t block_dim_x = input_shape_[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, BLOCK_SIZE / block_dim_x);
            const uint32_t blocks_y = noa::math::divideUp(input_shape_[2], threads.y);
            const dim3 blocks(blocks_y, input_shape_[1], input_shape_[0]);
            const LaunchConfig config{blocks, threads, BLOCK_SIZE * sizeof(U)};

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            uint32_t vec_size = input_strides_[3] == 1 ? util::maxVectorCount(input) : 1;
            if ((input_strides_[2] % vec_size && input_shape_[2] > 1) ||
                (input_strides_[1] % vec_size && input_shape_[1] > 1) ||
                (input_strides_[0] % vec_size && input_shape_[0] > 1))
                vec_size = 1; // TODO If not multiple of 4, try 2 before turning off vectorization?

            const AccessorRestrict<const T, 4, uint32_t> input_accessor(input, input_strides_);
            const AccessorRestrict<U, 3, uint32_t> output_accessor(output, output_strides_.get());
            if (threads.x == 256) {
                stream.enqueue(name,
                               vec_size == 4 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 256, 4> :
                               vec_size == 2 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 256, 2> :
                                               reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 256, 1>,
                               config, input_accessor, uint2_t{input_shape_[2], input_shape_[3]},
                               output_accessor, transform_op, reduce_op, init, post_process_op);
            } else {
                stream.enqueue(name,
                               vec_size == 4 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 64, 4> :
                               vec_size == 2 ? reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 64, 2> :
                                               reduceRows_<T, U, TransformOp, ReduceOp, PostProcess, 64, 1>,
                               config, input_accessor, uint2_t{input_shape_[2], input_shape_[3]},
                               output_accessor, transform_op, reduce_op, init, post_process_op);
            }

        } else if (mask[2]) {
            const uint32_t blocks_x = noa::math::divideUp(input_shape_[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape_[1], input_shape_[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(U)};
            const uint2_t shape{input_shape_[2], input_shape_[3]};
            stream.enqueue(name, reduceDim_<T, U, TransformOp, ReduceOp, PostProcess>, config,
                           input, input_strides_, shape, output, output_strides_,
                           transform_op, reduce_op, init, post_process_op);

        } else if (mask[1]) {
            const uint32_t blocks_x = noa::math::divideUp(input_shape_[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape_[2], input_shape_[0]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(U)};
            const uint4_t i_strides{input_strides_[0], input_strides_[2], input_strides_[1], input_strides_[3]};
            const uint4_t o_strides{output_strides_[0], output_strides_[2], output_strides_[1], output_strides_[3]};
            const uint2_t shape{input_shape_[1], input_shape_[3]};
            stream.enqueue(name, reduceDim_<T, U, TransformOp, ReduceOp, PostProcess>, config,
                           input, i_strides, shape, output, o_strides,
                           transform_op, reduce_op, init, post_process_op);

        } else if (mask[0]) {
            const uint32_t blocks_x = noa::math::divideUp(input_shape_[3], BLOCK_SIZE_2D.x);
            const dim3 blocks(blocks_x, input_shape_[2], input_shape_[1]);
            const LaunchConfig config{blocks, BLOCK_SIZE_2D, BLOCK_SIZE * sizeof(U)};
            const uint4_t i_strides{input_strides_[1], input_strides_[2], input_strides_[0], input_strides_[3]};
            const uint4_t o_strides{output_strides_[1], output_strides_[2], output_strides_[0], output_strides_[3]};
            const uint2_t shape{input_shape_[0], input_shape_[3]};
            stream.enqueue(name, reduceDim_<T, U, TransformOp, ReduceOp, PostProcess>, config,
                           input, i_strides, shape, output, o_strides,
                           transform_op, reduce_op, init, post_process_op);
        }
    }

    bool4_t getMask_(const char* func, dim4_t input_shape, dim4_t output_shape) {
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
    void min(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        const char* name = "math::min";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce(output_shape == 1 || mask);

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            T* null{};
            util::reduce(name, input.get(), input_strides, input_shape,
                         noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                         output.get(), output_strides[0], noa::math::copy_t{},
                         null, 0, noa::math::copy_t{}, is_or_should_reduce[0], true, stream);
        } else {
            reduceAxis_(name,
                        input.get(), input_strides, input_shape,
                        output.get(), output_strides, output_shape, mask,
                        noa::math::copy_t{}, noa::math::min_t{}, noa::math::Limits<T>::max(),
                        noa::math::copy_t{}, stream);
        }
        stream.attach(input, output);
    }

    template<typename T, typename>
    void max(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        const char* name = "math::max";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce(output_shape == 1 || mask);

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            T* null{};
            util::reduce(name, input.get(), input_strides, input_shape,
                         noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                         output.get(), output_strides[0], noa::math::copy_t{},
                         null, 0, noa::math::copy_t{}, is_or_should_reduce[0], true, stream);
        } else {
            reduceAxis_(name,
                        input.get(), input_strides, input_shape,
                        output.get(), output_strides, output_shape, mask,
                        noa::math::copy_t{}, noa::math::max_t{}, noa::math::Limits<T>::lowest(),
                        noa::math::copy_t{}, stream);
        }
        stream.attach(input, output);
    }

    template<typename T, typename>
    void sum(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
             const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        const char* name = "math::sum";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce(output_shape == 1 || mask);

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            T* null{};
            util::reduce(name, input.get(), input_strides, input_shape,
                         noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                         output.get(), output_strides[0], noa::math::copy_t{},
                         null, 0, noa::math::copy_t{}, is_or_should_reduce[0], true, stream);
        } else {
            reduceAxis_(name,
                        input.get(), input_strides, input_shape,
                        output.get(), output_strides, output_shape, mask,
                        noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                        noa::math::copy_t{}, stream);
        }
        stream.attach(input, output);
    }

    template<typename T, typename U>
    void mean(const shared_t<T[]>& input, dim4_t input_strides, dim4_t input_shape,
              const shared_t<T[]>& output, dim4_t output_strides, dim4_t output_shape, Stream& stream) {
        const char* name = "math::mean";
        const bool4_t mask = getMask_(name, input_shape, output_shape);
        const bool4_t is_or_should_reduce(output_shape == 1 || mask);
        using real_t = noa::traits::value_type_t<T>;

        if (!any(mask))
            return cuda::memory::copy(input, input_strides, output, output_strides, output_shape, stream);

        if (is_or_should_reduce[1] && is_or_should_reduce[2] && is_or_should_reduce[3]) {
            const uint32_t element_per_batch = input_shape[1] * input_shape[2] * input_shape[3] *
                                               (is_or_should_reduce[0] ? input_shape[0] : 1);
            const real_t inv_count = real_t(1) / static_cast<real_t>(element_per_batch);
            auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };

            T* null{};
            util::reduce(name, input.get(), input_strides, input_shape,
                         noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                         output.get(), output_strides[0], sum_to_mean_op,
                         null, 0, noa::math::copy_t{}, is_or_should_reduce[0], true, stream);
        } else {
            const real_t inv_count = real_t(1) / static_cast<real_t>(noa::math::sum(input_shape * size4_t{mask}));
            auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
            reduceAxis_(name,
                        input.get(), input_strides, input_shape,
                        output.get(), output_strides, output_shape, mask,
                        noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                        sum_to_mean_op, stream);
        }
        stream.attach(input, output);
    }

    #define NOA_INSTANTIATE_REDUCE_(T)                                                                                  \
    template void min<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);    \
    template void max<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);    \
    template void sum<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);    \
    template void mean<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_REDUCE_(float);
    NOA_INSTANTIATE_REDUCE_(double);
    NOA_INSTANTIATE_REDUCE_(uint32_t);
    NOA_INSTANTIATE_REDUCE_(uint64_t);
    NOA_INSTANTIATE_REDUCE_(int32_t);
    NOA_INSTANTIATE_REDUCE_(int64_t);

    #define NOA_INSTANTIATE_REDUCE_COMPLEX_(T)                                                                          \
    template void sum<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&);    \
    template void mean<T, void>(const shared_t<T[]>&, dim4_t, dim4_t, const shared_t<T[]>&, dim4_t, dim4_t, Stream&)

    NOA_INSTANTIATE_REDUCE_COMPLEX_(cfloat_t);
    NOA_INSTANTIATE_REDUCE_COMPLEX_(cdouble_t);
}
