#pragma once

#include "noa/common/Math.h"
#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Traits.h"

#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrManaged.h"

namespace noa::cuda::util::details {
    struct FindConfig {
        static constexpr uint ELEMENTS_PER_THREAD = 8;
        static constexpr uint BLOCK_SIZE = 512;
        static constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
        static constexpr uint MAX_GRID_SIZE = 4096;
    };

    // Grid.X is the number of blocks, at most MAX_GRID_SIZE.
    // Grid.Y is the number of batches. If the input was fully reduced to one element, there's only one batch.
    // The output should have the size of the grid, at minimum.
    template<typename value_t, typename transformed_t, typename offset_t,
            typename transform_op_t, typename find_op_t, int VEC_SIZE>
    __global__ __launch_bounds__(FindConfig::BLOCK_SIZE)
    void findLarge1DStart_(const value_t* __restrict__ input, uint2_t stride /* W,X */, uint elements_per_batch,
                           transform_op_t transform_op, find_op_t find_op, transformed_t init,
                           transformed_t* __restrict__ tmp_output_value,
                           offset_t* __restrict__ tmp_output_offset,
                           uint tmp_output_pitch) {
        constexpr uint BLOCK_SIZE = FindConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = FindConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = FindConfig::ELEMENTS_PER_THREAD;

        // Each block reduces chunks of BLOCK_WORK_SIZE elements at a time.
        // Batches are kept independent of each other.
        const uint tid = threadIdx.x;
        const uint base = blockIdx.x * BLOCK_WORK_SIZE;
        const uint batch = blockIdx.y;
        input += batch * stride[0];

        // Initial reduction to bring the input to BLOCK_SIZE * gridDim.x elements.
        Pair<transformed_t, offset_t> best{init, 0};
        for (uint gid = base; gid < elements_per_batch; gid += BLOCK_WORK_SIZE * gridDim.x) {
            util::block::findGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    input, gid, tid, stride[1], elements_per_batch, transform_op, find_op, &best);
        }

        // Share thread's result to the other threads.
        using uninitialized_t = util::traits::uninitialized_type_t<transformed_t>;
        __shared__ uninitialized_t s_values_[BLOCK_SIZE];
        __shared__ offset_t s_indexes[BLOCK_SIZE];
        auto* s_values = reinterpret_cast<transformed_t*>(s_values_);

        s_values[tid] = best.first;
        s_indexes[tid] = best.second;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        best = util::block::findShared1D<BLOCK_SIZE>(s_values, s_indexes, tid, find_op);
        if (tid == 0) {
            const uint offset = batch * tmp_output_pitch + blockIdx.x;
            tmp_output_value[offset] = best.first;
            tmp_output_offset[offset] = best.second;
        }
    }

    // Grid.X -> Blocks to reduce the 3 innermost dimensions.
    // Grid.Y -> Batch dimension.
    template<typename value_t, typename transformed_t, typename offset_t,
             typename transform_op_t, typename find_op_t, int BLOCK_DIM_X, int VEC_SIZE>
    __global__ __launch_bounds__(FindConfig::BLOCK_SIZE)
    void findLarge4DStart_(const value_t* __restrict__ input, uint4_t input_stride, uint4_t shape, uint rows_per_batch,
                           transform_op_t transform_op, find_op_t find_op, transformed_t init,
                           transformed_t* __restrict__ tmp_output_value,
                           offset_t* __restrict__ tmp_output_offset,
                           uint tmp_output_pitch) {
        constexpr uint EPT = FindConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = FindConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE_X = BLOCK_DIM_X * EPT;

        const uint rows_per_grid = blockDim.y * gridDim.x;
        const uint initial_row = blockDim.y * blockIdx.x + threadIdx.y;
        const uint batch = blockIdx.y;
        input += batch * input_stride[0];

        // Initial reduction. Loop until all rows are consumed.
        Pair<transformed_t, offset_t> best{init, 0};
        for (uint row = initial_row; row < rows_per_batch; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const uint3_t index = indexing::indexes(row, shape[1], shape[2]); // row -> W,Z,Y
            const uint offset = indexing::at(index, input_stride);

            // Consume the row:
            for (uint cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const uint gid = offset + cid;
                util::block::findGlobal1D<BLOCK_DIM_X, EPT, VEC_SIZE>(
                        input, gid, threadIdx.x, input_stride[3], gid + shape[3],
                        transform_op, find_op, &best);
            }
        }

        // Share thread's result to the other threads.
        const uint tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        using uninitialized_t = util::traits::uninitialized_type_t<transformed_t>;
        __shared__ uninitialized_t s_values_[BLOCK_SIZE];
        __shared__ offset_t s_offsets[BLOCK_SIZE];
        auto* s_values = reinterpret_cast<transformed_t*>(s_values_);

        s_values[tid] = best.first;
        s_offsets[tid] = best.second;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        best = util::block::findShared1D<BLOCK_SIZE>(s_values, s_offsets, tid, find_op);
        if (tid == 0) {
            const uint offset = batch * tmp_output_pitch + blockIdx.x;
            tmp_output_value[offset] = best.first;
            tmp_output_offset[offset] = best.second;
        }
    }

    template<typename transformed_t, typename offset_t, typename find_op_t, int VEC_SIZE>
    __global__ __launch_bounds__(FindConfig::BLOCK_SIZE)
    void findLargeEnd_(const transformed_t* __restrict__ tmp_output_value,
                       const offset_t* __restrict__ tmp_output_offset,
                       uint tmp_output_pitch, uint elements_per_batch, find_op_t find_op, transformed_t init,
                       offset_t* __restrict__ output_offset) {
        constexpr uint BLOCK_SIZE = FindConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = FindConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = FindConfig::ELEMENTS_PER_THREAD;

        const uint tid = threadIdx.x;
        const uint batch = blockIdx.x;
        tmp_output_value += tmp_output_pitch * batch;
        tmp_output_offset += tmp_output_pitch * batch;

        // elements -> one element per thread.
        Pair<transformed_t, offset_t> best{init, 0};
        for (uint gid = 0; gid < elements_per_batch; gid += BLOCK_WORK_SIZE) {
            util::block::findGlobal1D<BLOCK_SIZE, EPT, 1>(
                    tmp_output_value, tmp_output_offset, gid, tid, elements_per_batch,
                    noa::math::copy_t{}, find_op, &best);
        }

        // one element per thread -> one element per block.
        using uninitialized_t = util::traits::uninitialized_type_t<transformed_t>;
        __shared__ uninitialized_t s_values_[BLOCK_SIZE];
        __shared__ offset_t s_offsets[BLOCK_SIZE];
        auto* s_values = reinterpret_cast<transformed_t*>(s_values_);

        s_values[tid] = best.first;
        s_offsets[tid] = best.second;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        best = util::block::findShared1D<BLOCK_SIZE>(s_values, s_offsets, tid, find_op);
        if (tid == 0)
            output_offset[batch] = best.second;
    }

    template<typename value_t, typename transformed_t, typename offset_t,
             typename transform_op_t, typename find_op_t, int VEC_SIZE>
    __global__ __launch_bounds__(FindConfig::BLOCK_SIZE)
    void findSmall1D_(const value_t* __restrict__ input, uint2_t input_stride /* batch,X */,
                      uint elements_per_batch, transform_op_t transform_op, find_op_t update_op, transformed_t init,
                      offset_t* __restrict__ output_index) {
        constexpr uint BLOCK_SIZE = FindConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = FindConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = FindConfig::ELEMENTS_PER_THREAD;

        const uint tid = threadIdx.x;
        const uint batch = blockIdx.x;
        input += input_stride[0] * batch;

        // elements -> one element per thread.
        Pair<transformed_t, offset_t> best{init, 0};
        for (uint gid = 0; gid < elements_per_batch; gid += BLOCK_WORK_SIZE) {
            util::block::findGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    input, gid, tid, input_stride[1], elements_per_batch, transform_op, update_op, &best);
        }

        // one element per thread -> one element per block.
        using uninitialized_t = util::traits::uninitialized_type_t<transformed_t>;
        __shared__ uninitialized_t s_values_[BLOCK_SIZE];
        __shared__ offset_t s_offsets[BLOCK_SIZE];
        auto* s_values = reinterpret_cast<transformed_t*>(s_values_);

        s_values[tid] = best.first;
        s_offsets[tid] = best.second;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        best = util::block::findShared1D<BLOCK_SIZE>(s_values, s_offsets, tid, update_op);
        if (tid == 0)
            output_index[batch] = best.second;
    }
}

namespace noa::cuda::util {
    /// Finds the memory offset of the best element according to an update operator.
    /// \tparam REDUCE_BATCH        Whether the outermost dimension should be reduced.
    /// \param name                 Name of the function. Used for logging if a kernel launch fails.
    /// \param[in] input            On the \b device. Input array to search.
    /// \param stride               Rightmost stride of \p input.
    /// \param shape                Rightmost shape of \p input. The outermost dimension is the batch dimension.
    /// \param transform_op         Transform operator, op(\p value_t) -> \p transformed_t, applied when loading data.
    /// \param update_op            Update operator: op(\p pair_t current, \p pair_t candidate) -> pair_t,
    ///                             where pair_t is `Pair<transformed_t, offset_t>.
    /// \param init                 Initial current value (an offset of 0 is assigned to it).
    /// \param output_offset        Memory offset of the best value(s).
    ///                             If \p REDUCE_BATCH is false, the offset of the best value in each batch is returned
    ///                             and these offsets are relative to the beginning of the batch.
    /// \param[in,out] stream       Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       \p input and \p output_offset should stay valid until completion.
    template<bool REDUCE_BATCH,
             typename value_t, typename transformed_t, typename offset_t,
             typename transform_op_t, typename find_op_t>
    void find(const char* name,
              const value_t* input, uint4_t stride, uint4_t shape,
              transform_op_t transform_op, find_op_t update_op, transformed_t init,
              offset_t* output_offset, Stream& stream) {
        const uint batches = REDUCE_BATCH ? 1 : shape[0];
        const uint elements = REDUCE_BATCH ? shape.elements() : shape[1] * shape[2] * shape[3];
        const bool4_t is_contiguous = indexing::isContiguous(stride, shape);

        // The output pointer is allowed to not be on the stream's device,
        // so make sure device memory is allocated for the output.
        memory::PtrDevice<offset_t> buffer;
        offset_t* output_ptr = util::devicePointer(output_offset, stream.device());
        if (!output_ptr) {
            buffer = memory::PtrDevice<offset_t>{batches, stream};
            output_ptr = buffer.get();
        }

        // Small contiguous arrays (1 kernel launch):
        using namespace details;
        if (elements <= FindConfig::BLOCK_WORK_SIZE * 4 &&
            (!REDUCE_BATCH || is_contiguous[0]) && is_contiguous[1] && is_contiguous[2]) {
            uint2_t tmp_stride{stride[0], stride[3]};

            // Try to vectorize the loads for the input.
            uint vec_size = tmp_stride[1] == 1 ? util::maxVectorCount(input) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = tmp_stride[0] % vec_size ? 1 : vec_size;

            stream.enqueue(
                    name,
                    vec_size == 4 ? findSmall1D_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 4> :
                    vec_size == 2 ? findSmall1D_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 2> :
                                    findSmall1D_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 1>,
                    {batches, FindConfig::BLOCK_SIZE},
                    input, tmp_stride, elements, transform_op, update_op, init, output_ptr);

        } else if ((!REDUCE_BATCH || is_contiguous[0]) && is_contiguous[1] && is_contiguous[2])  {
            // In this config, the input can be interpreted as a 1D array. If the innermost dimension is contiguous,
            // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
            // Here we use 1D blocks to go through each batch (if REDUCE_BATCH=true, there's only one batch).
            // Each block reduces at least BLOCK_WORK_SIZE elements. Max to MAX_GRID_SIZE blocks per batch.
            const uint blocks_x = noa::math::min(noa::math::divideUp(elements, FindConfig::BLOCK_WORK_SIZE),
                                                 FindConfig::MAX_GRID_SIZE);
            const dim3 blocks(blocks_x, batches);

            // Try to vectorize the loads for the input.
            uint vec_size = stride[3] == 1 ? util::maxVectorCount(input) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = stride[0] % vec_size ? 1 : vec_size;

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrManaged<transformed_t> tmp_values{pitch * blocks.y, stream};
            memory::PtrManaged<offset_t> tmp_indexes{pitch * blocks.y, stream};

            // findLarge1DStart_: (batch * elements) -> (blocks.x * blocks.y) elements.
            // findLargeEnd_: (blocks.x * blocks.y) -> (blocks.y) elements.
            stream.enqueue(
                    name,
                    vec_size == 4 ? findLarge1DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 4> :
                    vec_size == 2 ? findLarge1DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 2> :
                                    findLarge1DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 1>,
                    {blocks, FindConfig::BLOCK_SIZE},
                    input, uint2_t{stride[0], stride[3]}, elements, transform_op,
                    update_op, init, tmp_values.get(), tmp_indexes.get(), pitch);

            stream.enqueue(name, findLargeEnd_<transformed_t, offset_t, find_op_t, 4>,
                           {batches, FindConfig::BLOCK_SIZE},
                           tmp_values.get(), tmp_indexes.get(), pitch, blocks.x, update_op, init, output_ptr);
        } else {
            // In this config, the input cannot be easily interpreted as a 1D array.
            // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
            // Since the reduceLarge4D_ kernel will decompose the "row index" back to a (W,Z,Y) index, the 3 outermost
            // dimensions can be strided. If the innermost dimension is contiguous, blocks can use vectorize loads
            // to read their row(s).

            // If rows are large, switch to more threads per row.
            const uint block_dim_x = shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, FindConfig::BLOCK_SIZE / block_dim_x);
            const uint rows = shape[2] * shape[1] * (REDUCE_BATCH ? shape[0] : 1);
            const dim3 blocks(noa::math::min(noa::math::divideUp(rows, threads.y), FindConfig::MAX_GRID_SIZE),
                              batches);

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            uint vec_size = stride[3] == 1 ? util::maxVectorCount(input) : 1;
            if ((stride[2] % vec_size && shape[2] == 1) ||
                (stride[1] % vec_size && shape[1] == 1) ||
                (stride[0] % vec_size && shape[0] == 1))
                vec_size = 1; // TODO If not multiple of 4, try 2 before turning off vectorization?

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrDevice<transformed_t> tmp_values{pitch * blocks.y, stream};
            memory::PtrDevice<offset_t> tmp_indexes{pitch * blocks.y, stream};

            if (threads.x == 256) {
                stream.enqueue(
                        name,
                        vec_size == 4 ? findLarge4DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 256, 4> :
                        vec_size == 2 ? findLarge4DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 256, 2> :
                                        findLarge4DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 256, 1>,
                        {blocks, threads},
                        input, stride, shape, rows, transform_op, update_op, init,
                        tmp_values.get(), tmp_indexes.get(), pitch);
            } else {
                stream.enqueue(
                        name,
                        vec_size == 4 ? findLarge4DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 64, 4> :
                        vec_size == 2 ? findLarge4DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 64, 2> :
                                        findLarge4DStart_<value_t, transformed_t, offset_t, transform_op_t, find_op_t, 64, 1>,
                        {blocks, threads},
                        input, stride, shape, rows, transform_op, update_op, init,
                        tmp_values.get(), tmp_indexes.get(), pitch);
            }
            stream.enqueue(name, findLargeEnd_<transformed_t, offset_t, find_op_t, 4>,
                           {batches, FindConfig::BLOCK_SIZE},
                           tmp_values.get(), tmp_indexes.get(), pitch, blocks.x, update_op, init, output_ptr);
        }

        if (!buffer.empty())
            memory::copy(output_ptr, output_offset, batches, stream);
    }
}
