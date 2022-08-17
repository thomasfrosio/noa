#pragma once

#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"

#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/Block.cuh"

// These reduction kernels are adapted from different sources, but the main logic come from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::cuda::util::details {
    struct ReduceUnaryConfig {
        static constexpr uint ELEMENTS_PER_THREAD = 8;
        static constexpr uint BLOCK_SIZE = 512;
        static constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
        static constexpr uint MAX_GRID_SIZE = 4096;
    };

    // Grid.X is the number of blocks, at most MAX_GRID_SIZE.
    // Grid.Y is the number of batches. If the input was fully reduced to one element, there's only one batch.
    // The output should have the size of the grid, at minimum.
    template<typename value_t, typename reduce_value_t, typename transform_op_t, typename reduce_op_t, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceUnaryConfig::BLOCK_SIZE)
    void reduceUnaryLarge1D_(
            const value_t* __restrict__ input, uint2_t input_strides /* BW */, uint elements_per_batch,
            transform_op_t transform_op, reduce_op_t reduce_op, reduce_value_t init,
            reduce_value_t* __restrict__ tmp_output, uint tmp_output_stride) {
        constexpr uint EPT = ReduceUnaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceUnaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = ReduceUnaryConfig::BLOCK_WORK_SIZE;

        // Each block reduces chunks of BLOCK_WORK_SIZE elements at a time.
        // Batches are kept independent of each other.
        const uint tid = threadIdx.x;
        const uint base = blockIdx.x * BLOCK_WORK_SIZE;
        const uint batch = blockIdx.y;
        input += batch * input_strides[0];

        // Initial reduction to bring the input to BLOCK_SIZE * gridDim.x elements.
        reduce_value_t reduced = init;
        for (uint cid = base; cid < elements_per_batch; cid += BLOCK_WORK_SIZE * gridDim.x) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceUnaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    input + cid * input_strides[1], input_strides[1], remaining,
                    transform_op, reduce_op, &reduced, tid);
        }

        // Share thread's result to the other threads.
        using uninitialized_t = util::traits::uninitialized_type_t<reduce_value_t>;
        __shared__ uninitialized_t s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<reduce_value_t*>(s_data_);

        s_data[tid] = reduced;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total.
        const reduce_value_t final = util::block::reduceShared1D<BLOCK_SIZE>(s_data, tid, reduce_op);
        if (tid == 0)
            tmp_output[batch * tmp_output_stride + blockIdx.x] = final;
    }

    // Grid.X -> Blocks to reduce the 3 innermost dimensions.
    // Grid.Y -> Batch dimension.
    template<typename value_t, typename reduce_value_t,
             typename transform_op_t, typename reduce_op_t,
             int BLOCK_DIM_X, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceUnaryConfig::BLOCK_SIZE)
    void reduceUnaryLarge4D_(
            const value_t* __restrict__ input, uint4_t input_strides, uint4_t shape, uint rows_per_batch,
            transform_op_t transform_op, reduce_op_t reduce_op, reduce_value_t init,
            reduce_value_t* __restrict__ tmp_output, uint tmp_output_stride) {
        constexpr uint EPT = ReduceUnaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceUnaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE_X = BLOCK_DIM_X * EPT;

        const uint rows_per_grid = blockDim.y * gridDim.x;
        const uint initial_row = blockDim.y * blockIdx.x + threadIdx.y;
        const uint batch = blockIdx.y;
        input += batch * input_strides[0];

        // Initial reduction. Loop until all rows are consumed.
        reduce_value_t reduced = init;
        for (uint row = initial_row; row < rows_per_batch; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const uint3_t index = indexing::indexes(row, shape[1], shape[2]); // row -> W,Z,Y
            const uint offset = indexing::at(index, input_strides);

            // Consume the row:
            for (uint cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const uint remaining = shape[3] - cid;
                util::block::reduceUnaryGlobal1D<BLOCK_DIM_X, EPT, VEC_SIZE>(
                        input + offset + cid * input_strides[3], input_strides[3], remaining,
                        transform_op, reduce_op, &reduced, threadIdx.x);
            }
        }

        // Share thread's result to the other threads.
        const uint tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        using uninitialized_t = util::traits::uninitialized_type_t<reduce_value_t>;
        __shared__ uninitialized_t s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<reduce_value_t*>(s_data_);

        s_data[tid] = reduced;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        const reduce_value_t final = util::block::reduceShared1D<BLOCK_SIZE>(s_data, tid, reduce_op);
        if (tid == 0)
            tmp_output[batch * tmp_output_stride + blockIdx.x] = final;
    }

    template<typename value_t, typename reduce_value_t, typename post_value_t,
             typename transform_op_t, typename reduce_op_t,
             typename post0_op_t, typename post1_op_t, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceUnaryConfig::BLOCK_SIZE)
    void reduceUnarySmall1D_(
            const value_t* __restrict__ input, uint2_t input_strides /* BW */,
            uint elements_per_batch, transform_op_t transform_op, reduce_op_t reduce_op, reduce_value_t init,
            post_value_t* __restrict__ output0, uint output0_stride, post0_op_t post0_op,
            post_value_t* __restrict__ output1, uint output1_stride, post1_op_t post1_op) {
        constexpr uint EPT = ReduceUnaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceUnaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = ReduceUnaryConfig::BLOCK_WORK_SIZE;

        const uint tid = threadIdx.x;
        const uint batch = blockIdx.x;
        input += input_strides[0] * batch;

        // elements -> one element per thread.
        reduce_value_t reduced = init;
        for (uint cid = 0; cid < elements_per_batch; cid += BLOCK_WORK_SIZE) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceUnaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    input + cid * input_strides[1], input_strides[1], remaining,
                    transform_op, reduce_op, &reduced, tid);
        }

        // one element per thread -> one element per block.
        using uninitialized_t = util::traits::uninitialized_type_t<reduce_value_t>;
        __shared__ uninitialized_t s_data_[BLOCK_SIZE];
        auto* s_data = reinterpret_cast<reduce_value_t*>(s_data_);

        s_data[tid] = reduced;
        util::block::synchronize();
        const reduce_value_t final_ = util::block::reduceShared1D<BLOCK_SIZE>(s_data, tid, reduce_op);

        // Save final element.
        if (tid == 0) {
            const post_value_t final = post0_op(final_);
            if (output0)
                output0[batch * output0_stride] = final;
            if (output1)
                output1[batch * output1_stride] = post1_op(final);
        }
    }
}

namespace noa::cuda::util {
    // Reduce the three or four innermost dimensions of input to one element.
    // name:            Name of the function. Used for logging if a kernel launch fails.
    // input:           On the device. Input array to reduce.
    // strides:         BDHW strides of input.
    // shape:           BDHW shape of input.
    // transform_op:    Transform operator, op(T) -> U, to apply on the input before reduction.
    // reduce_op:       Reduction operator: op(U, U) -> U.
    // init:            Per-thread initial value for the reduction.
    // output0:         On the host or device. Reduced element(s).
    //                  If reduce_batch is false, there should be one element per batch.
    // post_process0:   Post process operator. Takes the final reduced value(s) and transform it before
    //                  saving it into output0.
    // output1:         On the host or device, or nullptr. Optional secondary output.
    //                  If nullptr, ignore it. If reduce_batch is false, there should be one element per batch.
    // post_process1:   Post process operator. Takes the output0 and transform it before saving it
    //                  into output1. It is ignored if output1 is nullptr.
    // reduce_batch:    Whether the outermost dimension should be reduced.
    // swap_layout:     Whether the layout can be reordered for maximum performance.
    //                  Otherwise, assume rightmost is the fastest order.
    //                  If reduce_batch is false, only the DHW dimensions can be reordered.
    // stream:          Stream on which to enqueue this function.
    // This function is asynchronous relative to the host and may return before completion.
    // input, output0 and output1 should stay valid until completion.
    template<typename value_t, typename reduce_value_t, typename post_value_t,
             typename transform_op_t, typename reduce_op_t,
             typename post0_op_t, typename post1_op_t>
    void reduce(const char* name,
                const value_t* input, uint4_t strides, uint4_t shape,
                transform_op_t transform_op, reduce_op_t reduce_op, reduce_value_t init,
                post_value_t* output0, uint output0_stride, post0_op_t post_process0,
                post_value_t* output1, uint output1_stride, post1_op_t post_process1,
                bool reduce_batch, bool swap_layout, cuda::Stream& stream) {
        if (swap_layout) {
            if (reduce_batch) {
                const uint4_t order = indexing::order(strides, shape);
                shape = indexing::reorder(shape, order);
                strides = indexing::reorder(strides, order);
            } else {
                const uint3_t order_3d = indexing::order(uint3_t(strides.get(1)), uint3_t(shape.get(1))) + 1;
                const uint4_t order{0, order_3d[0], order_3d[1], order_3d[2]};
                shape = indexing::reorder(shape, order);
                strides = indexing::reorder(strides, order);
            }
        }

        const uint batches = reduce_batch ? 1 : shape[0];
        const uint elements = reduce_batch ? shape.elements() : shape[1] * shape[2] * shape[3];
        const bool4_t is_contiguous = indexing::isContiguous(strides, shape);

        // The output pointers are allowed to not be on the stream's device,
        // so make sure device memory is allocated for the output.
        memory::PtrDevice<post_value_t> buffer0;
        post_value_t* output0_ptr = util::devicePointer(output0, stream.device());
        post_value_t* output1_ptr = output1 ? util::devicePointer(output1, stream.device()) : nullptr;
        bool output0_was_copied{false}, output1_was_copied{false};
        uint output0_stride_{output0_stride}, output1_stride_{output1_stride};
        if (!output0_ptr) {
            output0_was_copied = true;
            output0_stride_ = 1;
            if (output1 && !output1_ptr) {
                buffer0 = memory::PtrDevice<post_value_t>(batches * 2, stream);
                output0_ptr = buffer0.get();
                output1_ptr = buffer0.get() + batches;
                output1_was_copied = true;
                output1_stride_ = 1;
            } else {
                buffer0 = memory::PtrDevice<post_value_t>(batches, stream);
                output0_ptr = buffer0.get();
            }
        } else if (output1 && !output1_ptr) {
            buffer0 = memory::PtrDevice<post_value_t>(batches, stream);
            output1_ptr = buffer0.get();
            output1_was_copied = true;
            output1_stride_ = 1;
        }

        // Small arrays (1 kernel launch):
        using namespace details;
        if (elements <= ReduceUnaryConfig::BLOCK_WORK_SIZE * 4) {
            const value_t* tmp;
            uint2_t tmp_strides;
            memory::PtrDevice<value_t> buffer1;
            // If not contiguous, don't bother and copy this (small) input to a contiguous buffer.
            if ((reduce_batch && !is_contiguous[0]) || !is_contiguous[1] || !is_contiguous[2]) {
                buffer1 = memory::PtrDevice<value_t>(shape.elements(), stream);
                memory::copy(buffer1.attach(const_cast<value_t*>(input)), size4_t(strides),
                             buffer1.share(), size4_t(shape).strides(),
                             size4_t(shape), stream);
                tmp = buffer1.get();
                tmp_strides = {elements, 1};
            } else {
                tmp = input;
                tmp_strides = {strides[0], strides[3]};
            }

            // Try to vectorize the loads for the input.
            uint vec_size = tmp_strides[1] == 1 ? util::maxVectorCount(tmp) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = tmp_strides[0] % vec_size ? 1 : vec_size;

            stream.enqueue(
                    name,
                    vec_size == 4 ? reduceUnarySmall1D_<value_t, reduce_value_t, post_value_t, transform_op_t, reduce_op_t, post0_op_t, post1_op_t, 4> :
                    vec_size == 2 ? reduceUnarySmall1D_<value_t, reduce_value_t, post_value_t, transform_op_t, reduce_op_t, post0_op_t, post1_op_t, 2> :
                                    reduceUnarySmall1D_<value_t, reduce_value_t, post_value_t, transform_op_t, reduce_op_t, post0_op_t, post1_op_t, 1>,
                    {batches, ReduceUnaryConfig::BLOCK_SIZE},
                    tmp, tmp_strides, elements, transform_op, reduce_op, init,
                    output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);

        } else if ((!reduce_batch || is_contiguous[0]) && is_contiguous[1] && is_contiguous[2]) {
            // In this config, the input can be interpreted as a 1D array. If the innermost dimension is contiguous,
            // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
            // Here we use 1D blocks to go through each batch (if reduce_batch=true, there's only one batch).
            // Each block reduces at least BLOCK_WORK_SIZE elements. Max to MAX_GRID_SIZE blocks per batch.
            const uint blocks_x = noa::math::min(noa::math::divideUp(elements, ReduceUnaryConfig::BLOCK_WORK_SIZE),
                                                 ReduceUnaryConfig::MAX_GRID_SIZE);
            const dim3 blocks(blocks_x, batches);

            // Try to vectorize the loads for the input.
            uint vec_size = strides[3] == 1 ? util::maxVectorCount(input) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = strides[0] % vec_size ? 1 : vec_size;

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrDevice<reduce_value_t> tmp(pitch * blocks.y, stream);

            // reduceUnaryLarge1D_: (batch * elements) -> (blocks.x * blocks.y) elements.
            // reduceUnarySmall1D_: (blocks.x * blocks.y) -> (blocks.y) elements.
            stream.enqueue(name,
                           vec_size == 4 ? reduceUnaryLarge1D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 4> :
                           vec_size == 2 ? reduceUnaryLarge1D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 2> :
                                           reduceUnaryLarge1D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 1>,
                           {blocks, ReduceUnaryConfig::BLOCK_SIZE},
                           input, uint2_t{strides[0], strides[3]}, elements, transform_op, reduce_op, init, tmp.get(), pitch);

            // Here the input is already transformed, so copy.
            stream.enqueue(name, reduceUnarySmall1D_<reduce_value_t, reduce_value_t, post_value_t, noa::math::copy_t, reduce_op_t, post0_op_t, post1_op_t, 4>,
                           {batches, ReduceUnaryConfig::BLOCK_SIZE},
                           tmp.get(), uint2_t{pitch, 1}, blocks.x, noa::math::copy_t{}, reduce_op, init,
                           output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);

        } else {
            // In this config, the input cannot be easily interpreted as a 1D array.
            // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
            // Since the reduceUnaryLarge4D_ kernel will decompose the "row index" back to a (W,Z,Y) index, the 3 outermost
            // dimensions can be stridesd. If the innermost dimension is contiguous, blocks can use vectorize loads
            // to read their row(s).

            // If rows are large, switch to more threads per row.
            const uint block_dim_x = shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, ReduceUnaryConfig::BLOCK_SIZE / block_dim_x);
            const uint rows = shape[2] * shape[1] * (reduce_batch ? shape[0] : 1);
            const dim3 blocks(noa::math::min(noa::math::divideUp(rows, threads.y), ReduceUnaryConfig::MAX_GRID_SIZE),
                              batches);

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            uint vec_size = strides[3] == 1 ? util::maxVectorCount(input) : 1;
            if ((strides[2] % vec_size && shape[2] != 1) ||
                (strides[1] % vec_size && shape[1] != 1) ||
                (strides[0] % vec_size && shape[0] != 1))
                vec_size = 1; // TODO If not multiple of 4, try 2 before turning off vectorization?

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrDevice<reduce_value_t> tmp(pitch * blocks.y, stream);

            if (threads.x == 256) {
                stream.enqueue(name,
                               vec_size == 4 ? reduceUnaryLarge4D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 256, 4> :
                               vec_size == 2 ? reduceUnaryLarge4D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 256, 2> :
                                               reduceUnaryLarge4D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 256, 1>,
                               {blocks, threads},
                               input, strides, shape, rows, transform_op, reduce_op, init, tmp.get(), pitch);
            } else {
                stream.enqueue(name,
                               vec_size == 4 ? reduceUnaryLarge4D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 64, 4> :
                               vec_size == 2 ? reduceUnaryLarge4D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 64, 2> :
                                               reduceUnaryLarge4D_<value_t, reduce_value_t, transform_op_t, reduce_op_t, 64, 1>,
                               {blocks, threads},
                               input, strides, shape, rows, transform_op, reduce_op, init, tmp.get(), pitch);
            }
            // Here the input is already transformed, so copy.
            stream.enqueue(name, reduceUnarySmall1D_<reduce_value_t, reduce_value_t, post_value_t, noa::math::copy_t, reduce_op_t, post0_op_t, post1_op_t, 4>,
                           {batches, ReduceUnaryConfig::BLOCK_SIZE},
                           tmp.get(), uint2_t{pitch, 1}, blocks.x, noa::math::copy_t{}, reduce_op, init,
                           output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);
        }

        // A temporary may have been allocated for the device to store the results.
        // In this case, copy back to the original output location.
        if (!buffer0.empty()) {
            const size4_t output_shape{1, 1, batches, 1};
            if (output0_was_copied)
                memory::copy(output0_ptr, output0_stride_, output0, output0_stride, output_shape, stream);
            if (output1_was_copied)
                memory::copy(output1_ptr, output0_stride_, output1, output1_stride, output_shape, stream);
        }
    }

    // Returns the variance of the input array.
    // STD:             Whether the standard deviation should be computed instead.
    // value_t:         float, double, cfloat_t, cdouble_t.
    // reduce_value_t:  If value_t is complex, should be the corresponding real type.
    //                  Otherwise, same as value_t.
    // input:           On the device. Input array to reduce.
    // input_strides:   BDHW strides of input.
    // shape:           BDHW shape of input.
    // output:          On the host or device. Output variance(s) (or stddev).
    // output_stride:   Stride of output.
    // ddof:            Delta Degree Of Freedom used to calculate the variance.
    // reduce_batch:    Whether the batch dimension should be reduced too.
    // swap_layout:     Whether the layout can be reordered for maximum performance.
    //                  Otherwise, assume rightmost is the fastest order.
    //                  If reduce_batch is false, only the DHW dimensions can be reordered.
    // stream:          Stream on which to enqueue this function.
    // This function is asynchronous relative to the host and may return before completion.
    // input and output should stay valid until completion.
    template<bool STD, typename value_t, typename reduce_value_t>
    void reduceVar(const char* name,
                   const value_t* input, uint4_t input_strides, uint4_t shape,
                   reduce_value_t* output, uint output_stride,
                   int ddof, bool reduce_batch, bool swap_layout, Stream& stream) {
        const uint batches = reduce_batch ? 1 : shape[0];
        const uint4_t shape_ = reduce_batch ? shape : uint4_t{1, shape[1], shape[2], shape[3]};
        const uint elements = shape_.elements();
        value_t* null0{};

        // Get the mean:
        memory::PtrPinned<value_t> means(batches);
        const reduce_value_t inv_count = reduce_value_t{1} / static_cast<reduce_value_t>(elements - ddof);
        auto sum_to_mean_op = [inv_count]__device__(value_t v) -> value_t { return v * inv_count; };
        util::reduce(name, input, input_strides, shape,
                     noa::math::copy_t{}, noa::math::plus_t{}, value_t{0},
                     means.get(), 1, sum_to_mean_op, null0, 0, noa::math::copy_t{}, reduce_batch, swap_layout, stream);

        // Get the variance:
        // util::reduce cannot batch this operation because the mean has to be embedded in the transform_op
        // which is fixed for every batch, whereas the mean is per-batch.
        stream.synchronize();
        auto dist2_to_var = [inv_count]__device__(reduce_value_t v) -> reduce_value_t {
            if constexpr (STD)
                return noa::math::sqrt(v * inv_count);
            return v * inv_count;
        };
        reduce_value_t* null1{};
        for (size_t batch = 0; batch < batches; ++batch) {
            value_t mean = means[batch];
            auto transform_op = [mean]__device__(value_t value) -> reduce_value_t {
                if constexpr (noa::traits::is_complex_v<value_t>) {
                    const reduce_value_t distance = noa::math::abs(value - mean);
                    return distance * distance;
                } else {
                    const reduce_value_t distance = value - mean;
                    return distance * distance;
                }
                return reduce_value_t{}; // unreachable
            };
            util::reduce(name, input + input_strides[0] * batch, input_strides, shape_,
                         transform_op, noa::math::plus_t{}, reduce_value_t{0},
                         output + output_stride * batch, 1, dist2_to_var,
                         null1, 0, noa::math::copy_t{}, true, swap_layout, stream);
        }
    }
}
