#pragma once

#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"
#include "noa/gpu/cuda/memory/PtrPinned.h"

#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/Block.cuh"

// These reduction kernels are similar to ReduceUnary.cuh,
// except that they take two inputs, combine them and then reduce.

namespace noa::cuda::util::details {
    struct ReduceBinaryConfig {
        static constexpr uint ELEMENTS_PER_THREAD = 8;
        static constexpr uint BLOCK_SIZE = 512;
        static constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
        static constexpr uint MAX_GRID_SIZE = 4096;
    };

    template<typename lhs_value_t, typename rhs_value_t, typename reduce_value_t,
             typename transform_op_lhs_t, typename transform_op_rhs_t,
             typename combine_op_t, typename reduce_op_t, bool RESTRICT, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceBinaryConfig::BLOCK_SIZE)
    void reduceBinaryLarge1D_(accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor, uint2_t lhs_strides /* BW */,
                              accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor, uint2_t rhs_strides /* BW */,
                              uint elements_per_batch,
                              transform_op_lhs_t transform_op_lhs, transform_op_rhs_t transform_op_rhs,
                              combine_op_t combine_op, reduce_op_t reduce_op, reduce_value_t init,
                              reduce_value_t* __restrict__ tmp_output, uint tmp_output_stride) {
        constexpr uint EPT = ReduceBinaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceBinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = ReduceBinaryConfig::BLOCK_WORK_SIZE;

        // Each block reduces chunks of BLOCK_WORK_SIZE elements at a time.
        // Batches are kept independent of each other.
        const uint tid = threadIdx.x;
        const uint base = blockIdx.x * BLOCK_WORK_SIZE;
        const uint batch = blockIdx.y;

        using lhs_ptr_t = typename accessor_t<RESTRICT, const lhs_value_t*>::ptr_type;
        using rhs_ptr_t = typename accessor_t<RESTRICT, const rhs_value_t*>::ptr_type;
        lhs_ptr_t lhs = lhs_accessor.get() + batch * lhs_strides[0];
        rhs_ptr_t rhs = rhs_accessor.get() + batch * rhs_strides[0];

        // Initial reduction to bring the lhs and rhs to BLOCK_SIZE * gridDim.x elements.
        reduce_value_t reduced = init;
        for (uint cid = base; cid < elements_per_batch; cid += BLOCK_WORK_SIZE * gridDim.x) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceBinaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    lhs + cid * lhs_strides[1], lhs_strides[1],
                    rhs + cid * rhs_strides[1], rhs_strides[1], remaining,
                    transform_op_lhs, transform_op_rhs, combine_op, reduce_op, &reduced, tid);
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

    template<typename lhs_value_t, typename rhs_value_t, typename reduce_value_t,
             typename transform_op_lhs_t, typename transform_op_rhs_t,
             typename combine_op_t, typename reduce_op_t,
             int BLOCK_DIM_X, bool RESTRICT, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceBinaryConfig::BLOCK_SIZE)
    void reduceBinaryLarge4D_(accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor, uint4_t lhs_strides,
                              accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor, uint4_t rhs_strides,
                              uint4_t shape, uint rows_per_batch,
                              transform_op_lhs_t transform_op_lhs, transform_op_rhs_t transform_op_rhs,
                              combine_op_t combine_op, reduce_op_t reduce_op, reduce_value_t init,
                              reduce_value_t* __restrict__ tmp_output, uint tmp_output_stride) {
        constexpr uint EPT = ReduceBinaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceBinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE_X = BLOCK_DIM_X * EPT;

        const uint rows_per_grid = blockDim.y * gridDim.x;
        const uint initial_row = blockDim.y * blockIdx.x + threadIdx.y;
        const uint batch = blockIdx.y;

        using lhs_ptr_t = typename accessor_t<RESTRICT, const lhs_value_t*>::ptr_type;
        using rhs_ptr_t = typename accessor_t<RESTRICT, const rhs_value_t*>::ptr_type;
        lhs_ptr_t lhs = lhs_accessor.get() + batch * lhs_strides[0];
        rhs_ptr_t rhs = rhs_accessor.get() + batch * rhs_strides[0];

        // Initial reduction. Loop until all rows are consumed.
        reduce_value_t reduced = init;
        for (uint row = initial_row; row < rows_per_batch; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const uint3_t index = indexing::indexes(row, shape[1], shape[2]); // row -> W,Z,Y
            const uint lhs_offset = indexing::at(index, lhs_strides);
            const uint rhs_offset = indexing::at(index, rhs_strides);

            // Consume the row:
            for (uint cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const uint remaining = shape[3] - cid;
                util::block::reduceBinaryGlobal1D<BLOCK_DIM_X, EPT, VEC_SIZE>(
                        lhs + lhs_offset + cid * lhs_strides[3], lhs_strides[3],
                        rhs + rhs_offset + cid * rhs_strides[3], rhs_strides[3], remaining,
                        transform_op_lhs, transform_op_rhs, combine_op, reduce_op, &reduced, threadIdx.x);
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

    template<typename reduce_value_t, typename post_value_t,
             typename reduce_op_t, typename post0_op_t, typename post1_op_t, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceBinaryConfig::BLOCK_SIZE)
    void reduceBinaryLargeFinal_(
            const reduce_value_t* __restrict__ tmp_output, uint tmp_output_stride /* batch,X */,
            uint elements_per_batch, reduce_op_t reduce_op, reduce_value_t init,
            post_value_t* __restrict__ output0, uint output0_stride, post0_op_t post0_op,
            post_value_t* __restrict__ output1, uint output1_stride, post1_op_t post1_op) {
        constexpr uint EPT = ReduceBinaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceBinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = ReduceBinaryConfig::BLOCK_WORK_SIZE;

        const uint tid = threadIdx.x;
        const uint batch = blockIdx.x;
        tmp_output += tmp_output_stride * batch;

        // elements -> one element per thread.
        reduce_value_t reduced = init;
        for (uint cid = 0; cid < elements_per_batch; cid += BLOCK_WORK_SIZE) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceUnaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    tmp_output + cid, 1, remaining,
                    noa::math::copy_t{}, reduce_op, &reduced, tid);
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

    template<typename lhs_value_t, typename rhs_value_t, typename reduce_value_t, typename post_value_t,
             typename transform_op_lhs_t, typename transform_op_rhs_t,
             typename combine_op_t, typename reduce_op_t, typename post0_op_t, typename post1_op_t,
             bool RESTRICT, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceBinaryConfig::BLOCK_SIZE)
    void reduceBinarySmall1D_(
            accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor, uint2_t lhs_strides /* BW */,
            accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor, uint2_t rhs_strides /* BW */,
            uint elements_per_batch, transform_op_lhs_t transform_op_lhs, transform_op_rhs_t transform_op_rhs,
            combine_op_t combine_op, reduce_op_t reduce_op, reduce_value_t init,
            post_value_t* __restrict__ output0, uint output0_stride, post0_op_t post0_op,
            post_value_t* __restrict__ output1, uint output1_stride, post1_op_t post1_op) {
        constexpr uint EPT = ReduceBinaryConfig::ELEMENTS_PER_THREAD;
        constexpr uint BLOCK_SIZE = ReduceBinaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = ReduceBinaryConfig::BLOCK_WORK_SIZE;

        const uint tid = threadIdx.x;
        const uint batch = blockIdx.x;

        using lhs_ptr_t = typename accessor_t<RESTRICT, const lhs_value_t*>::ptr_type;
        using rhs_ptr_t = typename accessor_t<RESTRICT, const rhs_value_t*>::ptr_type;
        lhs_ptr_t lhs = lhs_accessor.get() + batch * lhs_strides[0];
        rhs_ptr_t rhs = rhs_accessor.get() + batch * rhs_strides[0];

        // elements -> one element per thread.
        reduce_value_t reduced = init;
        for (uint cid = 0; cid < elements_per_batch; cid += BLOCK_WORK_SIZE) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceBinaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    lhs + cid * lhs_strides[1], lhs_strides[1],
                    rhs + cid * rhs_strides[1], rhs_strides[1], remaining,
                    transform_op_lhs, transform_op_rhs, combine_op, reduce_op, &reduced, tid);
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

    template<typename T>
    void makeContiguous_(const T* input, uint4_t strides, uint4_t shape, uint elements,
                         memory::PtrDevice<T>& allocator, Stream& stream,
                         const T** output, uint2_t* output_strides) {
        allocator = memory::PtrDevice<T>(elements, stream);
        memory::copy(allocator.attach(const_cast<T*>(input)), size4_t(strides),
                     allocator.share(), size4_t(shape).strides(),
                     size4_t(shape), stream);
        *output = allocator.get();
        *output_strides = {elements, 1};
    }
}

namespace noa::cuda::util {
    // Reduce the three or four innermost dimensions of lhs and rhs.
    // Reads element-wise lhs and rhs, transform and combine them, and then reduce.
    // RESTRICT:            Whether lhs and rhs can be accessed using the __restrict__ attribute.
    // name:                Name of the function. Used for logging if a kernel launch fails.
    // lhs:                 On the device. Left-hande side array to reduce.
    // lhs_strides:         BDHW strides of lhs.
    // rhs:                 On the device. Left-hande side array to reduce.
    // rhs_strides:         BDHW strides of rhs.
    // shape:               BDHW shape of lhs and rhs. Assume rightmost order is the fastest order.
    // transform_op_lhs:    Transform operator, op(lhs_value_t) -> Xl, to apply on lhs before combination.
    // transform_op_rhs:    Transform operator, op(rhs_value_t) -> Xr, to apply on rhs before combination.
    // combine_op:          Combine operator, op(Xl, Xr), to apply on the left and right transformed value before
    //                      reduction. The output value of this operator is cast to reduce_value_t.
    // reduce_op:           Reduction operator: op(reduce_value_t, reduce_value_t) -> reduce_value_t.
    // init:                Per-thread initial value for the reduction.
    // output0:             On the host or device. Reduced element(s).
    //                      If reduce_batch is false, there should be shape[0] elements.
    // post_process0:       Post process operator. Takes the final reduced value(s) and transform it before
    //                      saving it into output0.
    // output1:             On the host or device, or nullptr. Optional secondary output.
    //                      If nullptr, it is ignored. If reduce_batch is false, there should be shape[0] elements.
    // post_process1:       Post process operator. Takes the output0 and transform it before saving it
    //                      into output1. It is ignored if output1 is nullptr.
    // reduce_batch:        Whether the outermost dimension should be reduced.
    // stream:              Stream on which to enqueue this function.
    // This function is asynchronous relative to the host and may return before completion.
    // lhs, rhs, output0 and output1 should stay valid until completion.
    template<bool RESTRICT,
             typename lhs_value_t, typename rhs_value_t, typename reduce_value_t, typename post_value_t,
             typename transform_op_lhs_t, typename transform_op_rhs_t, typename combine_op_t, typename reduce_op_t,
             typename post0_op_t, typename post1_op_t>
    void reduce(const char* name,
                const lhs_value_t* lhs, uint4_t lhs_strides,
                const rhs_value_t* rhs, uint4_t rhs_strides, uint4_t shape,
                transform_op_lhs_t transform_op_lhs, transform_op_rhs_t transform_op_rhs, combine_op_t combine_op,
                reduce_op_t reduce_op, reduce_value_t init,
                post_value_t* output0, uint output0_stride, post0_op_t post_process0,
                post_value_t* output1, uint output1_stride, post1_op_t post_process1,
                bool reduce_batch, cuda::Stream& stream) {
        const uint batches = reduce_batch ? 1 : shape[0];
        const uint elements = reduce_batch ? shape.elements() : shape[1] * shape[2] * shape[3];
        const bool4_t is_lhs_contiguous = indexing::isContiguous(lhs_strides, shape);
        const bool4_t is_rhs_contiguous = indexing::isContiguous(rhs_strides, shape);
        const bool4_t is_contiguous = is_lhs_contiguous && is_rhs_contiguous;

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
        if (elements <= ReduceBinaryConfig::BLOCK_WORK_SIZE * 4) {
            const lhs_value_t* tmp_lhs = lhs;
            const rhs_value_t* tmp_rhs = rhs;
            uint2_t tmp_lhs_strides{lhs_strides[0], lhs_strides[3]};
            uint2_t tmp_rhs_strides{rhs_strides[0], rhs_strides[3]};
            memory::PtrDevice<lhs_value_t> buffer_lhs;
            memory::PtrDevice<rhs_value_t> buffer_rhs;

            // If not contiguous, don't bother and copy this (small) input to a contiguous buffer.
            if ((reduce_batch && !is_lhs_contiguous[0]) || !is_lhs_contiguous[1] || !is_lhs_contiguous[2])
                makeContiguous_(lhs, lhs_strides, shape, elements, buffer_lhs, stream, &tmp_lhs, &tmp_lhs_strides);
            if ((reduce_batch && !is_rhs_contiguous[0]) || !is_rhs_contiguous[1] || !is_rhs_contiguous[2])
                makeContiguous_(rhs, rhs_strides, shape, elements, buffer_rhs, stream, &tmp_rhs, &tmp_lhs_strides);

            // Try to vectorize the loads for the input.
            uint vec_size = tmp_lhs_strides[1] == 1 && tmp_rhs_strides[1] == 1 ?
                            std::min(maxVectorCount(tmp_lhs), maxVectorCount(tmp_rhs)) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = tmp_lhs_strides[0] % vec_size && tmp_rhs_strides[0] % vec_size ? 1 : vec_size;

            // If one of them was copied to a contiguous buffer, then RESTRICT can be true.
            // However, that means doubling the kernel instantiations, so for now, ignore this case.
            accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor(tmp_lhs);
            accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor(tmp_rhs);
            stream.enqueue(
                    name,
                    vec_size == 4 ? reduceBinarySmall1D_<lhs_value_t, rhs_value_t, reduce_value_t, post_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, post0_op_t, post1_op_t, RESTRICT, 4> :
                    vec_size == 2 ? reduceBinarySmall1D_<lhs_value_t, rhs_value_t, reduce_value_t, post_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, post0_op_t, post1_op_t, RESTRICT, 2> :
                                    reduceBinarySmall1D_<lhs_value_t, rhs_value_t, reduce_value_t, post_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, post0_op_t, post1_op_t, RESTRICT, 1>,
                    {batches, ReduceBinaryConfig::BLOCK_SIZE},
                    lhs_accessor, tmp_lhs_strides, rhs_accessor, tmp_rhs_strides, elements,
                    transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init,
                    output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);

        } else if ((!reduce_batch || is_contiguous[0]) && is_contiguous[1] && is_contiguous[2]) {
            // In this config, the input can be interpreted as a 1D array. If the innermost dimension is contiguous,
            // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
            // Here we use 1D blocks to go through each batch (if reduce_batch=true, there's only one batch).
            // Each block reduces at least BLOCK_WORK_SIZE elements. Max to MAX_GRID_SIZE blocks per batch.
            const uint blocks_x = noa::math::min(noa::math::divideUp(elements, ReduceBinaryConfig::BLOCK_WORK_SIZE),
                                                 ReduceBinaryConfig::MAX_GRID_SIZE);
            const dim3 blocks(blocks_x, batches);

            // Try to vectorize the loads for the input.
            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(rhs)) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = lhs_strides[0] % vec_size || rhs_strides[0] % vec_size ? 1 : vec_size;

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrDevice<reduce_value_t> tmp(pitch * blocks.y, stream);

            // reduceLarge1D_: (batch * elements) -> (blocks.x * blocks.y) elements.
            // reduceSmall1D_: (blocks.x * blocks.y) -> (blocks.y) elements.
            accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor(lhs);
            accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor(rhs);
            stream.enqueue(
                    name,
                    vec_size == 4 ? reduceBinaryLarge1D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, RESTRICT, 4> :
                    vec_size == 2 ? reduceBinaryLarge1D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, RESTRICT, 2> :
                                    reduceBinaryLarge1D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, RESTRICT, 1>,
                    {blocks, ReduceBinaryConfig::BLOCK_SIZE},
                    lhs_accessor, uint2_t(lhs_strides[0], lhs_strides[3]),
                    rhs_accessor, uint2_t(rhs_strides[0], rhs_strides[3]), elements,
                    transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init, tmp.get(), pitch);

            stream.enqueue(name, reduceBinaryLargeFinal_<reduce_value_t, post_value_t, reduce_op_t, post0_op_t, post1_op_t, 4>,
                           {batches, ReduceBinaryConfig::BLOCK_SIZE},
                           tmp.get(), pitch, blocks.x, reduce_op, init,
                           output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);

        } else {
            // In this config, the inputs cannot be easily interpreted as a 1D array.
            // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
            // Since the reduceBinaryLarge4D_ kernel will decompose the "row index" back to a (W,Z,Y) index, the 3
            // outermost dimensions can be stridesd. If the innermost dimension is contiguous, blocks can use vectorize
            // loads to read their row(s).

            // If rows are large, switch to more threads per row.
            const uint block_dim_x = shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, ReduceBinaryConfig::BLOCK_SIZE / block_dim_x);
            const uint rows = shape[2] * shape[1] * (reduce_batch ? shape[0] : 1);
            const dim3 blocks(noa::math::min(noa::math::divideUp(rows, threads.y), ReduceBinaryConfig::MAX_GRID_SIZE), batches);

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(rhs)) : 1;
            if (((lhs_strides[2] % vec_size || rhs_strides[2] % vec_size) && shape[2] != 1) ||
                ((lhs_strides[1] % vec_size || rhs_strides[1] % vec_size) && shape[1] != 1) ||
                ((lhs_strides[0] % vec_size || rhs_strides[0] % vec_size) && shape[0] != 1))
                vec_size = 1; // TODO If not multiple of 4, try 2 before turning off vectorization?

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrDevice<reduce_value_t> tmp(pitch * blocks.y, stream);
            accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor(lhs);
            accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor(rhs);
            if (threads.x == 256) {
                stream.enqueue(
                        name,
                        vec_size == 4 ? reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 256, RESTRICT, 4> :
                        vec_size == 2 ? reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 256, RESTRICT, 2> :
                                        reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 256, RESTRICT, 1>,
                        {blocks, threads},
                        lhs_accessor, lhs_strides, rhs_accessor, rhs_strides, shape, rows,
                        transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init, tmp.get(), pitch);
            } else {
                stream.enqueue(
                        name,
                        vec_size == 4 ? reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 64, RESTRICT, 4> :
                        vec_size == 2 ? reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 64, RESTRICT, 2> :
                                        reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 64, RESTRICT, 1>,
                        {blocks, threads},
                        lhs_accessor, lhs_strides, rhs_accessor, rhs_strides, shape, rows,
                        transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init, tmp.get(), pitch);
            }
            stream.enqueue(name, reduceBinaryLargeFinal_<reduce_value_t, post_value_t, reduce_op_t, post0_op_t, post1_op_t, 4>,
                           {batches, ReduceBinaryConfig::BLOCK_SIZE},
                           tmp.get(), pitch, blocks.x, reduce_op, init,
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
}
