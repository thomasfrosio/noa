/// \file noa/gpu/cuda/util/ReduceBinary.cuh
/// \brief Reduction utilities.
/// \author Thomas - ffyr2w
/// \date 13 Feb 2022
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
    void reduceBinaryLarge1D_(accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor, uint2_t lhs_stride /* W,X */,
                              accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor, uint2_t rhs_stride /* W,X */,
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
        lhs_ptr_t lhs = lhs_accessor.get() + batch * lhs_stride[0];
        rhs_ptr_t rhs = rhs_accessor.get() + batch * rhs_stride[0];

        // Initial reduction to bring the lhs and rhs to BLOCK_SIZE * gridDim.x elements.
        reduce_value_t reduced = init;
        for (uint cid = base; cid < elements_per_batch; cid += BLOCK_WORK_SIZE * gridDim.x) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceBinaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    lhs + cid, lhs_stride[1], rhs + cid, rhs_stride[1], remaining,
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
    void reduceBinaryLarge4D_(accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor, uint4_t lhs_stride,
                              accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor, uint4_t rhs_stride,
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
        lhs_ptr_t lhs = lhs_accessor.get() + batch * lhs_stride[0];
        rhs_ptr_t rhs = rhs_accessor.get() + batch * rhs_stride[0];

        // Initial reduction. Loop until all rows are consumed.
        reduce_value_t reduced = init;
        for (uint row = initial_row; row < rows_per_batch; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const uint3_t index = indexing::indexes(row, shape[1], shape[2]); // row -> W,Z,Y
            const uint lhs_offset = indexing::at(index, lhs_stride);
            const uint rhs_offset = indexing::at(index, rhs_stride);

            // Consume the row:
            for (uint cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const uint remaining = shape[3] - cid;
                util::block::reduceBinaryGlobal1D<BLOCK_DIM_X, EPT, VEC_SIZE>(
                        lhs + lhs_offset + cid, lhs_stride[3], rhs + rhs_offset + cid, rhs_stride[3], remaining,
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
            accessor_t<RESTRICT, const lhs_value_t*> lhs_accessor, uint2_t lhs_stride /* batch,X */,
            accessor_t<RESTRICT, const rhs_value_t*> rhs_accessor, uint2_t rhs_stride /* batch,X */,
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
        lhs_ptr_t lhs = lhs_accessor.get() + batch * lhs_stride[0];
        rhs_ptr_t rhs = rhs_accessor.get() + batch * rhs_stride[0];

        // elements -> one element per thread.
        reduce_value_t reduced = init;
        for (uint cid = 0; cid < elements_per_batch; cid += BLOCK_WORK_SIZE) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceBinaryGlobal1D<BLOCK_SIZE, EPT, VEC_SIZE>(
                    lhs + cid, lhs_stride[1], rhs + cid, rhs_stride[1], remaining,
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
    void makeContiguous_(const T* input, uint4_t stride, uint4_t shape, uint elements,
                         memory::PtrDevice<T>& allocator, Stream& stream,
                         const T** output, uint2_t* output_stride) {
        allocator = memory::PtrDevice<T>(elements, stream);
        memory::copy(allocator.attach(const_cast<T*>(input)), size4_t{stride},
                     allocator.share(), size4_t{shape}.stride(),
                     size4_t{shape}, stream);
        *output = allocator.get();
        *output_stride = {elements, 1};
    }
}

namespace noa::cuda::util {
    /// Reduce the three or four innermost dimensions of \p lhs and \p rhs.
    /// \details Reads element-wise \p lhs and \p rhs, transform and combine them, and then reduce.
    /// \tparam REDUCE_BATCH    Whether the outermost dimension should be reduced.
    /// \tparam RESTRICT        Whether \p lhs and \p rhs can be accessed using the __restrict__ attribute.
    /// \param[in] name         Name of the function. Used for logging if a kernel launch fails.
    /// \param[in] lhs          On the \b device. Left-hande side array to reduce.
    /// \param lhs_stride       Rightmost stride of \p lhs.
    /// \param[in] rhs          On the \b device. Left-hande side array to reduce.
    /// \param rhs_stride       Rightmost stride of \p rhs.
    /// \param shape            Rightmost shape of \p lhs and \p rhs.
    /// \param transform_op_lhs Transform operator, op(\p lhs_value_t) -> Xl, to apply on \p lhs before combination.
    /// \param transform_op_rhs Transform operator, op(\p rhs_value_t) -> Xr, to apply on \p rhs before combination.
    /// \param combine_op       Combine operator, op(Xl, Xr), to apply on the left and right transformed value before
    ///                         reduction. The output value of this operator is casted to \p reduce_value_t.
    /// \param reduce_op        Reduction operator: op(\p reduce_value_t, \p reduce_value_t) -> \p reduce_value_t.
    /// \param init             Per-thread initial value for the reduction.
    /// \param[out] output0     On the \b host or \b device. Reduced element(s).
    ///                         If REDUCE_BATCH is false, there should be \p shape[0] elements.
    /// \param post_process0    Post process operator. Takes the final reduced value(s) and transform it before
    ///                         saving it into \p output0.
    /// \param[out] output1     On the \b host or \b device, or nullptr. Optional secondary output.
    ///                         If nullptr, it is ignored. If REDUCE_BATCH is false, there should be \p shape[0] elements.
    /// \param post_process1    Post process operator. Takes the \p output0 and transform it before saving it
    ///                         into \p output1. It is ignored if \p output1 is nullptr.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       \p lhs, \p rhs, \p output0 and \p output1 should stay valid until completion.
    template<bool REDUCE_BATCH, bool RESTRICT,
             typename lhs_value_t, typename rhs_value_t, typename reduce_value_t, typename post_value_t,
             typename transform_op_lhs_t, typename transform_op_rhs_t, typename combine_op_t, typename reduce_op_t,
             typename post0_op_t, typename post1_op_t>
    void reduce(const char* name,
                const lhs_value_t* lhs, uint4_t lhs_stride,
                const rhs_value_t* rhs, uint4_t rhs_stride, uint4_t shape,
                transform_op_lhs_t transform_op_lhs, transform_op_rhs_t transform_op_rhs, combine_op_t combine_op,
                reduce_op_t reduce_op, reduce_value_t init,
                post_value_t* output0, uint output0_stride, post0_op_t post_process0,
                post_value_t* output1, uint output1_stride, post1_op_t post_process1,
                cuda::Stream& stream) {
        const uint batches = REDUCE_BATCH ? 1 : shape[0];
        const uint elements = REDUCE_BATCH ? shape.elements() : shape[1] * shape[2] * shape[3];
        const bool4_t is_lhs_contiguous = indexing::isContiguous(lhs_stride, shape);
        const bool4_t is_rhs_contiguous = indexing::isContiguous(rhs_stride, shape);
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
            uint2_t tmp_lhs_stride{lhs_stride[0], lhs_stride[3]};
            uint2_t tmp_rhs_stride{rhs_stride[0], rhs_stride[3]};
            memory::PtrDevice<lhs_value_t> buffer_lhs;
            memory::PtrDevice<rhs_value_t> buffer_rhs;

            // If not contiguous, don't bother and copy this (small) input to a contiguous buffer.
            if ((REDUCE_BATCH && !is_lhs_contiguous[0]) || !is_lhs_contiguous[1] || !is_lhs_contiguous[2])
                makeContiguous_(lhs, lhs_stride, shape, elements, buffer_lhs, stream, &tmp_lhs, &tmp_lhs_stride);
            if ((REDUCE_BATCH && !is_rhs_contiguous[0]) || !is_rhs_contiguous[1] || !is_rhs_contiguous[2])
                makeContiguous_(rhs, rhs_stride, shape, elements, buffer_rhs, stream, &tmp_rhs, &tmp_lhs_stride);

            // Try to vectorize the loads for the input.
            uint vec_size = tmp_lhs_stride[1] == 1 && tmp_rhs_stride[1] == 1 ?
                            std::min(maxVectorCount(tmp_lhs), maxVectorCount(tmp_rhs)) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = tmp_lhs_stride[0] % vec_size && tmp_rhs_stride[0] % vec_size ? 1 : vec_size;

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
                    lhs_accessor, tmp_lhs_stride, rhs_accessor, tmp_rhs_stride, elements,
                    transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init,
                    output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);

        } else if ((!REDUCE_BATCH || is_contiguous[0]) && is_contiguous[1] && is_contiguous[2]) {
            // In this config, the input can be interpreted as a 1D array. If the innermost dimension is contiguous,
            // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
            // Here we use 1D blocks to go through each batch (if REDUCE_BATCH=true, there's only one batch).
            // Each block reduces at least BLOCK_WORK_SIZE elements. Max to MAX_GRID_SIZE blocks per batch.
            const uint blocks_x = noa::math::min(noa::math::divideUp(elements, ReduceBinaryConfig::BLOCK_WORK_SIZE),
                                                 ReduceBinaryConfig::MAX_GRID_SIZE);
            const dim3 blocks(blocks_x, batches);

            // Try to vectorize the loads for the input.
            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(rhs)) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = lhs_stride[0] % vec_size || rhs_stride[0] % vec_size ? 1 : vec_size;

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
                    lhs_accessor, uint2_t{lhs_stride[0], lhs_stride[3]},
                    rhs_accessor, uint2_t{rhs_stride[0], rhs_stride[3]}, elements,
                    transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init, tmp.get(), pitch);

            // Here the input is already transformed, so copy.
            stream.enqueue(name, reduceBinaryLargeFinal_<reduce_value_t, post_value_t, reduce_op_t, post0_op_t, post1_op_t, 4>,
                           {batches, ReduceBinaryConfig::BLOCK_SIZE},
                           tmp.get(), pitch, blocks.x, reduce_op, init,
                           output0_ptr, output0_stride_, post_process0, output1_ptr, output1_stride_, post_process1);

        } else {
            // In this config, the input cannot be easily interpreted as a 1D array.
            // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
            // Since the reduceLarge4D_ kernel will decompose the "row index" back to a (W,Z,Y) index, the 3 outermost
            // dimensions can be strided. If the innermost dimension is contiguous, blocks can use vectorize loads
            // to read their row(s).

            // If rows are large, switch to more threads per row.
            const uint block_dim_x = shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, ReduceBinaryConfig::BLOCK_SIZE / block_dim_x);
            const uint rows = shape[2] * shape[1] * (REDUCE_BATCH ? shape[0] : 1);
            const dim3 blocks(noa::math::min(noa::math::divideUp(rows, threads.y), ReduceBinaryConfig::MAX_GRID_SIZE), batches);

            // Try to vectorize the loads within a row.
            // Check that the beginning of each row is at the same alignment. This is true for pitch2D arrays.
            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(lhs), maxVectorCount(rhs)) : 1;
            if (((lhs_stride[2] % vec_size || rhs_stride[2] % vec_size) && shape[2] != 1) ||
                ((lhs_stride[1] % vec_size || rhs_stride[1] % vec_size) && shape[1] != 1) ||
                ((lhs_stride[0] % vec_size || rhs_stride[0] % vec_size) && shape[0] != 1))
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
                        lhs_accessor, lhs_stride, rhs_accessor, rhs_stride, shape, rows,
                        transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init, tmp.get(), pitch);
            } else {
                stream.enqueue(
                        name,
                        vec_size == 4 ? reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 64, RESTRICT, 4> :
                        vec_size == 2 ? reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 64, RESTRICT, 2> :
                                        reduceBinaryLarge4D_<lhs_value_t, rhs_value_t, reduce_value_t, transform_op_lhs_t, transform_op_rhs_t, combine_op_t, reduce_op_t, 64, RESTRICT, 1>,
                        {blocks, threads},
                        lhs_accessor, lhs_stride, rhs_accessor, rhs_stride, shape, rows,
                        transform_op_lhs, transform_op_rhs, combine_op, reduce_op, init, tmp.get(), pitch);
            }
            // Here the input is already transformed, so copy.
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
                memory::copy(output1_ptr, output0_stride_, output1, output0_stride, output_shape, stream);
        }
    }
}
