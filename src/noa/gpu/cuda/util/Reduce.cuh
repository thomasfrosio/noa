/// \file noa/gpu/cuda/util/Reduce.cuh
/// \brief Reduction utilities.
/// \author Thomas - ffyr2w
/// \date 13 Feb 2022
#pragma once

#include "noa/common/Math.h"
#include "noa/gpu/cuda/math/Reduce.h"
#include "noa/gpu/cuda/memory/Copy.h"
#include "noa/gpu/cuda/memory/PtrDevice.h"

#include "noa/gpu/cuda/util/Pointers.h"
#include "noa/gpu/cuda/util/Block.cuh"

// These reduction kernels are adapted from different sources, but the main logic come from:
//  - https://github.com/NVIDIA/cuda-samples/tree/master/Samples/reduction
//  - https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

namespace noa::cuda::util::details {
    struct ReduceConfig {
        static constexpr uint ELEMENTS_PER_THREAD = 8;
        static constexpr uint BLOCK_SIZE = 512;
        static constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;
        static constexpr uint MAX_GRID_SIZE = 4096;
    };

    // Grid.X is the number of blocks, at most MAX_GRID_SIZE.
    // Grid.Y is the number of batches. If the input was fully reduced to one element, there's only one batch.
    // The output should have the size of the grid, at minimum.
    template<typename T, typename U, typename TransformOp, typename ReduceOp, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceConfig::BLOCK_SIZE)
    void reduceLarge1D_(const T* __restrict__ input, uint2_t input_stride /* W,X */, uint elements_per_batch,
                        TransformOp transform_op, ReduceOp reduce_op, U init, U* __restrict__ tmp_output) {
        // Each block reduces chunks of BLOCK_WORK_SIZE elements at a time.
        // Batches are kept independent of each other.
        const uint tid = threadIdx.x;
        const uint base = blockIdx.x * ReduceConfig::BLOCK_WORK_SIZE;
        const uint batch = blockIdx.y;
        input += batch * input_stride[0];

        // Initial reduction to bring the input to BLOCK_SIZE * gridDim.x elements.
        U reduced = init;
        for (uint cid = base; cid < elements_per_batch; cid += ReduceConfig::BLOCK_WORK_SIZE * gridDim.x) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceGlobal1D<ReduceConfig::BLOCK_SIZE, ReduceConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                    input + cid, input_stride[1], remaining, transform_op, reduce_op, &reduced, tid);
        }

        // Share thread's result to the other threads.
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        const U final = util::block::reduceShared1D<ReduceConfig::BLOCK_SIZE>(s_data, tid, reduce_op);
        if (tid == 0)
            tmp_output[batch * gridDim.x + blockIdx.x] = final;
    }

    // Grid.X -> Blocks to reduce the 3 innermost dimensions.
    // Grid.Y -> Batch dimension.
    template<typename T, typename U, typename TransformOp, typename ReduceOp, int BLOCK_DIM_X, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceConfig::BLOCK_SIZE)
    void reduceLarge4D_(const T* __restrict__ input, uint4_t input_stride, uint4_t shape, uint rows_per_batch,
                        TransformOp transform_op, ReduceOp reduce_op, U init, U* __restrict__ tmp_output) {
        constexpr uint BLOCK_WORK_SIZE_X = BLOCK_DIM_X * ReduceConfig::ELEMENTS_PER_THREAD;
        const uint rows_per_grid = blockDim.y * gridDim.x;
        const uint initial_row = blockDim.y * blockIdx.x;
        const uint batch = blockIdx.y;
        input += batch * input_stride[0];

        // Initial reduction. Loop until all rows are consumed.
        U reduced = init;
        for (uint row = initial_row; row < rows_per_batch; row += rows_per_grid) {
            // Retrieve the 3D block index from the linear Grid.X:
            const uint3_t index = indexing::indexes(row, shape[1], shape[2]); // row -> W,Z,Y
            const uint offset = indexing::at(index, input_stride);

            // Consume the row:
            for (uint cid = 0; cid < shape[3]; cid += BLOCK_WORK_SIZE_X) {
                const uint remaining = shape[3] - cid;
                util::block::reduceGlobal1D<ReduceConfig::BLOCK_SIZE, ReduceConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                        input + offset + cid, input_stride[3], remaining,
                        transform_op, reduce_op, &reduced, threadIdx.x);
            }
        }

        // Share thread's result to the other threads.
        const uint tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        util::block::synchronize();

        // Reduce shared data to one element, i.e. gridDim.x elements in total
        const U final = util::block::reduceShared1D<ReduceConfig::BLOCK_SIZE>(s_data, tid, reduce_op);
        if (tid == 0)
            tmp_output[batch * gridDim.x + blockIdx.x] = final;
    }

    template<typename T, typename U, typename TransformOp, typename ReduceOp,
             typename PostProcessOp0, typename PostProcessOp1, int VEC_SIZE>
    __global__ __launch_bounds__(ReduceConfig::BLOCK_SIZE)
    void reduceSmall1D_(const T* __restrict__ input, uint2_t input_stride /* batch,X */,
                        uint elements_per_batch, TransformOp transform_op, ReduceOp reduce_op, U init,
                        U* __restrict__ output0, PostProcessOp0 post_process0,
                        U* __restrict__ output1, PostProcessOp1 post_process1) {
        const uint tid = threadIdx.x;
        const uint batch = blockIdx.x;
        input += input_stride[0] * batch;

        // elements -> one element per thread.
        U reduced = init;
        for (uint cid = 0; cid < elements_per_batch; cid += ReduceConfig::BLOCK_WORK_SIZE) {
            const uint remaining = elements_per_batch - cid;
            util::block::reduceGlobal1D<ReduceConfig::BLOCK_SIZE, ReduceConfig::ELEMENTS_PER_THREAD, VEC_SIZE>(
                    input + cid, input_stride[1], remaining, transform_op, reduce_op, &reduced, tid);
        }

        // one element per thread -> one element per block.
        U* s_data = util::block::dynamicSharedResource<U>(); // BLOCK_SIZE elements.
        s_data[tid] = reduced;
        util::block::synchronize();
        U final0 = util::block::reduceShared1D<ReduceConfig::BLOCK_SIZE>(s_data, tid, reduce_op);

        // Save final element.
        if (tid == 0) {
            final0 = post_process0(final0);
            if (output0)
                output0[batch] = final0;
            if (output1)
                output1[batch] = post_process1(final0);
        }
    }
}

namespace noa::cuda::util {
    /// Reduce the three or four innermost dimensions of \p input to one element.
    /// \tparam REDUCE_BATCH    Whether the outermost dimension should be reduced.
    /// \param[in] name         Name of the function. Used for logging if a kernel launch fails.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost stride of \p input. Contiguous arrays are more efficient to reduce.
    /// \param shape            Rightmost shape of \p input.
    /// \param transform_op     Transform operator, op(\p T) -> \p U, to apply on the input before reduction.
    /// \param reduce_op        Reduction operator: op(\p U, \p U) -> \p U.
    /// \param init             Per-thread initial value for the reduction.
    /// \param[out] output0     On the \b host or \b device. Reduced element(s).
    ///                         If REDUCE_BATCH is false, there should be \p shape[0] elements.
    /// \param post_process0    Post process operator. Takes the final reduced value(s) and transform it before
    ///                         saving it into \p output0.
    /// \param[out] output1     On the \b host or \b device, or nullptr. Optional secondary output.
    ///                         If nullptr, ignore it. If REDUCE_BATCH is false, there should be \p shape[0] elements.
    /// \param post_process1    Post process operator. Takes the \p output0 and transform it before saving it
    ///                         into \p output1. It is ignored if \p output1 is nullptr.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       \p input, \p output0 and \p output1 should stay valid until completion.
    template<bool REDUCE_BATCH,
             typename T, typename U, typename TransformOp, typename ReduceOp,
             typename PostProcessOp0, typename PostProcessOp1>
    void reduce(const char* name,
                const T* input, uint4_t stride, uint4_t shape,
                TransformOp transform_op, ReduceOp reduce_op, U init,
                U* output0, PostProcessOp0 post_process0,
                U* output1, PostProcessOp1 post_process1,
                cuda::Stream& stream) {
        const uint batches = REDUCE_BATCH ? 1 : shape[0];
        const uint elements = REDUCE_BATCH ? shape.elements() : shape[1] * shape[2] * shape[3];
        const bool4_t is_contiguous = indexing::isContiguous(stride, shape);
        const uint ndim = shape.ndim();

        // The output pointers are allowed to not be on the stream's device,
        // so make sure device memory is allocated for the output.
        memory::PtrDevice<U> buffer0;
        U* output0_ptr = util::devicePointer(output0, stream.device());
        U* output1_ptr = output1 ? util::devicePointer(output1, stream.device()) : nullptr;
        bool output0_was_copied{false}, output1_was_copied{false};
        if (!output0_ptr) {
            output0_was_copied = true;
            if (output1 && !output1_ptr) {
                buffer0 = memory::PtrDevice<U>{batches * 2, stream};
                output0_ptr = buffer0.get();
                output1_ptr = buffer0.get() + batches;
                output1_was_copied = true;
            } else {
                buffer0 = memory::PtrDevice<U>{batches, stream};
                output0_ptr = buffer0.get();
            }
        } else if (output1 && !output1_ptr) {
            buffer0 = memory::PtrDevice<U>{batches, stream};
            output1_ptr = buffer0.get();
            output1_was_copied = true;
        }

        // Small arrays (1 kernel launch):
        using namespace details;
        if (elements <= ReduceConfig::BLOCK_WORK_SIZE * 4) {
            const T* tmp;
            uint2_t tmp_stride;
            memory::PtrDevice<T> buffer1;
            // If not contiguous, don't bother and copy this (small) input to a contiguous buffer.
            if ((REDUCE_BATCH && !is_contiguous[0]) || !is_contiguous[1] || !is_contiguous[2]) {
                buffer1 = memory::PtrDevice<T>{shape.elements(), stream};
                memory::copy(buffer1.attach(const_cast<T*>(input)), size4_t{stride},
                             buffer1.share(), size4_t{shape}.stride(),
                             size4_t{shape}, stream);
                tmp = buffer1.get();
                tmp_stride = {elements, 1};
            } else {
                tmp = input;
                tmp_stride = {stride[0], stride[3]};
            }

            // Try to vectorize the loads for the input.
            uint vec_size = tmp_stride[1] == 1 ? util::maxVectorCount(tmp) : 1;
            if (batches > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = tmp_stride[0] % vec_size ? 1 : vec_size;

            stream.enqueue(
                    name,
                    vec_size == 4 ? reduceSmall1D_<T, U, TransformOp, ReduceOp, PostProcessOp0, PostProcessOp1, 4> :
                    vec_size == 2 ? reduceSmall1D_<T, U, TransformOp, ReduceOp, PostProcessOp0, PostProcessOp1, 2> :
                                    reduceSmall1D_<T, U, TransformOp, ReduceOp, PostProcessOp0, PostProcessOp1, 1>,
                    {batches, ReduceConfig::BLOCK_SIZE, ReduceConfig::BLOCK_SIZE * sizeof(U)},
                    tmp, tmp_stride, elements, transform_op, reduce_op, init,
                    output0_ptr, post_process0, output1_ptr, post_process1);

        } else if ((!REDUCE_BATCH || is_contiguous[0]) && is_contiguous[1] && is_contiguous[2]) {
            // In this config, the input can be interpreted as a 1D array. If the innermost dimension is contiguous,
            // i.e. if all elements to reduce are contiguous, we can vectorize loads for the first kernel.
            // Here we use 1D blocks to go through each batch (if REDUCE_BATCH=true, there's only one batch).
            // Each block reduces at least BLOCK_WORK_SIZE elements. Max to MAX_GRID_SIZE blocks per batch.
            const uint blocks_x = noa::math::min(noa::math::divideUp(elements, ReduceConfig::BLOCK_WORK_SIZE),
                                                 ReduceConfig::MAX_GRID_SIZE);
            const dim3 blocks(blocks_x, batches);

            // Try to vectorize the loads for the input.
            uint vec_size = stride[3] == 1 ? util::maxVectorCount(input) : 1;
            if (batches) // make sure the beginning of each batch preserves the alignment
                vec_size = stride[0] % vec_size ? 1 : vec_size;

            // In the output (i.e. the input of the second kernel), preserve the alignment between batches.
            const uint pitch = noa::math::nextMultipleOf(blocks.x, 4u); // at most MAX_GRID_SIZE
            memory::PtrDevice<U> tmp{pitch * blocks.y, stream};

            // reduceLarge1D_: (batch * elements) -> (blocks.x * blocks.y) elements.
            // reduceSmall1D_: (blocks.x * blocks.y) -> (blocks.y) elements.
            stream.enqueue(name,
                           vec_size == 4 ? reduceLarge1D_<T, U, TransformOp, ReduceOp, 4> :
                           vec_size == 2 ? reduceLarge1D_<T, U, TransformOp, ReduceOp, 2> :
                                           reduceLarge1D_<T, U, TransformOp, ReduceOp, 1>,
                           {blocks, ReduceConfig::BLOCK_SIZE, ReduceConfig::BLOCK_SIZE * sizeof(U)},
                           input, uint2_t{stride[0], stride[3]}, elements, transform_op, reduce_op, init, tmp.get());

            // Here the input is already transformed, so copy.
            stream.enqueue(name, reduceSmall1D_<U, U, noa::math::copy_t, ReduceOp, PostProcessOp0, PostProcessOp1, 4>,
                           {batches, ReduceConfig::BLOCK_SIZE, ReduceConfig::BLOCK_SIZE * sizeof(U)},
                           tmp.get(), uint2_t{pitch, 1}, blocks.x, noa::math::copy_t{}, reduce_op, init,
                           output0_ptr, post_process0, output1_ptr, post_process1);

        } else {
            // In this config, the input cannot be easily interpreted as a 1D array.
            // As such, the 3 outermost dimensions are batched in a set of rows. Each block reduces at least one row.
            // Since the reduceLarge4D_ kernel will decompose the "row index" back to a (W,Z,Y) index, the 3 outermost
            // dimensions can be strided. If the innermost dimension is contiguous, blocks can use vectorize loads
            // to read their row(s).

            // If rows are large, switch to more threads per row.
            const uint block_dim_x = shape[3] > 512 ? 256 : 64;
            const dim3 threads(block_dim_x, ReduceConfig::BLOCK_SIZE / block_dim_x);
            const uint rows = shape[2] * shape[1] * (REDUCE_BATCH ? shape[0] : 1);
            const dim3 blocks(noa::math::min(noa::math::divideUp(rows, threads.y), ReduceConfig::MAX_GRID_SIZE),
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
            memory::PtrDevice<U> tmp(pitch * blocks.y, stream);

            if (threads.x == 256) {
                stream.enqueue(name,
                               vec_size == 4 ? reduceLarge4D_<T, U, TransformOp, ReduceOp, 256, 4> :
                               vec_size == 2 ? reduceLarge4D_<T, U, TransformOp, ReduceOp, 256, 2> :
                                               reduceLarge4D_<T, U, TransformOp, ReduceOp, 256, 1>,
                               {blocks, ReduceConfig::BLOCK_SIZE, ReduceConfig::BLOCK_SIZE * sizeof(U)},
                               input, stride, shape, rows, transform_op, reduce_op, init, tmp.get());
            } else {
                stream.enqueue(name,
                               vec_size == 4 ? reduceLarge4D_<T, U, TransformOp, ReduceOp, 64, 4> :
                               vec_size == 2 ? reduceLarge4D_<T, U, TransformOp, ReduceOp, 64, 2> :
                                               reduceLarge4D_<T, U, TransformOp, ReduceOp, 64, 1>,
                               {blocks, ReduceConfig::BLOCK_SIZE, ReduceConfig::BLOCK_SIZE * sizeof(U)},
                               input, stride, shape, rows, transform_op, reduce_op, init, tmp.get());
            }
            // Here the input is already transformed, so copy.
            stream.enqueue(name, reduceSmall1D_<U, U, noa::math::copy_t, ReduceOp, PostProcessOp0, PostProcessOp1, 4>,
                           {batches, ReduceConfig::BLOCK_SIZE, ReduceConfig::BLOCK_SIZE * sizeof(U)},
                           tmp.get(), uint2_t{pitch, 1}, blocks.x, noa::math::copy_t{}, reduce_op, init,
                           output0_ptr, post_process0, output1_ptr, post_process1);
        }

        // A temporary may have been allocated for the device to store the results.
        // In this case, copy back to the original output location.
        if (!buffer0.empty()) {
            if (output0_was_copied)
                memory::copy(output0_ptr, output0, batches, stream);
            if (output1_was_copied)
                memory::copy(output1_ptr, output1, batches, stream);
        }
    }

    /// Returns the variance of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Output variance.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       \p input and \p output should stay valid until completion.
    template<int DDOF, typename T, typename U>
    void reduceVar(const char* name,
                   const T* input, uint4_t stride, uint4_t shape,
                   U* output, Stream& stream) {
        // Get the mean:
        T h_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements() - DDOF);
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true, T, T>(name, input, stride, shape,
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &h_mean, sum_to_mean_op, nullptr, noa::math::copy_t{}, stream);

        // Get the variance:
        stream.synchronize();
        auto transform_op = [h_mean]__device__(T value) -> U {
                if constexpr (noa::traits::is_complex_v<T>) {
                    const U distance = noa::math::abs(value - h_mean);
                    return distance * distance;
                } else {
                    const U distance = value - h_mean;
                    return distance * distance;
                }
                return U(0); // unreachable
        };
        auto dist2_to_var = [inv_count]__device__(U v) -> U { return v * inv_count; };
        util::reduce<true, T, U>(name, input, stride, shape,
                                 transform_op, noa::math::plus_t{}, U(0),
                                 output, dist2_to_var, nullptr, noa::math::copy_t{}, stream);
    }

    /// Returns the standard-deviation of the input array.
    /// \tparam DDOF            Delta Degree Of Freedom used to calculate the variance. Should be 0 or 1.
    ///                         In standard statistical practice, DDOF=1 provides an unbiased estimator of the variance
    ///                         of a hypothetical infinite population. DDOF=0 provides a maximum likelihood estimate
    ///                         of the variance for normally distributed variables.
    /// \tparam T               float, double, cfloat_t, cdouble_t.
    /// \tparam U               If \p T is complex, should be the corresponding real type. Otherwise, same as \p T.
    /// \param[in] input        On the \b device. Input array to reduce.
    /// \param stride           Rightmost strides, in elements of \p input.
    /// \param shape            Rightmost shape of \p input.
    /// \param[out] output      On the \b host or \b device. Output stddev.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    ///       \p input and \p output should stay valid until completion.
    template<int DDOF, typename T, typename U>
    void reduceStddev(const char* name,
                      const T* input, uint4_t stride, uint4_t shape,
                      U* output, Stream& stream) {
        // Get the mean:
        T h_mean;
        const U inv_count = U(1) / static_cast<U>(shape.elements() - DDOF);
        auto sum_to_mean_op = [inv_count]__device__(T v) -> T { return v * inv_count; };
        util::reduce<true, T, T>(name, input, stride, shape,
                                 noa::math::copy_t{}, noa::math::plus_t{}, T(0),
                                 &h_mean, sum_to_mean_op, nullptr, noa::math::copy_t{}, stream);

        // Get the variance:
        stream.synchronize();
        auto transform_op = [h_mean]__device__(T value) -> U {
                if constexpr (noa::traits::is_complex_v<T>) {
                    const U distance = noa::math::abs(value - h_mean);
                    return distance * distance;
                } else {
                    const U distance = value - h_mean;
                    return distance * distance;
                }
                return U(0); // unreachable
        };
        auto dist2_to_std = [inv_count]__device__(U v) -> U { return noa::math::sqrt(v * inv_count); };
        util::reduce<true, T, U>(name, input, stride, shape,
                                 transform_op, noa::math::plus_t{}, U(0),
                                 output, dist2_to_std, nullptr, noa::math::copy_t{}, stream);
    }
}
