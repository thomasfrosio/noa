#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/util/Traits.h"
#include "noa/gpu/cuda/util/Block.cuh"
#include "noa/gpu/cuda/util/Pointers.h"

// -- Unary -- //
namespace noa::cuda::util::ewise::details {
    struct UnaryConfig {
        static constexpr uint ELEMENTS_PER_THREAD = 4;
        static constexpr uint BLOCK_SIZE = 128;
        static constexpr uint BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

        // Still the same threads per block and elements per thread, but using a 2D block.
        // The goal is waste as fewer threads as possible, assuming 2D/3D/4D arrays have a
        // similar number of elements in their two innermost dimensions.
        static constexpr uint ELEMENTS_PER_THREAD_2D = ELEMENTS_PER_THREAD / 2;
        static constexpr dim3 BLOCK_SIZE_2D{32, BLOCK_SIZE / 32, 1};
        static constexpr dim3 BLOCK_WORK_SIZE_2D{BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D,
                                                 BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D, 1};
    };

    template<typename T, typename U, typename UnaryOp, int VEC_SIZE, bool RESTRICT>
    __global__ __launch_bounds__(UnaryConfig::BLOCK_SIZE)
    void unary1D_(accessor_t<RESTRICT, const T*> input, uint2_t input_strides,
                  accessor_t<RESTRICT, U*> output, uint2_t output_strides,
                  uint elements, UnaryOp unary_op) {
        constexpr uint BLOCK_SIZE = UnaryConfig::BLOCK_SIZE;
        constexpr uint BLOCK_WORK_SIZE = UnaryConfig::BLOCK_WORK_SIZE;
        constexpr uint EPT = UnaryConfig::ELEMENTS_PER_THREAD;

        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, U*>::ptr_type;
        const uint base = BLOCK_WORK_SIZE * blockIdx.x;
        iptr_t input_ = input.get();
        optr_t output_ = output.get();

        input_ += blockIdx.y * input_strides[0];
        output_ += blockIdx.y * output_strides[0];

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (int i = 0; i < EPT; ++i) {
                const uint gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output_[gid * output_strides[1]] = static_cast<U>(unary_op(input_[gid * input_strides[1]]));
            }
        } else { // assume contiguous
            input_ += base;
            output_ += base;
            const uint remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (int i = 0; i < EPT; ++i) {
                    const uint gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        output_[gid] = static_cast<U>(unary_op(input_[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                T args[EPT];
                U results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(input_, args, threadIdx.x);
                #pragma unroll
                for (uint i = 0; i < EPT; ++i)
                    results[i] = static_cast<U>(unary_op(args[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, output_, threadIdx.x);
            }
        }
    }

    template<typename T, typename U, typename UnaryOp, bool RESTRICT>
    __global__ __launch_bounds__(UnaryConfig::BLOCK_SIZE)
    void unary4D_(accessor_t<RESTRICT, const T*> input, uint4_t input_strides,
                  accessor_t<RESTRICT, U*> output, uint4_t output_strides,
                  uint2_t shape, UnaryOp unary_op, uint blocks_x) {
        using iptr_t = typename accessor_t<RESTRICT, const T*>::ptr_type;
        using optr_t = typename accessor_t<RESTRICT, U*>::ptr_type;
        iptr_t input_ = input.get();
        optr_t output_ = output.get();

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const int4_t gid{blockIdx.z,
                         blockIdx.y,
                         UnaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                         UnaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        input_ += indexing::at(gid[0], gid[1], input_strides);
        output_ += indexing::at(gid[0], gid[1], output_strides);

        #pragma unroll
        for (int k = 0; k < UnaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (int l = 0; l < UnaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint ik = gid[2] + UnaryConfig::BLOCK_SIZE_2D.y * k;
                const uint il = gid[3] + UnaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output_[ik * output_strides[2] + il * output_strides[3]] =
                            static_cast<U>(unary_op(input_[ik * input_strides[2] + il * input_strides[3]]));
            }
        }
    }
}

namespace noa::cuda::util::ewise {
    // Apply an unary operator, element-wise.
    // RESTRICT:        Whether the pointers can be accessed using the __restrict__ attribute
    // name:            Name of the function. Used for logging if kernel launch fails.
    // input:           On the device. Input array to transform.
    // input_strides:   Strides, of input.
    // output:          On the device. Transformed array.
    // output_strides:  Strides, of output.
    // elements:        Shape of input and output.
    // swap_layout:     Swap the memory layout to optimize output writes.
    //                  If false, assume rightmost order is the fastest order.
    // stream:          Stream on which to enqueue this function.
    // unary_op:        Unary operator, such as, op(T) -> U.
    //                  The output is explicitly cast to U.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure input and output stay valid until completion.
    template<bool RESTRICT = false, typename T, typename U, typename UnaryOp>
    void unary(const char* name,
               const T* input, size4_t input_strides,
               U* output, size4_t output_strides,
               size4_t shape, bool swap_layout, Stream& stream, UnaryOp unary_op) {
        using namespace details;
        accessor_t<RESTRICT, const T*> input_accessor(input);
        accessor_t<RESTRICT, U*> output_accessor(output);

        if (swap_layout) {
            const size4_t order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            input_strides = indexing::reorder(input_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(input_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const uint4_t uint_shape(shape);
            const uint elements = is_contiguous[0] ? uint_shape.elements() : uint3_t{uint_shape.get() + 1}.elements();
            const dim3 blocks(noa::math::divideUp(elements, UnaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            uint vec_size = is_contiguous[3] ? std::min(maxVectorCount(input), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = input_strides[0] % vec_size || output_strides[0] % vec_size ? 1 : vec_size;

            const uint2_t uint_input_strides{input_strides[0], input_strides[3]};
            const uint2_t uint_output_strides{output_strides[0], output_strides[3]};
            const LaunchConfig config{blocks, UnaryConfig::BLOCK_SIZE};
            if (vec_size == 4) {
                return stream.enqueue(name, unary1D_<T, U, UnaryOp, 4, RESTRICT>, config,
                                      input_accessor, uint_input_strides,
                                      output_accessor, uint_output_strides,
                                      elements, unary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, unary1D_<T, U, UnaryOp, 2, RESTRICT>, config,
                                      input_accessor, uint_input_strides,
                                      output_accessor, uint_output_strides,
                                      elements, unary_op);
            } else {
                return stream.enqueue(name, unary1D_<T, U, UnaryOp, 1, RESTRICT>, config,
                                      input_accessor, uint_input_strides,
                                      output_accessor, uint_output_strides,
                                      elements, unary_op);
            }
        } else { // multi-dimensional, non-contiguous array
            const uint2_t i_shape(shape.get(2));
            const uint blocks_x = noa::math::divideUp(i_shape[1], UnaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint blocks_y = noa::math::divideUp(i_shape[0], UnaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, UnaryConfig::BLOCK_SIZE_2D};
            stream.enqueue(name, unary4D_<T, U, UnaryOp, RESTRICT>, config,
                           input_accessor, uint4_t(input_strides),
                           output_accessor, uint4_t(output_strides),
                           i_shape, unary_op, blocks_x);
        }
    }
}
