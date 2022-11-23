#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/utils/Traits.h"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.h"

// -- Unary -- //
namespace noa::cuda::utils::ewise::details {
    struct UnaryConfig {
        static constexpr uint32_t ELEMENTS_PER_THREAD = 4;
        static constexpr uint32_t BLOCK_SIZE = 128;
        static constexpr uint32_t BLOCK_WORK_SIZE = BLOCK_SIZE * ELEMENTS_PER_THREAD;

        // Still the same threads per block and elements per thread, but using a 2D block.
        // The goal is waste as fewer threads as possible, assuming 2D/3D/4D arrays have a
        // similar number of elements in their two innermost dimensions.
        static constexpr uint32_t ELEMENTS_PER_THREAD_2D = ELEMENTS_PER_THREAD / 2;
        static constexpr dim3 BLOCK_SIZE_2D{32, BLOCK_SIZE / 32, 1};
        static constexpr dim3 BLOCK_WORK_SIZE_2D{BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D,
                                                 BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D, 1};
    };

    template<typename InVal, typename OutVal, typename UnaryOp, int VEC_SIZE, AccessorTraits TRAITS>
    __global__ __launch_bounds__(UnaryConfig::BLOCK_SIZE)
    void unary1D_(Accessor<const InVal, 2, uint32_t, TRAITS> input,
                  Accessor<OutVal, 2, uint32_t, TRAITS> output,
                  uint32_t elements, UnaryOp unary_op) {

        constexpr uint32_t BLOCK_SIZE = UnaryConfig::BLOCK_SIZE;
        constexpr uint32_t BLOCK_WORK_SIZE = UnaryConfig::BLOCK_WORK_SIZE;
        constexpr uint32_t EPT = UnaryConfig::ELEMENTS_PER_THREAD;

        const auto input_ = input[blockIdx.y];
        const auto output_ = output[blockIdx.y];
        const uint32_t base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VEC_SIZE == 1) {
            #pragma unroll
            for (uint32_t i = 0; i < EPT; ++i) {
                const uint32_t gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output_[gid] = static_cast<OutVal>(unary_op(input_[gid]));
            }
        } else { // assume contiguous
            NOA_ASSERT(input_.stride(0) == 1 && output_.stride(0) == 1);
            using iptr_t = typename decltype(input)::pointer_type;
            using optr_t = typename decltype(output)::pointer_type;
            iptr_t input_ptr = input_.get() + base;
            optr_t output_ptr = output_.get() + base;

            const uint32_t remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i) {
                    const uint32_t gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        output_ptr[gid] = static_cast<OutVal>(unary_op(input_ptr[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                InVal args[EPT];
                OutVal results[EPT];
                block::vectorizedLoad<BLOCK_SIZE, EPT, VEC_SIZE>(input_ptr, args, threadIdx.x);
                #pragma unroll
                for (uint32_t i = 0; i < EPT; ++i)
                    results[i] = static_cast<OutVal>(unary_op(args[i]));
                block::vectorizedStore<BLOCK_SIZE, EPT, VEC_SIZE>(results, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename InVal, typename OutVal, typename UnaryOp, AccessorTraits TRAITS>
    __global__ __launch_bounds__(UnaryConfig::BLOCK_SIZE)
    void unary4D_(Accessor<const InVal, 4, uint32_t, TRAITS> input,
                  Accessor<OutVal, 4, uint32_t, TRAITS> output,
                  uint2_t shape, UnaryOp unary_op, uint32_t blocks_x) {

        const uint2_t index = indexing::indexes(blockIdx.x, blocks_x);
        const uint4_t gid{blockIdx.z,
                          blockIdx.y,
                          UnaryConfig::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                          UnaryConfig::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto input_ = input[gid[0]][gid[1]];
        const auto output_ = output[gid[0]][gid[1]];

        #pragma unroll
        for (uint32_t k = 0; k < UnaryConfig::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (uint32_t l = 0; l < UnaryConfig::ELEMENTS_PER_THREAD_2D; ++l) {
                const uint32_t ik = gid[2] + UnaryConfig::BLOCK_SIZE_2D.y * k;
                const uint32_t il = gid[3] + UnaryConfig::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output_(ik, il) = static_cast<OutVal>(unary_op(input_(ik, il)));
            }
        }
    }
}

namespace noa::cuda::utils::ewise {
    // Apply a unary operator, element-wise.
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
    // unary_op:        Unary operator, such as, op(InVal) -> OutVal.
    //                  The output is explicitly cast to OutVal.
    // This function is asynchronous relative to the host and may return before completion.
    // One must make sure input and output stay valid until completion.
    template<bool RESTRICT = false, typename InVal, typename OutVal, typename UnaryOp>
    void unary(const char* name,
               const InVal* input, dim4_t input_strides,
               OutVal* output, dim4_t output_strides,
               dim4_t shape, bool swap_layout,
               Stream& stream, UnaryOp unary_op) {
        using namespace details;
        constexpr AccessorTraits TRAITS = RESTRICT ? AccessorTraits::RESTRICT : AccessorTraits::DEFAULT;
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        if (swap_layout) {
            const dim4_t order = indexing::order(output_strides, shape);
            shape = indexing::reorder(shape, order);
            output_strides = indexing::reorder(output_strides, order);
            input_strides = indexing::reorder(input_strides, order);
        }

        const bool4_t is_contiguous = indexing::isContiguous(input_strides, shape) &&
                                      indexing::isContiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const auto elements = safe_cast<uint32_t>(
                    is_contiguous[0] ? shape.elements() : dim3_t(shape.get(1)).elements());
            const dim3 blocks(noa::math::divideUp(elements, UnaryConfig::BLOCK_WORK_SIZE),
                              is_contiguous[0] ? 1 : shape[0]);

            const auto uint_input_strides = safe_cast<uint2_t>(dim2_t{input_strides[0], input_strides[3]});
            const auto uint_output_strides = safe_cast<uint2_t>(dim2_t{output_strides[0], output_strides[3]});
            const LaunchConfig config{blocks, UnaryConfig::BLOCK_SIZE};

            uint32_t vec_size = is_contiguous[3] ? std::min(maxVectorCount(input), maxVectorCount(output)) : 1;
            if (blocks.y > 1) // make sure the beginning of each batch preserves the alignment
                vec_size = uint_input_strides[0] % vec_size || uint_output_strides[0] % vec_size ? 1 : vec_size;

            const Accessor<const InVal, 2, uint32_t, TRAITS> input_accessor(input, uint_input_strides);
            const Accessor<OutVal, 2, uint32_t, TRAITS> output_accessor(output, uint_output_strides);

            if (vec_size == 4) {
                return stream.enqueue(name, unary1D_<InVal, OutVal, UnaryOp, 4, TRAITS>, config,
                                      input_accessor, output_accessor, elements, unary_op);
            } else if (vec_size == 2) {
                return stream.enqueue(name, unary1D_<InVal, OutVal, UnaryOp, 2, TRAITS>, config,
                                      input_accessor, output_accessor, elements, unary_op);
            } else {
                return stream.enqueue(name, unary1D_<InVal, OutVal, UnaryOp, 1, TRAITS>, config,
                                      input_accessor, output_accessor, elements, unary_op);
            }
        } else { // multi-dimensional, non-contiguous array
            const auto i_shape = safe_cast<uint2_t>(dim2_t(shape.get(2)));
            const uint32_t blocks_x = noa::math::divideUp(i_shape[1], UnaryConfig::BLOCK_WORK_SIZE_2D.x);
            const uint32_t blocks_y = noa::math::divideUp(i_shape[0], UnaryConfig::BLOCK_WORK_SIZE_2D.y);
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, UnaryConfig::BLOCK_SIZE_2D};

            const Accessor<const InVal, 4, uint32_t, TRAITS> input_accessor(input, safe_cast<uint4_t>(input_strides));
            const Accessor<OutVal, 4, uint32_t, TRAITS> output_accessor(output, safe_cast<uint4_t>(output_strides));

            stream.enqueue(name, unary4D_<InVal, OutVal, UnaryOp, TRAITS>, config,
                           input_accessor, output_accessor, i_shape, unary_op, blocks_x);
        }
    }
}
