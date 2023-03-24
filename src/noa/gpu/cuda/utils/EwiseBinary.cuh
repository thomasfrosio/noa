#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"

namespace noa::cuda::utils::details {
    template<typename Lhs, typename Rhs, typename Output, typename Index, typename BinaryOp, typename Config,
             u32 VECTOR_SIZE, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_binary_1d(Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait> lhs_batched,
                         Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait> rhs_batched,
                         Accessor<Output, 2, Index, PointerTrait, StridesTrait> output_batched,
                         Index elements, BinaryOp binary_op) {

        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        const auto lhs = lhs_batched[blockIdx.y];
        const auto rhs = rhs_batched[blockIdx.y];
        const auto output = output_batched[blockIdx.y];
        const Index block_offset = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VECTOR_SIZE == 1) {
            #pragma unroll
            for (Index i = 0; i < EPT; ++i) {
                const Index gid = block_offset + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output[gid] = static_cast<Output>(binary_op(lhs[gid], rhs[gid]));
            }
        } else { // assume contiguous
            auto lhs_ptr = lhs.get() + block_offset;
            auto rhs_ptr = rhs.get() + block_offset;
            auto output_ptr = output.get() + block_offset;

            const Index remaining = elements - block_offset;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (Index i = 0; i < EPT; ++i) {
                    const Index gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        output_ptr[gid] = static_cast<Output>(binary_op(lhs_ptr[gid], rhs_ptr[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                Lhs lhs_values[EPT];
                Rhs rhs_values[EPT];
                Output output_values[EPT];
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(lhs_ptr, lhs_values, threadIdx.x);
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(rhs_ptr, rhs_values, threadIdx.x);
                #pragma unroll
                for (Index i = 0; i < EPT; ++i)
                    output_values[i] = static_cast<Output>(binary_op(lhs_values[i], rhs_values[i]));
                block_store<BLOCK_SIZE, EPT, VECTOR_SIZE>(output_values, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename Lhs, typename Rhs, typename Output, typename Index, typename BinaryOp,
             typename Config, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_binary_4d(Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait> lhs_batched,
                         Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait> rhs_batched,
                         Accessor<Output, 4, Index, PointerTrait, StridesTrait> output_batched,
                         Shape2<Index> shape, BinaryOp binary_op, Index blocks_x) {

        const auto index = noa::indexing::offset2index(static_cast<Index>(blockIdx.x), blocks_x);
        const auto gid = Vec4<Index>{
                blockIdx.z, blockIdx.y,
                Config::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                Config::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs = lhs_batched[gid[0]][gid[1]];
        const auto rhs = rhs_batched[gid[0]][gid[1]];
        const auto output = output_batched[gid[0]][gid[1]];

        #pragma unroll
        for (u32 k = 0; k < Config::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (u32 l = 0; l < Config::ELEMENTS_PER_THREAD_2D; ++l) {
                const auto ik = gid[2] + Config::BLOCK_SIZE_2D.y * k;
                const auto il = gid[3] + Config::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output(ik, il) = static_cast<Output>(binary_op(lhs(ik, il), rhs(ik, il)));
            }
        }
    }
}

namespace noa::cuda::utils {
    // Apply a unary operator op(Lhs, Rhs) -> Output, element-wise.
    // This function is asynchronous relative to the host and may return before completion.
    // The caller must make sure the input and output arrays stay valid until completion.
    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Rhs, typename Output,
             typename Index, typename BinaryOp>
    void ewise_binary(const char* name,
                      const Lhs* lhs, Strides4<Index> lhs_strides,
                      const Rhs* rhs, Strides4<Index> rhs_strides,
                      Output* output, Strides4<Index> output_strides,
                      Shape4<Index> shape, Stream& stream,
                      BinaryOp binary_op) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        shape = noa::indexing::effective_shape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::any(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = noa::indexing::reorder(shape, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            rhs_strides = noa::indexing::reorder(rhs_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
        }

        const auto is_contiguous =
                noa::indexing::is_contiguous(lhs_strides, shape) &&
                noa::indexing::is_contiguous(rhs_strides, shape) &&
                noa::indexing::is_contiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const auto batch_size = is_contiguous[0] ? 1 : shape[0];
            const auto elements = is_contiguous[0] ? shape.elements() : shape.pop_front().elements();
            const auto lhs_strides_2d = lhs_strides.filter(0, 3);
            const auto rhs_strides_2d = rhs_strides.filter(0, 3);
            const auto output_strides_2d = output_strides.filter(0, 3);

            u32 vector_size = is_contiguous[3] ?
                    std::min({max_vector_count(lhs), max_vector_count(rhs), max_vector_count(output), i64{8}}) : 1;
            if (batch_size > 1) {
                // Make sure the beginning of each batch preserves the alignment.
                // If not, try with a lower vector size
                for (; vector_size >= 2; vector_size /= 2) {
                    if (!(lhs_strides_2d[0] % vector_size) &&
                        !(rhs_strides_2d[0] % vector_size) &&
                        !(output_strides_2d[0] % vector_size))
                        break;
                }
            }

            const Index block_work_size = Config::BLOCK_SIZE * std::max(vector_size, Config::ELEMENTS_PER_THREAD);
            const dim3 blocks(noa::math::divide_up(elements, block_work_size), batch_size);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE};

            if (vector_size == 1) {
                using lhs_accessor_t = Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait>;
                using rhs_accessor_t = Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait>;
                using output_accessor_t = Accessor<Output, 2, Index, PointerTrait, StridesTrait>;
                const auto lhs_accessor = lhs_accessor_t(lhs, lhs_strides_2d);
                const auto rhs_accessor = rhs_accessor_t(rhs, rhs_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                return stream.enqueue(
                        name,
                        details::ewise_binary_1d<Lhs, Rhs, Output, Index, BinaryOp, Config, 1, PointerTrait, StridesTrait>,
                        config, lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
            } else {
                using lhs_accessor_t = AccessorContiguous<const Lhs, 2, Index, PointerTrait>;
                using rhs_accessor_t = AccessorContiguous<const Rhs, 2, Index, PointerTrait>;
                using output_accessor_t = AccessorContiguous<Output, 2, Index, PointerTrait>;
                const auto lhs_accessor = lhs_accessor_t(lhs, lhs_strides_2d);
                const auto rhs_accessor = rhs_accessor_t(rhs, rhs_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                if (vector_size == 2) {
                    return stream.enqueue(
                            name,
                            details::ewise_binary_1d<Lhs, Rhs, Output, Index, BinaryOp, Config, 2, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
                } else if (vector_size == 4) {
                    return stream.enqueue(
                            name,
                            details::ewise_binary_1d<Lhs, Rhs, Output, Index, BinaryOp, Config, 4, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
                } else {
                    return stream.enqueue(
                            name,
                            details::ewise_binary_1d<Lhs, Rhs, Output, Index, BinaryOp, Config, 8, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, lhs_accessor, rhs_accessor, output_accessor, elements, binary_op);
                }
            }
        } else {
            const Index blocks_x = noa::math::divide_up(shape[3], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.x));
            const Index blocks_y = noa::math::divide_up(shape[2], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.y));
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE_2D};

            using lhs_accessor_t = Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait>;
            using rhs_accessor_t = Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait>;
            using output_accessor_t = Accessor<Output, 4, Index, PointerTrait, StridesTrait>;
            const auto lhs_accessor = lhs_accessor_t(lhs, lhs_strides);
            const auto rhs_accessor = rhs_accessor_t(rhs, rhs_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            stream.enqueue(name,
                    details::ewise_binary_4d<Lhs, Rhs, Output, Index, BinaryOp, Config, PointerTrait, StridesTrait>,
                    config, lhs_accessor, rhs_accessor, output_accessor, shape.filter(2, 3), binary_op, blocks_x);
        }
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Rhs, typename Output,
             typename Index, typename BinaryOp>
    void ewise_binary(const char* name,
                      Lhs lhs,
                      const Rhs* rhs, const Strides4<Index>& rhs_strides,
                      Output* output, const Strides4<Index>& output_strides,
                      const Shape4<Index>& shape, Stream& stream,
                      const BinaryOp& binary_op) {
        ewise_unary<PointerTrait, StridesTrait>(
                name, rhs, rhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Rhs rhs_value) { return binary_op(lhs, rhs_value); });
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Rhs, typename Output,
             typename Index, typename BinaryOp>
    void ewise_binary(const char* name,
                      const Lhs* lhs, const Strides4<Index>& lhs_strides,
                      Rhs rhs,
                      Output* output, const Strides4<Index>& output_strides,
                      const Shape4<Index>& shape, Stream& stream,
                      const BinaryOp& binary_op) {
        ewise_unary<PointerTrait, StridesTrait>(
                name, lhs, lhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Lhs lhs_value) { return binary_op(lhs_value, rhs); });
    }
}
