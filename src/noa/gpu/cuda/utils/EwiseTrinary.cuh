#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"
#include "noa/gpu/cuda/utils/EwiseUnary.cuh"
#include "noa/gpu/cuda/utils/EwiseBinary.cuh"

namespace noa::cuda::utils::details {
    template<typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename TrinaryOp, typename Config,
             u32 VECTOR_SIZE, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_trinary_1d(
            Accessor<const Lhs, 2, Index, PointerTrait, StridesTrait> lhs_batched,
            Accessor<const Mhs, 2, Index, PointerTrait, StridesTrait> mhs_batched,
            Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait> rhs_batched,
            Accessor<Output, 2, Index, PointerTrait, StridesTrait> output_batched,
            Index elements, TrinaryOp trinary_op
    ) {
        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        const auto lhs = lhs_batched[blockIdx.y];
        const auto mhs = mhs_batched[blockIdx.y];
        const auto rhs = rhs_batched[blockIdx.y];
        const auto output = output_batched[blockIdx.y];
        const Index block_offset = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VECTOR_SIZE == 1) {
            #pragma unroll
            for (Index i = 0; i < EPT; ++i) {
                const Index gid = block_offset + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output[gid] = static_cast<Output>(trinary_op(lhs[gid], mhs[gid], rhs[gid]));
            }
        } else { // assume contiguous
            auto lhs_ptr = lhs.get() + block_offset;
            auto mhs_ptr = mhs.get() + block_offset;
            auto rhs_ptr = rhs.get() + block_offset;
            auto output_ptr = output.get() + block_offset;

            const Index remaining = elements - block_offset;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (Index i = 0; i < EPT; ++i) {
                    const Index gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        output_ptr[gid] = static_cast<Output>(trinary_op(lhs_ptr[gid], mhs_ptr[gid], rhs_ptr[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                Lhs lhs_values[EPT];
                Mhs mhs_values[EPT];
                Rhs rhs_values[EPT];
                Output output_values[EPT];
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(lhs_ptr, lhs_values, threadIdx.x);
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(mhs_ptr, mhs_values, threadIdx.x);
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(rhs_ptr, rhs_values, threadIdx.x);
                #pragma unroll
                for (Index i = 0; i < EPT; ++i)
                    output_values[i] = static_cast<Output>(trinary_op(lhs_values[i], mhs_values[i], rhs_values[i]));
                block_store<BLOCK_SIZE, EPT, VECTOR_SIZE>(output_values, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename Lhs, typename Mhs, typename Rhs, typename Output, typename Index, typename TrinaryOp,
             typename Config, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_trinary_4d(
            Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait> lhs_batched,
            Accessor<const Mhs, 4, Index, PointerTrait, StridesTrait> mhs_batched,
            Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait> rhs_batched,
            Accessor<Output, 4, Index, PointerTrait, StridesTrait> output_batched,
            Shape2<Index> shape, TrinaryOp trinary_op, Index blocks_x
    ) {
        const auto index = noa::indexing::offset2index(static_cast<Index>(blockIdx.x), blocks_x);
        const auto gid = Vec4<Index>{
                blockIdx.z, blockIdx.y,
                Config::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                Config::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto lhs = lhs_batched[gid[0]][gid[1]];
        const auto mhs = mhs_batched[gid[0]][gid[1]];
        const auto rhs = rhs_batched[gid[0]][gid[1]];
        const auto output = output_batched[gid[0]][gid[1]];

        #pragma unroll
        for (u32 k = 0; k < Config::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (u32 l = 0; l < Config::ELEMENTS_PER_THREAD_2D; ++l) {
                const auto ik = gid[2] + Config::BLOCK_SIZE_2D.y * k;
                const auto il = gid[3] + Config::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output(ik, il) = static_cast<Output>(trinary_op(lhs(ik, il), mhs(ik, il), rhs(ik, il)));
            }
        }
    }
}

namespace noa::cuda::utils {
    // Apply a unary operator op(Lhs, Mhs, Rhs) -> Output, element-wise.
    // This function is asynchronous relative to the host and may return before completion.
    // The caller must make sure the input and output arrays stay valid until completion.
    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename TrinaryOp>
    void ewise_trinary(
            const Lhs* lhs, Strides4<Index> lhs_strides,
            const Mhs* mhs, Strides4<Index> mhs_strides,
            const Rhs* rhs, Strides4<Index> rhs_strides,
            Output* output, Strides4<Index> output_strides,
            Shape4<Index> shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(lhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(mhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(rhs, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        shape = noa::indexing::effective_shape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::any(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = noa::indexing::reorder(shape, order);
            lhs_strides = noa::indexing::reorder(lhs_strides, order);
            mhs_strides = noa::indexing::reorder(mhs_strides, order);
            rhs_strides = noa::indexing::reorder(rhs_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
        }

        const auto is_contiguous =
                noa::indexing::is_contiguous(lhs_strides, shape) &&
                noa::indexing::is_contiguous(mhs_strides, shape) &&
                noa::indexing::is_contiguous(rhs_strides, shape) &&
                noa::indexing::is_contiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const auto batch_size = is_contiguous[0] ? 1 : shape[0];
            const auto elements = is_contiguous[0] ? shape.elements() : shape.pop_front().elements();
            const auto lhs_strides_2d = lhs_strides.filter(0, 3);
            const auto mhs_strides_2d = mhs_strides.filter(0, 3);
            const auto rhs_strides_2d = rhs_strides.filter(0, 3);
            const auto output_strides_2d = output_strides.filter(0, 3);

            u32 vector_size = is_contiguous[3] ?
                           std::min({max_vector_count(lhs), max_vector_count(mhs),
                                     max_vector_count(rhs), max_vector_count(output), i64{8}}) : 1;
            if (batch_size > 1) {
                // Make sure the beginning of each batch preserves the alignment.
                // If not, try with a lower vector size
                for (; vector_size >= 2; vector_size /= 2) {
                    if (!(lhs_strides_2d[0] % vector_size) &&
                        !(mhs_strides_2d[0] % vector_size) &&
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
                using mhs_accessor_t = Accessor<const Mhs, 2, Index, PointerTrait, StridesTrait>;
                using rhs_accessor_t = Accessor<const Rhs, 2, Index, PointerTrait, StridesTrait>;
                using output_accessor_t = Accessor<Output, 2, Index, PointerTrait, StridesTrait>;
                const auto lhs_accessor = lhs_accessor_t(lhs, lhs_strides_2d);
                const auto mhs_accessor = mhs_accessor_t(mhs, mhs_strides_2d);
                const auto rhs_accessor = rhs_accessor_t(rhs, rhs_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                return stream.enqueue(
                        details::ewise_trinary_1d<Lhs, Mhs, Rhs, Output, Index, TrinaryOp, Config, 1, PointerTrait, StridesTrait>,
                        config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
            } else {
                using lhs_accessor_t = AccessorContiguous<const Lhs, 2, Index, PointerTrait>;
                using mhs_accessor_t = AccessorContiguous<const Mhs, 2, Index, PointerTrait>;
                using rhs_accessor_t = AccessorContiguous<const Rhs, 2, Index, PointerTrait>;
                using output_accessor_t = AccessorContiguous<Output, 2, Index, PointerTrait>;
                const auto lhs_accessor = lhs_accessor_t(lhs, lhs_strides_2d);
                const auto mhs_accessor = mhs_accessor_t(mhs, mhs_strides_2d);
                const auto rhs_accessor = rhs_accessor_t(rhs, rhs_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                if (vector_size == 2) {
                    return stream.enqueue(
                            details::ewise_trinary_1d<Lhs, Mhs, Rhs, Output, Index, TrinaryOp, Config, 2, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
                } else if (vector_size == 4) {
                    return stream.enqueue(
                            details::ewise_trinary_1d<Lhs, Mhs, Rhs, Output, Index, TrinaryOp, Config, 4, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
                } else {
                    return stream.enqueue(
                            details::ewise_trinary_1d<Lhs, Mhs, Rhs, Output, Index, TrinaryOp, Config, 8, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, elements, trinary_op);
                }
            }
        } else {
            const Index blocks_x = noa::math::divide_up(shape[3], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.x));
            const Index blocks_y = noa::math::divide_up(shape[2], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.y));
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE_2D};

            using lhs_accessor_t = Accessor<const Lhs, 4, Index, PointerTrait, StridesTrait>;
            using mhs_accessor_t = Accessor<const Mhs, 4, Index, PointerTrait, StridesTrait>;
            using rhs_accessor_t = Accessor<const Rhs, 4, Index, PointerTrait, StridesTrait>;
            using output_accessor_t = Accessor<Output, 4, Index, PointerTrait, StridesTrait>;
            const auto lhs_accessor = lhs_accessor_t(lhs, lhs_strides);
            const auto mhs_accessor = mhs_accessor_t(mhs, mhs_strides);
            const auto rhs_accessor = rhs_accessor_t(rhs, rhs_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            stream.enqueue(
                    details::ewise_trinary_4d<Lhs, Mhs, Rhs, Output, Index, TrinaryOp, Config, PointerTrait, StridesTrait>,
                    config, lhs_accessor, mhs_accessor, rhs_accessor, output_accessor, shape.filter(2, 3), trinary_op, blocks_x);
        }
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename TrinaryOp>
    void ewise_trinary(
            const Lhs* lhs, const Strides4<Index>& lhs_strides,
            Mhs mhs,
            Rhs rhs,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        ewise_unary<PointerTrait, StridesTrait>(
                lhs, lhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Lhs lhs_value) { return trinary_op(lhs_value, mhs, rhs); });
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
            StridesTraits StridesTrait = StridesTraits::STRIDED,
            typename Config = EwiseStaticConfigDefault,
            typename Lhs, typename Mhs, typename Rhs, typename Output,
            typename Index, typename TrinaryOp>
    void ewise_trinary(
            Lhs lhs,
            const Mhs* mhs, const Strides4<Index>& mhs_strides,
            Rhs rhs,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        ewise_unary<PointerTrait, StridesTrait>(
                mhs, mhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Mhs mhs_value) { return trinary_op(lhs, mhs_value, rhs); });
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
            StridesTraits StridesTrait = StridesTraits::STRIDED,
            typename Config = EwiseStaticConfigDefault,
            typename Lhs, typename Mhs, typename Rhs, typename Output,
            typename Index, typename TrinaryOp>
    void ewise_trinary(
            Lhs lhs,
            Mhs mhs,
            const Rhs* rhs, const Strides4<Index>& rhs_strides,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        ewise_unary<PointerTrait, StridesTrait>(
                rhs, rhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Rhs rhs_value) { return trinary_op(lhs, mhs, rhs_value); });
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename TrinaryOp>
    void ewise_trinary(
            const Lhs* lhs, const Strides4<Index>& lhs_strides,
            const Mhs* mhs, const Strides4<Index>& mhs_strides,
            Rhs rhs,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        ewise_binary<PointerTrait, StridesTrait>(
                lhs, lhs_strides, mhs, mhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Lhs lhs_value, Mhs mhs_value) { return trinary_op(lhs_value, mhs_value, rhs); });
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
            StridesTraits StridesTrait = StridesTraits::STRIDED,
            typename Config = EwiseStaticConfigDefault,
            typename Lhs, typename Mhs, typename Rhs, typename Output,
            typename Index, typename TrinaryOp>
    void ewise_trinary(
            const Lhs* lhs, const Strides4<Index>& lhs_strides,
            Mhs mhs,
            const Rhs* rhs, const Strides4<Index>& rhs_strides,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        ewise_binary<PointerTrait, StridesTrait>(
                lhs, lhs_strides, rhs, rhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Lhs lhs_value, Rhs rhs_value) { return trinary_op(lhs_value, mhs, rhs_value); });
    }

    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Lhs, typename Mhs, typename Rhs, typename Output,
             typename Index, typename TrinaryOp>
    void ewise_trinary(
            Lhs lhs,
            const Mhs* mhs, const Strides4<Index>& mhs_strides,
            const Rhs* rhs, const Strides4<Index>& rhs_strides,
            Output* output, const Strides4<Index>& output_strides,
            const Shape4<Index>& shape, Stream& stream,
            TrinaryOp trinary_op
    ) {
        ewise_binary<PointerTrait, StridesTrait>(
                mhs, mhs_strides, rhs, rhs_strides, output, output_strides, shape, stream,
                [=] NOA_DEVICE(Mhs mhs_value, Rhs rhs_value) { return trinary_op(lhs, mhs_value, rhs_value); });
    }
}

#define NOA_CUDA_EWISE_TRINARY_GENERATE_API                                                             \
namespace noa::cuda {                                                                                   \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,                                \
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,                                \
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,                                \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs, lhs_strides,                                                                       \
                mhs, mhs_strides,                                                                       \
                rhs, rhs_strides,                                                                       \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
                                                                                                        \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(Lhs lhs,                                                                         \
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,                                \
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,                                \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs,                                                                                    \
                mhs, mhs_strides,                                                                       \
                rhs, rhs_strides,                                                                       \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
                                                                                                        \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,                                \
                       Mhs mhs,                                                                         \
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,                                \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs, lhs_strides,                                                                       \
                mhs,                                                                                    \
                rhs, rhs_strides,                                                                       \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
                                                                                                        \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,                                \
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,                                \
                       Rhs rhs,                                                                         \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs, lhs_strides,                                                                       \
                mhs, mhs_strides,                                                                       \
                rhs,                                                                                    \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
                                                                                                        \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(const Lhs* lhs, const Strides4<i64>& lhs_strides,                                \
                       Mhs mhs,                                                                         \
                       Rhs rhs,                                                                         \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs, lhs_strides,                                                                       \
                mhs,                                                                                    \
                rhs,                                                                                    \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
                                                                                                        \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(Lhs lhs,                                                                         \
                       const Mhs* mhs, const Strides4<i64>& mhs_strides,                                \
                       Rhs rhs,                                                                         \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs,                                                                                    \
                mhs, mhs_strides,                                                                       \
                rhs,                                                                                    \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
                                                                                                        \
    template<typename Lhs, typename Mhs, typename Rhs, typename Out, typename TrinaryOp, typename>      \
    void ewise_trinary(Lhs lhs,                                                                         \
                       Mhs mhs,                                                                         \
                       const Rhs* rhs, const Strides4<i64>& rhs_strides,                                \
                       Out* output, const Strides4<i64>& output_strides,                                \
                       const Shape4<i64>& shape, TrinaryOp trinary_op, Stream& stream) {                \
        noa::cuda::utils::ewise_trinary(                                                                \
                lhs,                                                                                    \
                mhs,                                                                                    \
                rhs, rhs_strides,                                                                       \
                output, output_strides,                                                                 \
                shape, stream, trinary_op);                                                             \
    }                                                                                                   \
}

#define NOA_CUDA_EWISE_TRINARY_INSTANTIATE_API(Lhs, Mhs ,Rhs, Out, TrinaryOp)   \
namespace noa::cuda {                                                           \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        const Lhs*, const Strides4<i64>&,                                       \
        const Mhs*, const Strides4<i64>&,                                       \
        const Rhs*, const Strides4<i64>&,                                       \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        Lhs,                                                                    \
        const Mhs*, const Strides4<i64>&,                                       \
        const Rhs*, const Strides4<i64>&,                                       \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        const Lhs*, const Strides4<i64>&,                                       \
        Mhs,                                                                    \
        const Rhs*, const Strides4<i64>&,                                       \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        const Lhs*, const Strides4<i64>&,                                       \
        const Mhs*, const Strides4<i64>&,                                       \
        Rhs,                                                                    \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        const Lhs*, const Strides4<i64>&,                                       \
        Mhs,                                                                    \
        Rhs,                                                                    \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        Lhs,                                                                    \
        const Mhs*, const Strides4<i64>&,                                       \
        Rhs,                                                                    \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
    template void ewise_trinary<Lhs,Mhs,Rhs,Out,TrinaryOp,void>(                \
        Lhs,                                                                    \
        Mhs,                                                                    \
        const Rhs*, const Strides4<i64>&,                                       \
        Out*, const Strides4<i64>&,                                             \
        const Shape4<i64>&, TrinaryOp, Stream&);                                \
}
