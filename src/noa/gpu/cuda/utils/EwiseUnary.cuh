#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/utils/Block.cuh"
#include "noa/gpu/cuda/utils/Pointers.hpp"

// -- Unary -- //
namespace noa::cuda::utils {
    template<u32 ElementsPerThread, u32 BlockSize>
    struct EwiseStaticConfig {
        static_assert(!(ElementsPerThread % 2) && !(BlockSize % 32));
        static constexpr u32 ELEMENTS_PER_THREAD = ElementsPerThread;
        static constexpr u32 BLOCK_SIZE = BlockSize;

        // Still the same threads per block and elements per thread, but using a 2D block.
        // The goal is to waste as fewer threads as possible, assuming 2D/3D/4D arrays have a
        // similar number of elements in their two innermost dimensions. Also, here we assume
        // there's no vectorization, so we can compute the actual block work size.
        static constexpr u32 ELEMENTS_PER_THREAD_2D = ELEMENTS_PER_THREAD / 2;
        static constexpr dim3 BLOCK_SIZE_2D{32, BLOCK_SIZE / 32, 1};
        static constexpr dim3 BLOCK_WORK_SIZE_2D{BLOCK_SIZE_2D.x * ELEMENTS_PER_THREAD_2D,
                                                 BLOCK_SIZE_2D.y * ELEMENTS_PER_THREAD_2D, 1};
    };

    using EwiseStaticConfigDefault = EwiseStaticConfig<4, 128>;
}

namespace noa::cuda::utils::details {
    template<typename Input, typename Output, typename Index, typename UnaryOp, typename Config,
             u32 VECTOR_SIZE, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_unary_1d(Accessor<const Input, 2, Index, PointerTrait, StridesTrait> input_batched,
                        Accessor<Output, 2, Index, PointerTrait, StridesTrait> output_batched,
                        Index elements, UnaryOp unary_op) {

        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        const auto input = input_batched[blockIdx.y];
        const auto output = output_batched[blockIdx.y];
        const Index block_offset = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VECTOR_SIZE == 1) {
            #pragma unroll
            for (Index i = 0; i < EPT; ++i) {
                const Index gid = block_offset + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    output[gid] = static_cast<Output>(unary_op(input[gid]));
            }
        } else { // assume contiguous
            auto input_ptr = input.get() + block_offset;
            auto output_ptr = output.get() + block_offset;

            const Index remaining = elements - block_offset;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (Index i = 0; i < EPT; ++i) {
                    const Index gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        output_ptr[gid] = static_cast<Output>(unary_op(input_ptr[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                Input input_values[EPT];
                Output output_values[EPT];
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(input_ptr, input_values, threadIdx.x);
                #pragma unroll
                for (Index i = 0; i < EPT; ++i)
                    output_values[i] = static_cast<Output>(unary_op(input_values[i]));
                block_store<BLOCK_SIZE, EPT, VECTOR_SIZE>(output_values, output_ptr, threadIdx.x);
            }
        }
    }

    template<typename Value, typename Index, typename UnaryOp, typename Config,
            u32 VECTOR_SIZE, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_unary_1d(Accessor<Value, 2, Index, PointerTrait, StridesTrait> array_batched,
                        Index elements, UnaryOp unary_op) {

        constexpr Index BLOCK_SIZE = Config::BLOCK_SIZE;
        constexpr Index EPT = noa::math::max(Config::ELEMENTS_PER_THREAD, VECTOR_SIZE);
        constexpr Index BLOCK_WORK_SIZE = BLOCK_SIZE * EPT;

        const auto array = array_batched[blockIdx.y];
        const Index base = BLOCK_WORK_SIZE * blockIdx.x;

        if constexpr (VECTOR_SIZE == 1) {
            #pragma unroll
            for (Index i = 0; i < EPT; ++i) {
                const Index gid = base + BLOCK_SIZE * i + threadIdx.x;
                if (gid < elements)
                    array[gid] = static_cast<Value>(unary_op(array[gid]));
            }
        } else { // assume contiguous
            auto array_ptr = array.get() + base;

            const Index remaining = elements - base;
            if (remaining < BLOCK_WORK_SIZE) {
                #pragma unroll
                for (Index i = 0; i < EPT; ++i) {
                    const Index gid = BLOCK_SIZE * i + threadIdx.x;
                    if (gid < remaining)
                        array_ptr[gid] = static_cast<Value>(unary_op(array_ptr[gid]));
                }
            } else { // this block has BLOCK_WORK_SIZE elements to handle, so we can use vectorized memory accesses
                Value values[EPT];
                // TODO If unary_op doesn't use its input (write only), the optimizer should still
                //      be able to optimize away the block_load since it only writes to the array
                //      of values, which end up being not read.
                block_load<BLOCK_SIZE, EPT, VECTOR_SIZE>(array_ptr, values, threadIdx.x);
                #pragma unroll
                for (Index i = 0; i < EPT; ++i)
                    values[i] = static_cast<Value>(unary_op(values[i]));
                block_store<BLOCK_SIZE, EPT, VECTOR_SIZE>(values, array_ptr, threadIdx.x);
            }
        }
    }

    template<typename Input, typename Output, typename Index, typename UnaryOp,
             typename Config, PointerTraits PointerTrait, StridesTraits StridesTrait>
    __global__ __launch_bounds__(Config::BLOCK_SIZE)
    void ewise_unary_4d(Accessor<const Input, 4, Index, PointerTrait, StridesTrait> input_batched,
                        Accessor<Output, 4, Index, PointerTrait, StridesTrait> output_batched,
                        Shape2<Index> shape, UnaryOp unary_op, Index blocks_x) {

        const auto index = noa::indexing::offset2index(static_cast<Index>(blockIdx.x), blocks_x);
        const auto gid = Vec4<Index>{
                blockIdx.z, blockIdx.y,
                Config::BLOCK_WORK_SIZE_2D.y * index[0] + threadIdx.y,
                Config::BLOCK_WORK_SIZE_2D.x * index[1] + threadIdx.x};
        const auto input = input_batched[gid[0]][gid[1]];
        const auto output = output_batched[gid[0]][gid[1]];

        #pragma unroll
        for (u32 k = 0; k < Config::ELEMENTS_PER_THREAD_2D; ++k) {
            #pragma unroll
            for (u32 l = 0; l < Config::ELEMENTS_PER_THREAD_2D; ++l) {
                const Index ik = gid[2] + Config::BLOCK_SIZE_2D.y * k;
                const Index il = gid[3] + Config::BLOCK_SIZE_2D.x * l;
                if (ik < shape[0] && il < shape[1])
                    output(ik, il) = static_cast<Output>(unary_op(input(ik, il)));
            }
        }
    }
}

namespace noa::cuda::utils {
    // Apply a unary operator op(Input) -> Output, element-wise.
    // This function is asynchronous relative to the host and may return before completion.
    // The caller must make sure the input and output arrays stay valid until completion.
    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Input, typename Output, typename Index, typename UnaryOp>
    void ewise_unary(const char* name,
                     const Input* input, Strides4<Index> input_strides,
                     Output* output, Strides4<Index> output_strides,
                     Shape4<Index> shape, Stream& stream,
                     UnaryOp unary_op) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        // Rearrange to rightmost order.
        shape = noa::indexing::effective_shape(shape, output_strides);
        const auto order = noa::indexing::order(output_strides, shape);
        if (noa::any(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = noa::indexing::reorder(shape, order);
            input_strides = noa::indexing::reorder(input_strides, order);
            output_strides = noa::indexing::reorder(output_strides, order);
        }

        const auto is_contiguous =
                noa::indexing::is_contiguous(input_strides, shape) &&
                noa::indexing::is_contiguous(output_strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated if they're not contiguous.
            const auto batch_size = is_contiguous[0] ? 1 : shape[0];
            const auto input_strides_2d = input_strides.filter(0, 3);
            const auto output_strides_2d =  output_strides.filter(0, 3);
            const auto elements = is_contiguous[0] ? shape.elements() : shape.pop_front().elements();

            u32 vector_size = is_contiguous[3] ? std::min({max_vector_count(input), max_vector_count(output), i64{8}}) : 1;
            if (batch_size > 1) {
                // Make sure the beginning of each batch preserves the alignment.
                // If not, try with a lower vector size
                for (; vector_size >= 2; vector_size /= 2) {
                    if (!(input_strides_2d[0] % vector_size) && !(output_strides_2d[0] % vector_size))
                        break;
                }
            }

            const Index block_work_size = Config::BLOCK_SIZE * std::max(vector_size, Config::ELEMENTS_PER_THREAD);
            const dim3 blocks(noa::math::divide_up(elements, block_work_size), batch_size);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE};

            if (vector_size == 1) {
                using input_accessor_t = Accessor<const Input, 2, Index, PointerTrait, StridesTrait>;
                using output_accessor_t = Accessor<Output, 2, Index, PointerTrait, StridesTrait>;
                const auto input_accessor = input_accessor_t(input, input_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                return stream.enqueue(name,
                        details::ewise_unary_1d<Input, Output, Index, UnaryOp, Config, 1, PointerTrait, StridesTrait>,
                        config, input_accessor, output_accessor, elements, unary_op);
            } else {
                using input_accessor_t = AccessorContiguous<const Input, 2, Index, PointerTrait>;
                using output_accessor_t = AccessorContiguous<Output, 2, Index, PointerTrait>;
                const auto input_accessor = input_accessor_t(input, input_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                if (vector_size == 2) {
                    return stream.enqueue(name,
                            details::ewise_unary_1d<Input, Output, Index, UnaryOp,Config, 2, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, input_accessor, output_accessor, elements, unary_op);
                } else if (vector_size == 4) {
                    return stream.enqueue(name,
                            details::ewise_unary_1d<Input, Output, Index, UnaryOp,Config, 4, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, input_accessor, output_accessor, elements, unary_op);
                } else {
                    return stream.enqueue(name,
                            details::ewise_unary_1d<Input, Output, Index, UnaryOp,Config, 8, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, input_accessor, output_accessor, elements, unary_op);
                }
            }
        } else { // multi-dimensional, non-contiguous array
            const auto blocks_x = noa::math::divide_up(shape[3], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.x));
            const auto blocks_y = noa::math::divide_up(shape[2], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.y));
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE_2D};

            using input_accessor_t = Accessor<const Input, 4, Index, PointerTrait, StridesTrait>;
            using output_accessor_t = Accessor<Output, 4, Index, PointerTrait, StridesTrait>;
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);

            stream.enqueue(
                    name,
                    details::ewise_unary_4d<Input, Output, Index, UnaryOp, Config, PointerTrait, StridesTrait>,
                    config, input_accessor, output_accessor, shape.filter(2, 3), unary_op, blocks_x);
        }
    }

    template<StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Value, typename Index, typename UnaryOp>
    void ewise_unary(const char* name,
                     Value* array, Strides4<Index> strides,
                     Shape4<Index> shape, Stream& stream,
                     UnaryOp unary_op) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(array, stream.device());
        constexpr PointerTraits PointerTrait = PointerTraits::RESTRICT;

        // Rearrange to rightmost order.
        shape = noa::indexing::effective_shape(shape, strides);
        const auto order = noa::indexing::order(strides, shape);
        if (noa::any(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = noa::indexing::reorder(shape, order);
            strides = noa::indexing::reorder(strides, order);
        }

        const auto is_contiguous = noa::indexing::is_contiguous(strides, shape);
        if (is_contiguous[1] && is_contiguous[2]) { // 1D-like
            // Keep batches separated in a different Grid.Y if they're not contiguous.
            const bool batch_size = is_contiguous[0] ? 1 : shape[0];
            const auto strides_2d = strides.filter(0, 3);
            const auto elements = is_contiguous[0] ? shape.elements() : shape.pop_front().elements();

            u32 vector_size = is_contiguous[3] ? std::min(max_vector_count(array), i64{8}) : 1;
            if (batch_size > 1) {
                // Make sure the beginning of each batch preserves the alignment.
                // If not, try with a lower vector size
                for (; vector_size >= 2; vector_size /= 2) {
                    if (!(strides[0] % vector_size))
                        break;
                }
            }

            const Index block_work_size = Config::BLOCK_SIZE * std::max(vector_size, Config::ELEMENTS_PER_THREAD);
            const dim3 blocks(noa::math::divide_up(elements, block_work_size), batch_size);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE};

            if (vector_size == 1) {
                using accessor_t = Accessor<Value, 2, Index, PointerTrait, StridesTrait>;
                const auto accessor = accessor_t(array, strides_2d);
                return stream.enqueue(name,
                        details::ewise_unary_1d<Value, Index, UnaryOp, Config, 1, PointerTrait, StridesTrait> ,
                        config, accessor, elements, unary_op);
            } else {
                using accessor_t = AccessorContiguous<Value, 2, Index, PointerTrait>;
                const auto accessor = accessor_t(array, strides_2d);
                if (vector_size == 2) {
                    return stream.enqueue(name,
                            details::ewise_unary_1d<Value, Index, UnaryOp, Config, 2, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, accessor, elements, unary_op);
                } else if (vector_size == 4) {
                    return stream.enqueue(name,
                            details::ewise_unary_1d<Value, Index, UnaryOp, Config, 4, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, accessor, elements, unary_op);
                } else {
                    return stream.enqueue(name,
                            details::ewise_unary_1d<Value, Index, UnaryOp, Config, 8, PointerTrait, StridesTraits::CONTIGUOUS>,
                            config, accessor, elements, unary_op);
                }
            }
        } else { // multi-dimensional, non-contiguous array
            const u32 blocks_x = noa::math::divide_up(shape[3], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.x));
            const u32 blocks_y = noa::math::divide_up(shape[2], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.y));
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE_2D};

            using accessor_t = Accessor<Value, 4, Index, PointerTrait, StridesTrait>;
            using accessor_t = Accessor<Value, 4, Index, PointerTrait, StridesTrait>;
            const auto accessor = accessor_t(array, strides);

            stream.enqueue(name,
                           details::ewise_unary_4d<Value, Value, Index, UnaryOp, Config, PointerTrait, StridesTrait>,
                           config, accessor, accessor, shape.filter(2, 3), unary_op, blocks_x);
        }
    }
}

#define NOA_CUDA_EWISE_UNARY_GENERATE_API
namespace noa::cuda {                                                               \
    template<typename In, typename Out, typename UnaryOp, typename>                 \
    void ewise_unary(const In* input, const Strides4<i64>& input_strides,           \
                     Out* output, const Strides4<i64>& output_strides,              \
                     const Shape4<i64>& shape, UnaryOp unary_op, Stream& stream) {  \
        noa::cuda::utils::ewise_unary(                                              \
                "ewise_unary",                                                      \
                input, input_strides,                                               \
                output, output_strides,                                             \
                shape, stream, unary_op);                                           \
    }                                                                               \
}

#define NOA_CUDA_EWISE_UNARY_INSTANTIATE_API(In, Out, UnaryOp)  \
namespace noa::cuda {                                           \
    template void ewise_unary<In,Out,UnaryOp,void>(             \
        const In*, const Strides4<i64>&,                        \
        Out*, const Strides4<i64>&,                             \
        const Shape4<i64>&, UnaryOp, Stream&);                  \
}
