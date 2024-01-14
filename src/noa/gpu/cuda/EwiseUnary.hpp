#pragma once

#include "noa/core/Config.hpp"

#if defined(NOA_IS_OFFLINE)
#include "noa/core/types/Accessor.hpp"
#include "noa/core/string/Reflect.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/utils/Pointers.hpp"

namespace noa::cuda {
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

namespace noa::cuda {
    void ewise_unary_launch(
            std::string_view kernel,
            LaunchConfig config,
            void** arguments,
            Stream& stream
    );

    /// Launches an element-wise kernel.
    template<PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             typename Config = EwiseStaticConfigDefault,
             typename Input, typename Output, typename Index, typename UnaryOp>
    void ewise_unary(
            const Input* input, Strides4<Index> input_strides,
            Output* output, Strides4<Index> output_strides,
            Shape4<Index> shape, Stream& stream,
            UnaryOp unary_op
    ) {
        NOA_ASSERT(all(shape > 0));
        NOA_ASSERT_DEVICE_PTR(input, stream.device());
        NOA_ASSERT_DEVICE_PTR(output, stream.device());

        // Rearrange to rightmost order.
        shape = noa::effective_shape(shape, output_strides);
        const auto order = noa::order(output_strides, shape);
        if (noa::any(order != Vec4<Index>{0, 1, 2, 3})) {
            shape = shape.reorder(order);
            input_strides = input_strides.reorder(order);
            output_strides = output_strides.reorder(order);
        }

        const auto is_contiguous =
                noa::is_contiguous(input_strides, shape) &&
                noa::is_contiguous(output_strides, shape);

        if (is_contiguous[1] && is_contiguous[2]) { // 1d-like
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
            const dim3 blocks(noa::divide_up(elements, block_work_size), batch_size);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE};

            std::string kernel_name = noa::ReflectedTemplate("::noa::cuda::guts::ewise_unary_1d")
                    .instantiate(
                            TypeWrapper<Input>{}, TypeWrapper<Output>{}, TypeWrapper<Index>{},
                            TypeWrapper<UnaryOp>{}, TypeWrapper<Config>{}, vector_size,
                            PointerTrait, vector_size == 1 ? StridesTrait : StridesTraits::CONTIGUOUS);

            if (vector_size == 1) {
                using input_accessor_t = Accessor<const Input, 2, Index, PointerTrait, StridesTrait>;
                using output_accessor_t = Accessor<Output, 2, Index, PointerTrait, StridesTrait>;
                const auto input_accessor = input_accessor_t(input, input_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                const auto arguments = CollectArgumentAddresses(input_accessor, output_accessor, elements, unary_op);
                ewise_unary_launch(kernel_name, config, arguments.pointers(), stream);
            } else {
                using input_accessor_t = AccessorContiguous<const Input, 2, Index, PointerTrait>;
                using output_accessor_t = AccessorContiguous<Output, 2, Index, PointerTrait>;
                const auto input_accessor = input_accessor_t(input, input_strides_2d);
                const auto output_accessor = output_accessor_t(output, output_strides_2d);
                const auto arguments = CollectArgumentAddresses(input_accessor, output_accessor, elements, unary_op);
                ewise_unary_launch(kernel_name, config, arguments.pointers(), stream);
            }

        } else { // multi-dimensional, non-contiguous array
            const auto blocks_x = noa::divide_up(shape[3], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.x));
            const auto blocks_y = noa::divide_up(shape[2], static_cast<Index>(Config::BLOCK_WORK_SIZE_2D.y));
            const dim3 blocks(blocks_x * blocks_y, shape[1], shape[0]);
            const LaunchConfig config{blocks, Config::BLOCK_SIZE_2D};

            std::string kernel_name = noa::ReflectedTemplate("::noa::cuda::guts::ewise_unary_4d")
                    .instantiate(
                            TypeWrapper<Input>{}, TypeWrapper<Output>{}, TypeWrapper<Index>{},
                            TypeWrapper<UnaryOp>{}, TypeWrapper<Config>{},
                            PointerTrait, StridesTrait);

            using input_accessor_t = Accessor<const Input, 4, Index, PointerTrait, StridesTrait>;
            using output_accessor_t = Accessor<Output, 4, Index, PointerTrait, StridesTrait>;
            const auto input_accessor = input_accessor_t(input, input_strides);
            const auto output_accessor = output_accessor_t(output, output_strides);
            const auto shape_hw = shape.filter(2, 3);

            const auto arguments = CollectArgumentAddresses(
                    input_accessor, output_accessor, shape_hw, unary_op, blocks);
            ewise_unary_launch(kernel_name, config, arguments.pointers(), stream);
        }
    }
}
#endif
