#pragma once

#include "noa/core/utils/Interfaces.hpp"
#include "noa/gpu/cuda/kernels/Block.cuh"
#include "noa/gpu/cuda/Types.hpp"

namespace noa::cuda {
    // TODO Atm, we set the block size at compile time, and prefer smaller blocks as they tend to "waste"
    //      less threads. We should maybe switch (or at least allow) to runtime block sizes and try to
    //      maximize the occupancy for the target GPU...
    template<bool ZipInput = false,
             bool ZipOutput = false,
             u32 BlockSize = 128,
             u32 ElementsPerThread = 4>
    struct EwiseConfig {
        static_assert(is_multiple_of(ElementsPerThread, 2u) and
                      is_multiple_of(BlockSize, Constant::WARP_SIZE) and
                      BlockSize <= Limits::MAX_THREADS);

        using interface = ng::EwiseInterface<ZipInput, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
    };
}

namespace noa::cuda::guts {
    template<typename EwiseConfig, u32 VectorSize>
    struct EwiseConfig1dBlock {
        using interface = EwiseConfig::interface;
        static constexpr u32 block_size = EwiseConfig::block_size;
        static constexpr u32 vector_size = VectorSize;
        static constexpr u32 n_elements_per_thread = max(EwiseConfig::n_elements_per_thread, VectorSize);
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
    };

    template<typename EwiseConfig>
    struct EwiseConfig2dBlock {
        using interface = EwiseConfig::interface;

        // The goal is to waste as fewer threads as possible, assuming 2d/3d/4d arrays have a
        // similar number of elements in their two innermost dimensions, so make the block
        // (and block work shape) as square as possible.
        static constexpr u32 block_size = EwiseConfig::block_size;
        static constexpr u32 block_size_x = min(block_size, Constant::WARP_SIZE);
        static constexpr u32 block_size_y = block_size / block_size_x;

        static constexpr u32 n_elements_per_thread_x = EwiseConfig::n_elements_per_thread / 2;
        static constexpr u32 n_elements_per_thread_y = EwiseConfig::n_elements_per_thread - n_elements_per_thread_x;

        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
        static constexpr u32 block_work_size_y = block_size_y * n_elements_per_thread_y;
    };

    // 2d grid of 1d blocks.
    template<typename Config, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void ewise_2d(Op op, Input input, Output output, Index width) {
        const Index batch = blockIdx.y;
        const Index gid = blockIdx.x * Config::block_work_size + threadIdx.x;

        Config::interface::init(op, thread_uid<2>());

        for (Index i = 0; i < Config::n_elements_per_thread; ++i) {
            const Index cid = gid + i * Config::block_size;
            if (cid < width)
                Config::interface::call(op, input, output, batch, cid);
        }
        Config::interface::final(op, thread_uid<2>());
    }

    template<typename Config, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void ewise_2d_vectorized(Op op, Input input, Output output, Index width) {
        Config::interface::init(op, thread_uid<2>());

        // Offset to the current row.
        // AccessorValue(s) are simply moved, and 2d Accessor(s) return 1d AccessorReference(s).
        // Note that AccessorReference points to the strides of the original Accessor, but since
        // the move is non-destructive (it's just a cast here), everything is fine.
        auto to_1d = []<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor);
            else
                return accessor[blockIdx.y]; // AccessorReference
        };
        auto input_1d = std::move(input).map(to_1d);
        auto output_1d = std::move(output).map(to_1d);

        // Offset to the current batch.
        const Index block_offset = blockIdx.x * Config::block_work_size;
        const Index remaining = width - block_offset;

        if (remaining < Config::block_work_size) {
            const Index gid = block_offset + threadIdx.x;
            for (Index i = 0; i < Config::n_elements_per_thread; ++i) {
                const Index cid = gid + i * Config::block_size;
                if (cid < width)
                    Config::interface::call(op, input_1d, output_1d, cid);
            }
        } else {
            // Offset the accessors to the start of the block workspace.
            input_1d.for_each([=](auto& accessor){ accessor.offset_accessor(block_offset); });
            output_1d.for_each([=](auto& accessor){ accessor.offset_accessor(block_offset); });

            // Load the inputs.
            using ivec_t = vectorized_tuple_t<Input>;
            ivec_t vectorized_input[Config::n_elements_per_thread];
            block_load<Config::block_size, Config::vector_size, Config::n_elements_per_thread>(
                    input_1d, vectorized_input, threadIdx.x);

            // Call the operator, store the results in the output buffer.
            // This implies that the operator does not modify the input(s).
            using ovec_t = vectorized_tuple_t<Output>;
            ovec_t vectorized_output[Config::n_elements_per_thread];
            for (Index i = 0; i < Config::n_elements_per_thread; ++i)
                Config::interface::call(op, vectorized_input[i], vectorized_output[i], 0);

            // Store the output values back to global memory.
            block_store<Config::block_size, Config::vector_size, Config::n_elements_per_thread>(
                    vectorized_output, output_1d, threadIdx.x);
        }

        Config::interface::final(op, thread_uid<2>());
    }

    // 3d grid of 2d blocks.
    template<typename Config, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Config::block_size)
    void ewise_4d(Op op, Input input, Output output, Shape2<Index> shape_hw, u32 blocks_x) {
        const auto index = ni::offset2index(blockIdx.x, blocks_x);
        const auto gid = Vec4<Index>::from_values(
                blockIdx.z,
                blockIdx.y,
                Config::block_work_size_y * index[0] + threadIdx.y,
                Config::block_work_size_x * index[1] + threadIdx.x
        );

        auto to_2d = [&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor); // move AccessorValue
            else
                return accessor[gid[0]][gid[1]]; // 4d Accessor -> 2d AccessorReference
        };
        auto input_2d = std::move(input).map(to_2d);
        auto output_2d = std::move(output).map(to_2d);

        Config::interface::init(op, thread_uid());

        for (u32 h = 0; h < Config::n_elements_per_thread_y; ++h) {
            for (u32 w = 0; w < Config::n_elements_per_thread_x; ++w) {
                const Index ih = gid[2] + Config::block_size_y * h;
                const Index iw = gid[3] + Config::block_size_x * w;
                if (ih < shape_hw[0] && iw < shape_hw[1])
                    Config::interface::call(op, input_2d, output_2d, ih, iw);
            }
        }
        Config::interface::final(op, thread_uid());
    }
}
