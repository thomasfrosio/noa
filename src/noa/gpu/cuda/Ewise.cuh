#pragma once
#include "noa/gpu/cuda/IncludeGuard.cuh"

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/Constants.hpp"
#include "noa/gpu/cuda/Pointers.hpp"
#include "noa/gpu/cuda/Stream.hpp"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

namespace noa::cuda::details {
    template<typename Block, typename Interface, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_2d(Op op, Input input, Output output, Index width) {
        const Vec<Index, 2> gid = global_indices_2d<Index, Block>();

        Interface::init(op, thread_uid<2>());

        for (Index i = 0; i < Block::n_elements_per_thread_x; ++i) {
            const Index cid = gid[1] + i * Block::block_size_x;
            if (cid < width)
                Interface::call(op, input, output, gid[0], cid);
        }
        Interface::final(op, thread_uid<2>());
    }

    template<typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffer,
             typename Output, typename OutputAlignedBuffer>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_2d_vectorized(Op op, Input input, Output output, Index width) {
        Interface::init(op, thread_uid<2>());

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
        const Index block_offset = blockIdx.x * Block::block_work_size_x;
        const Index remaining = width - block_offset;

        if (remaining < Block::block_work_size_x) {
            const Index gid = block_offset + threadIdx.x;
            for (Index i = 0; i < Block::n_elements_per_thread_x; ++i) {
                const Index cid = gid + i * Block::block_size_x;
                if (cid < width)
                    Interface::call(op, input_1d, output_1d, cid);
            }
        } else {
            // Offset the accessors to the start of the block workspace.
            input_1d.for_each([=](auto& accessor){ accessor.offset_inplace(block_offset); });
            output_1d.for_each([=](auto& accessor){ accessor.offset_inplace(block_offset); });

            // Load the inputs.
            vectorized_tuple_t<Input> vectorized_input[Block::n_elements_per_thread_x];
            block_load<Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffer>(
                input_1d, vectorized_input, threadIdx.x);

            // Call the operator, store the results in the output buffer.
            // This implies that the operator does not write to the input(s)
            // and does not read from the output(s).
            vectorized_tuple_t<Output> vectorized_output[Block::n_elements_per_thread_x];
            for (Index i = 0; i < Block::n_elements_per_thread_x; ++i)
                Interface::call(op, vectorized_input[i], vectorized_output[i], 0);

            // Store the output values back to global memory.
            block_store<Block::block_size_x, Block::n_elements_per_thread_x, OutputAlignedBuffer>(
                vectorized_output, output_1d, threadIdx.x);
        }

        Interface::final(op, thread_uid<2>());
    }

    // 3d grid of 2d blocks.
    template<typename Block, typename Interface, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_4d(Op op, Input input, Output output, Shape2<Index> shape_hw, u32 n_blocks_x) {
        const auto gid = global_indices_4d<Index, Block>(n_blocks_x);

        auto to_2d = [&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor); // move AccessorValue
            else
                return accessor[gid[0]][gid[1]]; // 4d Accessor -> 2d AccessorReference
        };
        auto input_2d = std::move(input).map(to_2d);
        auto output_2d = std::move(output).map(to_2d);

       Interface::init(op, thread_uid());

        for (u32 h = 0; h < Block::n_elements_per_thread_y; ++h) {
            for (u32 w = 0; w < Block::n_elements_per_thread_x; ++w) {
                const Index ih = gid[2] + Block::block_size_y * h;
                const Index iw = gid[3] + Block::block_size_x * w;
                if (ih < shape_hw[0] and iw < shape_hw[1])
                   Interface::call(op, input_2d, output_2d, ih, iw);
            }
        }
       Interface::final(op, thread_uid());
    }
}

namespace noa::cuda::details {
    // nvcc bug - this could be a lambda, but nvcc <=12.6 is broken...
    template<size_t ALIGNMENT, typename Config, typename Input, typename Output,  typename Op, typename Index>
    void launch_ewise_2d(
        Op&& op,
        Input&& input,
        Output&& output,
        Stream& stream,
        Index n_elements,
        u32 batch
    ) {
        using OpDecay = std::decay_t<Op>;
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input, Output>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using OutputVec = to_aligned_buffer_t<Output, ALIGNMENT, VEC_SIZE>;
        constexpr bool VECTORIZE = is_vectorized<InputVec, OutputVec>();

        constexpr u32 N_ELEMENTS_PER_THREAD = max(static_cast<size_t>(Config::n_elements_per_thread), VEC_SIZE);
        using Block = StaticBlock<Config::block_size, 1, 1, N_ELEMENTS_PER_THREAD, 1, 1>;
        using Interface = Config::interface;

        constexpr auto TO_2D = nd::AccessorConfig<2>{
            .enforce_contiguous = VECTORIZE,
            .enforce_restrict = false,
            .filter = {0, 3},
        };
        auto input_2d = nd::reconfig_accessors<TO_2D>(std::forward<Input>(input));
        auto output_2d = nd::reconfig_accessors<TO_2D>(std::forward<Output>(output));
        using Input2D = decltype(input_2d);
        using Output2D = decltype(output_2d);

        auto grid_x = GridX(n_elements, Block::block_work_size_x);
        auto grid_y = GridY(batch, 1);
        check(grid_x.n_launches() == 1);

        for (u32 y{}; y < grid_y.n_launches(); ++y) {
            nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, output_2d);

            const auto config = LaunchConfig{
                .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y)),
                .n_threads = dim3(Block::block_size, 1),
            };
            if constexpr (VECTORIZE) {
                stream.enqueue(
                    details::ewise_2d_vectorized<Block, Interface, OpDecay, Index, Input2D, InputVec, Output2D, OutputVec>,
                    config, op, input_2d, output_2d, n_elements
                );
            } else {
                stream.enqueue(
                    details::ewise_2d<Block, Interface, OpDecay, Input2D, Output2D, Index>,
                    config, op, input_2d, output_2d, n_elements
                );
            }
        }
    }
}

namespace noa::cuda {
    // TODO Atm, we set the block size at compile time, and prefer smaller blocks as they tend to "waste"
    //      less threads. We should maybe switch to (or at least allow) runtime block sizes and try to
    //      maximize the occupancy for the target GPU...
    template<bool ZipInput = false,
             bool ZipOutput = false,
             u32 BlockSize = 128,
             u32 ElementsPerThread = 4,
             bool EnableVectorization = true>
    struct EwiseConfig {
        static_assert(is_power_of_2(ElementsPerThread));
        static_assert(is_power_of_2(BlockSize) and BlockSize >= Constant::WARP_SIZE and BlockSize <= Limits::MAX_THREADS);

        using interface = nd::EwiseInterface<ZipInput, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
        static constexpr bool enable_vectorization = EnableVectorization;
    };

    template<typename Config = EwiseConfig<>,
             typename Input, typename Output, typename Index, typename Op>
    requires (nt::tuple_of_accessor_or_empty<std::decay_t<Input>> and
              nt::tuple_of_accessor_pure_or_empty<std::decay_t<Output>>)
    NOA_NOINLINE void ewise(
        const Shape<Index, 4>& shape,
        Op&& op,
        Input&& input,
        Output&& output,
        Stream& stream
    ) {
        const Vec<bool, 4> is_contiguous =
            ni::is_contiguous(input, shape) and
            ni::is_contiguous(output, shape);

        // TODO fused contiguous dimensions together, e.g. 2d unbatched should be ok here
        if (is_contiguous[1] and is_contiguous[2]) { // 2d-like
            // If batches are not contiguous to each other, keep them separated in a different grid.y.
            const auto batch = is_contiguous[0] ? 1u : safe_cast<u32>(shape[0]);
            const auto shape_i64 = shape.template as_safe<i64>();
            const auto n_elements_i64 = is_contiguous[0] ? shape_i64.n_elements() : shape_i64.pop_front().n_elements();
            const auto n_elements = safe_cast<Index>(n_elements_i64);

            if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
                const auto shape_3d = Shape{batch, 1u, 1u};
                size_t alignment = min(
                    min_address_alignment(input, shape_3d),
                    min_address_alignment(output, shape_3d)
                );
                if (alignment == 16) {
                    return details::launch_ewise_2d<16, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch
                    );
                } else if (alignment == 8) {
                    return details::launch_ewise_2d<8, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch
                    );
                } else if (alignment == 4) {
                    return details::launch_ewise_2d<4, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch
                    );
                } else if (alignment == 2) {
                    return details::launch_ewise_2d<2, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch
                    );
                }
            }
            details::launch_ewise_2d<1, Config>(
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                stream, n_elements, batch
            );
        } else {
            using InputDecay = std::decay_t<Input>;
            using OutputDecay = std::decay_t<Output>;
            using OpDecay = std::decay_t<Op>;
            using Interface = Config::interface;

            // 2d block.
            // The goal is to waste as fewer threads as possible, assuming 2d/3d/4d arrays have a
            // similar number of elements in their two innermost dimensions, so make the block
            // (and block work shape) as square as possible.
            static constexpr u32 BLOCK_SIZE_X = min(Config::block_size, Constant::WARP_SIZE);
            static constexpr u32 BLOCK_SIZE_Y = Config::block_size / BLOCK_SIZE_X;
            static constexpr u32 N_ELEMENTS_PER_THREAD_X = max(Config::n_elements_per_thread / 2, 1u);
            static constexpr u32 N_ELEMENTS_PER_THREAD_Y = max(Config::n_elements_per_thread - N_ELEMENTS_PER_THREAD_X, 1u);
            using Block = StaticBlock<
                BLOCK_SIZE_X, BLOCK_SIZE_Y, 1,
                N_ELEMENTS_PER_THREAD_X, N_ELEMENTS_PER_THREAD_Y, 1>;

            auto grid_x = GridXY(shape[3], shape[2], Block::block_work_size_x, Block::block_work_size_y);
            auto grid_y = GridY(shape[1], 1);
            auto grid_z = GridZ(shape[0], 1);
            check(grid_x.n_launches() == 1);

            // Save mutable versions of the accessors since we may need to offset them in-place.
            auto input_mut = std::forward<Input>(input);
            auto output_mut = std::forward<Output>(output);

            // Launch the grid.
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto offset = Vec{grid_z.offset_additive(z), grid_y.offset_additive(y)};
                    nd::offset_accessors(offset, input_mut, output_mut);

                    const auto config = LaunchConfig{
                        .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                        .n_threads = dim3(Block::block_size_x, Block::block_size_y),
                    };
                    stream.enqueue(
                        details::ewise_4d<Block, Interface, OpDecay, InputDecay, OutputDecay, Index>,
                        config, op, input_mut, output_mut, shape.filter(2, 3), grid_x.n_blocks_x()
                    );
                }
            }
        }
    }
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
