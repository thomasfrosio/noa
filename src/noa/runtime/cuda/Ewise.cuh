#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/ComputeHandle.cuh"
#include "noa/runtime/cuda/Constants.hpp"
#include "noa/runtime/cuda/Pointers.hpp"
#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/Utils.cuh"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

namespace noa::cuda::details {
    template<typename Block, typename Interface, typename InputAlignedBuffer, typename OutputAlignedBuffer,
             typename Op, typename Index, typename ComputeHandle, typename Input, typename Output>
    NOA_FD void ewise_vectorized(Op& op, Input& input_1d, Output& output_1d, Index width, const ComputeHandle& ch) {
        Interface::init(ch, op);

        // Offset to the current batch.
        const Index block_offset = blockIdx.x * Block::block_work_size_x;
        const Index remaining = width - block_offset;

        if (remaining < Block::block_work_size_x) {
            const Index gid = block_offset + threadIdx.x;
            for (Index i = 0; i < Block::n_elements_per_thread_x; ++i) {
                const Index cid = gid + i * Block::block_size_x;
                if (cid < width)
                    Interface::call(ch, op, input_1d, output_1d, cid);
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
                Interface::call(ch, op, vectorized_input[i], vectorized_output[i], Index{});

            // Store the output values back to global memory.
            block_store<Block::block_size_x, Block::n_elements_per_thread_x, OutputAlignedBuffer>(
                vectorized_output, output_1d, threadIdx.x);
        }

        Interface::deinit(ch, op);
    }

    template<typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffer,
             typename Output, typename OutputAlignedBuffer>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_1d_vectorized(Op op, Input input, Output output, Index width) {
        using compute_handle_t = ComputeHandle<Index,
            /*GridDim=*/ 1,
            /*BlockDim=*/ 1,
            /*IsMultiGridKernel=*/ true,
            /*IsUsingDynamicSharedMemory=*/ false,
            /*IsTwoPartReduction=*/ false
        >;
        const auto ci = compute_handle_t({}, {});
        ewise_vectorized<Block, Interface, InputAlignedBuffer, OutputAlignedBuffer>(op, input, output, width, ci);
    }

    template<typename Block, typename Interface, typename Op, typename Index,
         typename Input, typename InputAlignedBuffer,
         typename Output, typename OutputAlignedBuffer>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_2d_vectorized(
        Op op, Input input, Output output, Index width,
        Vec<u32, 1> grid_size_y, Vec<u32, 1> block_index_offset_y
    ) {
        using compute_handle_t = ComputeHandle<Index,
            /*GridDim=*/ 2,
            /*BlockDim=*/ 1,
            /*IsMultiGridKernel=*/ true,
            /*IsUsingDynamicSharedMemory=*/ false,
            /*IsTwoPartReduction=*/ false
        >;
        const auto ci = compute_handle_t(grid_size_y, block_index_offset_y);

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

        ewise_vectorized<Block, Interface, InputAlignedBuffer, OutputAlignedBuffer>(op, input_1d, output_1d, width, ci);
    }

    template<typename Block, typename Interface, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_1d(
        Op op, Input input, Output output, Index width
    ) {
        using compute_handle_t = ComputeHandle<Index,
            /*GridDim=*/ 1,
            /*BlockDim=*/ 1,
            /*IsMultiGridKernel=*/ true,
            /*IsUsingDynamicSharedMemory=*/ false,
            /*IsTwoPartReduction=*/ false
        >;
        const auto ci = compute_handle_t({}, {});
        Interface::init(ci, op);

        const auto gid = global_indices_1d<Index, Block>();
        for (Index i = 0; i < Block::n_elements_per_thread_x; ++i) {
            const Index cid = gid[0] + i * Block::block_size_x;
            if (cid < width)
                Interface::call(ci, op, input, output, cid);
        }
        Interface::deinit(ci, op);
    }

    template<typename Block, typename Interface, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_2d(
        Op op, Input input, Output output, Index width,
        Vec<u32, 1> grid_size_y, Vec<u32, 1> block_index_offset_y
    ) {
        using compute_handle_t = ComputeHandle<Index,
            /*GridDim=*/ 2,
            /*BlockDim=*/ 1,
            /*IsMultiGridKernel=*/ true,
            /*IsUsingDynamicSharedMemory=*/ false,
            /*IsTwoPartReduction=*/ false
        >;
        const auto ci = compute_handle_t(grid_size_y, block_index_offset_y);
        Interface::init(ci, op);

        const auto gid = global_indices_2d<Index, Block>();
        for (Index i = 0; i < Block::n_elements_per_thread_x; ++i) {
            const Index cid = gid[1] + i * Block::block_size_x;
            if (cid < width)
                Interface::call(ci, op, input, output, gid[0], cid);
        }
        Interface::deinit(ci, op);
    }

    // Element-wise kernel, for N > 2.
    template<usize N, typename Block, typename Interface, typename Op, typename Input, typename Output, typename Index>
    __global__ __launch_bounds__(Block::block_size)
    void ewise_nd(
        Op op, Input input, Output output, Shape<Index, 2> shape_hw,
        Vec<u32, 2> grid_shape_zy, Vec<u32, 2> block_index_offset_zy, Shape<u32, N - 3> fused_shape
    ) {
        using compute_handle = ComputeHandle<Index,
            /*GridDim=*/ 3,
            /*BlockDim=*/ Block::block_ndim,
            /*IsMultiGridKernel=*/ true,
            /*IsUsingDynamicSharedMemory=*/ false,
            /*IsTwoPartReduction=*/ false
        >;

        const auto ci = compute_handle(grid_shape_zy, block_index_offset_zy);
        Interface::init(ci, op);

        const auto gid = global_indices<Index, N, Block>(fused_shape);
        auto to_2d = [&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>) {
                return std::forward<T>(accessor); // move AccessorValue
            } else if constexpr (N == 3) {
                return accessor[gid[0]]; // 3d Accessor -> 2d AccessorReference
            } else if constexpr (N == 4) {
                return accessor[gid[0]][gid[1]]; // 4d Accessor -> 2d AccessorReference
            } else {
                static_assert(nt::always_false<T>);
            }
        };
        auto input_2d = std::move(input).map(to_2d);
        auto output_2d = std::move(output).map(to_2d);

        for (u32 h = 0; h < Block::n_elements_per_thread_y; ++h) {
            for (u32 w = 0; w < Block::n_elements_per_thread_x; ++w) {
                const Index ih = gid[N - 2] + Block::block_size_y * h;
                const Index iw = gid[N - 1] + Block::block_size_x * w;
                if (ih < shape_hw[0] and iw < shape_hw[1])
                    Interface::call(ci, op, input_2d, output_2d, ih, iw);
            }
        }
        Interface::deinit(ci, op);
    }

    template<usize ALIGNMENT, usize N, typename Config, typename Input, typename Output,  typename Op, typename Index>
    void launch_ewise_1d_or_2d(
        Op&& op,
        Input&& input,
        Output&& output,
        Stream& stream,
        Index n_elements,
        u32 batch
    ) {
        using OpDecay = std::decay_t<Op>;
        constexpr usize VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input, Output>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using OutputVec = to_aligned_buffer_t<Output, ALIGNMENT, VEC_SIZE>;
        constexpr bool VECTORIZE = is_vectorized<InputVec, OutputVec>();

        constexpr u32 N_ELEMENTS_PER_THREAD = max(static_cast<usize>(Config::n_elements_per_thread), VEC_SIZE);
        using Block = StaticBlock<Config::block_size, 1, 1, N_ELEMENTS_PER_THREAD, 1, 1>;
        using Interface = Config::interface;

        if constexpr (N == 1) {
            const auto grid_x = GridX(n_elements, Block::block_work_size_x);
            check(grid_x.n_launches() == 1);
            const auto config = LaunchConfig{
                .n_blocks = dim3(grid_x.n_blocks(0), 1, 1),
                .n_threads = dim3(Block::block_size, 1, 1),
            };
            if constexpr (VECTORIZE) {
                stream.enqueue(
                    details::ewise_1d_vectorized<
                        Block, Interface, OpDecay, Index,
                        std::decay_t<Input>, InputVec,
                        std::decay_t<Output>, OutputVec>,
                    config, op, std::forward<Input>(input), std::forward<Output>(output), n_elements
                );
            } else {
                stream.enqueue(
                    details::ewise_1d<Block, Interface, OpDecay, std::decay_t<Input>, std::decay_t<Output>, Index>,
                    config, op, std::forward<Input>(input), std::forward<Output>(output), n_elements
                );
            }
        } else {
            constexpr auto DIMENSIONS = Vec<usize, 2>{0, 1} + N - 2;
            constexpr auto CONFIG = nd::AccessorConfig<2>{
                .enforce_contiguous = VECTORIZE,
                .enforce_restrict = false,
            };
            auto input_2d = nd::reconfig_accessors<CONFIG>(std::forward<Input>(input), DIMENSIONS);
            auto output_2d = nd::reconfig_accessors<CONFIG>(std::forward<Output>(output), DIMENSIONS);
            using Input2D = decltype(input_2d);
            using Output2D = decltype(output_2d);

            const auto grid_x = GridX(n_elements, Block::block_work_size_x);
            const auto grid_y = GridY(batch, 1);
            check(grid_x.n_launches() == 1, "grid.x is larger than the maximum size currently allowed");

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, output_2d);
                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y)),
                    .n_threads = dim3(Block::block_size, 1),
                };
                const auto grid_size = Vec{grid_y.n_blocks_total()}.template as<u32>();
                const auto grid_offset = Vec{grid_y.offset(y)};
                if constexpr (VECTORIZE) {
                    stream.enqueue(
                        details::ewise_2d_vectorized<Block, Interface, OpDecay, Index, Input2D, InputVec, Output2D, OutputVec>,
                        config, op, input_2d, output_2d, n_elements, grid_size, grid_offset
                    );
                } else {
                    stream.enqueue(
                        details::ewise_2d<Block, Interface, OpDecay, Input2D, Output2D, Index>,
                        config, op, input_2d, output_2d, n_elements, grid_size, grid_offset
                    );
                }
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
             typename Input, typename Output, typename Index, usize N, typename Op>
    requires (nt::tuple_of_accessor_or_empty<std::decay_t<Input>> and
              nt::tuple_of_accessor_pure_or_empty<std::decay_t<Output>> and
              N >= 1)
    NOA_NOINLINE void ewise(
        const Shape<Index, N>& shape,
        Op&& op,
        Input&& input,
        Output&& output,
        Stream& stream
    ) {
        // Collapse contiguous dimensions.
        const auto shape_iz = shape.template as_safe<isize>();
        const auto contiguity = nd::accessors_contiguity(shape, input, output);
        const auto broadcasting = nd::accessors_broadcasting(shape, input, output);
        auto collapsed_shape = noa::collapse_contiguous_dimensions(shape_iz, contiguity, broadcasting);
        collapsed_shape = collapsed_shape.permute(squeeze_empty_dimensions_left(collapsed_shape));

        // Reshape the accessors to the new shape.
        auto input_collapsed = std::forward<Input>(input);
        auto output_collapsed = std::forward<Output>(output);
        check(nd::reshape_accessors<true>(shape_iz, collapsed_shape, input_collapsed, output_collapsed),
              "INTERNAL: reshape failed, please report this issue, shape={}, contiguity={}, broadcasting={}",
              shape_iz, contiguity, broadcasting);

        // Check whether we can use the 1d/2d kernels.
        // TODO Collapsing dimensions together may end up causing multi kernel launches. If so, revert back to original shape?
        bool use_rows{true};
        if constexpr (N >= 3) {
            for (usize i{}; i < N - 2; ++i)
                if (collapsed_shape[i] > 1)
                    use_rows = false;
        }

        if (use_rows) {
            const auto n_rows = N == 1 ? u32{1} : safe_cast<u32>(collapsed_shape[N - 2]);
            const auto width = safe_cast<Index>(collapsed_shape[N - 1]);
            if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
                // Find the minimum row alignment.
                auto shape_without_width = Shape<u32, N - 1>::from_value(1);
                if constexpr (N >= 2)
                    shape_without_width[N - 2] = n_rows;
                const auto alignment = min(
                    min_address_alignment(input_collapsed, shape_without_width),
                    min_address_alignment(output_collapsed, shape_without_width)
                );
                if (alignment == 16) {
                    return details::launch_ewise_1d_or_2d<16, N, Config>(
                        std::forward<Op>(op),
                        std::move(input_collapsed),
                        std::move(output_collapsed),
                        stream, width, n_rows
                    );
                } else if (alignment == 8) {
                    return details::launch_ewise_1d_or_2d<8, N, Config>(
                        std::forward<Op>(op),
                        std::move(input_collapsed),
                        std::move(output_collapsed),
                        stream, width, n_rows
                    );
                } else if (alignment == 4) {
                    return details::launch_ewise_1d_or_2d<4, N, Config>(
                        std::forward<Op>(op),
                        std::move(input_collapsed),
                        std::move(output_collapsed),
                        stream, width, n_rows
                    );
                } else if (alignment == 2) {
                    return details::launch_ewise_1d_or_2d<2, N, Config>(
                        std::forward<Op>(op),
                        std::move(input_collapsed),
                        std::move(output_collapsed),
                        stream, width, n_rows
                    );
                }
            }
            return details::launch_ewise_1d_or_2d<1, N, Config>(
                std::forward<Op>(op),
                std::move(input_collapsed),
                std::move(output_collapsed),
                stream, width, n_rows
            );
        }

        // Cases where the shape cannot be collapsed to 2d.
        if constexpr (N >= 3) {
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

            auto block_work_shape = Shape<u32, N>::from_value(1);
            block_work_shape[N - 2] = Block::block_work_size_y;
            block_work_shape[N - 1] = Block::block_work_size_x;

            auto grid = GridND(collapsed_shape, block_work_shape.template as<isize>());
            check(grid.n_launches_x() == 1, "grid.x is larger than the maximum size currently allowed");
            using Interface = Config::interface;

            // Launch the grid.
            for (u32 z{}; z < grid.n_launches_z(); ++z) {
                for (u32 y{}; y < grid.n_launches_y(); ++y) {
                    nd::offset_accessors(grid.incremental_block_offset_for_launch(z, y), input_collapsed, output_collapsed);
                    const auto config = LaunchConfig{
                        .n_blocks = grid.dim3_shape_for_launch(z, y, 0),
                        .n_threads = dim3(Block::block_size_x, Block::block_size_y, 1),
                    };
                    stream.enqueue(
                        details::ewise_nd<N, Block, Interface, std::decay_t<Op>, std::decay_t<Input>, std::decay_t<Output>, Index>,
                        config, op, input_collapsed, output_collapsed, collapsed_shape.template pop_front<N - 2>(),
                        grid.outer_shape().vec, grid.block_offset_for_launch(z, y), grid.fused_shape().pop_front()
                    );
                }
            }
        } else {
            unreachable();
        }
    }
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
