#pragma once
#include "noa/runtime/cuda/IncludeGuard.hpp"

#include "noa/base/Config.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Block.hpp"
#include "noa/runtime/cuda/ComputeHandle.hpp"
#include "noa/runtime/cuda/Constants.hpp"
#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/Utils.hpp"

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

namespace noa::gpu::details {
    template<usize N, typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffer,
             typename Output, typename OutputAlignedBuffer,
             bool IS_VECTORIZED>
    struct Ewise {
        static constexpr usize GRID_NDIM = N >= 3 ? 3 : N;
        using handle_type = ComputeHandle<Index, GRID_NDIM, Block, (N > 1), false, false>;
        using grid_fused_shape_type = std::conditional_t<(N >= 3), Vec<u32, N - 3>, Empty>;
        using shape_type = Shape<Index, 1 + (N > 2)>;

        NOA_NO_UNIQUE_ADDRESS Input input;
        NOA_NO_UNIQUE_ADDRESS Output output;
        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS grid_fused_shape_type grid_fused_shape_in_x_unbatched{};
        shape_type shape;

    public:
        template<typename Input1D, typename Output1D, typename Handle>
        NOA_FD void ewise_vectorized(Input1D input_1d, Output1D output_1d, const Handle& ch) {
            Interface::init(ch, op);

            // Offset to the current batch.
            const Index block_offset = ch.block_index_x() * static_cast<Index>(Block::block_work_size_x);
            const Index remaining = shape.back() - block_offset;

            if (remaining < static_cast<Index>(Block::block_work_size_x)) {
                const Index gid = block_offset + ch.thread_index_x();
                for (Index i = 0; i < static_cast<Index>(Block::n_elements_per_thread_x); ++i) {
                    const Index cid = gid + i * static_cast<Index>(Block::block_size_x);
                    if (cid < shape.back())
                        Interface::call(ch, op, input_1d, output_1d, cid);
                }
            } else {
                // Offset the accessors to the start of the block workspace.
                input_1d.for_each([=](auto& accessor){ accessor.offset_inplace(block_offset); });
                output_1d.for_each([=](auto& accessor){ accessor.offset_inplace(block_offset); });

                // Load the inputs.
                vectorized_tuple_t<Input> vectorized_input[Block::n_elements_per_thread_x];
                block_load<Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffer>(
                    input_1d, vectorized_input, ch.template thread_index_x<i32>());

                // Call the operator, store the results in the output buffer.
                // This implies that the operator does not write to the input(s)
                // and does not read from the output(s).
                vectorized_tuple_t<Output> vectorized_output[Block::n_elements_per_thread_x];
                for (Index i = 0; i < static_cast<Index>(Block::n_elements_per_thread_x); ++i)
                    Interface::call(ch, op, vectorized_input[i], vectorized_output[i], Index{});

                // Store the output values back to global memory.
                block_store<Block::block_size_x, Block::n_elements_per_thread_x, OutputAlignedBuffer>(
                    vectorized_output, output_1d, ch.template thread_index_x<i32>());
            }

            Interface::deinit(ch, op);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 1 and IS_VECTORIZED) {
            ewise_vectorized(input, output, handle);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 1 and not IS_VECTORIZED) {
            Interface::init(handle, op);

            const auto gid = handle.template global_indices<Index, 1>();
            for (Index i = 0; i < static_cast<Index>(Block::n_elements_per_thread_x); ++i) {
                const Index cid = gid[0] + i * static_cast<Index>(Block::block_size_x);
                if (cid < shape.back()) // width
                    Interface::call(handle, op, input, output, cid);
            }
            Interface::deinit(handle, op);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 2 and IS_VECTORIZED) {
            // Offset to the current row.
            // AccessorValue(s) are simply moved, and 2d Accessor(s) return 1d AccessorReference(s).
            // Note that AccessorReference points to the strides of the original Accessor, but since
            // the move is non-destructive (it's just a cast here), everything is fine.
            auto to_1d = [&]<typename T>(T&& accessor) {
                if constexpr (nt::is_accessor_value_v<T>)
                    return NOA_FWD(accessor);
                else
                    return accessor[handle.block_index_y()];
            };
            auto input_1d = std::move(input).map(to_1d);
            auto output_1d = std::move(output).map(to_1d);

            ewise_vectorized(input_1d, output_1d, handle);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N == 2 and not IS_VECTORIZED) {
            Interface::init(handle, op);

            const auto gid = handle.template global_indices<Index, 2>();
            for (Index i = 0; i < static_cast<Index>(Block::n_elements_per_thread_x); ++i) {
                const Index cid = gid[1] + i * static_cast<Index>(Block::block_size_x);
                if (cid < shape.back()) // width
                    Interface::call(handle, op, input, output, gid[0], cid);
            }
            Interface::deinit(handle, op);
        }

        NOA_FD void operator()(const handle_type& handle) requires (N >= 3) {
            Interface::init(handle, op);

            // Compute the global indices.
            // Unfused grid axes along the X axis.
            // Handle multigrid support by adding the grid offset.
            const auto gid = handle.template global_indices<Index, N>(grid_fused_shape_in_x_unbatched);

            auto to_2d = [&gid]<typename T>(T&& accessor) {
                if constexpr (nt::is_accessor_value_v<T>) {
                    return NOA_FWD(accessor);
                } else if constexpr (N >= 3) {
                    // Index the outer dimensions, leaving HW.
                    return accessor[gid.template pop_back<2>()];
                } else {
                    static_assert(nt::always_false<T>);
                }
            };
            auto input_2d = std::move(input).map(to_2d);
            auto output_2d = std::move(output).map(to_2d);

            for (u32 h = 0; h < static_cast<Index>(Block::n_elements_per_thread_y); ++h) {
                for (u32 w = 0; w < static_cast<Index>(Block::n_elements_per_thread_x); ++w) {
                    const Index ih = gid[N - 2] + static_cast<Index>(Block::block_size_y) * h;
                    const Index iw = gid[N - 1] + static_cast<Index>(Block::block_size_x) * w;
                    if (ih < shape[0] and iw < shape[1])
                        Interface::call(handle, op, input_2d, output_2d, ih, iw);
                }
            }
            Interface::deinit(handle, op);
        }
    };

    template<usize ALIGNMENT, usize N, typename Config, typename Input, typename Output,  typename Op, typename Index>
    void launch_ewise_1d_or_2d(
        Op&& op,
        Input&& input, // (((1..,)H,)W)
        Output&& output, // (((1..,)H,)W)
        Stream& stream,
        Index n_elements,
        u32 batch
    ) {
        constexpr usize VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input, Output>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using OutputVec = to_aligned_buffer_t<Output, ALIGNMENT, VEC_SIZE>;
        constexpr bool VECTORIZE = is_vectorized<InputVec, OutputVec>();

        constexpr u32 N_ELEMENTS_PER_THREAD = max(static_cast<usize>(Config::n_elements_per_thread), VEC_SIZE);
        using Block = StaticBlock<Config::block_size, 1, 1, N_ELEMENTS_PER_THREAD, 1, 1>;
        using Interface = Config::interface;
        using Op_ = std::decay_t<Op>;
        auto shape = Shape{n_elements};

        if constexpr (N == 1) {
            const auto grid_x = GridX(n_elements, Block::block_work_size_x);
            check(grid_x.n_launches() == 1);
            const auto config = LaunchConfig{
                .grid_shape = Vec<u32, 3>{1, 1, grid_x.n_blocks(0)},
                .block_shape = block_shape<Block>(),
            };
            using Input_ = std::decay_t<Input>;
            using Output_ = std::decay_t<Output>;
            using Ewise_ = Ewise<1, Block, Interface, Op_, Index, Input_, InputVec, Output_, OutputVec, VECTORIZE>;
            auto handle = typename Ewise_::handle_type{};
            stream.enqueue(config, handle, Ewise_{NOA_FWD(input), NOA_FWD(output), NOA_FWD(op), {}, shape});
        } else {
            constexpr auto CONFIG = nd::AccessorConfig{
                .enforce_contiguous = VECTORIZE,
                .enforce_restrict = false,
            };
            auto input_2d = nd::reconfig_accessors<CONFIG>(NOA_FWD(input), N - 2, N - 1);
            auto output_2d = nd::reconfig_accessors<CONFIG>(NOA_FWD(output), N - 2, N - 1);
            using Input_ = decltype(input_2d);
            using Output_ = decltype(output_2d);

            const auto grid_x = GridX(n_elements, Block::block_work_size_x);
            const auto grid_y = GridY(batch, 1);
            check(grid_x.n_launches() == 1, "grid.x is larger than the maximum size currently allowed");

            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                // We can offset the accessors directly because the grid Y has a block size of 1,
                // meaning the grid offset is equal to the actual index in the arrays.
                nd::offset_accessors(Vec{grid_y.offset_additive(y)}, input_2d, output_2d);

                const auto config = LaunchConfig{
                    .grid_shape = Vec<u32, 3>{1, grid_y.n_blocks(y), grid_x.n_blocks(0)},
                    .block_shape = block_shape<Block>(),
                };

                using Ewise_ = Ewise<2, Block, Interface, Op_, Index, Input_, InputVec, Output_, OutputVec, VECTORIZE>;
                auto handle = typename Ewise_::handle_type(
                    Vec{grid_y.n_blocks_total()}.template as<u32>(),
                    Vec{grid_y.offset(y)}
                );
                stream.enqueue(config, handle, Ewise_{input_2d, output_2d, op, {}, shape});
            }
        }
    }
}

namespace noa::gpu {
    // TODO Atm, we set the block size at compile time, and prefer smaller blocks as they tend to "waste"
    //      less threads. We should maybe switch to (or at least allow) runtime block sizes and try to
    //      maximize the occupancy for the target GPU...
    template<bool ZipInput = false,
             bool ZipOutput = false,
             u32 BlockSize = 128,
             u32 ElementsPerThread = 4,
             bool EnableVectorization = true>
    struct EwiseConfig {
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
        static_assert(is_power_of_2(Config::n_elements_per_thread));
        static_assert(is_power_of_2(Config::block_size) and Config::block_size >= Constant::WARP_SIZE);

        // Assume layout is optimized and shape is collapsed.
        bool use_row_kernels{true};
        if constexpr (N >= 3) {
            for (usize i{}; i < N - 2; ++i)
                if (shape[i] > 1)
                    use_row_kernels = false;
        }

        if (use_row_kernels) {
            const auto n_rows = N == 1 ? u32{1} : safe_cast<u32>(shape[N - 2]);
            const auto width = safe_cast<Index>(shape[N - 1]);
            if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
                // Find the minimum row alignment.
                auto shape_no_width = Shape<u32, N - 1>::from_value(1);
                if constexpr (N >= 2)
                    shape_no_width[N - 2] = n_rows;
                const auto alignment = min(
                    min_address_alignment(input, shape_no_width),
                    min_address_alignment(output, shape_no_width)
                );
                if (alignment == 16)
                    return details::launch_ewise_1d_or_2d<16, N, Config>(
                        NOA_FWD(op), NOA_FWD(input), NOA_FWD(output), stream, width, n_rows);
                if (alignment == 8)
                    return details::launch_ewise_1d_or_2d<8, N, Config>(
                        NOA_FWD(op), NOA_FWD(input), NOA_FWD(output), stream, width, n_rows);
                if (alignment == 4)
                    return details::launch_ewise_1d_or_2d<4, N, Config>(
                        NOA_FWD(op), NOA_FWD(input), NOA_FWD(output), stream, width, n_rows);
                if (alignment == 2)
                    return details::launch_ewise_1d_or_2d<2, N, Config>(
                        NOA_FWD(op), NOA_FWD(input), NOA_FWD(output), stream, width, n_rows);
            }
            return details::launch_ewise_1d_or_2d<1, N, Config>(
                NOA_FWD(op), NOA_FWD(input), NOA_FWD(output), stream, width, n_rows);
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

            auto grid = GridND(shape, block_work_shape);
            check(grid.n_launches_x() == 1);

            using Interface = Config::interface;
            using Op_ = std::decay_t<Op>;
            using Input_ = std::decay_t<Input>;
            using Output_ = std::decay_t<Output>;
            using Ewise = details::Ewise<N, Block, Interface, Op_, Index, Input_, Empty, Output_, Empty, false>;
            using Handle = Ewise::handle_type;

            auto ewise_op = Ewise{
                .input = NOA_FWD(input),
                .output = NOA_FWD(output),
                .op = NOA_FWD(op),
                .grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front(),
                .shape = shape.filter(N - 2, N - 1),
            };

            for (u32 z{}; z < grid.n_launches_z(); ++z) {
                for (u32 y{}; y < grid.n_launches_y(); ++y) {
                    const auto config = LaunchConfig{
                        .grid_shape = grid.shape_for_launch(z, y, 0),
                        .block_shape = details::block_shape<Block>(),
                    };
                    auto handle = Handle(grid.outer_shape().vec, grid.block_offset_for_launch(z, y));
                    stream.enqueue(config, handle, ewise_op);
                }
            }
        } else {
            unreachable();
        }
    }
}

#if defined(NOA_COMPILER_GCC)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
