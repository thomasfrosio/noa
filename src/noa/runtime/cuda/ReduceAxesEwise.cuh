#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Block.cuh"
#include "noa/runtime/cuda/ReduceEwise.cuh"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

namespace noa::cuda::details {
    template<typename Config, u32 BlockSizeX, u32 MaxVectorSize>
    struct ReduceAxesEwiseWidthConfig {
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 n_elements_per_thread_x = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
    };

    template<typename Block, typename Interface, typename Op, typename Index,
             typename Input, typename InputAlignedBuffer, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_width_ewise(Op op, Input input, Shape<Index, 2> shape_hw, Reduced reduced, Output output) {
        const auto tid = Vec<Index, 2>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec<Index, 4>::from_values(
            blockIdx.z,
            blockIdx.y,
            blockIdx.x * blockDim.y + tid[0],
            tid[1]
        );
        const bool is_valid_row = gid[2] < shape_hw[0];

        auto input_row = std::move(input).map([&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor);
            else
                return accessor[gid[0]][gid[1]][gid[2]];
        });

        // Initial reduction. Loop until the end of the row is reached.
        constexpr auto OFFSET = Block::block_work_size_x;
        for (Index cid = 0; cid < shape_hw[1] and is_valid_row; cid += OFFSET) {
            block_reduce_ewise_1d_init
                <Block::block_size_x, Block::n_elements_per_thread_x, InputAlignedBuffer, Interface>
                (op, input_row, shape_hw[1] - cid, reduced, tid[1]);

            // Offset to the next work space.
            input_row.for_each([](auto& accessor) { accessor.offset_inplace(OFFSET); });
        }

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Uninitialized<Reduced> shared_buffer_[Block::block_size];
        auto* shared_buffer = reinterpret_cast<Reduced*>(shared_buffer_);
        Reduced* joined = shared_buffer + tid[0] * Block::block_size_x;
        joined[tid[1]] = reduced;
        block_synchronize();

        // Reduce shared data to one element.
        reduced = block_reduce_shared<Interface, Block::block_size_x>(op, joined, tid[1]);
        if (gid[3] == 0 and is_valid_row)
            Interface::final(op, reduced, output, gid[0], gid[1], gid[2]);
    }

    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesEwiseHeightConfig {
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Block, typename Interface, typename Op, typename Index, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Block::block_size)
    void reduce_height_ewise(Op op, Input input, Shape<Index, 2> shape_hw, Reduced reduced, Output output) {
        const auto gid = Vec<Index, 4>::from_values(
            blockIdx.z,
            blockIdx.y,
            threadIdx.y, // one block along the height
            blockIdx.x * Block::block_size_x + threadIdx.x
        );
        const bool is_valid_column = gid[3] < shape_hw[1];

        // Process every row.
        input.for_each([&](auto& accessor) { accessor.offset_inplace(gid[0], gid[1]); });
        for (Index tidy = gid[2]; tidy < shape_hw[0] and is_valid_column; tidy += Block::block_size_y)
            Interface::init(op, input, reduced, 0, 0, tidy, gid[3]);

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Uninitialized<Reduced> shared_buffer_[Block::block_size];
        auto* shared_buffer = reinterpret_cast<Reduced*>(shared_buffer_);
        Reduced* joined = shared_buffer + threadIdx.y * Block::block_size_x + threadIdx.x;
        *joined = reduced;
        block_synchronize();

        // Reduce the height of the block.
        #pragma unroll
        for (u32 size = Block::block_size_y; size >= 2; size /= 2) {
            if (threadIdx.y < size / 2)
                Interface::join(op, joined[Block::block_size_x * size / 2], *joined);
            block_synchronize();
        }

        if (threadIdx.y == 0 and is_valid_column)
            Interface::final(op, *joined, output, gid[0], gid[1], gid[3]);
    }
}

namespace noa::cuda::details {
    template<size_t BLOCK_SIZE_X, size_t ALIGNMENT, typename Config,
             typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_width_(
        const Shape<Index, 4>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream,
        dim3 n_threads
    ) {
        // The width of the output is empty/reduced, remove it.
        constexpr auto TO_3D = nd::AccessorConfig<3>{.filter={0, 1, 2}};
        auto output_3d = nd::reconfig_accessors<TO_3D>(std::forward<Output>(output));
        using Output3D = decltype(output_3d);

        // Prepare the input for vectorization.
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using InputVec = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        constexpr auto TO_4D = nd::AccessorConfig<0>{.enforce_contiguous = is_vectorized<InputVec>()};
        auto input_4d = nd::reconfig_accessors<TO_4D>(std::forward<Input>(input));
        using Input4D = decltype(input_4d);

        // Grid shape.
        auto grid_x = GridX(shape[2], n_threads.y);
        auto grid_y = GridY(shape[1], 1);
        auto grid_z = GridZ(shape[0], 1);
        check(grid_x.n_launches() == 1);

        // Launch the kernel(s).
        using OpDecay = std::decay_t<Op>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Block = ReduceAxesEwiseWidthConfig<Config, BLOCK_SIZE_X, VEC_SIZE>;
        using Interface = Config::interface;

        for (u32 z{}; z < grid_z.n_launches(); z++) {
            for (u32 y{}; y < grid_y.n_launches(); y++) {
                nd::offset_accessors(Vec{grid_z.offset_additive(z), grid_y.offset_additive(y)}, input_4d, output_3d);

                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                    .n_threads = n_threads,
                };
                stream.enqueue(
                    reduce_width_ewise<Block, Interface, OpDecay, Index, Input4D, InputVec, ReducedDecay, Output3D>,
                    config, op, input_4d, shape.filter(2, 3), reduced, output_3d
                );
            }
        }
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_width(
        const Shape<Index, 4>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        // Block shape.
        u32 n_threads_x = shape[3] > 512 ? 256u : 64u;
        if (not is_multiple_of(Config::block_size, n_threads_x))
            n_threads_x = Constant::WARP_SIZE;
        const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
        const auto n_threads = dim3(n_threads_x, n_threads_y);

        // TODO Reduce the number of instantiations by adding support of a runtime block_size_x.
        //      That would mean using dynamic shared memory in block_reduce, which is fine...
        // Note that these if statements are likely to instantiate the same kernel. For instance, if the smallest
        // type has an alignment of 4, only 6 kernels are created (3 with n_threads_x=256, 3 with n_threads_x=64).
        // The worst case is when the smallest type is aligned to only one byte, in which case 10 kernels are created.
        if constexpr (Config::enable_vectorization and nt::enable_vectorization_v<Op>) {
            const size_t alignment = min_address_alignment(input, shape.pop_back());
            if (n_threads_x == 256) {
                if (alignment == 16) {
                    launch_reduce_ewise_width_<256, 16, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_width_<256, 8, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_width_<256, 4, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_width_<256, 2, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else {
                    launch_reduce_ewise_width_<256, 1, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                }
            } else {
                if (alignment == 16) {
                    launch_reduce_ewise_width_<64, 16, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 8) {
                    launch_reduce_ewise_width_<64, 8, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 4) {
                    launch_reduce_ewise_width_<64, 4, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else if (alignment == 2) {
                    launch_reduce_ewise_width_<64, 2, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                } else {
                    launch_reduce_ewise_width_<64, 1, Config>(
                        shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::forward<Output>(output), stream, n_threads
                    );
                }
            }
        } else {
            if (n_threads_x == 256) {
                launch_reduce_ewise_width_<256, 1, Config>(
                    shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), stream, n_threads
                );
            } else {
                launch_reduce_ewise_width_<64, 1, Config>(
                    shape, op, std::forward<Input>(input), std::forward<Reduced>(reduced),
                    std::forward<Output>(output), stream, n_threads
                );
            }
        }
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_bdh(
        const Shape<Index, 4>& shape,
        const Vec<bool, 4>& axes_to_reduce,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        // First copy|move the input and output since they'll need to be reordered.
        auto input_ = std::forward<Input>(input);
        auto output_ = std::forward<Output>(output);

        // The kernel needs the axis to reduce at the "height" position.
        // The width should still be at the rightmost dimension.
        auto order = squeeze_left(axes_to_reduce.as<i32>() + 1);
        order = order.filter(0, 1, 3, 2); // move the width back to rightmost

        // Reorder to (X, X, axis_to_reduce, width).
        const auto reordered_shape = shape.permute(order);
        input_.for_each([&order](auto& accessor) { accessor = accessor.permute(order); });
        output_.for_each([&order](auto& accessor) { accessor = accessor.permute(order); });

        // Remove the empty/reduced axis from the output.
        constexpr auto TO_3D = nd::AccessorConfig<3>{.filter = {0, 1, 3}};
        auto output_3d = nd::reconfig_accessors<TO_3D>(std::move(output_));
        using Output3D = decltype(output_3d);

        // Block shape.
        constexpr u32 N_THREADS_X = Constant::WARP_SIZE;
        constexpr u32 N_THREADS_Y = max(Config::block_size, N_THREADS_X) / N_THREADS_X;
        constexpr auto N_THREADS = dim3(N_THREADS_X, N_THREADS_Y);

        // Grid shape.
        const auto grid_x = GridX(reordered_shape[3], N_THREADS_X);
        const auto grid_y = GridY(reordered_shape[1], 1);
        const auto grid_z = GridZ(reordered_shape[0], 1);
        check(grid_x.n_launches() == 1);

        // Launch kernel.
        using OpDecay = std::decay_t<Op>;
        using InputDecay = std::decay_t<Input>;
        using ReducedDecay = std::decay_t<Reduced>;
        using Block = ReduceAxesEwiseHeightConfig<Config, N_THREADS_X>;
        using Interface = Config::interface;

        for (u32 z{}; z < grid_z.n_launches(); z++) {
            for (u32 y{}; y < grid_y.n_launches(); y++) {
                nd::offset_accessors(Vec{grid_z.offset_additive(z), grid_y.offset_additive(y)}, input_, output_3d);

                const auto config = LaunchConfig{
                    .n_blocks = dim3(grid_x.n_blocks(0), grid_y.n_blocks(y), grid_z.n_blocks(z)),
                    .n_threads = N_THREADS,
                };
                stream.enqueue(
                    reduce_height_ewise<Block, Interface, OpDecay, Index, InputDecay, ReducedDecay, Output3D>,
                    config, op, input_, reordered_shape.template pop_front<2>(), reduced, output_3d
                );
            }
        }
    }
}

namespace noa::cuda {
    template<bool ZipInput = false,
             bool ZipReduced = false,
             bool ZipOutput = false,
             u32 ElementsPerThread = 8,
             u32 BlockSize = 512,
             u32 MaxGridSize = 4096,
             bool EnableVectorization = true>
    struct ReduceAxesEwiseConfig {
        static_assert(is_power_of_2(ElementsPerThread));
        static_assert(is_power_of_2(BlockSize) and BlockSize <= Limits::MAX_THREADS);

        using interface = nd::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;
        static constexpr u32 max_grid_size = MaxGridSize;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool enable_vectorization = EnableVectorization;
    };

    template<typename Config = ReduceAxesEwiseConfig<>,
             typename Op, typename Input, typename Reduced, typename Output, typename Index>
    requires (nt::tuple_of_accessor_nd<std::decay_t<Input>, 4> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_pure_nd<std::decay_t<Output>, 4> and
              nt::tuple_of_accessor_value<std::decay_t<Reduced>>)
    NOA_NOINLINE void reduce_axes_ewise(
        const Shape<Index, 4>& input_shape,
        const Shape<Index, 4>& output_shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        const auto axes_to_reduce = input_shape.cmp_ne(output_shape);
        check((axes_to_reduce and output_shape.cmp_ne(1)) == false,
              "Dimensions should match the input shape, or be 1, "
              "indicating the dimension should be reduced to one element. "
              "Got shape input={}, output={}", input_shape, output_shape);
        check(axes_to_reduce.any_eq(true),
              "No reduction to compute. Got shape input={}, output={}. Please use ewise instead.",
              input_shape, output_shape);

        const auto axes_empty_or_to_reduce = output_shape.cmp_eq(1) or axes_to_reduce;
        if (axes_empty_or_to_reduce.pop_front() == true) { // reduce to one value or one value per batch
            const auto batches = safe_cast<Index>(output_shape[0]);
            const auto input_shape_iz = input_shape.template as_safe<isize>();
            const bool reduce_all = axes_empty_or_to_reduce[0];
            const auto n_elements_to_reduce_iz = reduce_all ? input_shape_iz.n_elements() : input_shape_iz.pop_front().n_elements();
            const auto n_elements_to_reduce = safe_cast<Index>(n_elements_to_reduce_iz);

            Vec<bool, 4> contiguity = nd::accessors_contiguity(input, input_shape);
            if (not reduce_all)
                contiguity[0] = true;

            auto output_1d = nd::reconfig_accessors<nd::AccessorConfig<1>{.filter = {0}}>(std::forward<Output>(output));

            constexpr auto SMALL_THRESHOLD = Config::block_work_size * 4;
            if (contiguity.pop_back() == true) {
                if (n_elements_to_reduce <= SMALL_THRESHOLD) {
                    details::launch_reduce_ewise_small_2d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), Shape{batches, n_elements_to_reduce}, stream
                    );
                } else {
                    details::launch_reduce_ewise_large_2d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), Shape{batches, n_elements_to_reduce}, stream
                    );
                }
            } else {
                if (n_elements_to_reduce <= SMALL_THRESHOLD) {
                    details::launch_reduce_ewise_small_4d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), input_shape, true, stream
                    );
                } else {
                    details::launch_reduce_ewise_large_4d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), input_shape, true, stream
                    );
                }
            }
            return;
        }

        const i32 nb_axes_to_reduce = sum(axes_to_reduce.template as<i32>());
        check(nb_axes_to_reduce == 1,
              "Reducing more than one axis at a time is currently limited to a reduction that would "
              "result in one value per batch, i.e. the DHW dimensions should empty after reduction. "
              "Got input_shape={}, output_shape={}, axes_to_reduce={}",
              input_shape, output_shape, axes_to_reduce);

        if (axes_to_reduce[3]) {
            details::launch_reduce_ewise_width<Config>(
                input_shape,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Reduced>(reduced),
                std::forward<Output>(output),
                stream
            );
        } else {
            details::launch_reduce_ewise_bdh<Config>(
                input_shape, axes_to_reduce,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Reduced>(reduced),
                std::forward<Output>(output),
                stream
            );
        }
    }
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
