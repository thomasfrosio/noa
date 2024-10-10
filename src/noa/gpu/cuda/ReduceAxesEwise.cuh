#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Interfaces.hpp"
#include "noa/gpu/cuda/Block.cuh"
#include "noa/gpu/cuda/ReduceEwise.cuh"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(push, 0)
#endif

namespace noa::cuda::guts {
    template<typename Config, u32 BlockSizeX, u32 MaxVectorSize>
    struct ReduceAxesEwiseWidthConfig {
        using interface = Config::interface;
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 n_elements_per_thread_x = max(MaxVectorSize, Config::n_elements_per_thread);
        static constexpr u32 block_work_size_x = block_size_x * n_elements_per_thread_x;
    };

    template<typename Config, typename Op, typename Index,
             typename Input, typename InputAlignedBuffer, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_width_ewise(Op op, Input input, Shape2<Index> shape_hw, Reduced reduced, Output output) {
        const auto tid = Vec2<Index>::from_values(threadIdx.y, threadIdx.x);
        const auto gid = Vec4<Index>::from_values(
            blockIdx.z,
            blockIdx.y,
            blockIdx.x * blockDim.y + tid[0],
            tid[1]);
        const bool is_valid_row = gid[2] < shape_hw[0];

        auto input_row = std::move(input).map([&gid]<typename T>(T&& accessor) {
            if constexpr (nt::is_accessor_value_v<T>)
                return std::forward<T>(accessor);
            else
                return accessor[gid[0]][gid[1]][gid[2]];
        });

        // Initial reduction. Loop until the end of the row is reached.
        constexpr auto OFFSET = Config::block_work_size_x;
        for (Index cid = 0; cid < shape_hw[1] and is_valid_row; cid += OFFSET) {
            block_reduce_ewise_1d_init
                <Config::block_size_x, Config::n_elements_per_thread_x, InputAlignedBuffer, Config::interface>
                (op, input_row, shape_hw[1] - cid, reduced, tid[1]);

            // Offset to the next work space.
            input_row.for_each([](auto& accessor) { accessor.offset_inplace(OFFSET); });
        }

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Config::block_size];
        Reduced* joined = shared_buffer + tid[0] * Config::block_size_x;
        joined[tid[1]] = reduced;
        block_synchronize();

        // Reduce shared data to one element.
        reduced = block_reduce_shared<Config::interface, Config::block_size_x>(op, joined, tid[1]);
        if (gid[3] == 0 and is_valid_row)
            Config::interface::final(op, reduced, output, gid[0], gid[1], gid[2]);
    }

    template<typename Config, u32 BlockSizeX>
    struct ReduceAxesEwiseHeightConfig {
        using interface = Config::interface;
        static constexpr u32 block_size = max(BlockSizeX, Config::block_size);
        static constexpr u32 block_size_x = BlockSizeX;
        static constexpr u32 block_size_y = block_size / block_size_x;
    };

    template<typename Config, typename Op, typename Index, typename Input, typename Reduced, typename Output>
    __global__ __launch_bounds__(Config::block_size)
    void reduce_height_ewise(Op op, Input input, Shape2<Index> shape_hw, Reduced reduced, Output output) {
        const auto gid = Vec4<Index>::from_values(
            blockIdx.z,
            blockIdx.y,
            threadIdx.y, // one block along the height
            blockIdx.x * Config::block_size_x + threadIdx.x);
        const bool is_valid_column = gid[3] < shape_hw[1];

        // Process every row.
        input.for_each([&](auto& accessor) { accessor.offset_inplace(gid[0], gid[1]); });
        for (Index tidy = gid[2]; tidy < shape_hw[0] and is_valid_column; tidy += Config::block_size_y)
            Config::interface::init(op, input, reduced, 0, 0, tidy, gid[3]);

        // Share the threads' initial reduction with the rest of the block.
        __shared__ Reduced shared_buffer[Config::block_size];
        Reduced* joined = shared_buffer + threadIdx.y * Config::block_size_x + threadIdx.x;
        *joined = reduced;
        block_synchronize();

        // Reduce the height of the block.
        #pragma unroll
        for (u32 size = Config::block_size_y; size >= 2; size /= 2) {
            if (threadIdx.y < size / 2)
                Config::interface::join(op, joined[Config::block_size_x * size / 2], *joined);
            block_synchronize();
        }

        if (threadIdx.y == 0 and is_valid_column)
            Config::interface::final(op, *joined, output, gid[0], gid[1], gid[3]);
    }
}

#ifdef NOA_IS_OFFLINE
namespace noa::cuda::guts {
    template<size_t BLOCK_SIZE_X, size_t ALIGNMENT, typename Config,
             typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_width_(
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream,
        const Shape2<Index>& shape_hw,
        LaunchConfig launch_config
    ) {
        // The width of the output is empty/reduced, remove it.
        constexpr auto to_3d = ng::AccessorConfig<3>{.filter={0, 1, 2}};
        auto output_3d = ng::reconfig_accessors<to_3d>(std::forward<Output>(output));
        using output_3d_t = decltype(output_3d);

        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input>();
        using iv_t = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        constexpr auto to_4d = ng::AccessorConfig<0>{.enforce_contiguous = is_vectorized<iv_t>()};
        auto input_4d = ng::reconfig_accessors<to_4d>(std::forward<Input>(input));

        using op_t = std::decay_t<Op>;
        using reduced_t = std::decay_t<Reduced>;
        using config_t = ReduceAxesEwiseWidthConfig<Config, BLOCK_SIZE_X, VEC_SIZE>;
        stream.enqueue(
            reduce_width_ewise<config_t, op_t, Index, decltype(input_4d), iv_t, reduced_t, output_3d_t>,
            launch_config, std::forward<Op>(op), std::move(input_4d), shape_hw,
            std::forward<Reduced>(reduced), output_3d
        );
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_width(
        const Shape4<Index>& shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        const auto shape_u32 = shape.template as_safe<u32>();
        const auto shape_hw = shape.filter(2, 3);
        u32 n_threads_x = shape_u32[3] > 512 ? 256u : 64u;
        if (not is_multiple_of(Config::block_size, n_threads_x))
            n_threads_x = Constant::WARP_SIZE;

        const u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
        const u32 n_blocks_x = divide_up(shape_u32[2], n_threads_y);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x, shape_u32[1], shape_u32[0]),
            .n_threads = dim3(n_threads_x, n_threads_y),
        };

        // TODO Reduce the number of instantiations by adding support of a runtime block_size_x.
        //      That would mean using dynamic shared memory in block_reduce, which is fine...
        // Note that these if statements are likely to instantiate the same kernel. For instance, if the smallest
        // type has an alignment of 4, only 6 kernels are created (3 with n_threads_x=256, 3 with n_threads_x=64).
        // The worst case is when the smallest type is aligned to only one byte, in which case 10 kernels are created.
        if constexpr (Config::enable_vectorization and
                      (nt::has_allow_vectorization_v<Op> or ng::are_accessors_const<std::decay_t<Input>>())) {
            const size_t alignment = min_address_alignment(input, shape_u32.pop_back());
            if (n_threads_x == 256) {
                if (alignment == 16) {
                    launch_reduce_ewise_width_<256, 16, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else if (alignment == 8) {
                    launch_reduce_ewise_width_<256, 8, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else if (alignment == 4) {
                    launch_reduce_ewise_width_<256, 4, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else if (alignment == 2) {
                    launch_reduce_ewise_width_<256, 2, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else {
                    launch_reduce_ewise_width_<256, 1, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                }
            } else {
                if (alignment == 16) {
                    launch_reduce_ewise_width_<64, 16, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else if (alignment == 8) {
                    launch_reduce_ewise_width_<64, 8, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else if (alignment == 4) {
                    launch_reduce_ewise_width_<64, 4, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else if (alignment == 2) {
                    launch_reduce_ewise_width_<64, 2, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                } else {
                    launch_reduce_ewise_width_<64, 1, Config>(
                        op, std::forward<Input>(input),
                        std::forward<Reduced>(reduced),
                        std::forward<Output>(output),
                        stream, shape_hw, launch_config);
                }
            }
        } else {
            if (n_threads_x == 256) {
                launch_reduce_ewise_width_<256, 1, Config>(
                    op, std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    stream, shape_hw, launch_config);
            } else {
                launch_reduce_ewise_width_<64, 1, Config>(
                    op, std::forward<Input>(input),
                    std::forward<Reduced>(reduced),
                    std::forward<Output>(output),
                    stream, shape_hw, launch_config);
            }
        }
    }

    template<typename Config, typename Op, typename Input, typename Reduced, typename Output, typename Index>
    void launch_reduce_ewise_bdh(
        const Shape4<Index>& shape,
        const Vec4<bool>& axes_to_reduce,
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
        auto order = ni::squeeze_left(axes_to_reduce.template as<i32>() + 1);
        order = order.filter(0, 1, 3, 2); // move the width back to rightmost

        // Reorder to (X, X, axis_to_reduce, width).
        auto reordered_shape = shape.reorder(order);
        input_.for_each([&order](auto& accessor) { accessor.reorder(order); });
        output_.for_each([&order](auto& accessor) { accessor.reorder(order); });

        // Remove the empty/reduced axis from the output.
        constexpr auto to_3d = ng::AccessorConfig<3>{.filter = {0, 1, 3}};
        auto output_3d = ng::reconfig_accessors<to_3d>(std::move(output_));

        // Launch config.
        const auto reordered_shape_u32 = reordered_shape.template as<u32>();
        constexpr u32 n_threads_x = Constant::WARP_SIZE;
        constexpr u32 n_threads_y = max(Config::block_size, n_threads_x) / n_threads_x;
        const u32 n_blocks_x = divide_up(reordered_shape_u32[3], n_threads_x);
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(n_blocks_x, reordered_shape_u32[1], reordered_shape_u32[0]),
            .n_threads = dim3(n_threads_x, n_threads_y),
        };

        using op_t = std::decay_t<Op>;
        using input_t = std::decay_t<Input>;
        using reduced_t = std::decay_t<Reduced>;
        using output_3d_t = decltype(output_3d);
        using kernel_config = ReduceAxesEwiseHeightConfig<Config, n_threads_x>;
        stream.enqueue(
            reduce_height_ewise<kernel_config, op_t, Index, input_t, reduced_t, output_3d_t>,
            launch_config, std::forward<Op>(op), std::move(input_), reordered_shape.template pop_front<2>(),
            std::forward<Reduced>(reduced), std::move(output_3d)
        );
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

        using interface = ng::ReduceEwiseInterface<ZipInput, ZipReduced, ZipOutput>;
        static constexpr u32 max_grid_size = MaxGridSize;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
        static constexpr u32 block_work_size = block_size * n_elements_per_thread;
        static constexpr bool enable_vectorization = EnableVectorization;
    };

    template<typename Config = ReduceAxesEwiseConfig<>,
             typename Op, typename Input, typename Reduced, typename Output, typename Index>
    requires (nt::tuple_of_accessor_nd<std::decay_t<Input>, 4> and
              nt::tuple_of_accessor_pure<std::decay_t<Output>> and
              nt::tuple_of_accessor_nd<std::decay_t<Output>, 4> and
              not nt::tuple_of_accessor_value<std::decay_t<Input>> and // at least one varray
              nt::tuple_of_accessor_value<std::decay_t<Reduced>>)
    constexpr void reduce_axes_ewise(
        const Shape4<Index>& input_shape,
        const Shape4<Index>& output_shape,
        Op&& op,
        Input&& input,
        Reduced&& reduced,
        Output&& output,
        Stream& stream
    ) {
        const auto axes_to_reduce = input_shape != output_shape;
        if (any(axes_to_reduce and (output_shape != 1))) {
            panic("Dimensions should match the input shape, or be 1, "
                  "indicating the dimension should be reduced to one element. "
                  "Got shape input={}, output={}", input_shape, output_shape);
        } else if (all(axes_to_reduce == false)) {
            panic("No reduction to compute. Got shape input={}, output={}. Please use ewise instead.",
                input_shape, output_shape);
        }

        const auto axes_empty_or_to_reduce = output_shape == 1 or axes_to_reduce;
        if (all(axes_empty_or_to_reduce.pop_front())) { // reduce to one value or one value per batch
            const auto n_batches = output_shape[0];
            const auto input_shape_i64 = input_shape.template as<i64>();
            const bool reduce_all = axes_empty_or_to_reduce[0];
            const auto n_elements_to_reduce = safe_cast<Index>(
                reduce_all ? input_shape_i64.n_elements() : input_shape_i64.pop_front().n_elements());

            Vec<bool, 4> is_contiguous = ni::is_contiguous(input, input_shape);
            if (not reduce_all)
                is_contiguous[0] = true;

            auto output_1d = ng::reconfig_accessors<ng::AccessorConfig<1>{.filter = {0}}>(std::forward<Output>(output));

            constexpr auto SMALL_THRESHOLD = Config::block_work_size * 4;
            if (all(is_contiguous.pop_back())) {
                if (n_elements_to_reduce <= SMALL_THRESHOLD) {
                    guts::launch_reduce_ewise_small_2d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), Shape2<Index>{n_batches, n_elements_to_reduce}, stream);
                } else {
                    guts::launch_reduce_ewise_large_2d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), Shape2<Index>{n_batches, n_elements_to_reduce}, stream);
                }
            } else {
                if (n_elements_to_reduce <= SMALL_THRESHOLD) {
                    guts::launch_reduce_ewise_small_4d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), input_shape, true, stream);
                } else {
                    guts::launch_reduce_ewise_large_4d<Config>(
                        std::forward<Op>(op), std::forward<Input>(input), std::forward<Reduced>(reduced),
                        std::move(output_1d), input_shape, true, stream);
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
            guts::launch_reduce_ewise_width<Config>(
                input_shape,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Reduced>(reduced),
                std::forward<Output>(output),
                stream);
        } else {
            guts::launch_reduce_ewise_bdh<Config>(
                input_shape, axes_to_reduce,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Reduced>(reduced),
                std::forward<Output>(output),
                stream);
        }
    }
}
#endif

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
