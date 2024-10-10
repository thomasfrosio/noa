#pragma once

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

namespace noa::cuda::guts {
    template<typename EwiseConfig, u32 MaxVectorSize>
    struct EwiseConfig1dBlock {
        using interface = EwiseConfig::interface;
        static constexpr u32 block_size = EwiseConfig::block_size;
        static constexpr u32 vector_size = MaxVectorSize;
        static constexpr u32 n_elements_per_thread = max(EwiseConfig::n_elements_per_thread, MaxVectorSize);
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

    template<typename Config, typename Op, typename Index,
             typename Input, typename InputAlignedBuffer,
             typename Output, typename OutputAlignedBuffer>
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
            input_1d.for_each([=](auto& accessor){ accessor.offset_inplace(block_offset); });
            output_1d.for_each([=](auto& accessor){ accessor.offset_inplace(block_offset); });

            // Load the inputs.
            using ivec_t = vectorized_tuple_t<Input>;
            ivec_t vectorized_input[Config::n_elements_per_thread];
            block_load<Config::block_size, Config::n_elements_per_thread, InputAlignedBuffer>(
                input_1d, vectorized_input, threadIdx.x);

            // Call the operator, store the results in the output buffer.
            // This implies that the operator does not write to the input(s)
            // and does not read from the output(s).
            using ovec_t = vectorized_tuple_t<Output>;
            ovec_t vectorized_output[Config::n_elements_per_thread];
            for (Index i = 0; i < Config::n_elements_per_thread; ++i)
                Config::interface::call(op, vectorized_input[i], vectorized_output[i], 0);

            // Store the output values back to global memory.
            block_store<Config::block_size, Config::n_elements_per_thread, OutputAlignedBuffer>(
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
                if (ih < shape_hw[0] and iw < shape_hw[1])
                    Config::interface::call(op, input_2d, output_2d, ih, iw);
            }
        }
        Config::interface::final(op, thread_uid());
    }
}

#ifdef NOA_IS_OFFLINE
namespace noa::cuda::guts {
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
        using op_t = std::decay_t<Op>;
        constexpr size_t VEC_SIZE = maximum_allowed_aligned_buffer_size<ALIGNMENT, Input, Output>();
        using iv_t = to_aligned_buffer_t<Input, ALIGNMENT, VEC_SIZE>;
        using ov_t = to_aligned_buffer_t<Output, ALIGNMENT, VEC_SIZE>;
        constexpr bool VECTORIZE = is_vectorized<iv_t, ov_t>();

        constexpr auto to_2d = ng::AccessorConfig<2>{
            .enforce_contiguous = VECTORIZE,
            .enforce_restrict = false,
            .filter = {0, 3},
        };
        auto input_2d = ng::reconfig_accessors<to_2d>(std::forward<Input>(input));
        auto output_2d = ng::reconfig_accessors<to_2d>(std::forward<Output>(output));

        using config_t = EwiseConfig1dBlock<Config, VEC_SIZE>;
        const auto n_blocks_x = divide_up(n_elements, static_cast<Index>(config_t::block_work_size));
        const auto launch_config = LaunchConfig{
            .n_blocks = dim3(static_cast<u32>(n_blocks_x), batch, 1u),
            .n_threads = dim3(config_t::block_size, 1u, 1u),
        };

        if constexpr (VECTORIZE) {
            stream.enqueue(
                guts::ewise_2d_vectorized
                <config_t, op_t, Index, decltype(input_2d), iv_t, decltype(output_2d), ov_t>,
                launch_config, std::forward<Op>(op), std::move(input_2d), std::move(output_2d), n_elements);
        } else {
            stream.enqueue(
                guts::ewise_2d<config_t, op_t, decltype(input_2d), decltype(output_2d), Index>,
                launch_config, std::forward<Op>(op), std::move(input_2d), std::move(output_2d), n_elements);
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
        static_assert(is_power_of_2(BlockSize) and BlockSize <= Limits::MAX_THREADS);

        using interface = ng::EwiseInterface<ZipInput, ZipOutput>;
        static constexpr u32 block_size = BlockSize;
        static constexpr u32 n_elements_per_thread = ElementsPerThread;
        static constexpr bool enable_vectorization = EnableVectorization;
    };

    template<typename Config = EwiseConfig<>,
             typename Input, typename Output, typename Index, typename Op>
    requires (nt::tuple_of_accessor_or_empty<std::decay_t<Input>> and
              (nt::empty_tuple<std::decay_t<Output>> or nt::tuple_of_accessor_pure<std::decay_t<Output>>))
    void ewise(
        const Shape4<Index>& shape,
        Op&& op,
        Input&& input,
        Output&& output,
        Stream& stream
    ) {
        const Vec4<bool> is_contiguous =
            ni::is_contiguous(input, shape) and
            ni::is_contiguous(output, shape);

        using input_t = std::decay_t<Input>;
        using output_t = std::decay_t<Output>;
        using op_t = std::decay_t<Op>;

        // TODO fused contiguous dimensions together, e.g. 2d unbatched should be ok here
        if (is_contiguous[1] and is_contiguous[2]) {
            // 2d-like
            // If batches are not contiguous to each other, keep them separated in a different grid.y.
            const auto batch = is_contiguous[0] ? 1u : static_cast<u32>(shape[0]);
            const auto shape_i64 = shape.template as<i64>();
            const auto n_elements = safe_cast<Index>(
                is_contiguous[0] ? shape_i64.n_elements() : shape_i64.pop_front().n_elements());

            if constexpr (Config::enable_vectorization and
                          (nt::has_allow_vectorization_v<Op> or
                           (output_t::SIZE == 0 and ng::are_accessors_const<input_t>()))) {
                const auto shape_3d = Shape3<u32>{batch, 1, 1};
                size_t alignment = min(
                    min_address_alignment(input, shape_3d),
                    min_address_alignment(output, shape_3d)
                );
                if (alignment == 16) {
                    return guts::launch_ewise_2d<16, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch);
                } else if (alignment == 8) {
                    return guts::launch_ewise_2d<8, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch);
                } else if (alignment == 4) {
                    return guts::launch_ewise_2d<4, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch);
                } else if (alignment == 2) {
                    return guts::launch_ewise_2d<2, Config>(
                        std::forward<Op>(op),
                        std::forward<Input>(input),
                        std::forward<Output>(output),
                        stream, n_elements, batch);
                }
            }

            guts::launch_ewise_2d<1, Config>(
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                stream, n_elements, batch);

        } else {
            using config_t = guts::EwiseConfig2dBlock<Config>;
            const auto shape_u32 = shape.template as<u32>();
            const u32 n_blocks_x = divide_up(shape_u32[3], config_t::block_work_size_x);
            const u32 n_blocks_y = divide_up(shape_u32[2], config_t::block_work_size_y);
            const auto launch_config = LaunchConfig{
                .n_blocks = dim3(n_blocks_x * n_blocks_y, shape_u32[1], shape_u32[0]),
                .n_threads = dim3(config_t::block_size_x, config_t::block_size_y, 1u),
            };
            stream.enqueue(
                guts::ewise_4d<config_t, op_t, input_t, output_t, Index>,
                launch_config,
                std::forward<Op>(op),
                std::forward<Input>(input),
                std::forward<Output>(output),
                shape.filter(2, 3), n_blocks_x);
        }
    }
}
#endif

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
