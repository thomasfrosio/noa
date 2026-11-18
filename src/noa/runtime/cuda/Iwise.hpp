#pragma once
#include "noa/runtime/cuda/IncludeGuard.hpp"

#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/cuda/Block.hpp"
#include "noa/runtime/cuda/Constants.hpp"
#include "noa/runtime/cuda/Stream.hpp"
#include "noa/runtime/cuda/ComputeHandle.hpp"

namespace noa::gpu::details {
    template<typename Block, typename Index>
    struct IwiseBounds {
        static constexpr bool BOUND_CHECK_Z = Block::enable_bound_check ? Block::block_size_z > 1 or Block::n_elements_per_thread_z > 1 : false;
        static constexpr bool BOUND_CHECK_Y = Block::enable_bound_check ? Block::block_size_y > 1 or Block::n_elements_per_thread_y > 1 : false;
        static constexpr bool BOUND_CHECK_X = Block::enable_bound_check ? Block::block_size_x > 1 or Block::n_elements_per_thread_x > 1 : false;

        static constexpr usize NB_AXES_NEED_BOUND_CHECK =
            static_cast<usize>(BOUND_CHECK_Z) +
            static_cast<usize>(BOUND_CHECK_Y) +
            static_cast<usize>(BOUND_CHECK_X);

        template<usize N>
        constexpr NOA_FHD explicit IwiseBounds(const Shape<Index, N>& shape) {
            if constexpr (BOUND_CHECK_Z and N >= 3) {
                bounds[0] = shape[N - 3];
            }
            if constexpr (BOUND_CHECK_Y and N >= 2) {
                bounds[BOUND_CHECK_Z] = shape[N - 2];
            }
            if constexpr (BOUND_CHECK_X) {
                bounds[BOUND_CHECK_Z + BOUND_CHECK_Y] = shape[N - 1];
            }
        }

        // I: {z,y,x}
        template<usize I> requires (I < 3)
        constexpr NOA_FHD auto is_within_upper_bound(Index index) const -> bool {
            if constexpr (Block::enable_bound_check) {
                if constexpr (I == 0) {
                    if constexpr (BOUND_CHECK_Z) {
                        return index < bounds[0];
                    } else {
                        return true;
                    }
                } else if constexpr (I == 1) {
                    if constexpr (BOUND_CHECK_Y) {
                        return index < bounds[BOUND_CHECK_Z];
                    } else {
                        return true;
                    }
                } else if constexpr (I == 2) {
                    if constexpr (BOUND_CHECK_X) {
                        return index < bounds[BOUND_CHECK_Z + BOUND_CHECK_Y];
                    } else {
                        return true;
                    }
                }
            }
            return true;
        }

        NOA_NO_UNIQUE_ADDRESS Vec<Index, NB_AXES_NEED_BOUND_CHECK> bounds{};
    };

    template<usize N, typename Index, typename Block, typename Op>
    struct Iwise {
        static constexpr usize GRID_NDIM = N >= 3 ? 3 : N;
        using interface_type = noa::details::IwiseInterface;
        using compute_handle_type = ComputeHandle<Index, GRID_NDIM, Block, (N > 1), false, false>;
        using grid_fused_shape_type = std::conditional_t<(N > 3), Vec<u32, N - 3>, Empty>;

        NOA_NO_UNIQUE_ADDRESS Op op;
        NOA_NO_UNIQUE_ADDRESS IwiseBounds<Block, Index> bounds;
        NOA_NO_UNIQUE_ADDRESS grid_fused_shape_type grid_fused_shape_in_x_unbatched{};

        NOA_FD void operator()(const compute_handle_type& handle) requires (N > 3) {
            interface_type::init(handle, op);
            const auto gid = handle.template global_indices<Index, N>(grid_fused_shape_in_x_unbatched);
            for (Index d = 0; d < static_cast<Index>(Block::n_elements_per_thread_z); ++d) {
                for (Index h = 0; h < static_cast<Index>(Block::n_elements_per_thread_y); ++h) {
                    for (Index w = 0; w < static_cast<Index>(Block::n_elements_per_thread_x); ++w) {
                        const Index id = gid[N - 3] + static_cast<Index>(Block::block_size_z) * d;
                        const Index ih = gid[N - 2] + static_cast<Index>(Block::block_size_y) * h;
                        const Index iw = gid[N - 1] + static_cast<Index>(Block::block_size_x) * w;
                        if (bounds.template is_within_upper_bound<0>(id) and
                            bounds.template is_within_upper_bound<1>(ih) and
                            bounds.template is_within_upper_bound<2>(iw)) {
                            if constexpr (N == 4)
                                interface_type::call(handle, op, gid[0], id, ih, iw);
                            else if constexpr (N == 5)
                                interface_type::call(handle, op, gid[0], gid[1], id, ih, iw);
                            else if constexpr (N == 6)
                                interface_type::call(handle, op, gid[0], gid[1], gid[2], id, ih, iw);
                            else
                                static_assert(nt::always_false<Op>);
                        }
                    }
                }
            }
            interface_type::deinit(handle, op);
        }

        NOA_FD void operator()(const compute_handle_type& ci) requires (N == 3) {
            interface_type::init(ci, op);
            const auto gid = ci.template global_indices<Index, 3>();
            for (Index d = 0; d < static_cast<Index>(Block::n_elements_per_thread_z); ++d) {
                for (Index h = 0; h < static_cast<Index>(Block::n_elements_per_thread_y); ++h) {
                    for (Index w = 0; w < static_cast<Index>(Block::n_elements_per_thread_x); ++w) {
                        const Index id = gid[0] + static_cast<Index>(Block::block_size_z) * d;
                        const Index ih = gid[1] + static_cast<Index>(Block::block_size_y) * h;
                        const Index iw = gid[2] + static_cast<Index>(Block::block_size_x) * w;
                        if (bounds.template is_within_upper_bound<0>(id) and
                            bounds.template is_within_upper_bound<1>(ih) and
                            bounds.template is_within_upper_bound<2>(iw))
                            interface_type::call(ci, op, id, ih, iw);
                    }
                }
            }
            interface_type::deinit(ci, op);
        }

        NOA_FD void operator()(const compute_handle_type& ci) requires (N == 2) {
            interface_type::init(ci, op);
            const auto gid = ci.template global_indices<Index, 2>();
            for (Index h = 0; h < static_cast<Index>(Block::n_elements_per_thread_y); ++h) {
                for (Index w = 0; w < static_cast<Index>(Block::n_elements_per_thread_x); ++w) {
                    const Index ih = gid[0] + static_cast<Index>(Block::block_size_y) * h;
                    const Index iw = gid[1] + static_cast<Index>(Block::block_size_x) * w;
                    if (bounds.template is_within_upper_bound<1>(ih) and
                        bounds.template is_within_upper_bound<2>(iw))
                        interface_type::call(ci, op, ih, iw);
                }
            }
            interface_type::deinit(ci, op);
        }

        NOA_FD void operator()(const compute_handle_type& ci) requires (N == 1) {
            interface_type::init(ci, op);
            const auto gid = ci.template global_indices<Index, 1>();
            for (Index w = 0; w < static_cast<Index>(Block::n_elements_per_thread_x); ++w) {
                const Index iw = gid[0] + static_cast<Index>(Block::block_size_x) * w;
                if (bounds.template is_within_upper_bound<2>(iw))
                    interface_type::call(ci, op, iw);
            }
            interface_type::deinit(ci, op);
        }
    };
}

namespace noa::gpu {
    template<usize N>
    using IwiseDefaultConfig = std::conditional_t<
        N == 1,
        StaticBlock<Constant::WARP_SIZE * 8, 1, 1>,
        StaticBlock<Constant::WARP_SIZE, 256 / Constant::WARP_SIZE, 1>>;

    template<usize N, typename Block = IwiseDefaultConfig<N>, typename Index, typename Op>
    NOA_NOINLINE void iwise(
        const Shape<Index, N>& shape,
        Op&& op,
        Stream& stream,
        usize scratch_size = 0
    ) {
        static_assert(N >= Block::block_ndim, "Shape n-dimensions must be greater than thread-block n-dimensions");
        using iwise_t = details::Iwise<N, Index, Block, std::decay_t<Op>>;
        using handle_t = iwise_t::compute_handle_type;
        auto iwise_op = iwise_t{.op = NOA_FWD(op), .bounds = details::IwiseBounds<Block, Index>(shape)};

        if constexpr (N >= 4) {
            auto block_work_shape = Shape<u32, N>::from_value(1);
            block_work_shape[N - 3] = Block::block_work_size_z;
            block_work_shape[N - 2] = Block::block_work_size_y;
            block_work_shape[N - 1] = Block::block_work_size_x;
            auto grid = GridND(shape, block_work_shape);
            check(grid.n_launches_x() == 1);
            iwise_op.grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();

            for (u32 z{}; z < grid.n_launches_z(); ++z) {
                for (u32 y{}; y < grid.n_launches_y(); ++y) {
                    const auto config = LaunchConfig{
                        .grid_shape = grid.shape_for_launch(z, y, 0),
                        .block_shape = details::block_shape<Block>(),
                        .n_bytes_of_scratch = scratch_size,
                    };
                    auto handle = handle_t(grid.outer_shape().vec, grid.block_offset_for_launch(z, y));
                    stream.enqueue(config, handle, iwise_op);
                }
            }
        } else if constexpr (N == 3) {
            auto grid_x = GridX(shape[2], Block::block_work_size_x);
            auto grid_y = GridY(shape[1], Block::block_work_size_y);
            auto grid_z = GridZ(shape[0], Block::block_work_size_z);
            check(grid_x.n_launches() == 1);
            for (u32 z{}; z < grid_z.n_launches(); ++z) {
                for (u32 y{}; y < grid_y.n_launches(); ++y) {
                    const auto config = LaunchConfig{
                        .grid_shape = Vec{grid_z.n_blocks(z), grid_y.n_blocks(y), grid_x.n_blocks(0)},
                        .block_shape = details::block_shape<Block>(),
                        .n_bytes_of_scratch = scratch_size,
                    };
                    auto handle = handle_t(
                        Vec{grid_z.n_blocks_total(), grid_y.n_blocks_total()}.template as<u32>(),
                        Vec{grid_z.offset(z), grid_y.offset(y)}
                    );
                    stream.enqueue(config, handle, iwise_op);
                }
            }
        } else if constexpr (N == 2) {
            auto grid_x = GridX(shape[1], Block::block_work_size_x);
            auto grid_y = GridY(shape[0], Block::block_work_size_y);
            check(grid_x.n_launches() == 1);
            for (u32 y{}; y < grid_y.n_launches(); ++y) {
                const auto config = LaunchConfig{
                    .grid_shape = Vec{u32{1}, grid_y.n_blocks(y), grid_x.n_blocks(0)},
                    .block_shape = details::block_shape<Block>(),
                    .n_bytes_of_scratch = scratch_size,
                };
                auto handle = handle_t(
                    Vec{grid_y.n_blocks_total()}.template as<u32>(),
                    Vec{grid_y.offset(y)}
                );
                stream.enqueue(config, handle, iwise_op);
            }
        } else if constexpr (N == 1) {
            auto grid_x = GridX(shape[0], Block::block_work_size_x);
            check(grid_x.n_launches() == 1);
            const auto config = LaunchConfig{
                .grid_shape = Vec<u32, 3>{1, 1, grid_x.n_blocks(0)},
                .block_shape = details::block_shape<Block>(),
                .n_bytes_of_scratch = scratch_size,
            };
            stream.enqueue(config, handle_t{}, iwise_op);
        }
    }
}
