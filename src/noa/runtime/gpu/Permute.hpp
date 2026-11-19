#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/base/Math.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/gpu/Block.hpp"
#include "noa/runtime/gpu/ComputeHandle.hpp"
#include "noa/runtime/gpu/Stream.hpp"

// Logic from:
// https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
// https://forums.developer.nvidia.com/t/efficient-in-place-transpose-of-multiple-square-float-matrices/34327/3
// https://www.aldapa.eus/res/cuTranspose/Readme.html
// Other transpositions are very similar to this one, but the tile might be in XZ along Y as opposed to XY along Z.
// Reads and writes to global memory should coalesce and there should be no shared memory bank conflict.

namespace noa::gpu::details {
    struct PermuteConfig {
        static constexpr u32 tile_size = Constant::WARP_SIZE;
        static constexpr u32 tile_stride = tile_size + 1; // +1 so that elements in a column map to different banks
        static constexpr u32 tile_scratch_size_single = tile_size * tile_stride;
        static constexpr u32 tile_scratch_size_outplace = tile_scratch_size_single;
        static constexpr u32 tile_scratch_size_inplace = 2 * tile_scratch_size_single;

        static constexpr u32 block_size = 256;
        static constexpr u32 block_size_x = tile_size;
        static constexpr u32 block_size_y = block_size / block_size_x;
        static constexpr u32 block_size_z = 1;
        static constexpr u32 block_work_size_x = tile_size;
        static constexpr u32 block_work_size_y = tile_size;
        static constexpr u32 block_work_size_z = 1;
        static constexpr u32 block_ndim = 2;
    };

    template<usize N>
    using permute_handle_type = ComputeHandle<u32, (N >= 3 ? 3 : 2), PermuteConfig, true, false, false>;

    // Out-of-place permutation of the two rightmost axes (height and width).
    // This is used to permute any axis to the rightmost and the rightmost/innermost axis to the second rightmost.
    template<bool IS_STATIC_SCRATCH_ENABLED, bool IS_MULTIPLE_OF_TILE, typename T, usize N>
    struct PermuteHW {
        AccessorRestrict<const T, N, u32> input;
        AccessorRestrict<T, N, u32> output;
        Shape<u32, 2> shape_hw;
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched;

        NOA_FD void permute(const permute_handle_type<N>& handle, T* tile) const {
            const auto [gid_batches, gid_hw] = handle.template global_indices<u32, N, false>(
                grid_fused_shape_in_x_unbatched).template split<N - 2>();
            const auto tid = handle.template thread_indices<u32, 2>();

            // Read tile to shared memory.
            const auto input_2d = input[gid_batches];
            const auto input_gid = gid_hw + tid;
            for (u32 repeat = 0; repeat < PermuteConfig::tile_size; repeat += PermuteConfig::block_size_y) {
                const u32 gy = input_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE or (input_gid[1] < shape_hw[1] and gy < shape_hw[0])) // x could be checked earlier
                    tile[(tid[0] + repeat) * PermuteConfig::tile_stride + tid[1]] = input_2d(gy, input_gid[1]);
            }

            handle.block_synchronize();

            // Write permuted tile to global memory.
            const auto output_hw = output[gid_batches];
            const auto output_gid = gid_hw.flip() + tid; // Y->X', X->Y'
            for (u32 repeat = 0; repeat < PermuteConfig::tile_size; repeat += PermuteConfig::block_size_y) {
                const u32 gy = output_gid[0] + repeat;
                if (IS_MULTIPLE_OF_TILE or (output_gid[1] < shape_hw[0] and gy < shape_hw[1]))
                    output_hw(gy, output_gid[1]) = tile[tid[1] * PermuteConfig::tile_stride + tid[0] + repeat];
            }
        }

        NOA_FD void operator()(const permute_handle_type<N>& handle) const {
            if constexpr (IS_STATIC_SCRATCH_ENABLED) {
                #ifdef NOA_ENABLE_SYCL
                static_assert(nt::always_false<T>); // TODO
                #else
                __shared__ Uninitialized<T> tile[PermuteConfig::tile_scratch_size_outplace];
                permute(handle, reinterpret_cast<T*>(tile));
                #endif
            } else {
                permute(handle, handle.template scratch_pointer<T>());
            }
        }
    };

    template<bool IS_STATIC_SCRATCH_ENABLED, bool IS_MULTIPLE_OF_TILE, typename T, usize N>
    struct PermuteInplaceHW {
        AccessorRestrict<T, N, u32> output;
        u32 size;
        Vec<u32, (N <= 3 ? 0 : N - 3)> grid_fused_shape_in_x_unbatched;

    public:
        NOA_FD void permute(const permute_handle_type<N>& handle, T* tile_src, T* tile_dst) const {
            const auto [gid_batches, gid_hw] = handle.template global_indices<u32, N, false>(
                grid_fused_shape_in_x_unbatched).template split<N - 2>();

            const auto tid = handle.template thread_indices<u32, 2>();
            const auto output_hw = output[gid_batches];

            if (gid_hw[0] > gid_hw[1]) { // lower triangle
                const auto input_gid = gid_hw + tid;
                const auto output_gid = gid_hw.flip() + tid; // Y->X', X->Y'

                // Read tiles to shared memory.
                for (u32 repeat = 0; repeat < PermuteConfig::tile_size; repeat += PermuteConfig::block_size_y) {
                    const u32 iy = input_gid[0] + repeat;
                    if (IS_MULTIPLE_OF_TILE or (input_gid[1] < size and iy < size))
                        tile_src[(tid[0] + repeat) * PermuteConfig::tile_stride + tid[1]] = output_hw(iy, input_gid[1]);

                    const u32 oy = output_gid[0] + repeat;
                    if (IS_MULTIPLE_OF_TILE or (output_gid[1] < size and oy < size))
                        tile_dst[(tid[0] + repeat) * PermuteConfig::tile_stride + tid[1]] = output_hw(oy, output_gid[1]);
                }

                handle.block_synchronize();

                // Write permuted tiles to global memory.
                for (u32 repeat = 0; repeat < PermuteConfig::tile_size; repeat += PermuteConfig::block_size_y) {
                    const u32 oy = output_gid[0] + repeat;
                    if (IS_MULTIPLE_OF_TILE or (output_gid[1] < size and oy < size))
                        output_hw(oy, output_gid[1]) = tile_src[tid[1] * PermuteConfig::tile_stride + tid[0] + repeat];

                    const u32 iy = input_gid[0] + repeat;
                    if (IS_MULTIPLE_OF_TILE or (input_gid[1] < size and iy < size))
                        output_hw(iy, input_gid[1]) = tile_dst[tid[1] * PermuteConfig::tile_stride + tid[0] + repeat];
                }

            } else if (gid_hw[0] == gid_hw[1]) { // diagonal
                const auto gid = gid_hw + tid;

                // Read tile to shared memory.
                for (u32 repeat = 0; repeat < PermuteConfig::tile_size; repeat += PermuteConfig::block_size_y) {
                    const u32 gy = gid[0] + repeat;
                    if (IS_MULTIPLE_OF_TILE or (gid[1] < size and gy < size))
                        tile_src[(tid[0] + repeat) * PermuteConfig::tile_stride + tid[1]] = output_hw(gy, gid[1]);
                }

                handle.block_synchronize();

                // Write permuted tile to global memory.
                for (u32 repeat = 0; repeat < PermuteConfig::tile_size; repeat += PermuteConfig::block_size_y) {
                    const u32 gy = gid[0] + repeat;
                    if (IS_MULTIPLE_OF_TILE or (gid[1] < size and gy < size))
                        output_hw(gy, gid[1]) = tile_src[tid[1] * PermuteConfig::tile_stride + tid[0] + repeat];
                }
            }
        }

        NOA_FD void operator()(const permute_handle_type<N>& handle) const {
            if constexpr (IS_STATIC_SCRATCH_ENABLED) {
                #ifdef NOA_ENABLE_SYCL
                static_assert(nt::always_false<T>); // TODO
                #else
                __shared__ Uninitialized<T> tile_src[PermuteConfig::tile_scratch_size_single];
                __shared__ Uninitialized<T> tile_dst[PermuteConfig::tile_scratch_size_single];
                permute(handle, reinterpret_cast<T*>(tile_src), reinterpret_cast<T*>(tile_dst));
                #endif
            } else {
                T* tile_src = handle.template scratch_pointer<T>();
                T* tile_dst = tile_src + PermuteConfig::tile_scratch_size_single;
                permute(handle, tile_src, tile_dst);
            }
        }
    };
}

namespace noa::gpu {
    template<typename T, usize N> requires (N >= 2)
    void permute_copy_hw(
        const T* input, const Strides<isize, N>& input_strides, const Shape<isize, N>& input_shape,
        T* output, const Strides<isize, N>& output_strides, Stream& stream
    ) {
        using Config = details::PermuteConfig;
        using Handle = details::permute_handle_type<N>;

        const auto shape_2d = input_shape.filter(N - 2, N - 1).template as_safe<u32>();
        const bool is_inplace = input == output;
        const bool are_multiple_tile =
            noa::is_multiple_of(shape_2d[0], Config::tile_size) and
            noa::is_multiple_of(shape_2d[1], Config::tile_size);

        const auto input_accessor = AccessorRestrict<const T, N, u32>(input, input_strides.template as_safe<u32>());
        const auto output_accessor = AccessorRestrict<T, N, u32>(output, output_strides.template as_safe<u32>());

        auto tile_shape = Shape<u32, N>::from_value(1);
        tile_shape[N - 2] = Config::tile_size;
        tile_shape[N - 1] = Config::tile_size;
        const auto grid = GridND(input_shape, tile_shape);
        const auto grid_fused_shape_in_x_unbatched = grid.fused_shape().vec.pop_front();
        check(grid.n_launches() == 1);

        for (u32 z{}; z < grid.n_launches_z(); ++z) {
            for (u32 y{}; y < grid.n_launches_y(); ++y) {
                constexpr bool IS_STATIC_SCRATCH_ENABLED = is_static_scratch_enabled();
                const auto handle = Handle(grid.outer_shape().vec, grid.block_offset_for_launch(z, y)); // TODO outer shape is not used
                const auto config = LaunchConfig{
                    .grid_shape = grid.shape_for_launch(z, y, 0), // for inplace about less than half will be no-op blocks...
                    .block_shape = details::block_shape<Config>(),
                    .n_bytes_of_scratch = IS_STATIC_SCRATCH_ENABLED ? 0 :
                        sizeof(T) * (is_inplace ? Config::tile_scratch_size_inplace : Config::tile_scratch_size_outplace),
                };

                if (are_multiple_tile) {
                    if (is_inplace) {
                        using Kernel = details::PermuteInplaceHW<IS_STATIC_SCRATCH_ENABLED, true, T, N>;
                        auto kernel = Kernel{output_accessor, shape_2d[1], grid_fused_shape_in_x_unbatched};
                        stream.enqueue(config, handle, kernel);
                    } else {
                        using Kernel = details::PermuteHW<IS_STATIC_SCRATCH_ENABLED, true, T, N>;
                        auto kernel = Kernel{input_accessor, output_accessor, shape_2d, grid_fused_shape_in_x_unbatched};
                        stream.enqueue(config, handle, kernel);
                    }
                } else {
                    if (is_inplace) {
                        using Kernel = details::PermuteInplaceHW<IS_STATIC_SCRATCH_ENABLED, false, T, N>;
                        auto kernel = Kernel{output_accessor, shape_2d[1], grid_fused_shape_in_x_unbatched};
                        stream.enqueue(config, handle, kernel);
                    } else {
                        using Kernel = details::PermuteHW<IS_STATIC_SCRATCH_ENABLED, false, T, N>;
                        auto kernel = Kernel{input_accessor, output_accessor, shape_2d, grid_fused_shape_in_x_unbatched};
                        stream.enqueue(config, handle, kernel);
                    }
                }
            }
        }
    }
}
