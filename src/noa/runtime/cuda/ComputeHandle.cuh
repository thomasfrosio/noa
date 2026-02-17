#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/base/Complex.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/cuda/Utils.cuh"

namespace noa::cuda::details {
    template<typename Index,
             u32 GridNDim, u32 BlockNDim,
             bool IsMultiGridKernel,
             bool IsUsingDynamicSharedMemory,
             bool IsTwoPartReduction>
    struct ComputeHandle {
        using vec_zy_type = std::conditional_t<IsMultiGridKernel, Vec<u32, GridNDim - 1>, Empty>;
        using scratch_type = std::conditional_t<IsUsingDynamicSharedMemory, u32, Empty>;
        using index_type = Index;

    public:
        static constexpr NOA_FD auto is_cpu() -> bool { return false; }
        static constexpr NOA_FD auto is_gpu() -> bool { return true; }

    public:
        NOA_FD explicit ComputeHandle() requires (not IsUsingDynamicSharedMemory and not IsMultiGridKernel) = default;

        NOA_FD explicit ComputeHandle(u32 scratch_size) requires (IsUsingDynamicSharedMemory and not IsMultiGridKernel)
            : m_scratch_size{scratch_size} {}

        NOA_FD explicit ComputeHandle(vec_zy_type grid_dim_zy, vec_zy_type block_idx_offset_zy) requires (not IsUsingDynamicSharedMemory and IsMultiGridKernel)
            : m_grid_size_zy{grid_dim_zy}, m_block_index_offset_zy{block_idx_offset_zy} {}

        NOA_FD explicit ComputeHandle(u32 scratch_size, vec_zy_type grid_dim_zy, vec_zy_type block_idx_offset_zy) requires (IsUsingDynamicSharedMemory and IsMultiGridKernel)
            : m_scratch_size{scratch_size}, m_grid_size_zy{grid_dim_zy}, m_block_index_offset_zy{block_idx_offset_zy} {}

    public:
        struct Grid {
            using index_type = Index;
            const ComputeHandle& handle;

            template<typename T = index_type>
            static constexpr NOA_FD auto ndim() -> T { return static_cast<T>(GridNDim); }

            template<typename T, usize N> requires (1 <= N and N <= 3)
            NOA_FD auto shape() const -> Shape<T, N> {
                return handle.template grid_shape<T, N>();
            }

            template<typename T = Index>
            NOA_FD auto size() const -> T {
                return handle.template grid_size<T>();
            }

            template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
            static NOA_FD auto atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
                auto pointer = accessor.offset_pointer(accessor.get(), indices...);
                ::noa::cuda::details::atomic_add(pointer, value);
            }

            static NOA_FD auto is_two_part_reduction() -> bool { return IsTwoPartReduction; }
        };

        struct Block {
            using index_type = Index;
            const ComputeHandle& handle;

        public:
            [[nodiscard]] static NOA_FD auto has_scratch() -> bool {
                return dynamic_shared_memory_size() > 0;
            }

            template<typename T = std::byte, typename I = index_type>
            [[nodiscard]] NOA_FD auto scratch() const -> SpanContiguous<T, 1, I> {
                return handle.template block_scratch<T, I>();
            }

            template<typename T = std::byte>
            [[nodiscard]] static NOA_FD auto scratch_pointer() -> T* {
                return dynamic_shared_memory_pointer<T>();
            }

            template<typename T = std::byte, typename I = index_type>
            NOA_FD auto zeroed_scratch() const -> SpanContiguous<T, 1, I> {
                return handle.template block_zeroed_scratch<T, I>();
            }

            template<typename T = index_type>
            NOA_FD auto lid() const -> T {
                return handle.template block_lid<T>();
            }

            template<typename T = index_type, usize N = BlockNDim> requires (1 <= N and N <= 3)
            NOA_FD auto id() const -> Vec<T, N> {
                return handle.template block_id<T, N>();
            }

            template<typename T, usize N> requires (1 <= N and N <= 3)
            static NOA_FD auto shape() -> Shape<T, N> {
                return block_shape<T, N>();
            }

            template<typename T = index_type>
            static NOA_FD auto size() -> T {
                return block_size<T>();
            }

            static constexpr NOA_FD auto ndim() -> Index { return static_cast<Index>(BlockNDim); }

            static NOA_FD void synchronize() {
                __syncthreads();
            }

            template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
            static NOA_FD auto atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
                auto pointer = accessor.offset_pointer(accessor.get(), indices...);
                ::noa::cuda::details::atomic_add(pointer, value);
            }
        };

        struct Thread {
            using index_type = Index;
            const ComputeHandle& handle;

            template<typename T = index_type>
            static NOA_FD auto lid() -> T {
                return thread_lid<T>();
            }

            template<typename T, usize N> requires (1 <= N and N <= 3)
            static NOA_FD auto id() -> Vec<T, N> {
                return thread_id<T, N>();
            }

            template<typename T = index_type>
            NOA_FD auto gid() const -> T {
                return handle.template thread_gid<T>();
            }
        };

        NOA_FD auto grid() const -> Grid { return Grid{*this}; }
        NOA_FD auto block() const -> Block { return Block{*this}; }
        NOA_FD auto thread() const -> Thread { return Thread{*this}; }

    private: // implementation
        template<i32 i>
        NOA_FD auto get_grid_size() const -> u32 {
            if constexpr (IsMultiGridKernel)
                return m_grid_size_zy[i];
            else
                return 0;
        }

        template<i32 i>
        NOA_FD auto get_block_index_offset() const -> u32 {
            if constexpr (IsMultiGridKernel)
                return m_block_index_offset_zy[i];
            else
                return 0;
        }

        template<typename T, usize N> requires (1 <= N and N <= 3)
        NOA_FD auto grid_shape() const -> Shape<T, N> {
            if constexpr (N == 1) {
                return Shape<T, N>::from_values(gridDim.x);

            } else if constexpr (N == 2) {
                if constexpr (GridNDim == 1)
                    return Shape<T, N>::from_values(0, gridDim.x);
                else if constexpr (GridNDim == 2)
                    return Shape<T, N>::from_values(get_grid_size<0>(), gridDim.x);
                else
                    return Shape<T, N>::from_values(get_grid_size<1>(), gridDim.x);

            } else if constexpr (N == 3) {
                if constexpr (GridNDim == 1)
                    return Shape<T, N>::from_values(0, 0, gridDim.x);
                else if constexpr (GridNDim == 2)
                    return Shape<T, N>::from_values(0, get_grid_size<0>(), gridDim.x);
                else
                    return Shape<T, N>::from_values(get_grid_size<0>(), get_grid_size<1>(), gridDim.x);

            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T>
        NOA_FD auto grid_size() -> T {
            u32 out = gridDim.x;
            if constexpr (GridNDim > 1)
                out *= get_grid_size<GridNDim - 2>();
            if constexpr (GridNDim == 3)
                out *= get_grid_size<0>();
            return static_cast<T>(out);
        }

        template<typename T, usize N> requires (1 <= N and N <= 3)
        static NOA_FD auto block_shape() -> Shape<T, N> {
            if constexpr (N == 1)
                return Shape<T, N>::from_values(blockDim.x);
            else if constexpr (N == 2)
                return Shape<T, N>::from_values(blockDim.y, blockDim.y);
            else if constexpr (N == 3)
                return Shape<T, N>::from_values(blockDim.z, blockDim.y, blockDim.x);
            else
                static_assert(nt::always_false<T>);
        }

        template<typename T>
        static NOA_FD auto block_size() -> T {
            if constexpr (BlockNDim == 1)
                return static_cast<T>(blockDim.x);
            else if constexpr (BlockNDim == 2)
                return static_cast<T>(blockDim.y * blockDim.x);
            else if constexpr (BlockNDim == 3)
                return static_cast<T>(blockDim.z * blockDim.y * blockDim.x);
            else
                static_assert(nt::always_false<T>);
        }

        template<typename T>
        NOA_FD auto block_lid() -> T {
            if constexpr (GridNDim == 1) {
                return static_cast<T>(blockIdx.x);
            } else if constexpr (GridNDim == 2) {
                const auto bid_y = blockIdx.y + get_block_index_offset<0>();
                return static_cast<T>(bid_y * gridDim.x + blockIdx.x);
            } else if constexpr (GridNDim == 3) {
                const auto bid_y = blockIdx.y + get_block_index_offset<1>();
                const auto bid_z = blockIdx.z + get_block_index_offset<0>();
                return static_cast<T>((bid_z * get_grid_size<1>() + bid_y) * gridDim.x + blockIdx.x);
            } else {
                static_assert(nt::always_false<T>);
            }
        }

        template<typename T, usize N> requires (1 <= N and N <= 3)
        NOA_FD auto block_id() -> Vec<T, N> {
            if constexpr (N == 1) {
                return Shape<T, N>::from_values(blockIdx.x);
            } else if constexpr (N == 2) {
                if constexpr (GridNDim == 1)
                    return Shape<T, N>::from_values(0, blockIdx.x);
                else
                    return Shape<T, N>::from_values(get_grid_size<GridNDim - 2>(), blockIdx.x);
            } else if constexpr (N == 3) {
                if constexpr (GridNDim == 1)
                    return Shape<T, N>::from_values(0, 0, blockIdx.x);
                else if constexpr (GridNDim == 2)
                    return Shape<T, N>::from_values(0, get_grid_size<0>(), blockIdx.x);
                else
                    return Shape<T, N>::from_values(get_grid_size<0>(), get_grid_size<1>(), blockIdx.x);
            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T>
        static NOA_FD auto thread_lid() -> T {
            if constexpr (BlockNDim == 1)
                return static_cast<T>(threadIdx.x);
            else if constexpr (BlockNDim == 2)
                return static_cast<T>(threadIdx.y * blockDim.x + threadIdx.x);
            else if constexpr (BlockNDim == 3)
                return static_cast<T>((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);
            else
                static_assert(nt::always_false<T>);
        }

        template<typename T, usize N> requires (1 <= N and N <= 3)
        static NOA_FD auto thread_id() -> Vec<T, N> {
            if constexpr (N == 1)
                return Vec<T, N>::from_value(threadIdx.x);
            else if constexpr (N == 2)
                return Vec<T, N>::from_values(threadIdx.y, threadIdx.x);
            else if constexpr (N == 3)
                return Vec<T, N>::from_values(threadIdx.z, threadIdx.y, threadIdx.x);
            else
                static_assert(nt::always_false<T>);
        }

        template<typename T>
        NOA_FD auto thread_gid() const -> T {
            if constexpr (GridNDim == 1) {
                if constexpr (BlockNDim == 1) {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    return static_cast<T>(index_x);
                } else if constexpr (BlockNDim == 2) {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = threadIdx.y;
                    const u32 size_x = gridDim.x * blockDim.x;
                    return static_cast<T>(index_y * size_x + index_x);
                } else {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = threadIdx.y;
                    const u32 index_z = threadIdx.z;
                    const u32 size_x = gridDim.x * blockDim.x;
                    const u32 size_y = blockDim.y;
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }

            } else if constexpr (GridNDim == 2) {
                if constexpr (BlockNDim == 1) {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = blockIdx.y + get_block_index_offset<0>();
                    const u32 size_x = gridDim.x * blockDim.x;
                    return static_cast<T>(index_y * size_x + index_x);
                } else if constexpr (BlockNDim == 2) {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = (blockIdx.y + get_block_index_offset<0>()) * blockDim.y + threadIdx.y;
                    const u32 size_x = gridDim.x * blockDim.x;
                    return static_cast<T>(index_y * size_x + index_x);
                } else {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = (blockIdx.y + get_block_index_offset<0>()) * blockDim.y + threadIdx.y;
                    const u32 index_z = threadIdx.z;
                    const u32 size_x = gridDim.x * blockDim.x;
                    const u32 size_y = blockDim.y;
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }

            } else {
                if constexpr (BlockNDim == 1) {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = blockIdx.y + get_block_index_offset<1>();
                    const u32 index_z = blockIdx.z + get_block_index_offset<0>();
                    const u32 size_x = gridDim.x * blockDim.x;
                    const u32 size_y = get_grid_size<1>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                } else if constexpr (BlockNDim == 2) {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = (blockIdx.y + get_block_index_offset<1>()) * blockDim.y + threadIdx.y;
                    const u32 index_z = blockIdx.z + get_block_index_offset<0>();
                    const u32 size_x = gridDim.x * blockDim.x;
                    const u32 size_y = get_grid_size<1>() * blockDim.y;
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                } else {
                    const u32 index_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const u32 index_y = (blockIdx.y + get_block_index_offset<1>()) * blockDim.y + threadIdx.y;
                    const u32 index_z = (blockIdx.z + get_block_index_offset<0>()) * blockDim.z + threadIdx.z;
                    const u32 size_x = gridDim.x * blockDim.x;
                    const u32 size_y = get_grid_size<1>() * blockDim.y;
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            }
        }

        template<typename T, typename I>
        [[nodiscard]] NOA_FD auto block_scratch() const -> SpanContiguous<T, 1, I> {
            auto ptr = dynamic_shared_memory_pointer<T>();
            if constexpr (IsUsingDynamicSharedMemory)
                return SpanContiguous<T, 1, I>(ptr, static_cast<I>(m_scratch_size / sizeof(T)));
            else
                return SpanContiguous<T, 1, I>(ptr, dynamic_shared_memory_size() / sizeof(T));
        }

        template<typename T, typename I>
        NOA_FD auto block_zeroed_scratch() const -> SpanContiguous<T, 1, I> {
            // TODO For trivial types, 32-bit bank accesses would be best,
            //      but the scratch size might not be a multiple of 4 bytes.
            auto span = block_scratch<T, I>();
            for (I tid = thread_lid<I>(); tid < span.n_elements(); tid += block_size<I>())
                span[tid] = T{};
            return span;
        }

    private:
        NOA_NO_UNIQUE_ADDRESS scratch_type m_scratch_size;
        NOA_NO_UNIQUE_ADDRESS vec_zy_type m_grid_size_zy;
        NOA_NO_UNIQUE_ADDRESS vec_zy_type m_block_index_offset_zy;
    };
}

namespace noa::traits {
    template<typename T, u32 G, u32 B, bool F0, bool F1, bool F2> struct proclaim_is_compute_handle<noa::cuda::details::ComputeHandle<T, G, B, F0, F1, F2>> : std::true_type {};
}
