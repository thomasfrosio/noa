#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/base/Complex.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/cuda/Utils.cuh"

namespace noa::cuda::details {
    template<typename Index,
             u32 GridNDim, u32 BlockNDim,
             bool IsUsingDynamicSharedMemory = false,
             bool IsTwoPartReduction = false>
    struct ComputeHandle {
        using scratch_type = std::conditional_t<IsUsingDynamicSharedMemory, u32, Empty>;
        using index_type = Index;

        static constexpr auto is_cpu() -> bool { return false; }
        static constexpr auto is_gpu() -> bool { return true; }

        struct Grid {
            using index_type = Index;

            static NOA_FD auto size() -> Index {
                if constexpr (GridNDim == 1)
                    return static_cast<Index>(gridDim.x);
                else if constexpr (GridNDim == 2)
                    return static_cast<Index>(gridDim.x * gridDim.y);
                else if constexpr (GridNDim == 3)
                    return static_cast<Index>(gridDim.x * gridDim.y * gridDim.z);
                else
                    static_assert(nt::always_false<Index>);
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
            using scratch_span_type = SpanContiguous<std::byte, 1, Index>;
            NOA_NO_UNIQUE_ADDRESS scratch_type scratch_size;

        public:
            [[nodiscard]] NOA_FD auto scratch() const -> scratch_span_type {
                auto ptr = dynamic_shared_memory_pointer();
                if constexpr (IsUsingDynamicSharedMemory)
                    return scratch_span_type(ptr, static_cast<Index>(scratch_size));
                else
                    return scratch_span_type(ptr, dynamic_shared_memory_size());
            }

            template<typename T>
            NOA_FD auto zeroed_scratch() const -> scratch_span_type {
                auto span = scratch();
                auto span_i32 = span.template as<i32>(); // assuming 32-bit bank access is best
                for (Index tid{}; tid < span_i32.n_elements(); tid += size())
                    span_i32[tid] = 0;
                return span;
            }

            [[nodiscard]] static NOA_FD auto has_scratch() -> bool {
                return dynamic_shared_memory_size() > 0;
            }

            static NOA_FD auto lid() -> Index {
                if constexpr (GridNDim == 1)
                    return static_cast<Index>(blockIdx.x);
                else if constexpr (GridNDim == 2)
                    return static_cast<Index>(blockIdx.y * gridDim.x + blockIdx.x);
                else if constexpr (GridNDim == 3)
                    return static_cast<Index>((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
                else
                    static_assert(nt::always_false<Index>);
            }

            static NOA_FD auto size() -> Index {
                if constexpr (BlockNDim == 1)
                    return static_cast<Index>(blockDim.x);
                else if constexpr (BlockNDim == 2)
                    return static_cast<Index>(blockDim.y * blockDim.x);
                else if constexpr (BlockNDim == 3)
                    return static_cast<Index>(blockDim.z * blockDim.y * blockDim.x);
                else
                    static_assert(nt::always_false<Index>);
            }

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

            static NOA_FD auto lid() -> Index {
                if constexpr (BlockNDim == 1)
                    return static_cast<Index>(threadIdx.x);
                else if constexpr (BlockNDim == 2)
                    return static_cast<Index>(threadIdx.y * blockDim.x + threadIdx.x);
                else if constexpr (BlockNDim == 3)
                    return static_cast<Index>((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x);
                else
                    static_assert(nt::always_false<Index>);
            }

            /// Returns a per-thread unique ID, i.e., each thread in the grid gets a unique value.
            static NOA_FD auto uid() -> Index {
                // TODO For multi-grid launches this doesn't work, we should take the offset as optional arg.
                if constexpr (BlockNDim == 1) {
                    return static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
                } else if constexpr (BlockNDim == 2) {
                    const auto bid_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const auto bid_y = blockIdx.y * blockDim.y + threadIdx.y;
                    return static_cast<Index>(bid_y * blockDim.x + bid_x);
                } else {
                    const auto bid_x = blockIdx.x * blockDim.x + threadIdx.x;
                    const auto bid_y = blockIdx.y * blockDim.y + threadIdx.y;
                    const auto bid_z = blockIdx.z * blockDim.z + threadIdx.z;
                    return static_cast<Index>((bid_z * blockDim.y + bid_y) * blockDim.x + bid_x);
                }
            }
        };

    public:
        NOA_FD constexpr explicit ComputeHandle() requires (not IsUsingDynamicSharedMemory) = default;
        NOA_FD explicit ComputeHandle(u32 scratch_size) requires (IsUsingDynamicSharedMemory) { m_scratch_size = scratch_size; }

        NOA_NO_UNIQUE_ADDRESS scratch_type m_scratch_size{};
        NOA_FD auto grid() const -> Grid { return {}; }
        NOA_FD auto block() const -> Block { return {m_scratch_size}; }
        NOA_FD auto thread() const -> Thread { return {}; }
    };
}

namespace noa::traits {
    template<typename T, u32 G, u32 B, bool F0, bool F1> struct proclaim_is_compute_handle<noa::cuda::details::ComputeHandle<T, G, B, F0, F1>> : std::true_type {};
}
