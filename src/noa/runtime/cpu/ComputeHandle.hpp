#pragma once

#include <omp.h>
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/core/Interfaces.hpp"

namespace noa::cpu {
    /// Compute handle for CPU code-paths.
    /// The grid is the OpenMP team (aka the parallel region). Blocks are made of one (OpenMP) thread.
    /// TODO Add scratch (dynamic and static).
    /// TODO Add SIMD.
    template<typename Index, bool PARALLEL>
    struct ComputeHandle {
        using index_type = Index;

        static constexpr auto is_cpu() -> bool { return true; }
        static constexpr auto is_gpu() -> bool { return false; }
        static constexpr auto is_two_part_reduction() -> bool { return false; }

    public: // grid
        template<nt::integer T = index_type>
        static constexpr auto grid_ndim() -> T {
            return 1;
        }

        template<nt::integer T = index_type>
        static constexpr auto raw_grid_size_x() -> T {
            if constexpr (PARALLEL)
                return static_cast<T>(omp_get_num_threads());
            else
                return static_cast<T>(1);
        }

        template<nt::integer T = index_type>
        static constexpr auto raw_grid_size_y() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto raw_grid_size_z() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto grid_size_x() -> T { return raw_grid_size_x<T>(); }

        template<nt::integer T = index_type>
        static constexpr auto grid_size_y() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto grid_size_z() -> T { return 1; }

        template<typename T, usize N = 1> requires (1 <= N and N <= 3)
        static constexpr auto raw_grid_shape() -> Shape<T, N> {
            auto shape = Shape<T, N>::from_value(1);
            if constexpr (PARALLEL)
                shape[N - 1] = raw_grid_size_x<T>();
            return shape;
        }

        template<typename T, usize N = 1> requires (1 <= N and N <= 3)
        static constexpr auto grid_shape() -> Shape<T, N> {
            return raw_grid_shape<T, N>();
        }

        template<nt::integer T = index_type>
        static constexpr auto raw_grid_size() -> T { return raw_grid_size_x<T>(); }

        template<nt::integer T = index_type>
        static constexpr auto grid_size() -> T { return raw_grid_size_x<T>(); }

        template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
        static auto grid_atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
            if constexpr (PARALLEL) {
                auto pointer = accessor.offset_pointer(accessor.data(), indices...);
                if constexpr (nt::complex<nt::mutable_value_type_t<T>>) {
                    #pragma omp atomic
                    (*pointer)[0] += value[0];
                    #pragma omp atomic
                    (*pointer)[1] += value[1];
                } else {
                    #pragma omp atomic
                    *pointer += value;
                }
            } else {
                accessor(indices...) += value;
            }
        }

    public: // block
        template<nt::integer T = index_type>
        static constexpr auto block_ndim() -> T { return 2; }

        template<nt::integer T = index_type>
        static constexpr auto raw_block_index_x() -> T {
            if constexpr (PARALLEL)
                return static_cast<T>(omp_get_thread_num());
            else
                return T{};
        }

        template<nt::integer T = index_type>
        static constexpr auto raw_block_index_y() -> T { return {}; }

        template<nt::integer T = index_type>
        static constexpr auto raw_block_index_z() -> T { return {}; }

        template<nt::integer T = index_type>
        static constexpr auto block_index_x() -> T { return raw_block_index_x<T>(); }

        template<nt::integer T = index_type>
        static constexpr auto block_index_y() -> T { return {}; }

        template<nt::integer T = index_type>
        static constexpr auto block_index_z() -> T { return {}; }

        template<typename T = index_type, usize N = 1> requires (1 <= N and N <= 3)
        static constexpr auto raw_block_indices() -> Vec<T, N> {
            if constexpr (N == 1) {
                return Vec<T, N>::from_values(raw_block_index_x<T>());
            } else if constexpr (N == 2) {
                return Vec<T, N>::from_values(0, raw_block_index_x<T>());
            } else if constexpr (N == 3) {
                return Vec<T, N>::from_values(0, 0, raw_block_index_x<T>());
            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T = index_type, usize N = 1> requires (1 <= N and N <= 3)
        static constexpr auto block_indices() -> Vec<T, N> { return raw_block_indices<T, N>(); }

        template<typename T>
        static constexpr auto raw_block_linear_index() -> T { return raw_block_index_x<T>(); }

        template<typename T>
        static constexpr auto block_linear_index() -> T { return raw_block_index_x<T>(); }

        template<nt::integer T = index_type>
        static constexpr auto block_size_x() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto block_size_y() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto block_size_z() -> T { return 1; }

        template<typename T = index_type, usize N = 1> requires (1 <= N and N <= 3)
        static constexpr auto block_shape() -> Shape<T, N> { return Shape<T, N>::from_value(1); }

        template<nt::integer T = index_type>
        static constexpr auto block_size() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto block_work_size_x() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto block_work_size_y() -> T { return 1; }

        template<nt::integer T = index_type>
        static constexpr auto block_work_size_z() -> T { return 1; }

        template<typename T = index_type, usize N = 1> requires (1 <= N and N <= 3)
        static constexpr auto block_work_shape() -> Shape<T, N> { return block_shape<T, N>(); }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_work_size() -> T { return 1; }

        static constexpr void block_synchronize() {}

        template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
        static auto block_atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
            accessor(indices...) += value;
        }

    public: // thread
        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr auto thread_index_x() -> T { return T{}; }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr auto thread_index_y() -> T { return T{}; }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr auto thread_index_z() -> T { return T{}; }

        template<nt::integer T = index_type, usize N = 1> requires (1 <= N and N <= 3)
        [[nodiscard]] static constexpr auto thread_indices() -> Vec<T, N> { return Vec<T, N>{}; }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr auto thread_linear_index() -> T { return {}; }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr auto raw_thread_uid() -> T { return raw_block_index_x<T>(); }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr auto thread_uid() -> T { return raw_block_index_x<T>(); }

    public: // scratch
        [[nodiscard]] static constexpr auto total_scratch_size() -> u32 { return {}; }

        [[nodiscard]] static constexpr auto scratch_size() -> u32 { return {}; }

        [[nodiscard]] static constexpr auto has_scratch() -> bool { return false; }

        template<typename T = std::byte>
        [[nodiscard]] static constexpr auto scratch_pointer() -> T* { return nullptr; }

        template<typename T = std::byte, typename I = index_type>
        [[nodiscard]] static constexpr auto scratch() -> SpanContiguous<T, 1, I> {
            return SpanContiguous<T, 1, I>(nullptr, I{});
        }

        template<typename T = std::byte, typename I = index_type>
        static constexpr auto zeroed_scratch_async() -> SpanContiguous<T, 1, I> {
            return scratch<T, I>();
        }

        template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
        static auto scratch_atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
            accessor(indices...) += value;
        }
    };
}

namespace noa::traits {
    template<typename I, bool P> struct proclaim_is_compute_handle<noa::cpu::ComputeHandle<I, P>> : std::true_type {};
}
