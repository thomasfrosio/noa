#pragma once

#include <omp.h>
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/core/Interfaces.hpp"

namespace noa::cpu::details {
    template<typename Index, bool PARALLEL>
    struct ComputeHandle {
        using index_type = Index;

        static constexpr auto is_cpu() -> bool { return true; }
        static constexpr auto is_gpu() -> bool { return false; }

        // The grid is the OpenMP team (aka the parallel region).
        // While this is quite different from, for instance, Kokkos, it works quite well for this library.
        struct Grid {
            using index_type = Index;

            template<typename T = index_type>
            static constexpr auto ndim() -> T { return static_cast<T>(1); }

            template<typename T = index_type>
            static constexpr auto size() -> T {
                if constexpr (PARALLEL)
                    return static_cast<T>(omp_get_num_threads());
                else
                    return static_cast<T>(1);
            }

            template<typename T, usize N> requires (1 <= N and N <= 3)
            static constexpr auto shape() -> Shape<T, N> {
                auto shape = Shape<T, N>::from_value(1);
                if constexpr (PARALLEL)
                    shape[N - 1] = size<T>();
                return shape;
            }

            template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
            static NOA_HD auto atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
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

            static constexpr auto is_two_part_reduction() -> bool { return false; }
        };

        // Blocks are made of one (OpenMP) thread.
        struct Block {
            using index_type = Index;

            template<typename T = index_type>
            static constexpr auto ndim() -> T { return static_cast<T>(1); }

            [[nodiscard]] static constexpr auto has_scratch() -> bool { return false; }

            template<typename T = std::byte, typename I = index_type>
            [[nodiscard]] static constexpr auto scratch() -> SpanContiguous<T, 1, I> { return {}; }

            template<typename T = std::byte>
            [[nodiscard]] static constexpr auto scratch_pointer() -> T* { return {}; }

            template<typename T = std::byte, typename I = index_type>
            static constexpr auto zeroed_scratch() -> SpanContiguous<T, 1, I> { return {}; }

            template<typename I = index_type>
            static constexpr auto lid() -> I {
                if constexpr (PARALLEL)
                    return static_cast<I>(omp_get_thread_num());
                else
                    return static_cast<I>(0);
            }

            template<typename I = index_type, usize N> requires (1 <= N and N <= 3)
            static constexpr auto id() -> Vec<I, N> {
                auto bid = Vec<I, N>{};
                if constexpr (PARALLEL)
                    bid[N - 1] = lid<I>();
                return bid;
            }

            template<typename I = index_type>
            static constexpr auto size() -> I {
                return static_cast<I>(1);
            }

            template<typename I, usize N> requires (1 <= N and N <= 3)
            static constexpr auto shape() -> Shape<I, N> {
                return Shape<I, N>::from_value(1);
            }

            static constexpr void synchronize() {}

            template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
            static constexpr auto atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
                accessor(indices...) += value;
            }
        };

        struct Thread {
            using index_type = Index;

            template<typename I = index_type>
            static constexpr auto lid() -> I { return static_cast<I>(0); }

            template<typename I, usize N> requires (1 <= N and N <= 3)
            static constexpr auto id() -> Vec<I, N> {
                return {};
            }

            template<typename I = index_type>
            static constexpr auto gid() -> I {
                return block().template lid<I>();
            }
        };

    public:
        static constexpr auto grid() -> Grid { return {}; }
        static constexpr auto block() -> Block { return {}; }
        static constexpr auto thread() -> Thread { return {}; }
    };
}

namespace noa::traits {
    template<typename I, bool P> struct proclaim_is_compute_handle<noa::cpu::details::ComputeHandle<I, P>> : std::true_type {};
}
