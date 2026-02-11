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

        /// The grid is the OpenMP team (aka the parallel region).
        /// While this is quite different from, for instance, Kokkos, it works quite well for this library.
        struct Grid {
            using index_type = Index;

            /// Grid size, i.e., the number of blocks in the grid.
            static constexpr auto size() -> Index {
                return static_cast<Index>(1);
                if constexpr (PARALLEL)
                    return static_cast<Index>(omp_get_num_threads());
                else
                    return static_cast<Index>(1);
            }

            /// Atomic add.
            /// This function guarantees no data-races between the threads of the grid.
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

        /// Blocks are made of one (OpenMP) thread.
        struct Block {
            using index_type = Index;

            [[nodiscard]] static constexpr auto has_scratch() -> bool { return false; }
            [[nodiscard]] static constexpr auto scratch() -> SpanContiguous<std::byte> { return {}; }
            static constexpr auto zeroed_scratch() -> SpanContiguous<std::byte> { return {}; }

            /// Linear index of the block within the grid.
            static constexpr auto lid() -> Index {
                if constexpr (PARALLEL)
                    return static_cast<Index>(omp_get_thread_num());
                else
                    return static_cast<Index>(0);
            }

            /// Block size, i.e. number of threads in the block.
            static constexpr auto size() -> Index {
                return static_cast<Index>(1);
            }

            /// Synchronizes the threads in the block.
            static constexpr void synchronize() {
                // no-op
            }

            /// Atomic add.
            /// This function guarantees no data-races between the threads of the same block.
            template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
            static constexpr auto atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
                accessor(indices...) += value;
            }
        };

        struct Thread {
            using index_type = Index;

            /// Linear index of the thread within the block.
            static constexpr auto lid() -> Index { return 0; }

            /// Returns a per-thread unique ID, i.e., each thread in the grid gets a unique value.
            static constexpr auto uid() -> Index {
                return block().lid();
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
