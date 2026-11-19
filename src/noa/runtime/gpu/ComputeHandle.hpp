#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/base/Config.hpp"
#include "noa/base/Complex.hpp"
#include "noa/base/Half.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/core/Interfaces.hpp"
#include "noa/runtime/gpu/Warp.hpp"

#if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
namespace noa::gpu::details {
    NOA_FD i32 atomic_add(i32* address, i32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD u32 atomic_add(u32* address, u32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD u64 atomic_add(u64* address, u64 val) {
        return ::atomicAdd(
            reinterpret_cast<unsigned long long*>(address),
            static_cast<unsigned long long>(val));
    }

    NOA_FD i64 atomic_add(i64* address, i64 val) {
        // CUDA has no signed 64-bit atomicAdd. Two's complement wraps identically, so use the unsigned one.
        return static_cast<i64>(::atomicAdd(
            reinterpret_cast<unsigned long long*>(address),
            static_cast<unsigned long long>(val)));
    }

    NOA_FD f16 atomic_add(f16* address, f16 val) {
        NOA_IF_TARGET_SM70(
            (return f16(::atomicAdd(reinterpret_cast<__half*>(address), val.native()));),
            (
                // Adds in a 16-bit value with a 32-bit CAS on the enclosing aligned word.
                // This works on ALL architectures (atomicCAS on 32-bit words has always existed), so it
                // is the answer to the "no 16-bit atomics on 5.3 and 6.x" comment. It also reads/writes
                // the neighbouring half, so allocations should be a multiple of 4 bytes.
                const auto addr = reinterpret_cast<std::uintptr_t>(address);
                auto* word = reinterpret_cast<unsigned int*>(addr & ~std::uintptr_t{3});
                const unsigned shift = (addr & 2) ? 16u : 0u;
                unsigned int old = *reinterpret_cast<volatile unsigned int*>(word);
                unsigned int assumed;
                do {
                    assumed = old;
                    const f16 cur = f16::from_bits(static_cast<u16>((assumed >> shift) & 0xFFFFu));
                    const f16 sum = cur + val;
                    const unsigned int updated = (assumed & ~(0xFFFFu << shift)) |
                    (static_cast<unsigned int>(__half_as_ushort(sum.native())) << shift);
                    old = ::atomicCAS(word, assumed, updated);
                } while (assumed != old);
                return f16::from_bits(static_cast<u16>((old >> shift) & 0xFFFFu));
            )
        )
    }

    NOA_FD f32 atomic_add(f32* address, f32 val) { return ::atomicAdd(address, val); }

    NOA_FD f64 atomic_add(f64* address, f64 val) {
        NOA_IF_TARGET_SM60(
            (return ::atomicAdd(address, val);),
            (
                using ull = unsigned long long;
                auto* address_as_ull = reinterpret_cast<ull*>(address);
                ull old = *address_as_ull, assumed;
                do {
                    assumed = old;
                    old = ::atomicCAS(
                        address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
                } while (assumed != old);
                return __longlong_as_double(old);
            )
        )
    }

    template<typename T>
    NOA_FD Complex<T> atomic_add(Complex<T>* address, Complex<T> val) {
        return {atomic_add(&address->real, val.real), atomic_add(&address->imag, val.imag)};
    }

    template<typename T>
    NOA_FD auto grid_atomic_add(T* a, T v) {
        return atomic_add(a, v);
    }

    // Block scope: atomicAdd_block exists from sm_60 (int, uint, ull, float, double).
    // The remaining types fall back to the device-scope version.
    template<typename T>
    NOA_FD T block_atomic_add(T* address, T val) {
        NOA_IF_TARGET_SM60((
            if constexpr (nt::any_of<T, i32, u32, f32, f64>)
                return ::atomicAdd_block(address, val);
            else if constexpr (nt::same_as<T, u64>)
                return ::atomicAdd_block(
                    reinterpret_cast<unsigned long long*>(address),
                    static_cast<unsigned long long>(val));
        ), ())
        return atomic_add(address, val);
    }

    template<typename T>
    NOA_FD auto scratch_atomic_add(T* a, T v) {
        return block_atomic_add(a, v);
    }
}
#endif

#ifdef NOA_ENABLE_SYCL
namespace noa::gpu::details {
    template<sycl::memory_scope Scope, sycl::access::address_space Space, typename T>
    NOA_FD auto atomic_add_impl(T* address, T val) {
        if constexpr (std::same_as<T, f16>) {
            // No half atomics in SYCL 2020, so use the same 32-bit CAS trick.
            // TODO AdaptiveCpp's CUDA backend, __acpp_if_target_cuda and use atomicAdd if SM70
            const auto addr = reinterpret_cast<std::uintptr_t>(address);
            auto* word = reinterpret_cast<std::uint32_t*>(addr & ~std::uintptr_t{3});
            const unsigned shift = (addr & 2) ? 16u : 0u;
            sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, Scope, Space> ref(*word);
            std::uint32_t old = ref.load();
            std::uint32_t updated;
            do {
                const f16 cur = f16::from_bits(static_cast<u16>((old >> shift) & 0xFFFFu));
                const f16 sum = cur + val;
                updated = (old & ~(0xFFFFu << shift)) |
                          (static_cast<std::uint32_t>(std::bit_cast<u16>(sum.native())) << shift); // FIXME check bit_cast?
            } while (!ref.compare_exchange_weak(old, updated)); // refreshes `old` on failure
            return f16::from_bits(static_cast<u16>((old >> shift) & 0xFFFFu));
        } else if constexpr (nt::complex<T>) {
            return T{atomic_add_impl<Scope>(&address->real, val.real),
                     atomic_add_impl<Scope>(&address->imag, val.imag)};
        } else {
            sycl::atomic_ref<T, sycl::memory_order::relaxed, Scope, Space> ref(*address);
            return ref.fetch_add(val);
        }
    }

    template<typename T>
    NOA_FD auto grid_atomic_add(T* a, T v) {
        return atomic_add_impl<sycl::memory_scope::device, sycl::access::address_space::generic_space>(a, v);
    }

    template<typename T>
    NOA_FD auto block_atomic_add(T* a, T v) {
        return atomic_add_impl<sycl::memory_scope::work_group, sycl::access::address_space::generic_space>(a, v);
    }

    template<typename T>
    NOA_FD auto scratch_atomic_add(T* a, T v) {
        return atomic_add_impl<sycl::memory_scope::work_group, sycl::access::address_space::local_space>(a, v);
    }
}
#endif

namespace noa::gpu {
    /// Compute handle for GPU code-paths.
    /// This offers a unified way to query various info about the compute kernel,
    /// including the grid/block shape and current indices, as well as atomic and
    /// synchronization barriers.
    ///
    /// \tparam Index:
    ///     Index type.
    /// \tparam Block:
    ///     Type specifying the block shape, such as:
    ///     Block::block_size_z: nb threads along the Z (the slowest/outer axis) of the block.
    ///     Block::block_size_y: nb threads along the Y (the middle axis) of the block.
    ///     Block::block_size_x: nb threads along the X (the fastest/inner axis) of the block.
    ///     Block::block_work_size_z: size covered by the block (block_size_z * iter_z) along Z.
    ///     Block::block_work_size_y: size covered by the block (block_size_y * iter_y) along Y.
    ///     Block::block_work_size_x: size covered by the block (block_size_x * iter_x) along X.
    /// \tparam IsMultiGridKernel:
    ///     Whether the kernel is part of a multi-grid kernel launch.
    ///     GPU kernels launches usually encode the Z and Y axis of the grid using u16, so the grid size along these
    ///     axes is limited to 2^16=65536, whereas the X axis can usually go up to 2^31=2147483648. These limits
    ///     depend on the underlying GPU driver (SYCL running on a CUDA device will ultimately share the CUDA limit).
    ///     If true, we can extend the Z and Y limits to match the X axis by launching kernels more than once.
    ///     Within a given kernel, we need to adjust the grid shape (gridDim in CUDA) and block indices (blockIdx)
    ///     to report the correct values. To do so, we store the grid outer shape (gridDim.x is omitted) as well as
    ///     the grid outer offset (offsets added to blockIdx.z and blockIdx.y).
    /// \tparam IsSharingScratch:
    ///     Whether the kernel implementation (not the user operator) is using dynamic shared memory.
    ///     Users may ask for some dynamic shared memory too, but its usage is limited by the compute interface
    ///     to the [init/call/deinit] steps. As such, to save on shared memory, the user can share its share memory
    ///     with the implementation as long as the implementation is using it before init or after deinit.
    ///     If true, we store the amount of bytes asked by the user so that the scratch span has the correct size.
    ///     For SYCL/HIP, we always store the total number of bytes too, whereas in CUDA that total can be retrieved
    ///     from registers directly.
    /// \tparam IsTwoPartReduction:
    ///     Whether the kernel is part of a two part reduction.
    ///     This is just a compile-time flag for the user to know whether the reduction is done in a single kernel
    ///     (ensuring the operator lives through the entire init/call/deinit/join/post reduction) or done in two kernels.
    template<typename Index,
             u32 GridNDim, typename Block,
             bool IsMultiGridKernel,
             bool IsSharingScratch,
             bool IsTwoPartReduction>
    struct ComputeHandle {
        static constexpr usize GRID_NDIM = GridNDim;
        static constexpr usize BLOCK_NDIM = Block::block_size_z > 1 ? 3 : Block::block_size_y > 1 ? 2 : 1;
        #ifdef NOA_ENABLE_SYCL
        // In SYCL, nd_item is parametrized using GRID_NDIM, so to query
        // the thread indices GRID_NDIM should be set to max(GRID_NDIM, BLOCK_NDIM).
        static_assert(GRID_NDIM >= BLOCK_NDIM);
        #endif

        using block_type = Block;
        using vec_zy_type = std::conditional_t<IsMultiGridKernel, Vec<u32, GRID_NDIM - 1>, Empty>;
        using scratch_size_type = std::conditional_t<IsSharingScratch, u32, Empty>;
        using index_type = Index;
        #ifdef NOA_ENABLE_SYCL
        using default_index_type = usize;
        #else
        using default_index_type = u32;
        #endif

    public:
        static constexpr NOA_FD auto is_cpu() -> bool { return false; }
        static constexpr NOA_FD auto is_gpu() -> bool { return true; }
        static constexpr NOA_FD auto is_two_part_reduction() -> bool { return IsTwoPartReduction; }

    public: // init
        explicit ComputeHandle() requires (not IsSharingScratch and not IsMultiGridKernel) = default;

        NOA_FHD explicit ComputeHandle(
            u32 user_scratch_size
        ) requires (IsSharingScratch and not IsMultiGridKernel) :
            m_user_scratch_size{user_scratch_size} {}

        NOA_FHD explicit ComputeHandle(
            vec_zy_type grid_dim_zy,
            vec_zy_type block_index_offset_zy
        ) requires (not IsSharingScratch and IsMultiGridKernel) :
            m_grid_size_zy{grid_dim_zy},
            m_block_index_offset_zy{block_index_offset_zy} {}

        NOA_FHD explicit ComputeHandle(
            Vec<u32, 2> grid_dim_zy,
            Vec<u32, 2> block_index_offset_zy
        ) requires (not IsSharingScratch and IsMultiGridKernel and GRID_NDIM < 3) :
            m_grid_size_zy{grid_dim_zy.pop_front<3 - GRID_NDIM>()},
            m_block_index_offset_zy{block_index_offset_zy.pop_front<3 - GRID_NDIM>()} {}

        NOA_FHD explicit ComputeHandle(
            u32 user_scratch_size,
            vec_zy_type grid_dim_zy,
            vec_zy_type block_index_offset_zy
        ) requires (IsSharingScratch and IsMultiGridKernel) :
            m_user_scratch_size{user_scratch_size},
            m_grid_size_zy{grid_dim_zy},
            m_block_index_offset_zy{block_index_offset_zy} {}

        NOA_FHD explicit ComputeHandle(
            u32 user_scratch_size,
            Vec<u32, 2> grid_dim_zy,
            Vec<u32, 2> block_index_offset_zy
        ) requires (IsSharingScratch and IsMultiGridKernel and GRID_NDIM < 3) :
            m_user_scratch_size{user_scratch_size},
            m_grid_size_zy{grid_dim_zy.pop_front<3 - GRID_NDIM>()},
            m_block_index_offset_zy{block_index_offset_zy.pop_front<3 - GRID_NDIM>()} {}

        #if defined(NOA_ENABLE_SYCL)
        // SYCL has its own compute-handle-like type, called nd_item.
        // Depending on the SYCL backend and implementation, it can technically be empty in SYCL2020
        // (previous version are required to store the offset) and does on-demand indexing like in CUDA
        // (querying blockIdx, threadIdx, etc.). Other backends/implementations may not do this on-demand
        // indexing and instead store all the indexing info (grid shape, block shape, indices, etc.) inside nd_item.
        // For us, since we know nd_item always outlives the compute handle, we can just store a pointer.
        NOA_FHD void set_sycl_nd_item(const sycl::nd_item<GRID_NDIM>* nd_item) {
            m_nd_item = nd_item;
        }

        // SYCL doesn't have a way to retrieve/declare the scratch pointer on-demand, like in CUDA/HIP, so we need to
        // store it inside the compute handle. The same is true for the total scratch size. If the launch config
        // doesn't allocate scratch, nullptr and zero are passed.
        // TODO AdaptiveCpp probably has ways to ge the on-demand pointer/size using the target macro
        //      (for sscp target, the jit if can also be used to get the underlying target).
        NOA_FHD void set_sycl_total_scratch_size(u32 total_scratch_size) { m_total_scratch_size = total_scratch_size; }
        NOA_FHD void set_sycl_scratch_pointer(std::byte* scratch) { m_scratch = scratch; }
        #elif defined(NOA_ENABLE_HIP)
        // In HIP, we don't have a way to retrieve the total scratch size on demand like in CUDA,
        // so we need to store it explicitly in the compute handle. TODO look at available HIP registers
        NOA_FHD void set_hip_total_scratch_size(u32 total_scratch_size) { m_total_scratch_size = total_scratch_size; }
        #endif

    public: // grid
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto grid_ndim() -> T {
            return static_cast<T>(GRID_NDIM);
        }

        #ifdef NOA_ENABLE_SYCL
        template<nt::integer T = index_type>
        NOA_FD auto raw_grid_size_x() const -> T {
            return static_cast<T>(m_nd_item->get_group_range(GRID_NDIM - 1));
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto raw_grid_size_y() const -> T {
            if constexpr (GRID_NDIM >= 2) {
                return static_cast<T>(m_nd_item->get_group_range(GRID_NDIM - 2));
            } else {
                return 1;
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto raw_grid_size_z() const -> T {
            if constexpr (GRID_NDIM == 3) {
                return static_cast<T>(m_nd_item->get_group_range(0));
            } else {
                return 1;
            }
        }

        template<nt::integer T = index_type>
        NOA_FD auto grid_size_x() const -> T {
            return static_cast<T>(m_nd_item->get_group_range(GRID_NDIM - 1));
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto grid_size_y() const -> T {
            if constexpr (GRID_NDIM >= 2) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(m_grid_size_zy[GRID_NDIM - 2]);
                else
                    return static_cast<T>(m_nd_item->get_group_range(GRID_NDIM - 2));
            } else {
                return T{1};
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto grid_size_z() const -> T {
            if constexpr (GRID_NDIM == 3) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(m_grid_size_zy[0]);
                else
                    return static_cast<T>(m_nd_item->get_group_range(0));
            } else {
                return T{1};
            }
        }
        #else
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto raw_grid_size_x() -> T {
            return static_cast<T>(gridDim.x);
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto raw_grid_size_y() -> T {
            if constexpr (GRID_NDIM >= 2) {
                return static_cast<T>(gridDim.y);
            } else {
                return T{1};
            }
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto raw_grid_size_z() -> T {
            if constexpr (GRID_NDIM == 3) {
                return static_cast<T>(gridDim.z);
            } else {
                return T{1};
            }
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto grid_size_x() -> T {
            return static_cast<T>(gridDim.x);
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto grid_size_y() const -> T {
            if constexpr (GRID_NDIM >= 2) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(m_grid_size_zy[GRID_NDIM - 2]);
                else
                    return static_cast<T>(gridDim.y);
            } else {
                return T{1};
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto grid_size_z() const -> T {
            if constexpr (GRID_NDIM == 3) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(m_grid_size_zy[GRID_NDIM - 3]);
                else
                    return static_cast<T>(gridDim.z);
            } else {
                return T{1};
            }
        }
        #endif

        /// Returns the grid shape of the current kernel launch (i.e., not accounting multi-grid kernels).
        template<typename T, usize N = GRID_NDIM> requires (1 <= N and N <= 3)
        NOA_FD auto raw_grid_shape() const -> Shape<T, N> {
            if constexpr (N == 1) {
                return Shape<T, N>::from_values(raw_grid_size_x<T>());
            } else if constexpr (N == 2) {
                return Shape<T, N>::from_values(raw_grid_size_y<T>(), raw_grid_size_x<T>());
            } else if constexpr (N == 3) {
                return Shape<T, N>::from_values(raw_grid_size_z<T>(), raw_grid_size_y<T>(), raw_grid_size_x<T>());
            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T, usize N = GRID_NDIM> requires (1 <= N and N <= 3)
        NOA_FD auto grid_shape() const -> Shape<T, N> {
            if constexpr (N == 1) {
                return Shape<T, N>::from_values(grid_size_x<T>());
            } else if constexpr (N == 2) {
                return Shape<T, N>::from_values(grid_size_y<T>(), grid_size_x<T>());
            } else if constexpr (N == 3) {
                return Shape<T, N>::from_values(grid_size_z<T>(), grid_size_y<T>(), grid_size_x<T>());
            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T>
        NOA_FD auto raw_grid_size() const -> T {
            return raw_grid_shape<T, GRID_NDIM>().n_elements();
        }

        template<typename T>
        NOA_FD auto grid_size() const -> T {
            return grid_shape<T, GRID_NDIM>().n_elements();
        }

        template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
        static NOA_FD auto grid_atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
            auto* pointer = accessor.offset_pointer(accessor.get(), indices...);
            return details::grid_atomic_add(pointer, value);
        }

    public: // block
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_ndim() -> T { return static_cast<T>(BLOCK_NDIM); }

        #ifdef NOA_ENABLE_SYCL
        template<nt::integer T = index_type>
        NOA_FD auto raw_block_index_x() const -> T {
            return static_cast<T>(m_nd_item->get_group(GRID_NDIM - 1));
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto raw_block_index_y() const -> T {
            if constexpr (GRID_NDIM >= 2) {
                return static_cast<T>(m_nd_item->get_group(GRID_NDIM - 2));
            } else {
                return T{};
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto raw_block_index_z() const -> T {
            if constexpr (GRID_NDIM == 3) {
                return static_cast<T>(m_nd_item->get_group(0));
            } else {
                return T{};
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto block_index_x() const -> T {
            return static_cast<T>(m_nd_item->get_group(GRID_NDIM - 1));
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto block_index_y() const -> T {
            if constexpr (GRID_NDIM >= 2) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(
                        static_cast<u32>(m_nd_item->get_group(GRID_NDIM - 2)) + m_block_index_offset_zy[GRID_NDIM - 2]);
                else
                    return static_cast<T>(m_nd_item->get_group(GRID_NDIM - 2));
            } else {
                return T{};
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto block_index_z() const -> T {
            if constexpr (GRID_NDIM == 3) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(
                        static_cast<u32>(m_nd_item->get_group(0)) + m_block_index_offset_zy[GRID_NDIM - 3]);
                else
                    return static_cast<T>(m_nd_item->get_group(0));
            } else {
                return T{};
            }
        }
        #else
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto raw_block_index_x() -> T {
            return static_cast<T>(blockIdx.x);
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto raw_block_index_y() -> T {
            if constexpr (GRID_NDIM >= 2) {
                return static_cast<T>(blockIdx.y);
            } else {
                return T{};
            }
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto raw_block_index_z() -> T {
            if constexpr (GRID_NDIM == 3) {
                return static_cast<T>(blockIdx.z);
            } else {
                return T{};
            }
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_index_x() -> T {
            return static_cast<T>(blockIdx.x);
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto block_index_y() const -> T {
            if constexpr (GRID_NDIM >= 2) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(blockIdx.y + m_block_index_offset_zy[GRID_NDIM - 2]);
                else
                    return static_cast<T>(blockIdx.y);
            } else {
                return T{};
            }
        }

        template<nt::integer T = index_type>
        constexpr NOA_FD auto block_index_z() const -> T {
            if constexpr (GRID_NDIM == 3) {
                if constexpr (IsMultiGridKernel)
                    return static_cast<T>(blockIdx.z + m_block_index_offset_zy[GRID_NDIM - 3]);
                else
                    return static_cast<T>(blockIdx.z);
            } else {
                return T{};
            }
        }
        #endif

        template<typename T = index_type, usize N = GRID_NDIM> requires (1 <= N and N <= 3)
        NOA_FD auto raw_block_indices() const -> Vec<T, N> {
            if constexpr (N == 1) {
                return Vec<T, N>::from_values(raw_block_index_x<T>());
            } else if constexpr (N == 2) {
                return Vec<T, N>::from_values(raw_block_index_y<T>(), raw_block_index_x<T>());
            } else if constexpr (N == 3) {
                return Vec<T, N>::from_values(raw_block_index_z<T>(), raw_block_index_y<T>(), raw_block_index_x<T>());
            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T = index_type, usize N = GRID_NDIM> requires (1 <= N and N <= 3)
        NOA_FD auto block_indices() const -> Vec<T, N> {
            if constexpr (N == 1) {
                return Vec<T, N>::from_values(block_index_x<T>());
            } else if constexpr (N == 2) {
                return Vec<T, N>::from_values(block_index_y<T>(), block_index_x<T>());
            } else if constexpr (N == 3) {
                return Vec<T, N>::from_values(block_index_z<T>(), block_index_y<T>(), block_index_x<T>());
            } else {
                static_assert(nt::always_false<Index>);
            }
        }

        template<typename T>
        NOA_FD auto raw_block_linear_index() const -> T {
            if constexpr (GRID_NDIM == 1) {
                return raw_block_index_x<T>();
            } else if constexpr (GRID_NDIM == 2) {
                return static_cast<T>(raw_block_index_y<u32>() * raw_grid_size_x<u32>() + raw_block_index_x<u32>());
            } else if constexpr (GRID_NDIM == 3) {
                return static_cast<T>((raw_block_index_z<u32>() * raw_grid_size_y<u32>() + raw_block_index_y<u32>()) * raw_grid_size_x<u32>() + raw_block_index_x<u32>());
            } else {
                static_assert(nt::always_false<T>);
            }
        }

        template<typename T>
        NOA_FD auto block_linear_index() const -> T {
            if constexpr (GRID_NDIM == 1) {
                return block_index_x<T>();
            } else if constexpr (GRID_NDIM == 2) {
                return static_cast<T>(block_index_y<u32>() * grid_size_x<u32>() + block_index_x<u32>());
            } else if constexpr (GRID_NDIM == 3) {
                return static_cast<T>((block_index_z<u32>() * grid_size_y<u32>() + block_index_y<u32>()) * grid_size_x<u32>() + block_index_x<u32>());
            } else {
                static_assert(nt::always_false<T>);
            }
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_size_x() -> T {
            return static_cast<T>(block_type::block_size_x);
        }
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_size_y() -> T {
            return static_cast<T>(block_type::block_size_y);
        }
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_size_z() -> T {
            return static_cast<T>(block_type::block_size_z);
        }

        template<typename T = index_type, usize N = BLOCK_NDIM> requires (1 <= N and N <= 3)
        static constexpr NOA_FD auto block_shape() -> Shape<T, N> {
            if constexpr (N == 1)
                return Shape<T, N>::from_values(block_size_x<T>());
            else if constexpr (N == 2)
                return Shape<T, N>::from_values(block_size_y<T>(), block_size_x<T>());
            else if constexpr (N == 3)
                return Shape<T, N>::from_values(block_size_z<T>(), block_size_y<T>(), block_size_x<T>());
            else
                static_assert(nt::always_false<T>);
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_size() -> T {
            if constexpr (BLOCK_NDIM == 1)
                return block_size_x<T>();
            else if constexpr (BLOCK_NDIM == 2)
                return static_cast<T>(block_size_y<u32>() * block_size_x<u32>());
            else if constexpr (BLOCK_NDIM == 3)
                return static_cast<T>(block_size_z<u32>() * block_size_y<u32>() * block_size_x<u32>());
            else
                static_assert(nt::always_false<T>);
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_work_size_x() -> T {
            return static_cast<T>(block_type::block_work_size_x);
        }
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_work_size_y() -> T {
            return static_cast<T>(block_type::block_work_size_y);
        }
        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_work_size_z() -> T {
            return static_cast<T>(block_type::block_work_size_z);
        }

        template<typename T = index_type, usize N = BLOCK_NDIM> requires (1 <= N and N <= 3)
        static constexpr NOA_FD auto block_work_shape() -> Shape<T, N> {
            if constexpr (N == 1)
                return Shape<T, N>::from_values(block_work_size_x<T>());
            else if constexpr (N == 2)
                return Shape<T, N>::from_values(block_work_size_y<T>(), block_work_size_x<T>());
            else if constexpr (N == 3)
                return Shape<T, N>::from_values(block_work_size_z<T>(), block_work_size_y<T>(), block_work_size_x<T>());
            else
                static_assert(nt::always_false<T>);
        }

        template<nt::integer T = index_type>
        static constexpr NOA_FD auto block_work_size() -> T {
            if constexpr (BLOCK_NDIM == 1)
                return block_work_size_x<T>();
            else if constexpr (BLOCK_NDIM == 2)
                return static_cast<T>(block_work_size_y<u32>() * block_work_size_x<u32>());
            else if constexpr (BLOCK_NDIM == 3)
                return static_cast<T>(block_work_size_z<u32>() * block_work_size_y<u32>() * block_work_size_x<u32>());
            else
                static_assert(nt::always_false<T>);
        }

        NOA_FD void block_synchronize() const {
            #ifdef NOA_ENABLE_SYCL
            sycl::group_barrier(m_nd_item->get_group());
            #else
            __syncthreads();
            #endif
        }

        template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
        static NOA_FD auto block_atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
            auto* pointer = accessor.offset_pointer(accessor.get(), indices...);
            return details::block_atomic_add(pointer, value);
        }

    public: // thread
        #ifdef NOA_ENABLE_SYCL
        template<nt::integer T = index_type>
        [[nodiscard]] NOA_FD auto thread_index_x() const -> T {
            return static_cast<T>(m_nd_item->get_local_id(GRID_NDIM - 1));
        }

        template<nt::integer T = index_type>
        [[nodiscard]] constexpr NOA_FD auto thread_index_y() const -> T {
            if constexpr (BLOCK_NDIM >= 2)
                return static_cast<T>(m_nd_item->get_local_id(GRID_NDIM - 2));
            return T{};
        }

        template<nt::integer T = index_type>
        [[nodiscard]] constexpr NOA_FD auto thread_index_z() const -> T {
            if constexpr (BLOCK_NDIM == 3)
                return static_cast<T>(m_nd_item->get_local_id(0));
            return T{};
        }
        #else
        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr NOA_FD auto thread_index_x() -> T {
            return static_cast<T>(threadIdx.x);
        }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr NOA_FD auto thread_index_y() -> T {
            if constexpr (BLOCK_NDIM >= 2)
                return static_cast<T>(threadIdx.y);
            return T{};
        }

        template<nt::integer T = index_type>
        [[nodiscard]] static constexpr NOA_FD auto thread_index_z() -> T {
            if constexpr (BLOCK_NDIM == 3)
                return static_cast<T>(threadIdx.z);
            return T{};
        }
        #endif

        template<nt::integer T = index_type, usize N = BLOCK_NDIM> requires (1 <= N and N <= 3)
        [[nodiscard]] NOA_FD auto thread_indices() const -> Vec<T, N> {
            if constexpr (N == 1)
                return Vec<T, N>::from_value(thread_index_x<T>());
            else if constexpr (N == 2)
                return Vec<T, N>::from_values(thread_index_y<T>(), thread_index_x<T>());
            else if constexpr (N == 3)
                return Vec<T, N>::from_values(thread_index_z<T>(), thread_index_y<T>(), thread_index_x<T>());
            else
                static_assert(nt::always_false<T>);
        }

        template<nt::integer T = index_type>
        [[nodiscard]] NOA_FD auto thread_linear_index() const -> T {
            if constexpr (BLOCK_NDIM == 1)
                return thread_index_x<T>();
            else if constexpr (BLOCK_NDIM == 2)
                return static_cast<T>(thread_index_y<u32>() * block_size_x<u32>() + thread_index_x<u32>());
            else if constexpr (BLOCK_NDIM == 3)
                return static_cast<T>((thread_index_z<u32>() * block_size_y<u32>() + thread_index_y<u32>()) * block_size_x<u32>() + thread_index_x<u32>());
            else
                static_assert(nt::always_false<T>);
        }

        template<nt::integer T = index_type>
        [[nodiscard]] NOA_FD auto raw_thread_uid() const -> T {
            if constexpr (GRID_NDIM == 1) {
                if constexpr (BLOCK_NDIM == 1) {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    return static_cast<T>(index_x);
                } else if constexpr (BLOCK_NDIM == 2) {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = thread_index_y<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    return static_cast<T>(index_y * size_x + index_x);
                } else {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = thread_index_y<u32>();
                    const auto index_z = thread_index_z<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            } else if constexpr (GRID_NDIM == 2) {
                if constexpr (BLOCK_NDIM == 1) {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = raw_block_index_y<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    return static_cast<T>(index_y * size_x + index_x);
                } else if constexpr (BLOCK_NDIM == 2) {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = raw_block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    return static_cast<T>(index_y * size_x + index_x);
                } else {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = raw_block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto index_z = thread_index_z<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = raw_grid_size_y<u32>() * block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            } else {
                if constexpr (BLOCK_NDIM == 1) {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = raw_block_index_y<u32>();
                    const auto index_z = raw_block_index_z<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = raw_grid_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                } else if constexpr (BLOCK_NDIM == 2) {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = raw_block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto index_z = raw_block_index_z<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = raw_grid_size_y<u32>() * block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                } else {
                    const auto index_x = raw_block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = raw_block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto index_z = raw_block_index_z<u32>() * block_size_z<u32>() + thread_index_z<u32>();
                    const auto size_x = raw_grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = raw_grid_size_y<u32>() * block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            }
        }

        template<nt::integer T = index_type>
        [[nodiscard]] NOA_FD auto thread_uid() const -> T {
            if constexpr (GRID_NDIM == 1) {
                if constexpr (BLOCK_NDIM == 1) {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    return static_cast<T>(index_x);
                } else if constexpr (BLOCK_NDIM == 2) {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = thread_index_y<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    return static_cast<T>(index_y * size_x + index_x);
                } else {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = thread_index_y<u32>();
                    const auto index_z = thread_index_z<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            } else if constexpr (GRID_NDIM == 2) {
                if constexpr (BLOCK_NDIM == 1) {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = block_index_y<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    return static_cast<T>(index_y * size_x + index_x);
                } else if constexpr (BLOCK_NDIM == 2) {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    return static_cast<T>(index_y * size_x + index_x);
                } else {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto index_z = thread_index_z<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = grid_size_y<u32>() * block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            } else {
                if constexpr (BLOCK_NDIM == 1) {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = block_index_y<u32>();
                    const auto index_z = block_index_z<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = grid_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                } else if constexpr (BLOCK_NDIM == 2) {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto index_z = block_index_z<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = grid_size_y<u32>() * block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                } else {
                    const auto index_x = block_index_x<u32>() * block_size_x<u32>() + thread_index_x<u32>();
                    const auto index_y = block_index_y<u32>() * block_size_y<u32>() + thread_index_y<u32>();
                    const auto index_z = block_index_z<u32>() * block_size_z<u32>() + thread_index_z<u32>();
                    const auto size_x = grid_size_x<u32>() * block_size_x<u32>();
                    const auto size_y = grid_size_y<u32>() * block_size_y<u32>();
                    return static_cast<T>((index_z * size_y + index_y) * size_x + index_x);
                }
            }
        }

    public:
        [[nodiscard]] NOA_FD auto total_scratch_size() const -> u32 {
            #ifdef NOA_ENABLE_CUDA
            u32 ret;
            asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r" (ret)); // PTX 4.1
            return ret;
            #else
            // TODO Is there a register in HIP to get the size?
            // TODO If SYCL backend is CUDA, use the PTX register?
            return m_total_scratch_size;
            #endif
        }

        [[nodiscard]] NOA_FD auto scratch_size() const -> u32 {
            if constexpr (IsSharingScratch)
                return m_user_scratch_size;
            else
                return total_scratch_size();
        }

        [[nodiscard]] NOA_FD auto has_scratch() const -> bool {
            return scratch_size() > 0;
        }

        template<typename T = std::byte>
        [[nodiscard]] NOA_FD auto scratch_pointer() const -> T* {
            #ifdef NOA_ENABLE_SYCL
            return reinterpret_cast<T*>(m_scratch);
            #else
            // This needs to be the only point in the program, otherwise nvcc (and hipcc?) complains.
            extern __shared__  std::byte shared_memory alignas(16)[];
            return reinterpret_cast<T*>(shared_memory);
            #endif
        }

        template<typename T = std::byte, typename I = index_type>
        [[nodiscard]] NOA_FD auto scratch() const -> SpanContiguous<T, 1, I> {
            return SpanContiguous<T, 1, I>(scratch_pointer<T>(), static_cast<I>(scratch_size() / sizeof(T)));
        }

        template<typename T = std::byte, typename I = index_type>
        NOA_FD auto zeroed_scratch_async() const -> SpanContiguous<T, 1, I> {
            // TODO For trivial types, 32-bit bank accesses would be best,
            //      but the scratch size might not be a multiple of 4 bytes.
            auto span = scratch<T, I>();
            for (I tid = thread_linear_index<I>(); tid < span.n_elements(); tid += block_size<I>())
                span[tid] = T{};
            return span;
        }

        template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
        static NOA_FD auto scratch_atomic_add(const nt::mutable_value_type_t<T>& value, const T& accessor, I... indices) {
            auto* pointer = accessor.offset_pointer(accessor.get(), indices...);
            return details::scratch_atomic_add(pointer, value);
        }

    public: // gpu-only
        /// Computes the ND indices within the grid.
        /// \param grid_fused_shape_in_x_unbatched:
        ///     Fused shape along grid.x, with the leftmost/outermost dimension popped out (see offset2index).
        ///     If N <= 3, the 3d grid is enough to contain the original shape,
        ///     so dimensions are not fused when constructing the grid, hence Vec<u32, 0>.
        template<nt::integer T, usize N, bool AddThreadIndices = true,
                 usize FusedShapePoppedN = (N <= 3 ? 0 : N - 3)>
        NOA_FD auto global_indices(const Vec<u32, FusedShapePoppedN>& grid_fused_shape_in_x_unbatched) const {
            if constexpr (N == 1) {
                auto gid = Vec{block_work_size_x<T>() * block_index_x<T>()};
                if constexpr (AddThreadIndices)
                    if constexpr (block_size_x<u32>() > 1)
                        gid[0] += thread_index_x<T>();
                return gid;

            } else if constexpr (N == 2) {
                auto gid = Vec{
                    static_cast<T>(block_work_size_y<u32>() * block_index_y<u32>()),
                    static_cast<T>(block_work_size_x<u32>() * block_index_x<u32>()),
                };
                if constexpr (AddThreadIndices) {
                    if constexpr (block_size_y<u32>() > 1)
                        gid[0] += thread_index_y<T>();
                    if constexpr (block_size_x<u32>() > 1)
                        gid[1] += thread_index_x<T>();
                }
                return gid;

            } else if constexpr (N == 3) {
                auto gid = Vec{
                    static_cast<T>(block_work_size_z<u32>() * block_index_z<u32>()),
                    static_cast<T>(block_work_size_y<u32>() * block_index_y<u32>()),
                    static_cast<T>(block_work_size_x<u32>() * block_index_x<u32>()),
                };
                if constexpr (AddThreadIndices) {
                    if constexpr (block_size_z<u32>() > 1)
                        gid[0] += thread_index_z<T>();
                    if constexpr (block_size_y<u32>() > 1)
                        gid[1] += thread_index_y<T>();
                    if constexpr (block_size_x<u32>() > 1)
                        gid[2] += thread_index_x<T>();
                }
                return gid;

            } else if constexpr (N >= 4) {
                // N=4: fused_shape=W   -> bid_unrolled=HW
                // N=5: fused_shape=HW  -> bid_unrolled=DHW
                // N=6: fused_shape=DHW -> bid_unrolled=BDHW
                const Vec<u32, N - 2> bid_unrolled = noa::offset2index(block_index_x<u32>(), grid_fused_shape_in_x_unbatched);
                Vec<T, N> gid;
                gid[0] = block_index_z<T>();
                if constexpr (N == 6) {
                    gid[1] = block_index_y<T>();
                    gid[2] = static_cast<T>(bid_unrolled[0]);
                    gid[3] = static_cast<T>(Block::block_work_size_z * bid_unrolled[1]);
                    gid[4] = static_cast<T>(Block::block_work_size_y * bid_unrolled[2]);
                    gid[5] = static_cast<T>(Block::block_work_size_x * bid_unrolled[3]);
                } else if constexpr (N == 5) {
                    gid[1] = block_index_y<T>();
                    gid[2] = static_cast<T>(Block::block_work_size_z * bid_unrolled[0]);
                    gid[3] = static_cast<T>(Block::block_work_size_y * bid_unrolled[1]);
                    gid[4] = static_cast<T>(Block::block_work_size_x * bid_unrolled[2]);
                } else if constexpr (N == 4) {
                    gid[1] = static_cast<T>(Block::block_work_size_z * block_index_y<u32>());
                    gid[2] = static_cast<T>(Block::block_work_size_y * bid_unrolled[0]);
                    gid[3] = static_cast<T>(Block::block_work_size_x * bid_unrolled[1]);
                }
                if constexpr (AddThreadIndices) {
                    if constexpr (block_size_z<u32>() > 1)
                        gid[N - 3] += thread_index_z<T>();
                    if constexpr (block_size_y<u32>() > 1)
                        gid[N - 2] += thread_index_y<T>();
                    if constexpr (block_size_x<u32>() > 1)
                        gid[N - 1] += thread_index_x<T>();
                }
                return gid;

            } else {
                static_assert(nt::always_false<T>);
            }
        }

        template<nt::integer T, usize N, bool AddThreadIndices = true> requires (N <= 3)
        NOA_FD auto global_indices() const {
            return global_indices<T, N, AddThreadIndices>(Vec<u32, 0>{});
        }

        [[nodiscard]] static constexpr auto warp_size() -> u32 {
            return Constant::WARP_SIZE;
        }

        /// Shuffles down by delta, so that thread i within the warp gets the value of thread i + delta.
        /// For i + delta >= warp_size(), the result is unspecified (CUDA/HIP return the threads' own value,
        /// SYCL is unspecified).
        template<nt::warp_shuffable T>
        [[nodiscard]] NOA_FD auto warp_shuffle_down(T value, u32 delta) const -> T {
            #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
            return details::warp_shuffle_down(value, delta);
            #elif defined(NOA_ENABLE_SYCL)
            return details::warp_shuffle_down(*m_nd_item, value, delta);
            #endif
        }

        /// Reduces one warp to one element.
        /// The first thread of the warp returns with the reduced value.
        /// The returned value is undefined for the other threads.
        template<typename BinaryOp, nt::warp_shuffable T>
        [[nodiscard]] NOA_FD auto warp_reduce(BinaryOp op, T value) const -> T {
            #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
            return details::warp_reduce(op, value);
            #elif defined(NOA_ENABLE_SYCL)
            return details::warp_reduce(*m_nd_item, op, value);
            #endif
        }

        /// Reduces one warp to one element.
        /// The first thread of the warp returns with the reduced value.
        /// The returned value is undefined for the other threads.
        template<typename Interface, typename Op, nt::tuple_of_accessor_value Reduced>
        [[nodiscard]] NOA_FD auto warp_reduce(Op op, Reduced reduced) const -> Reduced {
            #if defined(NOA_ENABLE_CUDA) || defined(NOA_ENABLE_HIP)
            return details::warp_reduce<Interface>(op, reduced);
            #elif defined(NOA_ENABLE_SYCL)
            return details::warp_reduce<Interface>(*m_nd_item, op, reduced);
            #endif
        }

    private:
        #if defined(NOA_ENABLE_SYCL)
        const sycl::nd_item<GRID_NDIM>* m_nd_item{};
        std::byte* m_scratch{};
        u32 m_total_scratch_size{};
        #elif defined(NOA_ENABLE_HIP)
        u32 m_total_scratch_size{};
        #endif
        NOA_NO_UNIQUE_ADDRESS scratch_size_type m_user_scratch_size{};
        NOA_NO_UNIQUE_ADDRESS vec_zy_type m_grid_size_zy{};
        NOA_NO_UNIQUE_ADDRESS vec_zy_type m_block_index_offset_zy{};
    };
}

namespace noa::traits {
    template<typename T, u32 G, typename B, bool F0, bool F1, bool F2>
    struct proclaim_is_compute_handle<noa::gpu::ComputeHandle<T, G, B, F0, F1, F2>> : std::true_type {};
}
