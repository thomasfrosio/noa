#pragma once
#include "noa/runtime/cuda/IncludeGuard.cuh"

#include "noa/base/Config.hpp"
#include "noa/base/Tuple.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/cuda/Pointers.hpp"
#include "noa/runtime/cuda/Warp.cuh"

namespace noa::cuda {
    /// Static initialization of shared variables is illegal in CUDA. Types that require an initialization
    /// cannot be used with the __shared__ attribute. This Uninitialized type bypasses this limit.
    /// \details Uninitialized<T> has the same size and alignment as T and is meant to be used for the
    /// declaration/allocation of static shared pointers/arrays. Then this type can be reinterpreted to T "safely"
    /// (we are most likely in C++ undefined behavior, but in CUDA C++ this is fine).
    /// \note nvcc 11.7 and above seem to support zero-initialization for __shared__, but still give warnings.
    template<typename T>
    struct alignas(alignof(T)) Uninitialized {
        unsigned char array[sizeof(T)];
    };

    /// Aligned array that generates vectorized load/store in CUDA.
    /// TODO Replace with Vec<T, VECTOR_SIZE, sizeof(T) * VECTOR_SIZE>?
    template<typename T, usize VECTOR_SIZE>
    struct alignas(sizeof(T) * VECTOR_SIZE) AlignedVector {
        T data[VECTOR_SIZE];
    };

    /// Aligned array used to generate vectorized load/store in CUDA.
    template<typename T, usize N, usize A>
    struct alignas(A) AlignedBuffer {
        using value_type = T;
        constexpr static usize SIZE = N;
        T data[N];
    };


    /// Collects and stores kernel argument addresses.
    /// This implies that the arguments should stay valid during the lifetime of this object.
    template<typename... Args>
    class CollectArgumentAddresses {
    public:
        /// The array of pointers is initialized with the arguments (which are not copied).
        /// Const-cast is required since CUDA expects void**.
        explicit CollectArgumentAddresses(Args&& ... args) :
            m_pointers{const_cast<void*>(static_cast<const void*>(&args))...} {}

        [[nodiscard]] auto pointers() -> void** { return static_cast<void**>(m_pointers); }

    private:
        void* m_pointers[max(usize{1}, sizeof...(Args))]{};
    };

    /// Returns the minimum address alignment for the given accessors.
    /// \details The vectorization happens along the width, so vectorization is turned off if any of the
    ///          width stride is not 1. This function checks that the alignment is preserved at the beginning
    ///          of every row. The block size and number of elements per thread is assumed to be a power of two
    ///          (in which case if the rows are aligned, the beginning of every block will be too).
    /// \param accessors    Tuple of 4d accessors, as this is intended for the *ewise core functions.
    ///                     AccessorValue is supported and preserves the vector size. Passing an empty
    ///                     tuple returns the maximum alignment (for global memory word-count), 16-byte.
    /// \param shape_bdh    BDH shape. Empty dimensions do not affect the alignment, so if certain
    ///                     dimensions are known to be contiguous, the dimension size can be set to 1
    ///                     to skip it.
    template<typename T, typename Index>
    requires (nt::tuple_of_accessor_nd<T, 4> or nt::empty_tuple<T>)
    constexpr auto min_address_alignment(
        const T& accessors,
        const Shape<Index, 3>& shape_bdh
    ) -> usize {
        auto get_alignment = [](const void* pointer) -> usize{
            // Global memory instructions support reading or
            // writing words of size equal to 1, 2, 4, 8, or 16 bytes.
            const auto address = reinterpret_cast<uintptr_t>(pointer);
            if (is_multiple_of(address, 16))
                return 16;
            if (is_multiple_of(address, 8))
                return 8;
            if (is_multiple_of(address, 4))
                return 4;
            if (is_multiple_of(address, 2))
                return 2;
            return 1;
        };

        usize alignment = 16;
        accessors.for_each([&]<typename U>(const U& accessor) {
            if constexpr (nt::accessor_pure<U>) {
                if (accessor.stride(3) == 1) {
                    usize i_alignment = get_alignment(accessor.get());
                    const auto strides = accessor.strides().template as_safe<usize>();

                    // Make sure every row is aligned to the current alignment.
                    // If not, try to decrease the alignment until reaching the minimum
                    // alignment for this type.
                    constexpr auto SIZE = sizeof(typename U::value_type);
                    for (; i_alignment >= 2; i_alignment /= 2) {
                        if ((shape_bdh[2] == 1 or is_multiple_of(strides[2] * SIZE, i_alignment)) and
                            (shape_bdh[1] == 1 or is_multiple_of(strides[1] * SIZE, i_alignment)) and
                            (shape_bdh[0] == 1 or is_multiple_of(strides[0] * SIZE, i_alignment)))
                            break;
                    }
                    alignment = min(alignment, i_alignment);
                } else {
                    // Since the vectorization is set up at compile time, we have no choice but
                    // to turn off the vectorization for everyone if one accessor is strided.
                    alignment = 1;
                }
            }
        });
        return alignment;
    }

    /// Computes the maximum vector size allowed for the given inputs/outputs.
    template<usize ALIGNMENT, typename... T>
    consteval auto maximum_allowed_aligned_buffer_size() -> usize {
        usize size{ALIGNMENT};
        auto get_size = [&]<typename V>() -> usize {
            using value_t = nt::mutable_value_type_t<V>;
            if constexpr (nt::accessor_value<V>) {
                return size; // AccessorValue shouldn't affect the vector size
            } else if constexpr (nt::accessor_pure<V> and is_power_of_2(sizeof(value_t))) {
                constexpr usize RATIO = sizeof(value_t) / alignof(value_t); // non naturally aligned types
                constexpr usize N = (ALIGNMENT / alignof(value_t)) / RATIO;
                return max(usize{1}, N); // clamp to one for cases where ALIGNMENT < alignof(value_t)
            } else {
                static_assert(nt::accessor_pure<V>);
                // If size is not a power of two, memory accesses cannot fully coalesce;
                // there's no point in increasing the word count.
                return 1;
            }
        };

        // To fully coalesce and to ensure that threads work on the same elements,
        // we have to use the same vector size for all inputs/outputs.
        constexpr auto accessors = (nt::type_list_t<T>{} + ...);
        [&]<typename... U>(nt::TypeList<U...>) {
            ((size = min(size, get_size.template operator()<U>())), ...);
        }(accessors);
        return size;
    }

    template<typename T, usize ALIGNMENT, usize N>
    struct to_aligned_buffer {
        template<typename U>
        static constexpr auto get_type() {
            using value_t = nt::mutable_value_type_t<U>;
            if constexpr (nt::accessor_pure<U> and is_power_of_2(sizeof(value_t))) {
                constexpr usize RATIO = sizeof(value_t) / alignof(value_t); // non naturally aligned types
                constexpr usize AA = min(alignof(value_t) * N * RATIO, ALIGNMENT);
                constexpr usize A = max(AA, alignof(value_t));
                return std::type_identity<AlignedBuffer<value_t, N, A>>{};
            } else {
                return std::type_identity<AlignedBuffer<value_t, N, alignof(value_t)>>{};
            }
        }

        template<typename... U>
        static constexpr auto get(nt::TypeList<U...>) {
            return std::type_identity<Tuple<typename decltype(get_type<U>())::type...>>{};
        }

        using type = decltype(get(nt::type_list_t<T>{}))::type;
    };
    template<typename T, usize ALIGNMENT, usize N>
    using to_aligned_buffer_t = to_aligned_buffer<T, ALIGNMENT, N>::type;

    /// Whether the aligned buffers are actually over-aligned compared to the original type.
    /// In other words, whether using vectorized loads/stores is useful.
    /// This is used to fall back on a non-vectorized implementation at compile time,
    /// reducing the number of kernels that need to be generated.
    template<typename... T>
    constexpr auto is_vectorized() -> usize {
        constexpr auto aligned_buffers = (nt::type_list_t<T>{} + ...);
        if constexpr (nt::empty_tuple<T...>) {
            return false; // no inputs and no outputs
        } else {
            return []<typename... U>(nt::TypeList<U...>) {
                return ((alignof(U) > alignof(typename U::value_type)) or ...);
            }(aligned_buffers);
        }
    }

    /// Returns the number of bytes of dynamic shared memory to allocate to statisfy the user (scratch size)
    /// and the internal block join (which needs block_size Reduced elements). The key here is that the lifetimes
    /// of these two do not overlap: init/call/deinit is where the scratch space is available, and it's only after
    /// that (during join) that we reduce the block behind the scene. As such, we can overlap the two buffers.
    template<typename Reduced>
    constexpr auto n_bytes_of_shared_memory_to_allocate_for_reduction(u32 block_size, usize scratch_size) {
        constexpr bool HAS_REDUCED = not nt::empty_tuple<std::remove_reference_t<Reduced>>;
        const auto n_bytes_for_join = HAS_REDUCED ? static_cast<usize>(block_size) * sizeof(Reduced) : usize{};
        return std::max(n_bytes_for_join, scratch_size);
    }

    /// Sets up the grid necessary for the block to loop through the problem size.
    /// This grid can be divided into multiple kernel launches to bypass CUDA's grid-size limits.
    /// \param T    Integer type used by the kernel for indexing.
    /// \param S    CUDA maximum grid size along the dimension.
    template<isize S>
    class Grid {
    public:
        constexpr Grid(nt::integer auto size, u32 block_work_size) {
            m_n_blocks_total = divide_up(safe_cast<isize>(size), static_cast<isize>(block_work_size));
            m_n_blocks_per_launch = min(m_n_blocks_total, S);
            m_n_launches = safe_cast<u32>(divide_up(m_n_blocks_total, m_n_blocks_per_launch));

            auto max_offset = m_n_blocks_per_launch * static_cast<isize>(m_n_launches - 1);
            check(is_safe_cast<u32>(max_offset),
                  "The grid offset is larger than the maximum supported offset. "
                  "size={}, block_work_size={}, max_grid_size={}",
                  size, block_work_size, S);
        }

        [[nodiscard]] constexpr auto n_launches() const -> u32 { return m_n_launches; }
        [[nodiscard]] constexpr auto n_blocks_total() const -> isize { return m_n_blocks_total; }
        [[nodiscard]] constexpr auto n_blocks(u32 launch) const -> u32 {
            check(launch < m_n_launches);
            auto offset = m_n_blocks_per_launch * static_cast<isize>(launch);
            auto left = m_n_blocks_total - offset;
            return static_cast<u32>(std::min(left, m_n_blocks_per_launch));
        }
        [[nodiscard]] constexpr auto offset(u32 launch) const -> u32 {
            return static_cast<u32>(m_n_blocks_per_launch * static_cast<isize>(launch));
        }
        [[nodiscard]] constexpr auto offset_additive(u32 launch) const -> u32 {
            check(launch < m_n_launches);
            if (launch)
                return static_cast<u32>(m_n_blocks_per_launch);
            return 0;
        }

    private:
        isize m_n_blocks_total;
        isize m_n_blocks_per_launch;
        u32 m_n_launches;
    };

    template<isize S>
    class GridFused {
    public:
        constexpr GridFused(
            nt::integer auto size_x,
            nt::integer auto size_y,
            u32 block_work_size_x,
            u32 block_work_size_y
        ) {
            const isize n_blocks_x = divide_up(safe_cast<isize>(size_x), static_cast<isize>(block_work_size_x));
            const isize n_blocks_y = divide_up(safe_cast<isize>(size_y), static_cast<isize>(block_work_size_y));
            m_n_blocks_x = safe_cast<u32>(n_blocks_x);
            m_n_blocks_total = n_blocks_x * n_blocks_y;
            m_n_blocks_per_launch = min(m_n_blocks_total, S);
            m_n_launches = safe_cast<u32>(divide_up(m_n_blocks_total, m_n_blocks_per_launch));

            auto max_offset = m_n_blocks_per_launch * static_cast<isize>(m_n_launches - 1);
            check(is_safe_cast<u32>(max_offset),
                  "The grid offset is larger than the maximum supported offset. "
                  "size_x={}, size_y={}, block_work_size_x={}, block_work_size_y={}, max_grid_size={}",
                  size_x, size_y, block_work_size_x, block_work_size_y, S);
        }
        [[nodiscard]] constexpr auto n_launches() const -> u32 { return m_n_launches; }
        [[nodiscard]] constexpr auto n_blocks_total() const -> isize { return m_n_blocks_total; }
        [[nodiscard]] constexpr auto n_blocks_x() const -> u32 { return m_n_blocks_x; }
        [[nodiscard]] constexpr auto n_blocks(u32 launch) const -> u32 {
            auto offset = m_n_blocks_per_launch * static_cast<isize>(launch);
            auto left = m_n_blocks_total - offset;
            return static_cast<u32>(std::min(left, m_n_blocks_per_launch));
        }
        [[nodiscard]] constexpr auto offset(u32 launch) const -> u32 {
            return static_cast<u32>(m_n_blocks_per_launch * static_cast<isize>(launch));
        }

    private:
        isize m_n_blocks_per_launch;
        isize m_n_blocks_total;
        u32 m_n_launches;
        u32 m_n_blocks_x;
    };

    using GridX = Grid<2'147'483'647>;
    using GridY = Grid<65'535>;
    using GridZ = Grid<65'535>;
    using GridXY = GridFused<2'147'483'647>;
}

namespace noa::cuda::details {
        template<typename Tup>
    struct vectorized_tuple { using type = Tup; };

    template<typename... T>
    struct vectorized_tuple<Tuple<T...>> { using type = Tuple<AccessorValue<typename T::value_type>...>; };

    /// Convert a tuple of accessors to a tuple of AccessorValue.
    /// This is used to store the values for vectorized load/stores, while preserving compatibility with the core
    /// interfaces. Note that the constness is preserved, so to access the values from the AccessorValue,
    /// .ref_() should be used.
    template<typename T>
    using vectorized_tuple_t = vectorized_tuple<std::decay_t<T>>::type;

    template<usize N, typename Index, typename T>
    struct joined_tuple { using type = T; };

    template<usize N, typename Index, typename... T>
    struct joined_tuple<N, Index, Tuple<T...>> {
        using type = Tuple<AccessorRestrictContiguous<typename T::mutable_value_type, N, Index>...>;
    };

    /// Convert a tuple of AccessorValue(s) to a tuple of N-d AccessorsContiguousRestrict.
    /// This is used to store the reduced values of reduce_ewise or reduce_axes_ewise in a struct of arrays,
    /// which can then be loaded as an input using vectorized loads. Note that the constness of the AccessorValue(s)
    /// has to be dropped (because kernels do expect to write then read from these accessors), but this is fine since
    /// the reduced AccessorValue(s) should not be const anyway.
    template<usize N, typename Index, typename T>
    using joined_tuple_t = joined_tuple<N, Index, std::decay_t<T>>::type;

    /// Synchronizes the block.
    /// TODO Cooperative groups may be the way to go and do offer more granularity.
    NOA_FD void block_synchronize() {
        __syncthreads();
    }

    template<nt::integer T = u32, usize N = 3>
    NOA_FD auto block_indices() -> Vec<T, N> {
        if constexpr (N == 3)
            return Vec<T, N>::from_values(blockIdx.z, blockIdx.y, blockIdx.x);
        else if constexpr (N == 2)
            return Vec<T, N>::from_values(blockIdx.y, blockIdx.x);
        else if constexpr (N == 1)
            return Vec<T, N>::from_values(blockIdx.x);
        else
            return Vec<T, 0>{};
    }

    template<nt::integer T = u32, usize N = 3>
    NOA_FD auto thread_indices() -> Vec<T, N> {
        if constexpr (N == 3)
            return Vec<T, N>::from_values(threadIdx.z, threadIdx.y, threadIdx.x);
        else if constexpr (N == 2)
            return Vec<T, N>::from_values(threadIdx.y, threadIdx.x);
        else if constexpr (N == 1)
            return Vec<T, N>::from_values(threadIdx.x);
        else
            return Vec<T, 0>{};
    }

    template<nt::integer T, typename Block>
    NOA_FD auto global_indices_4d(u32 grid_size_x, const Vec<u32, 2>& block_offset_zy = Vec<u32, 2>{}) {
        auto bid = block_indices<T, 3>();
        bid[0] += block_offset_zy[0];
        bid[1] += block_offset_zy[1];

        const Vec<T, 2> bid_yx = offset2index(bid[2], static_cast<T>(grid_size_x));
        auto gid = Vec{
            bid[0],
            static_cast<T>(Block::block_work_size_z) * bid[1],
            static_cast<T>(Block::block_work_size_y) * bid_yx[0],
            static_cast<T>(Block::block_work_size_x) * bid_yx[1],
        };
        if constexpr (Block::block_ndim == 3)
            gid[1] += static_cast<T>(threadIdx.z);
        if constexpr (Block::block_ndim >= 2)
            gid[2] += static_cast<T>(threadIdx.y);
        if constexpr (Block::block_ndim >= 1)
            gid[3] += static_cast<T>(threadIdx.x);
        return gid;
    }

    template<nt::integer T, typename Config>
    NOA_FD auto global_indices_3d(const Vec<u32, 2>& block_offset_zy = Vec<u32, 2>{}) {
        auto bid = block_indices<T, 3>();
        bid[0] += block_offset_zy[0];
        bid[1] += block_offset_zy[1];

        auto gid = Vec{
            static_cast<T>(Config::block_work_size_z) * static_cast<T>(bid[0]),
            static_cast<T>(Config::block_work_size_y) * static_cast<T>(bid[1]),
            static_cast<T>(Config::block_work_size_x) * static_cast<T>(bid[2]),
        };
        if constexpr (Config::block_ndim == 3)
            gid[0] += static_cast<T>(threadIdx.z);
        if constexpr (Config::block_ndim >= 2)
            gid[1] += static_cast<T>(threadIdx.y);
        if constexpr (Config::block_ndim >= 1)
            gid[2] += static_cast<T>(threadIdx.x);
        return gid;
    }

    template<nt::integer T, typename Config>
    NOA_FD auto global_indices_2d(const Vec<u32, 1>& block_offset_y = Vec<u32, 1>{}) {
        auto bid = block_indices<T, 2>();
        bid[0] += block_offset_y[0];

        auto gid = Vec{
            static_cast<T>(Config::block_work_size_y) * bid[0],
            static_cast<T>(Config::block_work_size_x) * bid[1],
        };
        if constexpr (Config::block_ndim >= 2)
            gid[0] += static_cast<T>(threadIdx.y);
        if constexpr (Config::block_ndim >= 1)
            gid[1] += static_cast<T>(threadIdx.x);
        return gid;
    }

    template<nt::integer T, typename Config>
    NOA_FD auto global_indices_1d() {
        const auto bid = block_indices<T, 1>();
        const auto tid = thread_indices<T, 1>();
        return Vec{
            static_cast<T>(Config::block_work_size_x) * bid[0] + tid[0],
        };
    }

    /// Returns a per-thread unique ID, i.e., each thread in the grid gets a unique value.
    template<usize N = 0>
    NOA_FD u32 thread_uid() {
        if constexpr (N == 1) {
            return blockIdx.x * blockDim.x + threadIdx.x;
        } else if constexpr (N == 2) {
            const auto bid_x = blockIdx.x * blockDim.x + threadIdx.x;
            const auto bid_y = blockIdx.y * blockDim.y + threadIdx.y;
            return bid_y * blockDim.x + bid_x;
        } else {
            const auto bid_x = blockIdx.x * blockDim.x + threadIdx.x;
            const auto bid_y = blockIdx.y * blockDim.y + threadIdx.y;
            const auto bid_z = blockIdx.z * blockDim.z + threadIdx.z;
            return (bid_z * blockDim.y + bid_y) * blockDim.x + bid_x;
        }
    }

        NOA_FD i32 atomic_add(i32* address, i32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD u32 atomic_add(u32* address, u32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD u64 atomic_add(u64* address, u64 val) {
        return ::atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
    }

    #if __CUDA_ARCH__ >= 700
    NOA_FD f16 atomic_add(f16* address, f16 val) {
        return f16(::atomicAdd(reinterpret_cast<__half*>(address), val.native()));
        // atomicCAS for ushort requires 700 as well, so I don't think there's an easy way to do atomics
        // on 16-bits values on 5.3 and 6.X devices...
    }
    #endif

    NOA_FD f32 atomic_add(f32* address, f32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_ID f64 atomic_add(f64* address, f64 val) {
        #if __CUDA_ARCH__ < 600
        using ull = unsigned long long;
        auto* address_as_ull = (ull*) address;
        ull old = *address_as_ull;
        ull assumed;

        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old); // uses integer comparison to avoid hanging in case of NaN (since NaN != NaN)

        return __longlong_as_double(old); // like every other atomicAdd, return old
        #else
        return ::atomicAdd(address, val);
        #endif
    }

    template<typename T>
    NOA_FD Complex<T> atomic_add(Complex<T>* address, Complex<T> val) {
        return {atomicAdd(&(address->real), val.real), atomicAdd(&(address->imag), val.imag)};
    }

    /// Returns the pointer to dynamic shared memory.
    template<typename T = std::byte>
    [[nodiscard]] NOA_FD auto dynamic_shared_memory_pointer() -> T* {
        // This needs to be the only point in the program, otherwise nvcc complains.
        extern __shared__  std::byte shared_memory alignas(std::max_align_t)[];
        return reinterpret_cast<T*>(shared_memory);
    }

    /// Returns the number of bytes of shared memory.
    [[nodiscard]] NOA_FD auto dynamic_shared_memory_size() -> u32 {
        u32 ret;
        asm volatile ("mov.u32 %0, %dynamic_smem_size;" : "=r" (ret)); // PTX 4.1
        return ret;
    }
}
