#pragma once
#include "noa/runtime/gpu/IncludeGuard.hpp"

#include "noa/base/Config.hpp"
#include "noa/base/Tuple.hpp"
#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Span.hpp"
#include "noa/runtime/gpu/Pointers.hpp"
#include "noa/runtime/gpu/Warp.hpp"
#include "noa/runtime/gpu/Constants.hpp"

namespace noa::gpu {
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

    /// Returns the minimum address alignment of any given row for the given accessors.
    /// \warning
    ///     The block size and number of elements per thread is assumed to be a power of two.
    ///     This guarantees that if the rows are aligned, the beginning of every block will be too.
    /// \note
    ///     If the returned alignment is > 1, the accessors can safely be set to StridesTraits::CONTIGUOUS.
    ///     Indeed, if the width is not contiguous, vectorization isn't possible and 1 is returned.
    /// \param accessors:
    ///     Tuple of rightmost accessors.
    ///     AccessorValue is supported and preserves the vector size.
    ///     Passing an empty tuple returns the maximum alignment of 16 bytes (for global memory word-count).
    /// \param shape_without_width:
    ///     Rightmost shape, without the width.
    ///     Empty dimensions do not affect the alignment, so if certain dimensions are known to be contiguous,
    ///     the dimension size can be set to 1 to skip it.
    template<typename T, typename Index, usize N>
        requires (nt::tuple_of_accessor_nd<T, N + 1> or nt::empty_tuple<T>)
    constexpr auto min_address_alignment(
        const T& accessors,
        const Shape<Index, N>& shape_without_width
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
                // If rightmost dimension is contiguous, find the minimum alignment of any given row.
                if (accessor.stride(N) == 1) {
                    usize i_alignment = get_alignment(accessor.get());

                    if constexpr (N >= 1) {
                        // If one row was found with a lower alignment, start over by decreasing
                        // the alignment and repeat until the minimum alignment for this type is reached.
                        constexpr auto SIZE = sizeof(typename U::value_type);
                        const auto strides = accessor.strides().template as_safe<usize>();
                        for (; i_alignment >= 2; i_alignment /= 2) {
                            i32 aligned{};
                            for (usize i{}; i < N; ++i)
                                if (shape_without_width[i] == 1 or noa::is_multiple_of(strides[i] * SIZE, i_alignment))
                                    ++aligned;
                            if (aligned == N)
                                break;
                        }
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

    constexpr auto is_static_scratch_enabled() -> bool {
#ifdef NOA_ENABLE_SYCL
        return false;
#else
        return true;
#endif
    }

    /// Returns the number of bytes of dynamic shared memory to allocate to statisfy the user (scratch size)
    /// and the internal block join (which needs block_size Reduced elements). The key here is that the lifetimes
    /// of these two do not overlap: init/call/deinit is where the scratch space is available, and it's only after
    /// that (during join) that we reduce the block behind the scene. As such, we can overlap the two buffers.
    template<typename Reduced, nt::any_of<u32, usize> T>
    constexpr auto n_bytes_of_scratch_to_allocate_for_reduction(u32 block_size, T scratch_size) {
        constexpr bool HAS_REDUCED = not nt::empty_tuple<std::remove_reference_t<Reduced>>;
        const auto n_bytes_for_join = HAS_REDUCED ? static_cast<usize>(block_size) * sizeof(Reduced) : usize{};
        return std::max(n_bytes_for_join, static_cast<usize>(scratch_size));
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

    /// Construct a 3d grid from a nd-shape by fusing the rightmost dimensions along the x dimension of the grid.
    template<usize N>
    class GridND {
    public:
        static constexpr u32 N_DIMENSIONS_IN_X = N >= 4 ? N - 2 : 1;

        template<typename T>
        constexpr GridND(
            Shape<T, N> shape,
            Shape<u32, N> block_work_shape
        ) {
            auto n_blocks = Vec<u32, N>{};
            for (usize i{}; i < N; ++i)
                n_blocks[i] = safe_cast<u32>(divide_up(
                    safe_cast<usize>(shape[i]), static_cast<usize>(block_work_shape[i])));

            if constexpr (N == 1) {
                m_fused_shape = n_blocks[0];
                m_shape = {1, 1, n_blocks[0]};
            } else if constexpr (N == 2) {
                m_fused_shape = n_blocks[1];
                m_shape = {1, n_blocks[0], n_blocks[1]};
            } else if constexpr (N >= 3) {
                m_fused_shape.vec = n_blocks.template pop_front<2>();
                m_shape = {n_blocks[0], n_blocks[1], m_fused_shape.n_elements()};
            }
            m_n_blocks_per_launch = {
                min(m_shape[0], Limits::MAX_YZ_BLOCKS),
                min(m_shape[1], Limits::MAX_YZ_BLOCKS),
                min(m_shape[2], Limits::MAX_X_BLOCKS),
            };
            for (usize i{}; i < 3; ++i) {
                m_n_launches[i] = safe_cast<u32>(divide_up(
                    static_cast<usize>(m_shape[i]),
                    static_cast<usize>(m_n_blocks_per_launch[i])
                ));
            }

            const auto max_offset = m_n_blocks_per_launch.as<isize>() * (m_n_launches.as<isize>() - 1);
            check(noa::is_safe_cast<Vec<u32, 3>>(max_offset),
                  "The grid is larger than the maximum allowed. shape={}, block_work_shape={}, grid_shape={}",
                  shape, block_work_shape, m_shape);
        }

        [[nodiscard]] constexpr auto shape() const -> const auto& { return m_shape; }
        [[nodiscard]] constexpr auto outer_shape() const -> auto { return m_shape.pop_back(); }
        [[nodiscard]] constexpr auto fused_shape() const -> const auto& { return m_fused_shape; }

        [[nodiscard]] constexpr auto n_launches_z() const -> u32 { return m_n_launches[0]; }
        [[nodiscard]] constexpr auto n_launches_y() const -> u32 { return m_n_launches[1]; }
        [[nodiscard]] constexpr auto n_launches_x() const -> u32 { return m_n_launches[2]; }
        [[nodiscard]] constexpr auto n_launches() const -> const auto& { return m_n_launches; }

        template<usize D>
        [[nodiscard]] constexpr auto n_blocks(u32 launch) const -> u32 {
            auto offset = static_cast<isize>(m_n_blocks_per_launch[D]) * static_cast<isize>(launch);
            auto left = static_cast<isize>(m_shape[D]) - offset;
            return safe_cast<u32>(std::min(left, static_cast<isize>(m_n_blocks_per_launch[D])));
        }
        [[nodiscard]] constexpr auto n_blocks_z(u32 launch) const -> u32 { return n_blocks<0>(launch); }
        [[nodiscard]] constexpr auto n_blocks_y(u32 launch) const -> u32 { return n_blocks<1>(launch); }
        [[nodiscard]] constexpr auto n_blocks_x(u32 launch) const -> u32 { return n_blocks<2>(launch); }
        [[nodiscard]] constexpr auto shape_for_launch(u32 z, u32 y, u32 x) const -> Vec<u32, 3> {
            return Vec{n_blocks_z(z), n_blocks_y(y), n_blocks_x(x)};
        }

        [[nodiscard]] constexpr auto block_offset_for_launch(u32 z, u32 y) const -> Vec<u32, 2> {
            return Vec{
                static_cast<u32>(m_n_blocks_per_launch[0] * static_cast<isize>(z)),
                static_cast<u32>(m_n_blocks_per_launch[1] * static_cast<isize>(y)),
            };
        }
        [[nodiscard]] constexpr auto block_offset_for_launch(u32 z, u32 y, u32 x) const -> Vec<u32, 3> {
            return Vec{
                static_cast<u32>(m_n_blocks_per_launch[0] * static_cast<isize>(z)),
                static_cast<u32>(m_n_blocks_per_launch[1] * static_cast<isize>(y)),
                static_cast<u32>(m_n_blocks_per_launch[2] * static_cast<isize>(x)),
            };
        }

        [[nodiscard]] constexpr auto incremental_block_offset_for_launch(u32 z, u32 y) const -> Vec<u32, 2> {
            check(z < m_n_launches[0] and y < m_n_launches[1]);
            return Vec{
                z ? static_cast<u32>(m_n_blocks_per_launch[0]) : 0,
                y ? static_cast<u32>(m_n_blocks_per_launch[1]) : 0,
            };
        }
        [[nodiscard]] constexpr auto incremental_block_offset_for_launch(u32 z, u32 y, u32 x) const -> Vec<u32, 3> {
            check(z < m_n_launches[0] and y < m_n_launches[1] and x < m_n_launches[2]);
            return Vec{
                z ? static_cast<u32>(m_n_blocks_per_launch[0]) : 0,
                y ? static_cast<u32>(m_n_blocks_per_launch[1]) : 0,
                x ? static_cast<u32>(m_n_blocks_per_launch[2]) : 0,
            };
        }

    private:
        Shape<u32, 3> m_shape;
        Vec<u32, 3> m_n_blocks_per_launch;
        Vec<u32, 3> m_n_launches;
        Shape<u32, N_DIMENSIONS_IN_X> m_fused_shape;
    };
}

namespace noa::gpu::details {
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
}

// TODO Some of these functions could be moved to ComputeHandle.
namespace noa::gpu::details {
    // Get a warp count that makes block_size_x a multiple of block_size.
    template<typename Config, u32 WarpCountBlockSizeX>
    constexpr auto best_block_size_x() {
        u32 c = max(1u, WarpCountBlockSizeX);
        for (;;) {
            if (Constant::WARP_SIZE * c <= Config::block_size and
                is_multiple_of(Config::block_size, Constant::WARP_SIZE * c)) {
                break;
                }
            --c; // try a block size smaller by one warp
        }
        return Constant::WARP_SIZE * c;
    }

    template<typename Block>
    constexpr auto block_shape() -> Vec<u32, 3> {
        return Vec{Block::block_size_z, Block::block_size_y, Block::block_size_x};
    }
}
