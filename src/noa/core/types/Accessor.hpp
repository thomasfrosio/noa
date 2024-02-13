#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/indexing/Offset.hpp"

namespace noa::guts {
    template<typename A, size_t... Is, typename Pointer, typename... Indices>
    requires (nt::is_accessor_v<A> and nt::are_int_v<Indices...>)
    [[nodiscard]] NOA_FHD constexpr auto offset_pointer(
            A&& accessor, Pointer pointer,
            std::index_sequence<Is...>,
            Indices... indices
    ) noexcept {
        ((pointer += ni::offset_at(indices, accessor.template stride<Is>())), ...);
        return pointer;
    }

    template<typename A, size_t... Is, typename Pointer, typename Integer>
    requires nt::is_accessor_v<A>
    [[nodiscard]] NOA_FHD constexpr auto offset_pointer(
            A&& accessor, Pointer pointer,
            std::index_sequence<Is...>,
            const Vec<Integer, sizeof...(Is)>& indexes
    ) noexcept {
        ((pointer += ni::offset_at(indexes[Is], accessor.template stride<Is>())), ...);
        return pointer;
    }

    template<typename A, size_t... Is, typename... Indices>
    requires (nt::is_accessor_v<A> and nt::are_int_v<Indices...>)
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            A&& accessor,
            std::index_sequence<Is...>,
            Indices... indices
    ) noexcept {
        nt::index_type_t<A> offset{};
        ((offset += ni::offset_at(indices, accessor.template stride<Is>())), ...);
        return offset;
    }

    template<typename A, size_t... Is, typename Integer>
    requires nt::is_accessor_v<A>
    [[nodiscard]] NOA_FHD constexpr auto offset_at(
            A&& accessor,
            std::index_sequence<Is...>,
            const Vec<Integer, sizeof...(Is)>& indexes
    ) noexcept {
        nt::index_type_t<A> offset{};
        ((offset += ni::offset_at(indexes[Is], accessor.template stride<Is>())), ...);
        return offset;
    }
}

namespace noa::inline types {
    enum class PointerTraits { DEFAULT, RESTRICT }; // TODO ATOMIC?
    enum class StridesTraits { STRIDED, CONTIGUOUS };

    template<typename T, size_t N, typename I,
             PointerTraits PointerTrait,
             StridesTraits StridesTrait>
    class AccessorReference;

    /// Multidimensional accessor; wraps a pointer and nd-strides, and provides nd-indexing.
    /// \details
    /// Accessors are mostly intended for internal use, which affected some design choices. Noticeable features:
    /// >>> The size of the dimensions are not stored, so accessors cannot bound-check the indexes against
    ///     their dimension size. In a lot of cases, the input/output arrays have the same size/shape and
    ///     the size/shape is often not needed by the compute kernels, leading to storing useless data.
    ///     If the extents of the region are required, use (md)spans.
    /// >>> \b Pointer-traits. By default, the pointers are not marked with any attributes, but the "restrict"
    ///     traits can be added. This is useful to signify that pointers don't alias, which helps generating
    ///     better code. Unfortunately, only g++ seem to acknowledge the restrict attribute on pointers
    ///     inside structs (details below)...
    /// >>> \b Strides-traits. Strides are fully dynamic (one dynamic stride per dimension) by default,
    ///     but the rightmost dimension can be marked contiguous. Accessors (and the library internals) uses
    ///     the rightmost convention, so that the innermost dimension is the rightmost dimension.
    ///     As such, StridesTraits::CONTIGUOUS implies C-contiguous. F-contiguous layouts are not supported
    ///     by the accessors, as these layouts should be reordered to C-contiguous before creating the
    ///     contiguous accessor.
    ///     With StridesTraits::CONTIGUOUS, the innermost/rightmost stride is fixed to 1 and is not stored,
    ///     resulting in the strides being truncated by 1 (Strides<I,N-1>). In case of a 1d contiguous
    ///     accessor, this mean that the strides are empty (Strides<I,0>) and the indexing is equivalent
    ///     to pointer/array indexing.
    template<typename T, size_t N, typename I,
             PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED>
    class Accessor {
    public:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);
        static_assert(std::is_integral_v<I>);
        static_assert(N <= 4);

        static constexpr PointerTraits pointer_trait = PointerTrait;
        static constexpr StridesTraits strides_trait = StridesTrait;
        static constexpr bool IS_RESTRICT = PointerTrait == PointerTraits::RESTRICT;
        static constexpr bool IS_CONTIGUOUS = StridesTrait == StridesTraits::CONTIGUOUS;
        static constexpr size_t SIZE = N;
        static constexpr int64_t SSIZE = N;

        // nvcc ignores __restrict__ when applied to member variables... https://godbolt.org/z/9GY5hGEzr
        // gcc/msvc optimizes it, but clang ignores it as well... https://godbolt.org/z/r869cb34s
        #if defined(__CUDACC__)
        using pointer_type = std::conditional_t<IS_RESTRICT, T* __restrict__, T*>;
        #else
        using pointer_type = std::conditional_t<IS_RESTRICT, T* __restrict, T*>;
        #endif
        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using index_type = I;
        using reference_type = T&;
        using strides_type = std::conditional_t<!IS_CONTIGUOUS, Strides<index_type, N>, Strides<index_type, N - 1>>;
        using accessor_reference_type = AccessorReference<value_type, SIZE, index_type, PointerTrait, StridesTrait>;

    public: // Constructors
        NOA_HD constexpr Accessor() = default;

        /// Creates a strided or contiguous accessor.
        /// If the accessor is contiguous, the width stride (strides[SIZE-1]) is ignored and assumed to be 1.
        NOA_HD constexpr Accessor(pointer_type pointer, const Strides<index_type, SIZE>& strides) noexcept
                : m_ptr{pointer},
                  m_strides{strides_type::from_pointer(strides.data())} {}

        /// Creates a contiguous accessor from contiguous strides.
        NOA_HD constexpr Accessor(pointer_type pointer, const strides_type& strides) noexcept requires IS_CONTIGUOUS
                : m_ptr{pointer},
                  m_strides{strides} {}

        /// Creates an accessor from an accessor reference.
        NOA_HD constexpr explicit Accessor(accessor_reference_type accessor_reference) noexcept
                : m_ptr{accessor_reference.get()},
                  m_strides{strides_type::from_pointer(accessor_reference.strides())} {}

        /// Creates a contiguous 1d accessor, assuming the stride is 1.
        NOA_HD constexpr explicit Accessor(pointer_type pointer) noexcept requires (SIZE == 1 && IS_CONTIGUOUS)
                : m_ptr{pointer} {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename U> requires nt::is_mutable_value_type_v<U, value_type>
        NOA_HD constexpr /* implicit */ Accessor(const Accessor<U, N, I, PointerTrait, StridesTrait>& accessor)
                : m_ptr{accessor.get()}, m_strides{accessor.strides()} {}

    public: // Accessing strides
        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr index_type stride() const noexcept {
            static_assert(INDEX < N);
            if constexpr (IS_CONTIGUOUS && INDEX == SIZE - 1)
                return index_type{1};
            else
                return m_strides[INDEX];
        }

        [[nodiscard]] NOA_HD constexpr index_type stride(std::integral auto index) const noexcept {
            NOA_ASSERT(!is_empty() and static_cast<i64>(index) < SSIZE);
            if (IS_CONTIGUOUS and index == SIZE - 1)
                return index_type{1};
            else
                return m_strides[index];
        }

        [[nodiscard]] NOA_HD constexpr strides_type& strides() noexcept { return m_strides; }
        [[nodiscard]] NOA_HD constexpr const strides_type& strides() const noexcept { return m_strides; }

    public:
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !is_empty(); }

        NOA_HD constexpr void reset_pointer(pointer_type new_pointer) noexcept { m_ptr = new_pointer; }

        template<std::integral Int>
        NOA_FHD constexpr Accessor& reorder(const Vec<Int, N>& order) noexcept {
            strides() = strides().reorder(order);
            return *this;
        }

    public:
        /// C-style indexing operator, decrementing the dimensionality of the accessor by 1.
        template<typename Int> requires (SIZE > 1 && std::is_integral_v<Int>)
        [[nodiscard]] NOA_HD auto operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            using output_type = AccessorReference<value_type, (SIZE - 1), index_type, PointerTrait, StridesTrait>;
            return output_type(m_ptr + ni::offset_at(index, stride<0>()), strides().data() + 1);
        }

        /// C-style indexing operator, decrementing the dimensionality of the accessor by 1.
        /// When done on a 1d accessor, this acts as a pointer/array indexing and dereferences the data.
        template<typename Int> requires (SIZE == 1 && std::is_integral_v<Int>)
        [[nodiscard]] NOA_HD reference_type operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            return m_ptr[ni::offset_at(index, stride<0>())];
        }

    public:
        template<typename... Indexes>
        requires ((SIZE >= sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>))
        NOA_HD constexpr Accessor& offset_accessor(Indexes... indexes) noexcept {
            NOA_ASSERT(!is_empty());
            if constexpr (nt::are_int_v<Indexes...>) {
                m_ptr = guts::offset_pointer(*this, m_ptr, std::make_index_sequence<sizeof...(Indexes)>{}, indexes...);
            } else {
                constexpr size_t VEC_SIZE = std::decay_t<nt::first_t<Indexes...>>::SIZE;
                static_assert(VEC_SIZE <= SIZE);
                m_ptr = guts::offset_pointer(*this, m_ptr, std::make_index_sequence<VEC_SIZE>{}, indexes...);
            }
            return *this;
        }

        template<typename P, typename... Indexes>
        requires (((SIZE >= sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                   (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>)) and
                  std::is_pointer_v<P>)
        [[nodiscard]] NOA_HD constexpr P offset_pointer(P pointer, Indexes&&... indexes) const noexcept {
            if constexpr (nt::are_int_v<Indexes...>) {
                return guts::offset_pointer(*this, pointer, std::make_index_sequence<sizeof...(Indexes)>{}, indexes...);
            } else {
                constexpr size_t VEC_SIZE = std::decay_t<nt::first_t<Indexes...>>::SIZE;
                static_assert(VEC_SIZE <= SIZE);
                return guts::offset_pointer(*this, pointer, std::make_index_sequence<VEC_SIZE>{}, indexes...);
            }
        }

        template<typename... Indexes>
        requires ((SIZE >= sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>))
        [[nodiscard]] NOA_HD constexpr index_type offset_at(Indexes&&... indexes) const noexcept {
            if constexpr (nt::are_int_v<Indexes...>) {
                return guts::offset_at(*this, std::make_index_sequence<sizeof...(Indexes)>{}, indexes...);
            } else {
                constexpr size_t VEC_SIZE = std::decay_t<nt::first_t<Indexes...>>::SIZE;
                static_assert(VEC_SIZE <= SIZE);
                return guts::offset_at(*this, std::make_index_sequence<VEC_SIZE>{}, indexes...);
            }
        }

        template<typename... Indexes>
        requires ((SIZE == sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_size_v<N, Indexes...>))
        [[nodiscard]] NOA_HD constexpr reference_type operator()(Indexes&&... indexes) const noexcept {
            NOA_ASSERT(!is_empty());
            return *guts::offset_pointer(*this, m_ptr, std::make_index_sequence<SIZE>{}, indexes...);
        }

    private:
        pointer_type m_ptr{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

    /// Reference to Accessor.
    ///
    /// \details This is similar to Accessor, except that this type does not store the strides, it simply points to
    ///          existing ones. Usually AccessorReference is not constructed explicitly, and is instead the result of
    ///          Accessor::operator[] used to decrement the dimensionality of the Accessor, emulating C-style multi-
    ///          dimensional indexing, e.g. a[b][d][h][w] = value.
    template<typename T, size_t N, typename I,
             PointerTraits PointerTrait = PointerTraits::DEFAULT,
             StridesTraits StridesTrait = StridesTraits::STRIDED>
    class AccessorReference {
    public:
        static constexpr PointerTraits pointer_trait = PointerTrait;
        static constexpr StridesTraits strides_trait = StridesTrait;
        static constexpr bool IS_RESTRICT = PointerTrait == PointerTraits::RESTRICT;
        static constexpr bool IS_CONTIGUOUS = StridesTrait == StridesTraits::CONTIGUOUS;
        static constexpr size_t SIZE = N;
        static constexpr int64_t SSIZE = N;

        using accessor_type = Accessor<T, N, I, PointerTrait, StridesTrait>;
        using value_type = typename accessor_type::value_type;
        using mutable_value_type = typename accessor_type::mutable_value_type;
        using index_type = typename accessor_type::index_type;
        using reference_type = typename accessor_type::reference_type;
        using pointer_type = typename accessor_type::pointer_type;
        using strides_type = const index_type*;

    public:
        // Creates an empty view.
        NOA_HD constexpr AccessorReference() = default;

        /// Creates a reference to an accessor.
        /// For the contiguous case, the rightmost stride is ignored and never read from
        /// the \p strides strides pointer (so \p strides[N-1] is never accessed).
        /// As such, in the 1d contiguous case, a nullptr can be passed.
        NOA_HD constexpr AccessorReference(pointer_type pointer, strides_type strides) noexcept
                : m_ptr(pointer), m_strides(strides) {}

        NOA_HD constexpr explicit AccessorReference(accessor_type accessor) noexcept
                : AccessorReference(accessor.ptr, accessor.strides().data()) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<typename U> requires nt::is_mutable_value_type_v<U, value_type>
        NOA_HD constexpr /* implicit */ AccessorReference(
                const AccessorReference<U, N, I, PointerTrait, StridesTrait>& accessor
        ) : m_ptr(accessor.get()), m_strides(accessor.strides()) {}

    public: // Accessing strides
        template<size_t INDEX>
        [[nodiscard]] NOA_HD constexpr index_type stride() const noexcept {
            static_assert(INDEX < N);
            if constexpr (IS_CONTIGUOUS and INDEX == SIZE - 1)
                return index_type{1};
            else
                return m_strides[INDEX];
        }

        [[nodiscard]] NOA_HD constexpr index_type stride(std::integral auto index) const noexcept {
            NOA_ASSERT(!is_empty() and static_cast<i64>(index) < SSIZE);
            if (IS_CONTIGUOUS and index == SIZE - 1)
                return index_type{1};
            else
                return m_strides[index];
        }

        [[nodiscard]] NOA_HD constexpr strides_type strides() noexcept { return m_strides; }

    public:
        [[nodiscard]] NOA_HD constexpr pointer_type get() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() const noexcept { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !is_empty(); }

        NOA_HD constexpr void reset_pointer(pointer_type new_pointer) noexcept { m_ptr = new_pointer; }

    public:
        // Indexing operator, on 1D accessor. 1D -> ref
        template<typename Int> requires (SIZE == 1 and std::is_integral_v<Int>)
        [[nodiscard]] NOA_HD reference_type operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            return m_ptr[ni::offset_at(index, stride<0>())];
        }

        // Indexing operator, multidimensional accessor. ND -> ND-1
        template<typename Int> requires (SIZE > 1 and std::is_integral_v<Int>)
        [[nodiscard]] NOA_HD auto operator[](Int index) const noexcept {
            NOA_ASSERT(!is_empty());
            using output_type = AccessorReference<value_type, SIZE - 1, index_type, PointerTrait, StridesTrait>;
            return output_type(m_ptr + ni::offset_at(index, stride<0>()), m_strides + 1);
        }

    public:
        template<typename... Indexes>
        requires ((SIZE >= sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>))
        NOA_HD constexpr AccessorReference& offset_accessor(Indexes... indexes) noexcept {
            NOA_ASSERT(!is_empty());
            if constexpr (nt::are_int_v<Indexes...>) {
                m_ptr = guts::offset_pointer(*this, m_ptr, std::make_index_sequence<sizeof...(Indexes)>{}, indexes...);
            } else {
                constexpr size_t VEC_SIZE = std::decay_t<nt::first_t<Indexes...>>::SIZE;
                static_assert(VEC_SIZE <= SIZE);
                m_ptr = guts::offset_pointer(*this, m_ptr, std::make_index_sequence<VEC_SIZE>{}, indexes...);
            }
            return *this;
        }

        template<typename P, typename... Indexes>
        requires (((SIZE >= sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                   (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>)) and
                  std::is_pointer_v<P>)
        [[nodiscard]] NOA_HD constexpr P offset_pointer(P pointer, Indexes&&... indexes) const noexcept {
            if constexpr (nt::are_int_v<Indexes...>) {
                return guts::offset_pointer(*this, pointer, std::make_index_sequence<sizeof...(Indexes)>{}, indexes...);
            } else {
                constexpr size_t VEC_SIZE = std::decay_t<nt::first_t<Indexes...>>::SIZE;
                static_assert(VEC_SIZE <= SIZE);
                return guts::offset_pointer(*this, pointer, std::make_index_sequence<VEC_SIZE>{}, indexes...);
            }
        }

        template<typename... Indexes>
        requires ((SIZE >= sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>))
        [[nodiscard]] NOA_HD constexpr index_type offset_at(Indexes&&... indexes) const noexcept {
            if constexpr (nt::are_int_v<Indexes...>) {
                return guts::offset_at(*this, std::make_index_sequence<sizeof...(Indexes)>{}, indexes...);
            } else {
                constexpr size_t VEC_SIZE = std::decay_t<nt::first_t<Indexes...>>::SIZE;
                static_assert(VEC_SIZE <= SIZE);
                return guts::offset_at(*this, std::make_index_sequence<VEC_SIZE>{}, indexes...);
            }
        }

        template<typename... Indexes>
        requires ((SIZE == sizeof...(Indexes) and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_size_v<N, Indexes...>))
        [[nodiscard]] NOA_HD constexpr reference_type operator()(Indexes&&... indexes) const noexcept {
            NOA_ASSERT(!is_empty());
            return *guts::offset_pointer(*this, m_ptr, std::make_index_sequence<SIZE>{}, indexes...);
        }

    private:
        pointer_type m_ptr{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

// Some compilers complain about the output pointer being marked restrict and say it is ignored...
#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-qualifiers"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

    /// Stores a value and provide an nd-accessor interface of that value.
    /// \details This is poorly named because as opposed to Accessor(Reference), this type is the owner of the
    ///          object being accessed. The original goal is to provide the accessor interface so we can index
    ///          a value as if it was a nd-array (something like Scalar would have been have better choice),
    ///          but we want to emphasize that the goal here is to support the accessor indexing, while referring
    ///          to a single _local_ single value (we don't want to refer to a value on the heap for example).
    ///
    /// \note As opposed to the Accessor, the const-ness is also enforced by the accessor.
    ///       With AccessorValue<const T>, the value cannot be mutated(*).
    ///       With AccessorValue<f64>, the value can be mutated, iif the accessor itself is not const.
    ///       This is because the AccessorValue stores the value, so const-ness is transferred to the member variable.
    ///       (*): using deref_unsafe() allows to access the stored T and bypass the const-ness of an
    ///            AccessorValue<const T> if the accessor itself is not const. This is intended for the library
    ///            to be able to reassign AccessorValue<const T> while still preserving the const-ness in the main
    ///            API (notably in the core interface and operators).
    ///
    /// \note This can be treated as an Accessor of any dimension. So a(2) or a(2,3,4,5) are equivalent and
    ///       are both returning a reference of the value. The only constraint is on operator[], which always
    ///       returns a reference of the value (like array/pointer indexing).
    ///
    /// \note The pointer_type (e.g. returned from .data()) is always marked restrict because the value is owned
    ///       by the accessor. This shouldn't be relevant since the compiler can see that, but it's just to keep
    ///       things coherent.
    ///
    /// \tparam T Any value. A value of that type is stored in the accessor.
    /// \tparam I Index type. This defines the type for the strides (which are empty).
    template<typename T, typename I = i64>
    class AccessorValue {
    public:
        static_assert(!std::is_pointer_v<T>);
        static_assert(!std::is_reference_v<T>);

        static constexpr PointerTraits pointer_trait = PointerTraits::RESTRICT;
        static constexpr StridesTraits strides_trait = StridesTraits::CONTIGUOUS;
        static constexpr bool IS_RESTRICT = true;
        static constexpr bool IS_CONTIGUOUS = true;
        static constexpr size_t SIZE = 1;
        static constexpr int64_t SSIZE = 1;

        using value_type = T;
        using index_type = I;
        using mutable_value_type = std::remove_const_t<T>;
        using const_reference_type = const mutable_value_type&;
        using reference_type = T&;
        using strides_type = Strides<index_type, 0>;
        #if defined(__CUDACC__)
        using pointer_type = T* __restrict__;
        using const_pointer_type = const mutable_value_type* __restrict__;
        #else
        using pointer_type = T* __restrict;
        using const_pointer_type = const mutable_value_type* __restrict;
        #endif

    public: // Constructors
        NOA_HD constexpr AccessorValue() = default;

        NOA_HD constexpr explicit AccessorValue(mutable_value_type&& value) noexcept
                : m_value{std::move(value)} {}

        NOA_HD constexpr explicit AccessorValue(const mutable_value_type& value) noexcept
                : m_value{value} {}

        template<typename U, typename J> requires nt::is_mutable_value_type_v<U, value_type>
        NOA_HD constexpr /* implicit */ AccessorValue(const AccessorValue<U, J>& accessor)
                : m_value{accessor.deref()} {}

    public: // Accessing strides
        template<size_t>
        [[nodiscard]] NOA_HD constexpr index_type stride() const noexcept { return 0; }
        [[nodiscard]] NOA_HD constexpr index_type stride(std::integral auto) const noexcept { return 0; }
        [[nodiscard]] NOA_HD constexpr const strides_type& strides() const noexcept { return m_strides; }

    public:
        // Note: restrict is a type qualifier and is ignored in the return type? -Wignored-qualifiers
        [[nodiscard]] NOA_HD constexpr pointer_type get() noexcept { return &m_value; }
        [[nodiscard]] NOA_HD constexpr pointer_type data() noexcept { return &m_value; }
        [[nodiscard]] NOA_HD constexpr const_pointer_type get() const noexcept { return &m_value; }
        [[nodiscard]] NOA_HD constexpr const_pointer_type data() const noexcept { return &m_value; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return false; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return !is_empty(); }

        NOA_HD constexpr void reset_pointer(pointer_type) noexcept {}

        template<std::integral Int, size_t N>
        NOA_FHD constexpr AccessorValue& reorder(const Vec<Int, N>&) noexcept {
            return *this;
        }

    public:
        template<typename Int> requires std::is_integral_v<Int>
        [[nodiscard]] NOA_HD const_reference_type operator[](Int) const noexcept {
            return m_value;
        }

        template<typename Int> requires std::is_integral_v<Int>
        [[nodiscard]] NOA_HD reference_type operator[](Int) noexcept {
            return m_value;
        }

    public:
        template<typename... Indexes>
        requires ((sizeof...(Indexes) < 4 and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>))
        NOA_HD constexpr AccessorValue& offset_accessor(Indexes...) noexcept {
            return *this;
        }

        template<typename P, typename... Indexes>
        requires (((sizeof...(Indexes) < 4 and nt::are_int_v<Indexes...>) or
                   (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>)) and
                   std::is_pointer_v<P>)
        [[nodiscard]] NOA_HD constexpr P offset_pointer(P pointer, Indexes&&...) const noexcept {
            return pointer;
        }

        template<typename... Indexes>
        requires ((sizeof...(Indexes) < 4 and nt::are_int_v<Indexes...>) or
                  (sizeof...(Indexes) == 1 and nt::are_vec_int_v<Indexes...>))
        [[nodiscard]] NOA_HD constexpr index_type offset_at(Indexes&&...) const noexcept {
            return static_cast<index_type>(0);
        }

        template<typename... Indexes> requires nt::are_int_v<Indexes...>
        [[nodiscard]] NOA_HD constexpr const_reference_type operator()(Indexes&&...) const noexcept {
            return m_value;
        }
        template<typename... Indexes> requires nt::are_int_v<Indexes...>
        [[nodiscard]] NOA_HD constexpr reference_type operator()(Indexes&&...) noexcept {
            return m_value;
        }

        template<size_t N0, typename Integer> requires nt::is_int_v<Integer>
        [[nodiscard]] NOA_HD constexpr const_reference_type operator()(const Vec<Integer, N0>&) const noexcept {
            return m_value;
        }
        template<size_t N0, typename Integer> requires nt::is_int_v<Integer>
        [[nodiscard]] NOA_HD constexpr reference_type operator()(const Vec<Integer, N0>&) noexcept {
            return m_value;
        }

    public: // Additional methods
        [[nodiscard]] NOA_HD constexpr const_reference_type deref() const noexcept { return m_value; }
        [[nodiscard]] NOA_HD constexpr reference_type deref() noexcept { return m_value; }

        // "private" method to access the underlying type without const (if the accessor is const, deref_ == deref)
        [[nodiscard]] NOA_HD constexpr const_reference_type deref_() const noexcept { return m_value; }
        [[nodiscard]] NOA_HD constexpr mutable_value_type& deref_() noexcept { return m_value; }

    private:
        mutable_value_type m_value;
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides;
    };

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif

    template<typename T, size_t N>
    using AccessorI64 = Accessor<T, N, int64_t>;
    template<typename T, size_t N>
    using AccessorI32 = Accessor<T, N, int32_t>;
    template<typename T, size_t N>
    using AccessorU64 = Accessor<T, N, uint64_t>;
    template<typename T, size_t N>
    using AccessorU32 = Accessor<T, N, uint32_t>;

    template<typename T, size_t N, typename I, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrict = Accessor<T, N, I, PointerTraits::RESTRICT, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictI64 = AccessorRestrict<T, N, int64_t, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictI32 = AccessorRestrict<T, N, int32_t, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictU64 = AccessorRestrict<T, N, uint64_t, StridesTrait>;
    template<typename T, size_t N, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrictU32 = AccessorRestrict<T, N, uint32_t, StridesTrait>;

    template<typename T, size_t N, typename I, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguous = Accessor<T, N, I, PointerTrait, StridesTraits::CONTIGUOUS>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousI64 = AccessorContiguous<T, N, int64_t, PointerTrait>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousI32 = AccessorContiguous<T, N, int32_t, PointerTrait>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousU64 = AccessorContiguous<T, N, uint64_t, PointerTrait>;
    template<typename T, size_t N, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguousU32 = AccessorContiguous<T, N, uint32_t, PointerTrait>;

    template<typename T, size_t N, typename I>
    using AccessorRestrictContiguous = Accessor<T, N, I, PointerTraits::RESTRICT, StridesTraits::CONTIGUOUS>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousI64 = AccessorRestrictContiguous<T, N, int64_t>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousI32 = AccessorRestrictContiguous<T, N, int32_t>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousU64 = AccessorRestrictContiguous<T, N, uint64_t>;
    template<typename T, size_t N>
    using AccessorRestrictContiguousU32 = AccessorRestrictContiguous<T, N, uint32_t>;

    template<typename T>
    using AccessorValueI64 = AccessorValue<T, int64_t>;
    template<typename T>
    using AccessorValueI32 = AccessorValue<T, int32_t>;
    template<typename T>
    using AccessorValueU64 = AccessorValue<T, uint64_t>;
    template<typename T>
    using AccessorValueU32 = AccessorValue<T, uint32_t>;
}

// Accessor, AccessorReference and AccessorValue are all accessors.
// AccessorReference is accessor_reference and AccessorValue is accessor_value.
// AccessorValue is accessor_restrict, accessor_contiguous and accessor_nd for any N.
namespace noa::traits {
    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor<Accessor<T, N, I, PointerTrait, StridesTrait>> : std::true_type {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor_pure<Accessor<T, N, I, PointerTrait, StridesTrait>> : std::true_type {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor_restrict<Accessor<T, N, I, PointerTrait, StridesTrait>> : std::bool_constant<PointerTrait == PointerTraits::RESTRICT> {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor_contiguous<Accessor<T, N, I, PointerTrait, StridesTrait>> : std::bool_constant<StridesTrait == StridesTraits::CONTIGUOUS> {};

    template<typename T, size_t N1, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait, size_t N2>
    struct proclaim_is_accessor_nd<Accessor<T, N1, I, PointerTrait, StridesTrait>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor<AccessorReference<T, N, I, PointerTrait, StridesTrait>> : std::true_type {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor_restrict<AccessorReference<T, N, I, PointerTrait, StridesTrait>> : std::bool_constant<PointerTrait == PointerTraits::RESTRICT> {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor_contiguous<AccessorReference<T, N, I, PointerTrait, StridesTrait>> : std::bool_constant<StridesTrait == StridesTraits::CONTIGUOUS> {};

    template<typename T, size_t N1, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait, size_t N2>
    struct proclaim_is_accessor_nd<AccessorReference<T, N1, I, PointerTrait, StridesTrait>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, size_t N, typename I, PointerTraits PointerTrait, StridesTraits StridesTrait>
    struct proclaim_is_accessor_reference<AccessorReference<T, N, I, PointerTrait, StridesTrait>> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor<AccessorValue<T, I>> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor_restrict<AccessorValue<T, I>> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor_contiguous<AccessorValue<T, I>> : std::true_type {};

    template<typename T, typename I, size_t N>
    struct proclaim_is_accessor_nd<AccessorValue<T, I>, N> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor_value<AccessorValue<T, I>> : std::true_type {};
}

namespace noa::guts {
    /// Reconfigures the Accessor(s).
    /// \param enforce_contiguous   Whether the output Accessor should be made contiguous.
    /// \param enforce_restrict     Whether the output Accessor should be made restrict.
    /// \param ndim                 Number of output dimensions. Should be less or equal than the current ndim of the
    ///                             input accessor(s). Zero (the default) indicates that the dimensionality should be
    ///                             left unchanged.
    /// \param collapse_leftmost    Collapse leftmost means that the dimensions are collapsed starting from the left.
    ///                             Collapse rightmost means that the dimensions are collapsed starting from the right.
    ///                             Regardless, collapsed dimensions are assumed to be C-contiguous, so the stride of
    ///                             the resulting collapsed dimension is taken from the rightmost collapsed dimension.
    template<size_t N = 0>
    struct AccessorConfig {
        bool enforce_contiguous{false};
        bool enforce_restrict{false};
        Vec<size_t, N> filter{};
    };
    template<AccessorConfig config, typename Accessors>
    requires nt::is_tuple_of_accessor_or_empty_v<Accessors>
    [[nodiscard]] constexpr auto reconfig_accessors(Accessors&& accessors) {
        return std::forward<Accessors>(accessors).map(
                []<typename T>(T&& accessor) {
                    if constexpr (nt::is_accessor_value_v<T>) {
                        // Forward the value into the new tuple, ie the caller decides whether we copy or move.
                        // std::forward to guarantee a move (g++ doesn't move otherwise).
                        return std::forward<T>(accessor);
                    } else if constexpr (not nt::is_accessor_reference_v<T>) {
                        using accessor_t = std::decay_t<T>;
                        using value_t = typename accessor_t::value_type;
                        using index_t = typename accessor_t::index_type;
                        constexpr size_t original_ndim = accessor_t::SIZE;
                        constexpr size_t vec_ndim = decltype(config.filter)::SIZE;
                        constexpr size_t new_ndim = vec_ndim == 0 ? original_ndim : vec_ndim;
                        constexpr auto pointer_trait =
                                config.enforce_restrict ? PointerTraits::RESTRICT : accessor_t::pointer_trait;
                        constexpr auto strides_trait =
                                config.enforce_contiguous ? StridesTraits::CONTIGUOUS : accessor_t::strides_trait;
                        using new_accessor_t = Accessor<value_t, new_ndim, index_t, pointer_trait, strides_trait>;

                        if constexpr (new_ndim == original_ndim) {
                            return new_accessor_t(accessor.get(), accessor.strides());
                        } else {
                            auto strides = [&accessor]<size_t... I>(std::index_sequence<I...>, const auto& filter) {
                                return Strides{accessor.stride(filter[I])...};
                            }(std::make_index_sequence<new_ndim>{}, config.filter);
                            return new_accessor_t(accessor.get(), strides);
                        }
                    } else {
                        static_assert(nt::always_false_v<T>);
                    }
                });
    }

    /// Whether the accessors are aliases of each others.
    /// TODO Take a shape and see if end of array doesn't overlap with other arrays? like are_overlapped
    template<typename... TuplesOfAccessors>
    requires nt::are_tuple_of_accessor_v<TuplesOfAccessors...>
    [[nodiscard]] constexpr auto are_accessors_aliased(const TuplesOfAccessors&... tuples_of_accessors) -> bool {
        auto tuple_of_pointers = tuple_cat(tuples_of_accessors.map([](const auto& accessor) {
            if constexpr (nt::is_accessor_value_v<decltype(accessor)>)
                return nullptr;
            else
                return accessor.get();
        })...);

        return tuple_of_pointers.any_enumerate([&]<size_t I, typename T>(T ei) {
            return tuple_of_pointers.any_enumerate([ei]<size_t J, typename U>(U ej) {
                if constexpr (I != J) {
                    // If nullptr, whether because it's an AccessorValue or because
                    // the accessor actually points to a nullptr, it does not alias.
                    auto* pi = static_cast<const void*>(ei);
                    auto* pj = static_cast<const void*>(ej);
                    return pi != nullptr and pj != nullptr and pi == pj;
                }
                return false;
            });
        });
    }

    /// Whether the accessors point to const data, i.e. their value_type is const.
    template<typename T> requires nt::is_tuple_of_accessor_v<T>
    [[nodiscard]] constexpr auto are_accessors_const() -> bool {
        constexpr bool are_all_const = []<typename... A>(nt::TypeList<A...>) {
            return (std::is_const_v<nt::value_type_t<A>> and ...);
        }(nt::type_list_t<T>{});
        return are_all_const;
    }
}
