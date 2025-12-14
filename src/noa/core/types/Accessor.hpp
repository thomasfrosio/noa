#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/indexing/Offset.hpp"

namespace noa::inline types {
    template<typename T, usize N, typename I,
            StridesTraits StridesTrait,
            PointerTraits PointerTrait>
    class AccessorReference;

    /// Multidimensional accessor; wraps a pointer and nd-strides, and provides nd-indexing.
    /// \details
    /// Accessors are mostly intended for internal use, which affected some design choices. Noticeable features:
    /// >>> The sizes of the dimensions are not stored, so accessors cannot bound-check the indexes against
    ///     their dimension size. In a lot of cases, the input/output arrays have the same size/shape and
    ///     the size/shape is often not needed by the compute kernels, leading to storing useless data.
    ///     If the extents of the region are required, use (md)spans.
    /// >>> \b Pointer-traits. By default, the pointers are not marked with any attributes, but the "restrict"
    ///     traits can be added. This is useful to signify that pointers don't alias, which helps generate
    ///     better code. Unfortunately, only g++ seems to acknowledge the restrict attribute on pointers
    ///     inside structs (details below)...
    /// >>> \b Strides-traits. Strides are fully dynamic (one dynamic stride per dimension) by default,
    ///     but the rightmost dimension can be marked contiguous. Accessors (and the library internals) use
    ///     the rightmost convention, so that the innermost dimension is the rightmost dimension. As such,
    ///     StridesTraits::CONTIGUOUS implies C-contiguous. Accessors don't support the F-contiguous layout,
    ///     but of course this layout can be reordered to C-contiguous before creating the contiguous accessor.
    ///     With StridesTraits::CONTIGUOUS, the innermost/rightmost stride is fixed to 1 and is not stored,
    ///     resulting in the strides being truncated by 1 (Strides<I,N-1>). In the case of a 1d contiguous
    ///     accessor, this means that the strides are empty (Strides<I,0>) and the indexing is equivalent
    ///     to pointer/array indexing.
    template<typename T, usize N, typename I = isize,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             PointerTraits PointerTrait = PointerTraits::DEFAULT>
    class Accessor : public ni::Indexer<Accessor<T, N, I, StridesTrait, PointerTrait>, N> {
    public:
        static_assert(not std::is_reference_v<T> and
                      not std::is_pointer_v<T> and
                      not std::extent_v<T> and
                      std::is_integral_v<I>);

        static constexpr StridesTraits STRIDES_TRAIT = StridesTrait;
        static constexpr PointerTraits POINTER_TRAIT = PointerTrait;
        static constexpr bool IS_CONTIGUOUS = STRIDES_TRAIT == StridesTraits::CONTIGUOUS;
        static constexpr bool IS_RESTRICT = POINTER_TRAIT == PointerTraits::RESTRICT;
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;

        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = const mutable_value_type;
        using pointer_type = value_type*;
        using const_pointer_type = const_value_type*;
        using index_type = I;
        using shape_type = Shape<index_type, N>;
        using strides_type = Strides<index_type, N - IS_CONTIGUOUS>;
        using reference_type = value_type&;
        using const_reference_type = const mutable_value_type&;
        using accessor_reference_type = AccessorReference<value_type, SIZE, index_type, STRIDES_TRAIT, POINTER_TRAIT>;

    public: // Constructors
        /// Creates an empty accessor.
        NOA_HD constexpr Accessor() = default;

        /// Creates a strided or contiguous accessor.
        /// If the accessor is contiguous, the width stride (strides[SIZE-1]) is ignored and assumed to be 1.
        NOA_HD constexpr Accessor(pointer_type pointer, const Strides<index_type, SIZE>& strides) noexcept :
            m_ptr{pointer},
            m_strides{strides_type::from_pointer(strides.data())} {}

        /// Creates a contiguous accessor from contiguous strides.
        NOA_HD constexpr Accessor(pointer_type pointer, const strides_type& strides) noexcept requires IS_CONTIGUOUS :
            m_ptr{pointer},
            m_strides{strides} {}

        /// Creates an accessor from an accessor reference.
        NOA_HD constexpr explicit Accessor(accessor_reference_type accessor_reference) noexcept :
            m_ptr{accessor_reference.get()},
            m_strides{strides_type::from_pointer(accessor_reference.strides())} {}

        /// Creates a contiguous 1d accessor, assuming the stride is 1.
        NOA_HD constexpr explicit Accessor(pointer_type pointer) noexcept requires (SIZE == 1 and IS_CONTIGUOUS) :
            m_ptr{pointer} {}

        /// Implicitly creates a const accessor from an existing non-const accessor.
        template<nt::mutable_of<value_type> U>
        NOA_HD constexpr Accessor(const Accessor<U, N, index_type, STRIDES_TRAIT, POINTER_TRAIT>& accessor) noexcept :
            m_ptr{accessor.get()}, m_strides{accessor.strides()} {}

    public: // Accessing strides
        template<usize INDEX> requires (INDEX < N)
        [[nodiscard]] NOA_HD constexpr auto stride() const noexcept -> index_type {
            if constexpr (IS_CONTIGUOUS and INDEX == SIZE - 1)
                return index_type{1};
            else
                return m_strides[INDEX];
        }

        [[nodiscard]] NOA_HD constexpr auto stride(nt::integer auto index) const noexcept -> index_type {
            NOA_ASSERT(not is_empty() and static_cast<isize>(index) < SSIZE);
            if (IS_CONTIGUOUS and index == SIZE - 1)
                return index_type{1};
            else
                return m_strides[index];
        }

        [[nodiscard]] NOA_HD constexpr auto strides() noexcept -> strides_type& { return m_strides; }
        [[nodiscard]] NOA_HD constexpr auto strides() const noexcept -> const strides_type& { return m_strides; }
        [[nodiscard]] NOA_HD constexpr auto strides_full() const noexcept -> decltype(auto) {
            if constexpr (IS_CONTIGUOUS) {
                return [this]<usize... J>(std::index_sequence<J...>){
                    return Strides{this->stride<J>()...}; // returns by value
                }(std::make_index_sequence<SIZE>{});
            } else {
                return strides(); // returns a lvalue const ref
            }
        }

    public: // Accessing data pointer
        [[nodiscard]] NOA_HD constexpr auto get() const noexcept -> pointer_type { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr auto data() const noexcept -> pointer_type { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return not is_empty(); }

    public:
        /// C-style indexing operator, decrementing the dimensionality of the accessor by 1.
        template<typename Int> requires (SIZE > 1)
        [[nodiscard]] NOA_HD constexpr auto operator[](Int index) const noexcept {
            NOA_ASSERT(not is_empty());
            using output_t = AccessorReference<value_type, SIZE - 1, index_type, STRIDES_TRAIT, POINTER_TRAIT>;
            return output_t(get() + ni::offset_at(stride<0>(), index), strides().data() + 1);
        }

        /// C-style indexing operator, decrementing the dimensionality of the accessor by 1.
        /// When done on a 1d accessor, this acts as a pointer/array indexing and dereferences the data.
        template<typename Int> requires (SIZE == 1)
        [[nodiscard]] NOA_HD constexpr auto& operator[](Int index) const noexcept {
            NOA_ASSERT(not is_empty());
            return get()[ni::offset_at(stride<0>(), index)];
        }

        template<nt::integer Int>
        NOA_FHD constexpr auto reorder(const Vec<Int, N>& order) noexcept -> Accessor& {
            strides() = strides().reorder(order);
            return *this;
        }

        NOA_HD constexpr void reset_pointer(pointer_type new_pointer) noexcept { m_ptr = new_pointer; }

    private:
        // nvcc ignores __restrict__ when applied to member variables... https://godbolt.org/z/9GY5hGEzr
        // gcc/msvc optimizes it, but clang ignores it as well... https://godbolt.org/z/r869cb34s
        using storage_ptr_type = std::conditional_t<IS_RESTRICT, value_type* NOA_RESTRICT_ATTRIBUTE, value_type*>;
        storage_ptr_type m_ptr{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

    /// Reference to Accessor.
    /// \details This is similar to Accessor, except that this type does not store the strides, it simply points to
    ///          existing ones. Usually AccessorReference is not constructed explicitly, and is instead the result of
    ///          Accessor::operator[] used to decrement the dimensionality of the Accessor, emulating C-style multi-
    ///          dimensional indexing, e.g. a[b][d][h][w] = value.
    template<typename T, usize N, typename I = isize,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             PointerTraits PointerTrait = PointerTraits::DEFAULT>
    class AccessorReference: public ni::Indexer<AccessorReference<T, N, I, StridesTrait, PointerTrait>, N> {
    public:
        static constexpr StridesTraits STRIDES_TRAIT = StridesTrait;
        static constexpr PointerTraits POINTER_TRAIT = PointerTrait;
        static constexpr bool IS_CONTIGUOUS = STRIDES_TRAIT == StridesTraits::CONTIGUOUS;
        static constexpr bool IS_RESTRICT = POINTER_TRAIT == PointerTraits::RESTRICT;
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;

        using accessor_type = Accessor<T, N, I, STRIDES_TRAIT, POINTER_TRAIT>;
        using value_type = accessor_type::value_type;
        using mutable_value_type = accessor_type::mutable_value_type;
        using index_type = accessor_type::index_type;
        using reference_type = accessor_type::reference_type;
        using pointer_type = accessor_type::pointer_type;
        using strides_type = const index_type*;

    public:
        /// Creates a reference to an accessor.
        /// For the contiguous case, the rightmost stride is ignored and never read from
        /// the stride pointer (i.e., strides[N-1] is never accessed).
        /// As such, in the 1d contiguous case, a nullptr can be passed.
        NOA_HD constexpr AccessorReference(pointer_type pointer, strides_type strides) noexcept :
            m_ptr(pointer), m_strides(strides) {}

        NOA_HD constexpr explicit AccessorReference(accessor_type accessor) noexcept :
            AccessorReference(accessor.ptr, accessor.strides().data()) {}

        /// Creates a const accessor from an existing non-const accessor.
        template<nt::mutable_of<value_type> U>
        NOA_HD constexpr AccessorReference(const AccessorReference<U, N, I, STRIDES_TRAIT, POINTER_TRAIT>& accessor) :
            m_ptr(accessor.get()), m_strides(accessor.strides()) {}

    public: // Accessing strides
        template<usize INDEX> requires (INDEX < N)
        [[nodiscard]] NOA_HD constexpr auto stride() const noexcept -> index_type {
            if constexpr (IS_CONTIGUOUS and INDEX == SIZE - 1)
                return index_type{1};
            else
                return m_strides[INDEX];
        }

        [[nodiscard]] NOA_HD constexpr auto stride(nt::integer auto index) const noexcept -> index_type {
            NOA_ASSERT(not is_empty() and static_cast<isize>(index) < SSIZE);
            if (IS_CONTIGUOUS and index == SIZE - 1)
                return index_type{1};
            else
                return m_strides[index];
        }

        [[nodiscard]] NOA_HD constexpr auto strides() const noexcept -> strides_type { return m_strides; }
        [[nodiscard]] NOA_HD constexpr auto strides_full() const noexcept {
            return [this]<usize... J>(std::index_sequence<J...>){
                return Strides{stride<J>()...}; // returns by value
            }(std::make_index_sequence<SIZE>{});
        }

    public: // Accessing data pointer
        [[nodiscard]] NOA_HD constexpr auto get() const noexcept -> pointer_type { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr auto data() const noexcept -> pointer_type { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return m_ptr == nullptr; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return not is_empty(); }

    public:
        /// C-style indexing operator, decrementing the dimensionality of the accessor by 1.
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto index) const noexcept requires (SIZE > 1){
            NOA_ASSERT(not is_empty());
            using output_t = AccessorReference<value_type, SIZE - 1, index_type, STRIDES_TRAIT, POINTER_TRAIT>;
            return output_t(get() + ni::offset_at(stride<0>(), index), strides() + 1);
        }

        [[nodiscard]] NOA_HD constexpr auto& operator[](nt::integer auto index) const noexcept requires (SIZE == 1) {
            NOA_ASSERT(not is_empty());
            return get()[ni::offset_at(stride<0>(), index)];
        }

        NOA_HD constexpr void reset_pointer(pointer_type new_pointer) noexcept { m_ptr = new_pointer; }

    private:
        using storage_ptr_type = std::conditional_t<IS_RESTRICT, value_type* NOA_RESTRICT_ATTRIBUTE, value_type*>;
        storage_ptr_type m_ptr{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

// Some compilers complain about the output pointer being marked restrict and say it is ignored...
#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-qualifiers"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

    /// Stores a value and provide an accessor-like interface of that value.
    /// \details This is poorly named because as opposed to Accessor(Reference), this type is the owner of the
    ///          object being accessed. The original goal is to provide the accessor interface so we can index
    ///          a value as if it was a nd-array (something like Scalar would have been a better choice).
    ///          However, we want to emphasize that the goal here is to support the accessor interface while
    ///          referring to a single _local_ value (we don't want to refer to a value on the heap, for example).
    ///
    /// \note As opposed to Accessor, the const-ness is also enforced by the wrapper type:
    ///    1. With AccessorValue<T>, the referred value can be mutated, iif the accessor itself is not const.
    ///       This is because the AccessorValue stores the value, so const-ness is transferred to the member variable.
    ///       This is different from const Accessor<T>, which refers to mutable T values.
    ///    2. With AccessorValue<const T>, the referenced value cannot be mutated (like Accessor<const T>)
    ///       using the classic accessor interface. However, AccessorValue has an extra member function, called
    ///       deref_unsafe(), which allows accessing the stored T, bypassing the const-ness of a mutable
    ///       AccessorValue<const T>. This is intended for the library to be able to reassign AccessorValue<const T>
    ///       while still preserving the const-ness in the main API (notably in the core interface and operators).
    ///
    /// \note This can be treated mostly as an Accessor of any dimension and size. So a(2) or a(2,3,4,5) are equivalent
    ///       and are both returning a reference of the value. However, AccessorValue<T> most equivalent to an
    ///       Accessor<T, 1, StridesTraits::CONTIGUOUS, PointerTraits::RESTRICT>.
    ///
    /// \note The pointer_type (returned from .data()) is always marked restrict because the value is owned
    ///       by the accessor. This shouldn't be relevant since the compiler can see that, but it's just to keep
    ///       things coherent.
    ///
    /// \tparam T Any value. A value of that type is stored in the accessor.
    /// \tparam I Index type. This defines the type for the strides (which are empty).
    template<typename T, typename I = isize>
    class AccessorValue {
    public:
        static_assert(not std::is_pointer_v<T> and not std::is_reference_v<T>);

        static constexpr StridesTraits STRIDES_TRAIT = StridesTraits::CONTIGUOUS;
        static constexpr PointerTraits POINTER_TRAIT = PointerTraits::RESTRICT;
        static constexpr bool IS_CONTIGUOUS = true;
        static constexpr bool IS_RESTRICT = true;
        static constexpr usize SIZE = 1;
        static constexpr isize SSIZE = 1;

        using value_type = T;
        using index_type = I;
        using mutable_value_type = std::remove_const_t<T>;
        using const_reference_type = const mutable_value_type&;
        using reference_type = T&;
        using strides_type = Strides<index_type, 0>;
        using pointer_type = T*;
        using const_pointer_type = const mutable_value_type*;

    public: // Constructors
        NOA_HD constexpr AccessorValue() = default;

        NOA_HD constexpr explicit AccessorValue(mutable_value_type&& value) : // TODO noexcept
            m_value{std::move(value)} {}

        NOA_HD constexpr explicit AccessorValue(const mutable_value_type& value) : // TODO noexcept
            m_value{value} {}

        template<nt::mutable_of<value_type> U, nt::integer J>
        NOA_HD constexpr AccessorValue(const AccessorValue<U, J>& accessor) : // TODO noexcept
            m_value{accessor.ref()} {}

    public: // Accessing strides
        template<usize>
        [[nodiscard]] NOA_HD constexpr auto stride() const noexcept -> index_type { return 0; }
        [[nodiscard]] NOA_HD constexpr auto stride(nt::integer auto) const noexcept -> index_type { return 0; }
        [[nodiscard]] NOA_HD constexpr auto strides() const noexcept -> strides_type { return strides_type{}; }
        [[nodiscard]] NOA_HD constexpr auto strides_full() const noexcept {
            return Strides<index_type, 1>{1}; // appear 1d contiguous
        }

    public:
        // Note: restrict is a type qualifier and is ignored in the return type? -Wignored-qualifiers
        [[nodiscard]] NOA_HD constexpr auto get() noexcept -> pointer_type { return &m_value; }
        [[nodiscard]] NOA_HD constexpr auto data() noexcept -> pointer_type { return &m_value; }
        [[nodiscard]] NOA_HD constexpr auto get() const noexcept -> const_pointer_type { return &m_value; }
        [[nodiscard]] NOA_HD constexpr auto data() const noexcept -> const_pointer_type { return &m_value; }
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return false; }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return not is_empty(); }

        NOA_HD constexpr void reset_pointer(pointer_type) noexcept {}
        NOA_FHD constexpr auto reorder(const nt::vec_integer auto&) noexcept -> AccessorValue& { return *this; }

    public:
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto) const noexcept -> const_reference_type { return m_value; }
        [[nodiscard]] NOA_HD constexpr auto operator[](nt::integer auto) noexcept -> reference_type { return m_value; }

    public:
        template<typename... U> requires nt::offset_indexing<4, U...>
        NOA_HD constexpr auto offset_inplace(U...) noexcept -> AccessorValue& {
            return *this;
        }

        template<typename... U> requires nt::offset_indexing<4, U...>
        [[nodiscard]] NOA_HD constexpr auto offset_pointer(auto* pointer, U...) const noexcept {
            return pointer;
        }

        template<typename... U> requires nt::offset_indexing<4, U...>
        [[nodiscard]] NOA_HD constexpr auto offset_at(U...) const noexcept -> index_type {
            return static_cast<index_type>(0);
        }

        [[nodiscard]] NOA_HD constexpr auto operator()(nt::integer auto...) const noexcept -> const_reference_type { return m_value; }
        [[nodiscard]] NOA_HD constexpr auto operator()(nt::integer auto...) noexcept -> reference_type { return m_value; }
        [[nodiscard]] NOA_HD constexpr auto operator()(const nt::vec_integer auto&) const noexcept -> const_reference_type { return m_value; }
        [[nodiscard]] NOA_HD constexpr auto operator()(const nt::vec_integer auto&) noexcept -> reference_type { return m_value; }

    public: // Additional member functions
        [[nodiscard]] NOA_HD constexpr auto ref() const& noexcept -> const_reference_type { return m_value; }
        [[nodiscard]] NOA_HD constexpr auto ref() && noexcept -> value_type&& { return std::move(m_value); }
        [[nodiscard]] NOA_HD constexpr auto ref() & noexcept -> value_type& { return m_value; }

        // "private" method to access the underlying type without const (if the accessor is const, ref_ == ref)
        [[nodiscard]] NOA_HD constexpr auto ref_() const& noexcept -> const_reference_type { return m_value; }
        [[nodiscard]] NOA_HD constexpr auto ref_() && noexcept -> mutable_value_type&& { return std::move(m_value); }
        [[nodiscard]] NOA_HD constexpr auto ref_() & noexcept -> mutable_value_type& { return m_value; }

    private:
        NOA_NO_UNIQUE_ADDRESS mutable_value_type m_value;
    };

    template<typename T>
    AccessorValue(T&& value) -> AccessorValue<std::decay_t<T>>;

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif

    template<typename T, usize N, typename I = isize, StridesTraits StridesTrait = StridesTraits::STRIDED>
    using AccessorRestrict = Accessor<T, N, I, StridesTrait, PointerTraits::RESTRICT>;

    template<typename T, usize N, typename I = isize, PointerTraits PointerTrait = PointerTraits::DEFAULT>
    using AccessorContiguous = Accessor<T, N, I, StridesTraits::CONTIGUOUS, PointerTrait>;

    template<typename T, usize N, typename I = isize>
    using AccessorRestrictContiguous = Accessor<T, N, I, StridesTraits::CONTIGUOUS, PointerTraits::RESTRICT>;
}

// Accessor, AccessorReference and AccessorValue are all accessors.
// Accessor is accessor_pure, AccessorReference is accessor_reference and AccessorValue is accessor_value.
// AccessorValue is accessor_restrict, accessor_contiguous and accessor_nd for any N.
namespace noa::traits {
    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor<noa::Accessor<T, N, I, S, P>> : std::true_type {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor_pure<noa::Accessor<T, N, I, S, P>> : std::true_type {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor_restrict<noa::Accessor<T, N, I, S, P>> : std::bool_constant<P == PointerTraits::RESTRICT> {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor_contiguous<noa::Accessor<T, N, I, S, P>> : std::bool_constant<S == StridesTraits::CONTIGUOUS> {};

    template<typename T, usize N1, typename I, StridesTraits S, PointerTraits P, usize N2>
    struct proclaim_is_accessor_nd<noa::Accessor<T, N1, I, S, P>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor<noa::AccessorReference<T, N, I, S, P>> : std::true_type {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor_restrict<noa::AccessorReference<T, N, I, S, P>> : std::bool_constant<P == PointerTraits::RESTRICT> {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor_contiguous<noa::AccessorReference<T, N, I, S, P>> : std::bool_constant<S == StridesTraits::CONTIGUOUS> {};

    template<typename T, usize N1, typename I, StridesTraits S, PointerTraits P, usize N2>
    struct proclaim_is_accessor_nd<noa::AccessorReference<T, N1, I, S, P>, N2> : std::bool_constant<N1 == N2> {};

    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    struct proclaim_is_accessor_reference<noa::AccessorReference<T, N, I, S, P>> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor<noa::AccessorValue<T, I>> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor_restrict<noa::AccessorValue<T, I>> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor_contiguous<noa::AccessorValue<T, I>> : std::true_type {};

    template<typename T, typename I, usize N>
    struct proclaim_is_accessor_nd<noa::AccessorValue<T, I>, N> : std::true_type {};

    template<typename T, typename I>
    struct proclaim_is_accessor_value<noa::AccessorValue<T, I>> : std::true_type {};
}

namespace noa::details {
    template<usize N = 0>
    struct AccessorConfig {
        /// Whether the reconfigured accessor(s) should be made const
        bool enforce_const{false};

        /// Whether the reconfigured accessor(s) should be made contiguous.
        bool enforce_contiguous{false};

        /// Whether the reconfigured accessor(s) should be made restrict.
        bool enforce_restrict{false};

        /// Whether the input accessor(s) can be an empty type.
        /// If so, empty inputs are allowed but ignored.
        bool allow_empty{false};

        /// Dimensions to store in the reconfigured accessor(s).
        /// This is equivalent to the Vec::filter(...) member function.
        /// If an empty filter is passed (default), the original dimensions are preserved.
        Vec<usize, N> filter{};
    };

    /// Constructs an accessor from the input.
    template<AccessorConfig config = AccessorConfig{}, typename Index = void, typename T>
    [[nodiscard]] constexpr auto to_accessor(T&& input) {
        using input_t = std::decay_t<T>;
        if constexpr (config.allow_empty and nt::empty<input_t>) {
            return input;
        } else if constexpr (nt::accessor_value<input_t>) {
            return input_t{std::forward<T>(input).ref()};
        } else { // Span, Accessor(Reference), View, Array, Texture
            using index_t = std::conditional_t<std::is_void_v<Index>, nt::index_type_t<input_t>, Index>;
            using value_t = std::conditional_t<config.enforce_const, nt::const_value_type_t<input_t>, nt::value_type_t<input_t>>;
            constexpr auto strides_traits = config.enforce_contiguous ? StridesTraits::CONTIGUOUS : input_t::STRIDES_TRAIT;
            constexpr auto pointer_traits = config.enforce_restrict ? PointerTraits::RESTRICT : input_t::POINTER_TRAIT;

            constexpr usize original_ndim = input_t::SIZE;
            constexpr usize vec_ndim = decltype(config.filter)::SIZE;
            constexpr usize new_ndim = vec_ndim == 0 ? original_ndim : vec_ndim;
            using accessor_t = Accessor<value_t, new_ndim, index_t, strides_traits, pointer_traits>;

            if constexpr (original_ndim == new_ndim) {
                return accessor_t(input.get(), input.strides_full().template as<index_t>());
            } else {
                auto strides = [&input]<usize... I>(std::index_sequence<I...>, const auto& filter) {
                    auto&& tmp = input.strides_full();
                    return Strides{static_cast<index_t>(tmp[filter[I]])...};
                }(std::make_index_sequence<new_ndim>{}, config.filter);
                return accessor_t(input.get(), strides.template as<index_t>());
            }
        }
    }

    template<bool ENFORCE_CONST = false, typename T>
    [[nodiscard]] constexpr auto to_accessor_contiguous(const T& v) {
        using value_t = std::conditional_t<ENFORCE_CONST, nt::const_value_type_t<T>, nt::value_type_t<T>>;
        using type_t = std::decay_t<T>;
        using index_t = nt::index_type_t<T>;
        using accessor_t = Accessor<value_t, type_t::SIZE, index_t, StridesTraits::CONTIGUOUS, type_t::POINTER_TRAIT>;
        return accessor_t(v.get(), v.strides());
    }

    template<bool ENFORCE_CONST = false, typename T>
    [[nodiscard]] constexpr auto to_accessor_contiguous_1d(const T& v) {
        using value_t = std::conditional_t<ENFORCE_CONST, nt::const_value_type_t<T>, nt::value_type_t<T>>;
        using type_t = std::decay_t<T>;
        using index_t = nt::index_type_t<T>;
        using accessor_t = Accessor<value_t, 1, index_t, StridesTraits::CONTIGUOUS, type_t::POINTER_TRAIT>;
        return accessor_t(v.get());
    }

    template<bool ENFORCE_CONST = false, typename T>
    [[nodiscard]] constexpr auto to_accessor_value(T&& v) {
        using decay_t = std::decay_t<T>;
        using value_t = std::conditional_t<ENFORCE_CONST, const decay_t, decay_t>;
        return AccessorValue<value_t>(std::forward<T>(v));
    }

    /// Reconfigures the Accessor(s).
    template<AccessorConfig config, typename T>
    requires nt::tuple_of_accessor_or_empty<std::decay_t<T>>
    [[nodiscard]] constexpr auto reconfig_accessors(T&& accessors) {
        return std::forward<T>(accessors).map(
            []<typename U>(U&& accessor) {
                return to_accessor<config>(std::forward<U>(accessor));
            });
    }

    /// Whether the accessors are aliases of each others. If empty, return false.
    /// TODO Take a shape and see if end of array doesn't overlap with other arrays? like are_overlapped
    template<nt::tuple_of_accessor_or_empty... T>
    [[nodiscard]] constexpr auto are_accessors_aliased(const T&... tuples_of_accessors) -> bool {
        auto tuple_of_pointers = tuple_cat(tuples_of_accessors.map([]<typename U>(const U& accessor) {
            if constexpr (nt::accessor_value<std::remove_reference_t<U>>)
                return nullptr;
            else
                return accessor.get();
        })...);

        return tuple_of_pointers.any_enumerate([&]<usize I>(auto ei) {
            return tuple_of_pointers.any_enumerate([ei]<usize J>(auto ej) {
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

    /// Whether the accessors point to const data, i.e., their value_type is const.
    template<nt::tuple_of_accessor_or_empty T>
    [[nodiscard]] consteval auto are_accessors_const() -> bool {
        return []<typename... A>(nt::TypeList<A...>) {
            return (std::is_const_v<nt::value_type_t<A>> and ...);
        }(nt::type_list_t<T>{});
    }

    template<typename T, usize N, nt::tuple_of_accessor_or_empty... U>
    constexpr void offset_accessors(const Vec<T, N>& offset, U&... accessors) {
        auto add_offset = [&]<typename A>(A& accessor) {
            if constexpr (nt::accessor_pure<A>) {
                accessor.offset_inplace(offset);
            } else if constexpr (nt::accessor_value<A>) {
                // do nothing
            } else {
                static_assert(nt::always_false<A>);
            }
        };
        (accessors.for_each(add_offset), ...);
    }
}
