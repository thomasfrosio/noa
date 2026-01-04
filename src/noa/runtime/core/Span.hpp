#pragma once

#include "noa/runtime/core/Access.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/core/Subregion.hpp"
#include "noa/runtime/core/Utils.hpp"

namespace noa::details {
    template<typename T, typename I, StridesTraits StrideTrait>
    struct SpanIterator {
    public:
        // iterator traits
        using difference_type = I;
        using value_type = T;
        using pointer = T*;
        using reference = T&;
        using iterator_category = std::forward_iterator_tag;

        // other
        static constexpr bool IS_CONTIGUOUS = StrideTrait == StridesTraits::CONTIGUOUS;
        using index_type = I;
        using stride_type = std::conditional_t<IS_CONTIGUOUS, Empty, index_type>;

    public:
        constexpr SpanIterator() = default;
        constexpr explicit SpanIterator(pointer p, I = {}) noexcept requires (IS_CONTIGUOUS) : m_pointer(p) {}
        constexpr explicit SpanIterator(pointer p, I stride) noexcept requires (not IS_CONTIGUOUS) : m_pointer(p), m_stride(stride) {}
        constexpr reference operator*() const noexcept { return *m_pointer; }
        constexpr pointer operator->() const noexcept { return m_pointer; }

        constexpr SpanIterator& operator++() noexcept {
            if constexpr (IS_CONTIGUOUS)
                ++m_pointer;
            else
                m_pointer += m_stride;
            return *this;
        }

        constexpr SpanIterator operator++(int) noexcept {
            const auto copy = *this;
            ++*this;
            return copy;
        }

        constexpr bool operator==(const SpanIterator& other) const noexcept { return m_pointer == other.m_pointer; }
        constexpr bool operator!=(const SpanIterator& other) const noexcept { return m_pointer != other.m_pointer; }

    protected:
        pointer m_pointer;
        NOA_NO_UNIQUE_ADDRESS stride_type m_stride;
    };
}

namespace noa::inline types {
    /// Multidimensional span.
    template<typename T, usize N = 1, typename I = isize,
             StridesTraits StridesTrait = StridesTraits::STRIDED,
             PointerTraits PointerTrait = PointerTraits::DEFAULT>
    class Span : public details::Indexer<Span<T, N, I, StridesTrait>, N> {
    public:
        static_assert(not std::is_reference_v<T> and
                      not std::is_pointer_v<T> and
                      not std::extent_v<T> and
                      std::is_integral_v<I>);

        static constexpr StridesTraits STRIDES_TRAIT = StridesTrait;
        static constexpr PointerTraits POINTER_TRAIT = PointerTrait;
        static constexpr bool IS_CONTIGUOUS = STRIDES_TRAIT == StridesTraits::CONTIGUOUS;
        static constexpr bool IS_RESTRICT = POINTER_TRAIT == PointerTraits::DEFAULT;
        static constexpr usize SIZE = N;
        static constexpr isize SSIZE = N;

        using value_type = T;
        using mutable_value_type = std::remove_const_t<T>;
        using const_value_type = const mutable_value_type;
        using pointer_type = value_type*;
        using const_pointer_type = const_value_type*;
        using reference_type = std::add_lvalue_reference_t<value_type>;
        using index_type = I;
        using shape_type = Shape<index_type, N>;
        using strides_type = Strides<index_type, N - IS_CONTIGUOUS>;
        using strides_full_type = Strides<index_type, N>;
        using span_iterator_type = nd::SpanIterator<value_type, index_type, STRIDES_TRAIT>;
        using contiguous_span_type = Span<value_type, SIZE, index_type, StridesTraits::CONTIGUOUS, POINTER_TRAIT>;

    public: // Constructors
        /// Creates an empty span.
        constexpr Span() = default;

        /// Creates a span of contiguous 1d data.
        NOA_HD constexpr Span(
            pointer_type data,
            index_type size
        ) noexcept :
            m_ptr{data},
            m_shape{shape_type::from_value(1).template set<N - 1>(size)},
            m_strides{strides_type::from_value(1)} {}

        /// Creates a span of rightmost contiguous nd data.
        template<usize A>
        NOA_HD constexpr Span(
            pointer_type data,
            const Shape<index_type, SIZE, A>& shape
        ) noexcept : Span(data, shape, shape.strides()) {}

        /// Creates a span of nd data.
        /// For contiguous spans, since the width stride (strides[N-1]) is ignored, it doesn't have to be specified.
        template<usize A, usize B> requires IS_CONTIGUOUS
        NOA_HD constexpr Span(
            pointer_type pointer,
            const Shape<index_type, SIZE, A>& shape,
            const Strides<index_type, SIZE - 1, B>& strides
        ) noexcept :
            m_ptr{pointer},
            m_shape{shape_type::from_shape(shape)},
            m_strides{strides_type::from_strides(strides)} {}

        /// Creates a span of nd data.
        /// If the span is contiguous, the width stride (strides[N-1]) is ignored and assumed to be 1.
        template<usize A, usize B>
        NOA_HD constexpr Span(
            pointer_type pointer,
            const Shape<index_type, SIZE, A>& shape,
            const Strides<index_type, SIZE, B>& strides
        ) noexcept :
            m_ptr{pointer},
            m_shape{shape_type::from_shape(shape)},
            m_strides{strides_type::from_pointer(strides.data())} {}

        /// Creates a span of 1d contiguous data.
        template<typename U, usize S> requires (not std::is_void_v<value_type>)
        NOA_HD constexpr explicit Span(U (& data)[S]) noexcept: Span(data, S) {}

        /// Creates a const span from an existing non-const span, or from/to a void span.
        template<typename U> requires (nt::mutable_of<U, value_type> or std::is_void_v<U> or std::is_void_v<value_type>)
        NOA_HD constexpr /*implicit*/ Span(const Span<U, SIZE, index_type, STRIDES_TRAIT, POINTER_TRAIT>& span) noexcept :
            m_ptr{span.get()},
            m_shape{span.shape()},
            m_strides{span.strides()} {}

        /// Creates a non-contiguous span from an otherwise identical contiguous span.
        template<typename U> requires (nt::same_as<U, contiguous_span_type> and (not IS_CONTIGUOUS))
        NOA_HD constexpr /*implicit*/ Span(const U& span) noexcept :
            m_ptr{span.get()},
            m_shape{span.shape()},
            m_strides{span.strides_full()} {}

    public: // Accessing strides
        template<usize INDEX>
        [[nodiscard]] NOA_HD constexpr auto stride() const noexcept -> index_type {
            static_assert(INDEX < N);
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
                    return Strides{(*this).template stride<J>()...}; // returns by value
                }(std::make_index_sequence<SIZE>{});
            } else {
                return strides(); // returns a lvalue const ref
            }
        }

        [[nodiscard]] NOA_HD constexpr auto shape() noexcept -> shape_type& { return m_shape; }
        [[nodiscard]] NOA_HD constexpr auto shape() const noexcept -> const shape_type& { return m_shape; }

        [[nodiscard]] NOA_HD constexpr auto get() const noexcept -> pointer_type { return m_ptr; }
        [[nodiscard]] NOA_HD constexpr auto data() const noexcept -> pointer_type { return m_ptr; }

    public: // Range
        [[nodiscard]] NOA_HD constexpr auto n_elements() const noexcept -> index_type { return m_shape.n_elements(); }
        [[nodiscard]] NOA_HD constexpr auto size() const noexcept -> usize { return static_cast<usize>(n_elements()); };
        [[nodiscard]] NOA_HD constexpr auto ssize() const noexcept -> isize { return static_cast<isize>(n_elements()); };

        [[nodiscard]] NOA_HD constexpr auto begin() const noexcept -> span_iterator_type requires (N == 1) {
            return span_iterator_type(get(), stride<0>());
        }
        [[nodiscard]] NOA_HD constexpr auto end() const noexcept -> span_iterator_type requires (N == 1) {
            return span_iterator_type(get() + ssize(), stride<0>());
        }

        [[nodiscard]] NOA_HD constexpr auto front() const noexcept -> reference_type requires (N == 1 and not std::is_void_v<value_type>) {
            NOA_ASSERT(not is_empty());
            return (*this)[0];
        }

        [[nodiscard]] NOA_HD constexpr auto back() const noexcept -> reference_type requires (N == 1 and not std::is_void_v<value_type>) {
            NOA_ASSERT(not is_empty());
            return (*this)[ssize() - 1];
        }

        /// C-style indexing operator, decrementing the dimensionality of the span by 1.
        [[nodiscard]] NOA_HD constexpr auto operator[](
            nt::integer auto index
        ) const noexcept requires (N > 1) {
            NOA_ASSERT(not is_empty());
            noa::bounds_check(shape(), index);
            using output_t = Span<value_type, N - 1, index_type, STRIDES_TRAIT, POINTER_TRAIT>;
            return output_t(get() + noa::offset_at(stride<0>(), index), shape().pop_front(), strides().pop_front());
        }

        /// C-style indexing operator, decrementing the dimensionality of the span by 1.
        /// When done on a 1d span, this acts as a pointer/array indexing and dereferences the data.
        [[nodiscard]] NOA_HD constexpr auto& operator[](
            nt::integer auto index
        ) const noexcept requires (N == 1 and not std::is_void_v<value_type>) {
            NOA_ASSERT(not is_empty());
            noa::bounds_check(shape(), index);
            return get()[noa::offset_at(stride<0>(), index)];
        }

    public:
        template<char ORDER = 'C'>
        [[nodiscard]] constexpr bool is_contiguous() const noexcept requires (SIZE == 4) {
            return strides_full().template is_contiguous<ORDER>(shape());
        }

        /// Whether the span is empty. A span is empty if not initialized,
        /// or if the viewed data is null, or if one of its dimension is 0.
        [[nodiscard]] NOA_HD constexpr bool is_empty() const noexcept { return not get() or shape().is_empty(); }
        [[nodiscard]] NOA_HD constexpr explicit operator bool() const noexcept { return not is_empty(); }

        /// Returns a new Span.
        /// \details While constructing the span, this function can also reinterpret the current value type.
        ///          This is only well-defined in cases where Span::as<U>() is well-defined.
        ///          If N < NewN, the outer-dimensions are stacked together.
        template<typename NewT = T, usize NewN = N, typename NewI = index_type,
                 StridesTraits NewStridesTrait = STRIDES_TRAIT,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto span() const {
            using output_span_t = Span<NewT, NewN, NewI, NewStridesTrait, NewPointerTrait>;

            const auto reinterpreted = details::ReinterpretLayout(shape(), strides_full(), get()).template as<NewT>();

            if constexpr (STRIDES_TRAIT != StridesTraits::CONTIGUOUS and
                          NewStridesTrait == StridesTraits::CONTIGUOUS) {
                check(reinterpreted.strides[N - 1] == 1,
                      "Cannot convert a non-contiguous span (strides={}) to a contiguous span",
                      reinterpreted.strides);
            }

            if constexpr (NewN == N) {
                return output_span_t(
                    reinterpreted.ptr,
                    reinterpreted.shape.template as_safe<NewI>(),
                    reinterpreted.strides.template as_safe<NewI>());
            } else if constexpr (NewN > N) {
                // Add empty dimensions on the left.
                constexpr usize n_dimensions_to_add = NewN - N;
                auto new_truncated_shape = reinterpreted.shape.template as_safe<NewI>();
                auto new_truncated_strides = reinterpreted.strides.template as_safe<NewI>();
                auto new_leftmost_stride = new_truncated_strides[0] * new_truncated_shape[0];
                return output_span_t(
                    reinterpreted.ptr,
                    new_truncated_shape.template push_front<n_dimensions_to_add>(1),
                    new_truncated_strides.template push_front<n_dimensions_to_add>(new_leftmost_stride));
            } else {
                // Construct the new shape by stacking the outer dimensions together.
                constexpr usize OFFSET = N - NewN;
                auto new_shape = Shape<index_type, N>::filled_with(1);
                for (usize i{}; i < N; ++i)
                    new_shape[max(i, OFFSET)] *= reinterpreted.shape[i];

                // Reshape.
                Strides<index_type, N> new_stride{};
                check(details::reshape(reinterpreted.shape, reinterpreted.strides, new_shape, new_stride),
                      "An array of shape {} and strides {} cannot be reshaped to shape {}",
                      reinterpreted.shape, reinterpreted.strides, new_shape);

                // Then remove the outer empty dimensions.
                return output_span_t(
                    reinterpreted.ptr,
                    new_shape.template pop_front<OFFSET>().template as_safe<NewI>(),
                    new_stride.template pop_front<OFFSET>().template as_safe<NewI>());
            }
        }

        template<typename NewT, usize NewN = N,
                 typename NewI = index_type,
                 StridesTraits NewStridesTrait = STRIDES_TRAIT,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as() const {
            return span<NewT, NewN, NewI, NewStridesTrait, NewPointerTrait>();
        }

        template<usize NewN = N,
                 typename NewI = index_type,
                 StridesTraits NewStridesTrait = STRIDES_TRAIT,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_const() const {
            return Span<const_value_type, NewN, NewI, NewStridesTrait, NewPointerTrait>(*this);
        }

        template<usize NewN = N,
                 typename NewI = index_type,
                 StridesTraits NewStridesTrait = STRIDES_TRAIT,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_bytes() const {
            using output_t = std::conditional_t<std::is_const_v<value_type>, const Byte, Byte>;
            return span<output_t, NewN, NewI, NewStridesTrait, NewPointerTrait>();
        }

        template<typename U = value_type, usize NewN = N,
                 typename NewI = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_strided() const {
            return span<U, NewN, NewI, StridesTraits::STRIDED, NewPointerTrait>();
        }

        template<typename U = value_type, usize NewN = N,
                 typename NewI = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_contiguous() const {
            return span<U, NewN, NewI, StridesTraits::CONTIGUOUS, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename NewI = index_type,
                 StridesTraits NewStridesTrait = StridesTraits::CONTIGUOUS,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_1d() const {
            return span<U, 1, NewI, NewStridesTrait, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename NewI = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_1d_contiguous() const {
            return span<U, 1, NewI, StridesTraits::CONTIGUOUS, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename NewI = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_1d_strided() const {
            return span<U, 1, NewI, StridesTraits::STRIDED, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename NewI = index_type,
                 StridesTraits NewStridesTrait = STRIDES_TRAIT,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_4d() const {
            return span<U, 4, NewI, NewStridesTrait, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename NewI = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_4d_contiguous() const {
            return span<U, 4, NewI, StridesTraits::CONTIGUOUS, NewPointerTrait>();
        }

        template<typename U = value_type,
                 typename NewI = index_type,
                 PointerTraits NewPointerTrait = POINTER_TRAIT>
        [[nodiscard]] constexpr auto as_4d_strided() const {
            return span<U, 4, NewI, StridesTraits::STRIDED, NewPointerTrait>();
        }

        /// Reshapes the view (must have the same number of elements as the current view).
        [[nodiscard]] auto reshape(shape_type new_shape) const -> Span {
            // Infer the size, if needed.
            check(nd::infer_size(new_shape, n_elements()),
                  "The desired shape {} is not compatible with the current shape {}, "
                  "or the size inference is invalid or ambiguous", new_shape, shape());

            // Then reshape.
            strides_full_type new_stride;
            check(nd::reshape(shape(), strides_full(), new_shape, new_stride),
                  "An memory region of shape {} and stride {} cannot be reshaped to a shape of {}",
                  shape(), strides_full(), new_shape);

            return Span(get(), new_shape, new_stride);
        }

        /// Reshapes the array in a vector along a particular axis.
        /// Returns a row vector by default (axis = 3).
        [[nodiscard]] auto flat(i32 axis = N - 1) const -> Span {
            bounds_check<true>(N, axis);
            auto output_shape = shape_type::filled_with(1);
            output_shape[axis] = shape().n_elements();
            return reshape(output_shape);
        }

        /// Permutes the dimensions of the view.
        /// \param permutation  Permutation with the axes numbered from 0 to 3.
        [[nodiscard]] constexpr auto permute(const Vec<i32, N>& permutation) const -> Span {
            return Span(get(), shape().permute(permutation), strides_full().permute(permutation));
        }

        /// Returns a span with the given axes.
        template<nt::integer... U> requires (STRIDES_TRAIT == StridesTraits::STRIDED)
        [[nodiscard]] constexpr auto filter(U... axes) const {
            return Span<value_type, sizeof...(U), index_type, STRIDES_TRAIT>(
                get(), shape().filter(axes...), strides().filter(axes...));
        }

        /// Subregion indexing. Extracts a subregion from the current span.
        template<typename... U>
        [[nodiscard]] constexpr auto subregion(const Subregion<SIZE, U...>& subregion) const -> Span {
            auto [new_shape, new_strides, offset] = subregion.extract_from(
                shape().template as_safe<isize>(),
                strides_full().template as_safe<isize>());
            return Span(get() + offset,
                        new_shape.template as_safe<index_type>(),
                        new_strides.template as_safe<index_type>());
        }

        /// Subregion indexing. Extracts a subregion from the current span.
        /// \see noa::indexing::Subregion for more details on the variadic parameters to enter.
        template<typename... Ts> requires nt::subregion_access_sequence<N, Ts...>
        [[nodiscard]] constexpr auto subregion(const Ts&... access_sequence) const -> Span {
            return subregion(Subregion<N, Ts...>(access_sequence...));
        }

    private:
        pointer_type m_ptr{};
        shape_type m_shape{};
        NOA_NO_UNIQUE_ADDRESS strides_type m_strides{};
    };

    /// Deduction guide. Span(ptr, size) creates a contiguous span.
    template<typename T, nt::integer I>
    Span(T*, I) -> Span<T, 1, I, StridesTraits::CONTIGUOUS>;

    /// Deduction guide. Span(ptr, shape) creates a contiguous span.
    template<typename T, nt::integer I, usize N, usize A>
    Span(T*, const Shape<I, N, A>&) -> Span<T, N, I, StridesTraits::CONTIGUOUS>;

    /// Deduction guide. Span(c_array) creates a contiguous span.
    template<typename T, usize S>
    Span(T (&)[S]) -> Span<T, 1, isize, StridesTraits::CONTIGUOUS>;

    template<typename T, usize N = 1, typename I = isize>
    using SpanContiguous = Span<T, N, I, StridesTraits::CONTIGUOUS>;
}

namespace noa::traits {
    template<typename T, usize N, typename I, StridesTraits S> struct proclaim_is_span<noa::Span<T, N, I, S>> : std::true_type {};
    template<typename T, usize N, typename I> struct proclaim_is_span_contiguous<noa::Span<T, N, I, StridesTraits::CONTIGUOUS>> : std::true_type {};

    template<typename T, usize N1, typename I, StridesTraits S, usize N2> struct proclaim_is_span_nd<noa::Span<T, N1, I, S>, N2> : std::bool_constant<N1 == N2> {};
    template<typename T, usize N1, typename I, usize N2> struct proclaim_is_span_contiguous_nd<noa::Span<T, N1, I, StridesTraits::CONTIGUOUS>, N2> : std::bool_constant<N1 == N2> {};
}

namespace noa {
    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    [[nodiscard]] constexpr auto is_broadcastable(
        const Span<T, N, I, S, P>& span,
        const Shape<I, N>& shape
    ) -> bool {
        return is_broadcastable(span.shape(), shape);
    }

    /// Broadcasts an array to a given shape.
    template<typename T, usize N, typename I, StridesTraits S, PointerTraits P>
    [[nodiscard]] auto broadcast(
        const Span<T, N, I, S, P>& span,
        const Shape<I, N>& shape
    ) {
        auto strides = span.strides();
        if (not broadcast(span.shape(), strides, shape))
            panic("Cannot broadcast shape={} into a shape={}", span.shape(), shape);
        return Span<T, N, I, S, P>(span.get(), shape, strides);
    }

    /// Whether \p lhs and \p rhs overlap in memory.
    [[nodiscard]] bool are_overlapped(const nt::span auto& lhs, const nt::span auto& rhs) {
        if (lhs.is_empty() or rhs.is_empty())
            return false;
        auto const lhs_start = reinterpret_cast<uintptr_t>(lhs.get());
        auto const rhs_start = reinterpret_cast<uintptr_t>(rhs.get());
        auto const lhs_end = reinterpret_cast<uintptr_t>(lhs.get() + offset_at(lhs.strides_full(), (lhs.shape() - 1).vec));
        auto const rhs_end = reinterpret_cast<uintptr_t>(rhs.get() + offset_at(rhs.strides_full(), (rhs.shape() - 1).vec));
        return details::are_overlapped(lhs_start, lhs_end, rhs_start, rhs_end);
    }

    /// Returns the multidimensional indices of \p span corresponding to a memory \p offset.
    /// \note 0 indicates the beginning of the span. The span should not have any broadcast dimension.
    template<typename T, usize N, typename I, StridesTraits S>
    [[nodiscard]] constexpr auto offset2index(isize offset, const Span<T, N, I, S>& span) -> Vec<isize, N> {
        check(span.strides() > 0,
              "Cannot retrieve the indices from broadcast strides. Got strides={}",
              span.strides());
        return offset2index(offset, span.strides(), span.shape());
    }
}
