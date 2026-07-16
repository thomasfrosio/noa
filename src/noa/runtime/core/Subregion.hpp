#pragma once

#include "noa/base/Error.hpp"
#include "noa/base/Tuple.hpp"
#include "noa/base/ClampCast.hpp"
#include "noa/runtime/core/Shape.hpp"

namespace noa::inline types {
    /// Ellipsis or "..." operator, which selects the full extent of the remaining outermost dimension(s).
    struct Ellipsis {};

    /// Selects the entire dimension.
    template<usize N = 1>
    struct Full {
        static constexpr usize SIZE = N;
    };

    /// Selects a slice.
    /// Negative indexes are valid and start from the end like in python.
    /// Indexes will be clamped to the dimension size.
    /// The step must be non-zero positive (negative strides are not supported).
    struct Slice {
        static constexpr usize SIZE = 1;
        isize start{};
        isize end{std::numeric_limits<isize>::max()};
        isize step{1};
    };

    template<usize N>
    struct Slices {
        static constexpr usize SIZE = N;
        Vec<isize, N> start{};
        Vec<isize, N> end{Vec<isize, N>::from_value(std::numeric_limits<isize>::max())};
        Vec<isize, N> step{Vec<isize, N>::from_value(1)};
    };

    /// Adds a positive offset, equivalent to Slice{start}.
    struct Offset {
        static constexpr usize SIZE = 1;
        isize start{};
    };

    template<usize N>
    struct Offsets {
        static constexpr usize SIZE = N;
        Vec<isize, N> start{};
    };
}

namespace noa::traits {
    namespace details {
        template<typename> struct ProclaimSubregionIndexer : std::false_type {};
        template<> struct ProclaimSubregionIndexer<Ellipsis> : std::true_type {};
        template<> struct ProclaimSubregionIndexer<Slice> : std::true_type {};
        template<> struct ProclaimSubregionIndexer<Offset> : std::true_type {};
        template<usize N> struct ProclaimSubregionIndexer<Full<N>> : std::true_type {};
        template<usize N> struct ProclaimSubregionIndexer<Slices<N>> : std::true_type {};
        template<usize N> struct ProclaimSubregionIndexer<Offsets<N>> : std::true_type {};

        template<typename> struct ProclaimSubregionSlices : std::false_type {};
        template<usize N> struct ProclaimSubregionSlices<Slices<N>> : std::true_type {};

        template<typename> struct ProclaimSubregionOffsets : std::false_type {};
        template<usize N> struct ProclaimSubregionOffsets<Offsets<N>> : std::true_type {};

        template<typename> struct ProclaimSubregionFull : std::false_type {};
        template<usize N> struct ProclaimSubregionFull<Full<N>> : std::true_type {};
    }
    template<typename T> concept subregion_indexer = details::ProclaimSubregionIndexer<T>::value;
    template<typename T> concept subregion_slices = details::ProclaimSubregionSlices<T>::value;
    template<typename T> concept subregion_offsets = details::ProclaimSubregionOffsets<T>::value;
    template<typename T> concept subregion_full = details::ProclaimSubregionFull<T>::value;
}

namespace noa::details {
    template<typename... T>
    struct SubregionAccessSequence {
        template<typename... U>
        static consteval auto are_indexer_no_ellipsis() {
            return ((nt::integer<std::remove_reference_t<U>> or
                     (nt::subregion_indexer<U> and not nt::almost_same_as<U, Ellipsis>)) and ...);
        }

        template<typename U, typename... V>
        static consteval auto parse_() {
            // TODO For simplicity, Ellipsis is only allowed as first argument. Add support for Ellipsis everywhere.
            return (nt::almost_same_as<U, Ellipsis> or are_indexer_no_ellipsis<U>()) and are_indexer_no_ellipsis<V...>();
        }

        static consteval auto is_valid() {
            if constexpr (sizeof...(T) == 0)
                return true;
            else {
                return parse_<T...>();
            }
        }

        template<std::size_t>
        using Full1D = Full<1>;

        template<typename U>
        static constexpr auto unroll_indexer(const U& indexer) {
            if constexpr (nt::integer<U> or nt::any_of<U, Ellipsis, Slice, Offset>) {
                return noa::make_tuple(indexer);
            } else {
                if constexpr (nt::subregion_full<U>) {
                    return [&]<usize... I>(std::index_sequence<I...>) {
                        return noa::make_tuple(Full1D<I>{}...);
                    }(std::make_index_sequence<U::SIZE>{});
                } else if constexpr (nt::subregion_slices<U>) {
                    return [&]<usize... I>(std::index_sequence<I...>) {
                        return noa::make_tuple(Slice{indexer.start[I], indexer.end[I], indexer.step[I]}...);
                    }(std::make_index_sequence<U::SIZE>{});
                } else if constexpr (nt::subregion_offsets<U>) {
                    return [&]<usize... I>(std::index_sequence<I...>) {
                        return noa::make_tuple(Offset{indexer.start[I]}...);
                    }(std::make_index_sequence<U::SIZE>{});
                } else {
                    static_assert(nt::always_false<U>);
                }
            }
        }

        static constexpr auto create_tuple(const T&... indexers) {
            if constexpr (sizeof...(T) == 0)
                return noa::make_tuple(Ellipsis{});
            else
                return noa::tuple_cat(unroll_indexer(indexers)...);
        }
    };
}

namespace noa::traits {
    template<typename... T> concept subregion_access_sequence = nd::SubregionAccessSequence<T...>::is_valid();
}

namespace noa {
    template<typename T, usize N>
    struct SubregionResult {
        Shape<T, N> shape;
        Strides<T, N> strides;
        std::uintptr_t offset;
    };

    /// Utility to create and extract subregions.
    /// Dimensions can be extracted using either:
    /// - Integer:
    ///     A single index value. This is bound-checked.
    ///     Negative values up to -size are allowed mapping to index=size+(-index). As such, -1 maps to the last element.
    /// - Slices:
    ///     Slice operator. Slices are clamped to the dimension size.
    ///     Negative values are allowed and behaves as when specifying an integer.
    /// - Offset(s):
    ///     Offset the axis or axes. Equivalent to Slice(s) with slice.end and slice.step left defaulted.
    /// - Full:
    ///     Select the entire dimension.
    /// - Ellipsis:
    ///     Fills all unspecified axes with Full. If all axes are specified, this maps to nothing and is ignored.
    ///     Can only be specified as first argument. Note that there's an implicit ellipsis at the end, so Subregion()
    ///     is equivalent to Subregion(Ellipsis{}). Similarly, Subregion(0) selects the 0th index of the leftmost axis,
    ///     and all the other axis are set to Full.
    template<typename... T>
        requires nt::subregion_access_sequence<T...>
    struct Subregion {
    public:
        using helper_type = details::SubregionAccessSequence<T...>;
        using tuple_type = decltype(helper_type::create_tuple(std::declval<T>()...));
        static constexpr usize SIZE = std::tuple_size_v<tuple_type>;
        static constexpr bool IS_FIRST_ELLIPSIS = nt::same_as<std::tuple_element_t<0, tuple_type>, Ellipsis>;
        static constexpr usize MIN_AXES = IS_FIRST_ELLIPSIS ? SIZE - 1 : SIZE;

        /// Creates a new subregion.
        /// If none are specified, the implicit right side ellipsis map
        constexpr explicit Subregion(const T&... access_sequence) noexcept :
            m_indexers{helper_type::create_tuple(access_sequence...)} {}

        /// Extracts the subregion from the provided layout.
        /// Both the shape and strides are clamp casted to isize.
        template<nt::sinteger I, usize N>
            requires (N >= MIN_AXES)
        [[nodiscard]] constexpr auto extract_from(
            const Shape<I, N>& shape,
            const Strides<I, N>& strides,
            std::uintptr_t offset = 0
        ) const {
            return [&, this]<I... J>(std::integer_sequence<I, J...>) {
                SubregionResult<I, N> output{};
                (this->extract_dim_(
                    dim<N, J>(), J, clamp_cast<isize>(shape[J]), clamp_cast<isize>(strides[J]),
                    output.shape[J], output.strides[J], offset), ...);
                output.offset = offset;
                return output;
            }(std::make_integer_sequence<I, N>{});
        }

    private:
        template<usize N, usize I>
        constexpr decltype(auto) dim() const {
            if constexpr (SIZE == N) {
                return m_indexers[Tag<I>{}];
            } else if constexpr (SIZE == N + 1) {
                // Initial Ellipsis maps to no axis, ignore it.
                // Other axes are mapped explicitly so:
                return m_indexers[Tag<I + 1>{}];
            } else if constexpr (SIZE < N) {
                if constexpr (IS_FIRST_ELLIPSIS) {
                    constexpr usize J = N - SIZE;
                    if constexpr (I <= J)
                        return Full{};
                    else
                        return m_indexers[Tag<I - J>{}];
                } else {
                    if constexpr (I < SIZE)
                        return m_indexers[Tag<I>{}];
                    else
                        return Full{};
                }
            } else {
                static_assert(nt::always_false<Tag<I>>);
            }
        }

        // Compute the new size, strides and offset, for one dimension,
        // given an indexing mode (integral, slice or full).
        template<typename Op, nt::integer I>
        static constexpr void extract_dim_(
            Op op, I dim,
            I old_size, I old_strides,
            I& new_size, I& new_strides, std::uintptr_t& new_offset
        ) {
            if constexpr (nt::integer<Op>) {
                auto index = clamp_cast<I>(op);
                check(-old_size <= index and index < old_size,
                      "Index {} is out of range for a size of {} at dimension {}",
                      index, old_size, dim);

                if (index < 0)
                    index += old_size;
                new_strides = old_strides; // or 0
                new_size = 1;
                new_offset += safe_cast<std::uintptr_t>(old_strides * index);

            } else if constexpr(nt::any_of<Op, Ellipsis, Full<1>>) {
                new_strides = old_strides;
                new_size = old_size;
                new_offset += 0;
                (void) op;
                (void) dim;

            } else if constexpr(nt::same_as<Slice, Op>) {
                check(op.step > 0, "Slice step must be positive, got {}", op.step);

                if (op.start < 0)
                    op.start += old_size;
                if (op.end < 0)
                    op.end += old_size;

                op.start = clamp(op.start, I{}, old_size);
                op.end = clamp(op.end, op.start, old_size);

                new_size = divide_up(op.end - op.start, op.step);
                new_strides = old_strides * op.step;
                new_offset += safe_cast<std::uintptr_t>(op.start * old_strides);
                (void) dim;

            } else if constexpr(nt::same_as<Offset, Op>) {
                if (op.start < 0)
                    op.start += old_size;
                op.start = clamp(op.start, I{}, old_size);

                new_size = old_size - op.start;
                new_strides = old_strides;
                new_offset += safe_cast<std::uintptr_t>(op.start * old_strides);
                (void) dim;
            } else {
                static_assert(nt::always_false<Op>);
            }
        }

        tuple_type m_indexers{};
    };
}
