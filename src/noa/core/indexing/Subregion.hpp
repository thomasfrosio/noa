#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/utils/ClampCast.hpp"

namespace noa::indexing {
    /// Ellipsis or "..." operator, which selects the full extent of the remaining outermost dimension(s).
    struct Ellipsis {};

    /// Selects the entire dimension.
    struct Full {};

    /// Slice operator.
    /// Negative indexes are valid and start from the end like in python.
    /// Indexes will be clamped to the dimension size.
    /// The step must be non-zero positive (negative strides are not supported).
    struct Slice {
        template<nt::integer T = isize, nt::integer U = isize, nt::integer V = isize>
        constexpr explicit Slice(
            T start_ = 0,
            U end_ = std::numeric_limits<isize>::max(),
            V step_ = V{1}
        ) noexcept :
            start{static_cast<isize>(start_)},
            end{static_cast<isize>(end_)},
            step{static_cast<isize>(step_)} {}

        isize start{};
        isize end{};
        isize step{};
    };

    struct Offset {
        template<nt::integer T = isize>
        constexpr explicit Offset(T start_ = 0) noexcept : start{static_cast<isize>(start_)} {}
        isize start{};
    };
}

namespace noa::indexing::details {
    template<usize N, typename... T>
    struct SubregionParser {
        template<typename U>
        static consteval auto is_subregion_indexer_no_ellipsis() {
            return nt::integer<std::remove_reference_t<U>> or nt::almost_any_of<U, Full, Slice, Offset>;
        }

        template<typename... U>
        static consteval auto are_subregion_indexer_no_ellipsis() {
            return (is_subregion_indexer_no_ellipsis<U>() and ...);
        }

        template<typename U, typename... V>
        static consteval auto parse() {
            return (nt::almost_same_as<U, Ellipsis> or is_subregion_indexer_no_ellipsis<U>()) and
                   are_subregion_indexer_no_ellipsis<V...>();
        }

        static constexpr bool value = sizeof...(T) <= N and (sizeof...(T) == 0 or parse<T...>());
    };
}

namespace noa::traits {
    template<usize N, typename... T>
    concept subregion_indexing = ni::details::SubregionParser<N, T...>::value;
}

namespace noa::indexing {
    template<typename T, usize N>
    struct SubregionResult {
        Shape<T, N> shape;
        Strides<T, N> strides;
        std::uintptr_t offset;
    };

    /// Utility to create and extract subregions.
    /// Dimensions can be extracted using either:
    /// -   A single index value: This is bound-checked. Negative values are allowed.
    /// -   Full: Select the entire dimension.
    /// -   Slice: Slice operator. Slices are clamped to the dimension size. Negative values are allowed.
    /// -   Offset: Offset the dimension. Equivalent to Slice{offset}.
    /// -   Ellipsis: Fills all unspecified dimensions with Full.
    template<usize N, typename... T> requires nt::subregion_indexing<N, T...>
    struct Subregion {
        /// Creates a new subregion.
        constexpr explicit Subregion(const T&... indices) noexcept : m_ops{make_tuple(indices...)} {}

        /// Extracts the subregion from the provided layout.
        template<nt::sinteger I>
        [[nodiscard]] constexpr auto extract_from(
            const Shape<I, N>& shape,
            const Strides<I, N>& strides,
            std::uintptr_t offset = 0
        ) const {
            return [&, this]<I... J>(std::integer_sequence<I, J...>) {
                SubregionResult<I, N> output{};
                (this->extract_dim_(dim<J>(), J, shape[J], strides[J], output.shape[J], output.strides[J], offset), ...);
                output.offset = offset;
                return output;
            }(std::make_integer_sequence<I, N>{});
        }

    private:
        template<usize I>
        constexpr decltype(auto) dim() const {
            constexpr usize COUNT = sizeof...(T);
            if constexpr (COUNT == N) {
                return m_ops[Tag<I>{}];
            } else if constexpr (COUNT < N) {
                constexpr usize J = N - COUNT;
                if constexpr (nt::almost_same_as<decltype(m_ops[Tag<0>{}]), Ellipsis>) {
                    if constexpr (I <= J)
                        return Full{};
                    else
                        return m_ops[Tag<I - J>{}];
                } else {
                    if constexpr (I < COUNT)
                        return m_ops[Tag<I>{}];
                    else
                        return Full{};
                }
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

            } else if constexpr(nt::any_of<Op, Ellipsis, Full>) {
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

        Tuple<std::decay_t<T>...> m_ops{};
    };

    template<usize N, typename... T> requires nt::subregion_indexing<N, T...>
    constexpr auto make_subregion(const T&... indices) noexcept {
        return Subregion<N, T...>(indices...);
    }
}
