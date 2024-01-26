#pragma once

#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Tuple.hpp"

namespace noa {
    /// Index-wise reduce operator to find the first minimum value.
    template<typename Accessor, typename Offset>
    struct FirstMin {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first < reduced.first or (current.first == reduced.first and current.second < reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };

    /// Index-wise reduce operator to find the first maximum value.
    template<typename Accessor, typename Offset>
    struct FirstMax {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first > reduced.first or (reduced.first == current.first and current.second < reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };

    /// Index-wise reduce operator to find the last minimum value.
    template<typename Accessor, typename Offset>
    struct LastMin {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first < reduced.first or (current.first == reduced.first and current.second > reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };

    /// Index-wise reduce operator to find the last maximum value.
    template<typename Accessor, typename Offset>
    struct LastMax {
        using accessor_type = Accessor;
        using value_type = accessor_type::value_type;
        using offset_type = Offset;
        using reduced_type = Pair<value_type, offset_type>;

        accessor_type accessor;

        constexpr void init(const auto& indices, reduced_type& reduced) const noexcept {
            reduced.first = accessor(indices);
            reduced.second = static_cast<offset_type>(accessor.offset_at(indices));
        }

        constexpr void join(const reduced_type& current, reduced_type& reduced) const noexcept {
            if (current.first > reduced.first or (reduced.first == current.first and current.second > reduced.second))
                reduced = current;
        }

        constexpr void final(const reduced_type& reduced, value_type& output) const noexcept {
            output = reduced.first;
        }
    };
}
