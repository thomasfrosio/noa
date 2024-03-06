#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/math/Comparison.hpp"

#ifdef NOA_IS_OFFLINE
#include <type_traits>
#include <iterator>
#else
#include <cuda/std/type_traits>
#include <cuda/std/iterator>
#endif

// Adapted from https://github.com/pytorch/pytorch/blob/master/c10/util/irange.h
namespace noa::guts {
    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    struct IntIterator {
    public:
        // iterator traits
        using difference_type = I;
        using value_type = I;
        using pointer = const I*;
        using reference = const I&;
        using iterator_category = std::forward_iterator_tag;

    public:
        constexpr explicit IntIterator(I value_) noexcept: value(value_) {}
        constexpr I operator*() const noexcept { return value; }
        constexpr I const* operator->() const noexcept { return &value; }

        constexpr IntIterator& operator++() noexcept {
            ++value;
            return *this;
        }

        constexpr IntIterator operator++(int) noexcept {
            const auto copy = *this;
            ++*this;
            return copy;
        }

        constexpr bool operator==(const IntIterator& other) const noexcept { return value == other.value; }
        constexpr bool operator!=(const IntIterator& other) const noexcept { return value != other.value; }

    protected:
        I value;
    };
}

namespace noa {
    template<typename I, typename = std::enable_if_t<std::is_integral_v<I>>>
    struct IntRange {
    public:
        constexpr IntRange(I begin, I end) noexcept: m_begin(begin), m_end(end) {}
        constexpr guts::IntIterator<I> begin() const noexcept { return m_begin; }
        constexpr guts::IntIterator<I> end() const noexcept { return m_end; }
    private:
        guts::IntIterator<I> m_begin;
        guts::IntIterator<I> m_end;
    };

    /// Creates an integer range for the half-open interval [begin, end)
    /// If end<=begin, then the range is empty.
    /// The range has the type of the `end` integer; `begin` integer is
    /// cast to this type.
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    constexpr IntRange<T> irange(T begin, T end) noexcept {
        // If end<=begin then the range is empty; we can achieve this effect by
        // choosing the larger of {begin, end} as the loop terminator
        return {begin, max(begin, end)};
    }

    /// Creates an integer range for the half-open interval [0, end)
    /// If end<=begin, then the range is empty
    template<typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    constexpr IntRange<T> irange(T end) noexcept {
        // If end<=begin then the range is empty; we can achieve this effect by
        // choosing the larger of {0, end} as the loop terminator
        // Handles the case where end<0. irange only works for ranges >=0
        return {T{}, max(T{}, end)};
    }
}
