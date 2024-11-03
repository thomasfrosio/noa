#pragma once

#include <ranges> // std::ranges::begin, std::ranges::end
#include <noa/core/types/Tuple.hpp>

namespace noa::guts {
    template<typename... T>
    class ZipIterator {
    public:
        ZipIterator() = delete;

        constexpr explicit ZipIterator(T&&... iterators)
            : m_iterators{std::move(iterators)...} {
        }

    public:
        constexpr auto operator++() -> ZipIterator& {
            m_iterators.for_each([](auto& arg) { ++arg; });
            return *this;
        }

        constexpr auto operator++(int) -> ZipIterator {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        template<typename... U>
        constexpr auto operator!=(const ZipIterator<U...>& other) const {
            return not (*this == other);
        }

        template<typename... U> requires (sizeof...(U) == sizeof...(T))
        constexpr auto operator==(const ZipIterator<U...>& other) const {
            return [&]<size_t... I>(std::index_sequence<I...>) {
                return ((m_iterators[Tag<I>{}] == other.m_iterators[Tag<I>{}]) or ...);
            }(std::make_index_sequence<sizeof...(T)>{});
        }

        constexpr auto operator*() {
            return m_iterators.apply([](auto&... args) {
                // std::ranges::range_reference_t -> decltype(*args)
                return Tuple<decltype(*args)...>{*args...};
            });
        }

    public:
        Tuple<T...> m_iterators;
    };

    template<typename... T>
    class ZipRange {
    public:
        template<typename... U>
        explicit constexpr ZipRange(U&&... ranges) : m_ranges{std::forward<U>(ranges)...} {}

        constexpr auto begin() {
            return m_ranges.apply([](auto&... ranges) {
                return ZipIterator(std::ranges::begin(ranges)...);
            });
        }
        constexpr auto end() {
            return m_ranges.apply([](auto&... ranges) {
                return ZipIterator(std::ranges::end(ranges)...);
            });
        }

    private:
        Tuple<T...> m_ranges;
    };
}

namespace noa {
    /// Zip ranges ala std::views::zip (C++23).
    template<typename... T>
    constexpr auto zip(T&&... r) {
        return guts::ZipRange<T...>(std::forward<T>(r)...);
    }
}