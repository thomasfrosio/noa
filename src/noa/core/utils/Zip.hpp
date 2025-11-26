#pragma once

#include <ranges> // std::ranges::begin/end
#include <noa/core/types/Tuple.hpp>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmissing-braces"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

namespace noa::details {
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
        constexpr bool operator!=(const ZipIterator<U...>& other) const {
            return not (*this == other);
        }

        template<typename... U> requires (sizeof...(U) == sizeof...(T))
        constexpr bool operator==(const ZipIterator<U...>& other) const {
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
        return details::ZipRange<T...>(std::forward<T>(r)...);
    }
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#pragma warning(pop)
#endif
