#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

// C++20 backport
namespace noa {
    template<typename T>
    NOA_FHD constexpr auto to_underlying(T value) noexcept {
        return static_cast<std::underlying_type_t<T>>(value);
    }
}

#if defined(NOA_IS_OFFLINE)
#include <algorithm>

namespace noa {
    template<typename T, typename... Args>
    constexpr bool any(T value, std::initializer_list<T> args) noexcept {
        return std::any_of(args.begin(), args.end(), [value](T x) { return value == x; });
    }
}

// C++20 backport
namespace noa {
    template<class T, class Alloc, class U>
    constexpr size_t erase(std::vector<T, Alloc>& c, const U& value) {
        auto it = std::remove(c.begin(), c.end(), value);
        auto r = std::distance(it, c.end());
        c.erase(it, c.end());
        return static_cast<size_t>(r);
    }

    template<class T, class Alloc, class Pred>
    constexpr size_t erase_if(std::vector<T, Alloc>& c, Pred pred) {
        auto it = std::remove_if(c.begin(), c.end(), pred);
        auto r = std::distance(it, c.end());
        c.erase(it, c.end());
        return static_cast<size_t>(r);
    }
}
#endif
