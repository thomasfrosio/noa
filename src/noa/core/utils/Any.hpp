#pragma once

#include <algorithm>

namespace noa {
    template<typename T, typename... Args>
    constexpr bool any(T value, std::initializer_list<T> args) noexcept {
        return std::any_of(args.begin(), args.end(), [value](T x) { return value == x; });
    }
}
