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

// Static for each
namespace noa {
    template<size_t... I, typename Integer = size_t, typename Op, typename... Args>
    void static_for_each(std::integer_sequence<Integer, I...>, Op&& op, Args&&... args) {
        (op.template operator()<I>(std::forward<Args>(args)...), ...);
    }

    template<size_t N, typename Integer = size_t, typename Op, typename... Args>
    void static_for_each(Op&& op, Args&&... args) {
        static_for_each(std::make_integer_sequence<Integer, N>{}, std::forward<Op>(op), std::forward<Args>(args)...);
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
#endif
