#pragma once

#include <algorithm>

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

// C++20 backport
namespace noa {
    template<typename T>
    NOA_FHD constexpr auto to_underlying(T value) noexcept {
        return static_cast<std::underlying_type_t<T>>(value);
    }
}

// C++23 backport
namespace noa {
    template<class T, class U>
    [[nodiscard]] constexpr auto&& forward_like(U&& x) noexcept {
        constexpr bool is_adding_const = std::is_const_v<std::remove_reference_t<T>>;
        if constexpr (std::is_lvalue_reference_v<T&&>) {
            if constexpr (is_adding_const)
                return std::as_const(x);
            else
                return static_cast<U&>(x);
        } else {
            if constexpr (is_adding_const)
                return std::move(std::as_const(x));
            else
                return std::move(x);
        }
    }
}

// Static for each
namespace noa {
    template<usize... I, typename Integer = usize, typename Op, typename... Args>
    void static_for_each(std::integer_sequence<Integer, I...>, Op&& op, Args&&... args) {
        (op.template operator()<I>(args...), ...);
    }

    template<usize N, typename Integer = usize, typename Op, typename... Args>
    void static_for_each(Op&& op, Args&&... args) {
        static_for_each(std::make_integer_sequence<Integer, N>{}, std::forward<Op>(op), args...);
    }
}

namespace noa {
    template<typename T>
    constexpr bool any(T value, std::initializer_list<T> args) noexcept {
        return std::any_of(args.begin(), args.end(), [value](T x) { return value == x; });
    }
}
