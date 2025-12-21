#pragma once

#include "noa/runtime/core/Traits.hpp"

namespace noa::details {
    template<typename T>
    struct Batch {
        static constexpr bool IS_BATCHED = nt::pointer<T> or nt::accessor_nd<T, 1> or nt::span_nd<T, 1>;
        using type = T;
        using value_type = std::conditional_t<IS_BATCHED, nt::value_type_t<T>, T>;

        constexpr auto& operator[](nt::integer auto i) const noexcept {
            if constexpr (IS_BATCHED)
                return value[i];
            else
                return value;
        }

        constexpr auto& operator[](nt::integer auto i) noexcept {
            if constexpr (IS_BATCHED)
                return value[i];
            else
                return value;
        }

        NOA_NO_UNIQUE_ADDRESS T value;
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_batch<nd::Batch<T>> : std::true_type {};
}
