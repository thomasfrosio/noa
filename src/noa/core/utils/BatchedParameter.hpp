#pragma once

#include "noa/core/Traits.hpp"

namespace noa {
    template<typename T>
    class BatchedParameter {
    public:
        using type = T;
        static constexpr bool IS_BATCHED = nt::pointer<type> or nt::accessor_nd<type, 1>;
        using value_type = std::conditional_t<IS_BATCHED, nt::value_type_t<T>, T>;

    public:
        constexpr decltype(auto) operator[](nt::integer auto batch) const noexcept {
            if constexpr (IS_BATCHED)
                return value[batch];
            else
                return value;
        }

        NOA_NO_UNIQUE_ADDRESS T value;
    };
}

namespace noa::traits {
    template<typename T> struct proclaim_is_batched_parameter<BatchedParameter<T>> : std::true_type {};
}
