#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa {
    template<typename T>
    struct AccurateVariance {
        using accurate_input_type = std::conditional_t<nt::is_complex_v<T>, c64, f64>;
        using reduced_type = nt::value_type_t<T>;
        accurate_input_type mean{};

        template<typename I>
        NOA_FHD constexpr void init(const I& input, reduced_type& output) const noexcept {
            const auto tmp = static_cast<accurate_input_type>(input);
            if constexpr (nt::is_complex_v<I>) {
                const auto distance = abs(tmp - mean);
                output += distance * distance;
            } else {
                const auto distance = tmp - mean;
                output += distance * distance;
            }
        }

        NOA_FHD constexpr void join(const reduced_type& to_reduce, reduced_type& output) const noexcept {
            output += to_reduce;
        }
    };
}
