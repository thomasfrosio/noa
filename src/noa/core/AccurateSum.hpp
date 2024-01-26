#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Complex.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa {
    /// Accurate sum reduction operator for (complex) floating-points using Kahan summation, with Neumaier variation.
    template<typename T>
    struct AccurateSum {
        using remove_defaulted_final = bool; // just to make sure the final() function is picked up
        using reduced_type = std::conditional_t<nt::is_real_v<T>, f64, c64>;

        constexpr void operator()(const auto& input, reduced_type& sum, reduced_type& error) const noexcept {
            auto value = static_cast<reduced_type>(input);
            auto sum_value = value + sum;
            if constexpr (nt::is_real_v<reduced_type>) {
                error += abs(sum) >= abs(value) ?
                         (sum - sum_value) + value :
                         (value - sum_value) + sum;
            } else if constexpr (nt::is_complex_v<reduced_type>) {
                for (i64 i = 0; i < 2; ++i) {
                    error[i] += abs(sum[i]) >= abs(value[i]) ?
                                (sum[i] - sum_value[i]) + value[i] :
                                (value[i] - sum_value[i]) + sum[i];
                }
            }
            sum = sum_value;
        }

        constexpr void join(
                const reduced_type& local_sum, const reduced_type& local_error,
                reduced_type& global_sum, reduced_type& global_error
        ) const noexcept {
            global_sum += local_sum;
            global_error += local_error;
        }

        constexpr void final(const reduced_type& global_sum, const reduced_type& global_error, auto& final) {
            final = static_cast<decltype(final)>(global_sum + global_error);
        }
    };
}
