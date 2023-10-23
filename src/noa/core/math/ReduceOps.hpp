#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/math/Generic.hpp"

namespace noa::math {
    // Accurate sum reduction for floating-points using Kahan summation, with Neumaier variation.
    // It accumulates a local (per-thread) error, which should then be safely added to the global
    // error thanks to the closure() function.
    template<typename T>
    struct KahanOp {
        using value_type = std::conditional_t<nt::is_real_v<T>, f64, c64>;
        value_type* global_error{};
        value_type local_error{};

        auto operator()(value_type sum, value_type value) noexcept -> value_type {
            auto sum_value = sum + value;
            if constexpr (nt::is_real_v<value_type>) {
                local_error += noa::abs(sum) >= noa::abs(value) ?
                               (sum - sum_value) + value :
                               (value - sum_value) + sum;
            } else {
                for (i64 i = 0; i < 2; ++i) {
                    local_error[i] += noa::abs(sum[i]) >= noa::abs(value[i]) ?
                                      (sum[i] - sum_value[i]) + value[i] :
                                      (value[i] - sum_value[i]) + sum[i];
                }
            }
            return sum_value;
        }

        constexpr void closure(i64) const noexcept { // must be in thread-safe scope
            NOA_ASSERT(global_error != nullptr);
            *global_error += local_error;
        }
    };
    static_assert(nt::is_detected_v<nt::has_closure, KahanOp<f32>>);
    static_assert(nt::is_detected_v<nt::has_closure, KahanOp<f64>>);
    static_assert(nt::is_detected_v<nt::has_closure, KahanOp<c32>>);
    static_assert(nt::is_detected_v<nt::has_closure, KahanOp<c64>>);

    // This is used as preprocessing operator to compute the variance.
    // The actual reduction operator is just noa::plus_t{}.
    template<typename Value>
    struct AccurateVariance {
        using output_type = nt::value_type_t<Value>;
        Value mean{};

        template<typename Input>
        NOA_FHD constexpr output_type operator()(Input value) const noexcept {
            if constexpr (nt::is_complex_v<Input>) {
                const auto tmp = static_cast<c64>(value);
                const auto distance = noa::abs(tmp - mean);
                return distance * distance;
            } else {
                const auto tmp = static_cast<f64>(value);
                const auto distance = tmp - mean;
                return distance * distance;
            }
        }
    };
}
