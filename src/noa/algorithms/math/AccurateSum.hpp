#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::math {
    // Accurate sum reduction for floating-points using Kahan summation, with Neumaier variation.
    // This is the reduction operator for the reduce_ewise functions, in the CPU backend.
    // It accumulates a local (per-thread) error, which should then be safely added to the
    // global error thanks to the closure() function.
    // The CUDA backend doesn't support the closure() function for the reduction operators,
    // but that's fine because in CUDA the simple noa::plus_t{} reduction operator
    // is effectively a very fine "partial sum" thanks to the number of threads, and is as accurate.
    struct AccuratePlusReal {
        f64* global_error{};
        f64 local_error{0};

        f64 operator()(f64 sum, f64 value) noexcept {
            auto sum_value = sum + value;
            local_error += noa::math::abs(sum) >= noa::math::abs(value) ?
                           (sum - sum_value) + value :
                           (value - sum_value) + sum;
            return sum_value;
        }

        constexpr void closure(i64) const noexcept { // must be in thread-safe scope
            NOA_ASSERT(global_error != nullptr);
            *global_error += local_error;
        }
    };
    static_assert(noa::traits::is_detected_v<noa::traits::has_closure, AccuratePlusReal>);

    // Same but for complex numbers.
    struct AccuratePlusComplex {
        c64* global_error{};
        c64 local_error{0};

        c64 operator()(c64 sum, c64 value) noexcept {
            auto sum_value = sum + value;
            for (i64 i = 0; i < 2; ++i) {
                local_error[i] += noa::math::abs(sum[i]) >= noa::math::abs(value[i]) ?
                                  (sum[i] - sum_value[i]) + value[i] :
                                  (value[i] - sum_value[i]) + sum[i];
            }
            return sum_value;
        }

        constexpr void closure(i64) const noexcept { // must be in thread-safe scope
            NOA_ASSERT(global_error != nullptr);
            *global_error += local_error;
        }
    };
    static_assert(noa::traits::is_detected_v<noa::traits::has_closure, AccuratePlusComplex>);

    // This is used as preprocessing operator to compute the variance.
    // The actual reduction operator is just noa::plus_t{}.
    template<typename Value>
    struct AccurateVariance {
        using output_type = noa::traits::value_type_t<Value>;
        Value mean{};

        template<typename Input>
        NOA_FHD constexpr output_type operator()(Input value) const noexcept {
            if constexpr (noa::traits::is_complex_v<Input>) {
                const auto tmp = static_cast<c64>(value);
                const auto distance = noa::math::abs(tmp - mean);
                return distance * distance;
            } else {
                const auto tmp = static_cast<f64>(value);
                const auto distance = tmp - mean;
                return distance * distance;
            }
        }
    };
}
