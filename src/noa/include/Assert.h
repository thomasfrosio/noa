/**
 * @file Assert.h
 * @brief Various predefined assertions.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */

#pragma once

#include "Core.h"
#include "Traits.h"

namespace Noa::Assert {
    /**
     * @fn range
     * @short               Check that scalars are within a given range, i.e min <= x <= max.
     *
     * @tparam [in] T       Type of a_value. Can be a scalar (is_arith_v) or a sequence
     *                      (is_sequence_of_arith_v). In this case, each element will be asserted.
     * @tparam [in] U       Type of a_min and a_max. Can be a scalar (is_arith_v) or a sequence
     *                      (is_sequence_of_arith_v). In this case, the ith element of a_value
     *                      will be compared with the ith element of a_min and a_max.
     * @param [in] a_value  One or multiple value(s) to assert.
     * @param [in] a_min    One or multiple value(s) to use as min.
     * @param [in] a_max    One or multiple value(s) to use as max.
     *
     * @example
     * @code
     * float a{3.45f};
     * std::array<int, 2> b{0.5, 0.7};
     * std::vector<int> min{0, 0};
     * std::vector<int> max{0.5, 2};
     *
     * range(a, 0f, 10f) // OK
     * range(b, 0, 0.5) // FAILED
     * range(b, min, max) // OK
     * @endcode
     */
    template<typename T, typename U>
    static void range(T&& a_value, U&& a_min, U&& a_max) {
        static_assert((Traits::is_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) ||
                      (Traits::is_sequence_of_arith_v<T> && Traits::is_sequence_of_arith_v<U>));

        if constexpr(Traits::is_arith_v<T> && Traits::is_arith_v<U>) {
            if (a_min > a_value || a_max < a_value) {
                NOA_CORE_ERROR("Assert::range: failed assertion; {} < {} < {} is not true",
                               a_min, a_value, a_max);
            }
        } else if constexpr(Traits::is_sequence_of_arith_v<T> && Traits::is_arith_v<U>) {
            for (auto& value : a_value) {
                if (a_min > value || a_max < value) {
                    NOA_CORE_ERROR("Assert::range: failed assertion; {} < {} < {} is not true",
                                   a_min, a_value, a_max);
                }
            }
        } else {
            if (a_value.size() != a_min.size() != a_max.size()) {
                NOA_CORE_ERROR("Assert::range: comparing sequences with different sizes, "
                               "got {} values for {} min and {} max",
                               a_value.size(), a_min.size(), a_max.size());
            }
            for (unsigned int i{0}; i < a_value.size(); ++i) {
                if (a_min[i] > a_value[i] || a_max[i] < a_value[i]) {
                    NOA_CORE_ERROR("Assert::range: failed assertion; {} < {} < {} is not true",
                                   a_min[i], a_value[i], a_max[i]);
                }
            }
        }
    }
}
