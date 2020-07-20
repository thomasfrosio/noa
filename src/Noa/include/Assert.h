/**
 * @file Assert.h
 * @brief Various predefined assertions.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */

#pragma once

#include "Noa.h"

namespace Noa {
    /**
     *
     * @tparam t_type
     * @tparam t_number
     * @tparam t_type_range
     * @param a_to_check
     * @param a_range_min
     * @param a_range_max
     */
    template<typename t_type, int t_number, typename t_type_range>
    static void range(const t_type& a_to_check,
                      const t_type_range& a_range_min,
                      const t_type_range& a_range_max) {
        if constexpr(std::is_same_v<t_type, t_type_range>) {
            if (a_range_min > a_to_check || a_range_max < a_to_check)
                std::cerr << "Error" << '\n';
        } else if constexpr(std::is_same_v<t_type, std::vector<t_type_range>> ||
                            std::is_same_v<t_type, std::array<t_type_range, t_number>>) {
            for (auto& value : a_to_check) {
                if (a_range_min > value || a_range_max < value)
                    std::cerr << "Error" << '\n';
            }
        } else {
            std::cerr << "Error" << std::endl;
        }
    }
}
