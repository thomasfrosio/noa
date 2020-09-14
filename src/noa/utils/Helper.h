/**
 * @file Helper.h
 * @brief Some helper functions.
 * @author Thomas - ffyr2w
 * @date 24 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/Assert.h"


namespace Noa::Helper {
    /**
     * @short                       Convenient function to assign a new value to a sequence.
     *                              std::vector - use emplace_back()
     *                              std::array - use operator[]
     *
     * @tparam S                    Sequence (std::vector|std::array).
     * @tparam T                    Type of `value`.
     * @param[in,out] sequence      Sequence to assign something to.
     * @param[in] value             Value to assign to a sequence.
     * @param[i] i                  Index at which the assign should be. Only used for arrays.
     */
    template<typename S, typename T>
    constexpr inline auto sequenceAssign(S&& sequence, T&& value, size_t idx = 0) {
        static_assert(Noa::Traits::is_sequence_v<S>);
        if constexpr(Noa::Traits::is_vector_v<S>)
            return sequence.emplace_back(std::forward<T>(value));
        else
            return sequence[idx] = std::forward<T>(value);
    }
}
