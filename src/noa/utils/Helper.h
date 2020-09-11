/**
 * @file Helper.h
 * @brief Some helper functions.
 * @author Thomas - ffyr2w
 * @date 24 Jul 2020
 */
#pragma once

#include "noa/noa.h"
#include "noa/utils/Assert.h"


namespace Noa::Helper {
    /**
     * @fn noa::Helper::sequenceAssign
     * @short                       Convenient function to assign a new value to a sequence.
     *                              std::vector - use emplace_back()
     *                              std::array - use operator[]
     *
     * @tparam [in,out] Sequence    Sequence (std::vector|std::array).
     * @tparam [in] T               value_type of Sequence.
     * @param [in,out] a_sequence   Sequence to assign something to.
     * @param [in] a_value          Value to assign to a sequence.
     * @param [i] i                 Index at which the assign should be. Only used for arrays.
     */
    template<typename Sequence, typename T>
    constexpr inline auto sequenceAssign(Sequence&& a_sequence,
                                         T&& a_value,
                                         unsigned int i = 0) {
        static_assert(Noa::Traits::is_sequence_v<Sequence>);
        if constexpr(Noa::Traits::is_vector_v<Sequence>)
            return a_sequence.emplace_back(std::forward<T>(a_value));
        else
            return a_sequence[i] = std::forward<T>(a_value);
    }
}
