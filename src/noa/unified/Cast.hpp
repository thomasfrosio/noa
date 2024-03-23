#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Ewise.hpp"
#include "noa/unified/Ewise.hpp"

namespace noa {
    /// Casts an array.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the casted values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    template<typename Input, typename Output>
    requires nt::are_varray_v<Input, Output>
    void cast(const Input& input, const Output& output, bool clamp = false) {
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");
        ewise(input, output, Cast{clamp});
    }
}
#endif
