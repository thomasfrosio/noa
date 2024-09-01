#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/Ewise.hpp"
#include "noa/unified/Ewise.hpp"

namespace noa {
    /// Casts an array.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the cast values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    template<nt::readable_varray_decay Input, nt::writable_varray_decay Output>
    requires nt::varray_decay_with_compatible_or_spectrum_types<Input, Output>
    void cast(Input&& input, Output&& output, bool clamp = false) {
        check(not ni::are_overlapped(input, output), "The input and output arrays should not overlap");
        ewise(std::forward<Input>(input), std::forward<Output>(output), Cast{clamp});
    }
}
#endif
