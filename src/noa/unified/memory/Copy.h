#pragma once

#include "noa/unified/Array.h"

namespace noa::memory {
    /// (Deep-)Copies of arrays.
    /// \details Contiguous arrays have no copy restrictions and can be copied to any device. This is also true for
    ///          pitched arrays. However, other non-contiguous memory layouts can only be copied if the source and
    ///          destination are both on the same device.
    /// \param[in] input    Source.
    /// \param[out] output  Destination. It should not overlap \p input.
    template<typename T>
    void copy(const Array<T>& input, const Array<T>& output) {
        input.to(output);
    }
}
