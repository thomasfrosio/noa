#pragma once

#include "noa/Array.h"

namespace noa::memory {
    /// (Deep-)Copies arrays.
    /// \details Contiguous regions of memory have no copy restrictions and can be copied to any device. This is
    ///          also true for pitched layouts and colum or row vectors. However, other non-contiguous memory
    ///          layouts can only be copied if the source and destination are both on the same GPU or on the CPU.
    /// \param[in] input    Source.
    /// \param[out] output  Destination. It should not overlap with \p input.
    template<typename T>
    void copy(const Array<T>& input, const Array<T>& output) {
        input.to(output);
    }

    /// (Deep-)Copies arrays.
    template<typename Value, typename Vector>
    void copy(const Array<Value>& input, const Array<Value>& output,
              const Vector& batch_indexes) {
        // If the batches to copy into output are next to each other,
        // this becomes a slice operation.

        // If the arrays are on the same device, use extract for
        // better performance (only one ewise call).

        // Worst case, copy batches across devices.

    }
}
