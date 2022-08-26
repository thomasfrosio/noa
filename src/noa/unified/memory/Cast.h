#pragma once

#include "noa/unified/Array.h"

namespace noa::memory {
    /// Returns evenly spaced values within a given interval.
    /// \tparam T           Any data type.
    /// \tparam U           Any data type. If \p T is complex, \p U should be complex as well.
    /// \param[in] input    Array to convert.
    /// \param[out] output  Array with the converted values.
    /// \param clamp        Whether the input values should be clamped to the output range before casting.
    template<typename T, typename U>
    void cast(const Array<T>& input, const Array<U>& output, bool clamp = false);
}

#define NOA_UNIFIED_CAST_
#include "noa/unified/memory/Cast.inl"
#undef NOA_UNIFIED_CAST_
