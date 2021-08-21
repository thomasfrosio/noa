/// \file noa/cpu/memory/Copy.h
/// \brief Copy memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::cpu::memory {
    /// Copies all elements in the range [\p first, \p last) starting from \p first and proceeding to \p last - 1.
    /// The behavior is undefined if \p dst_first is within the range [\p first, \p last).
    /// \tparam T                Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] first         On the \b host. The beginning of range to copy.
    /// \param[in] last          On the \b host. The end of range to copy.
    /// \param[out] dst_first    On the \b host. The beginning of the destination range.
    template<typename T>
    NOA_IH void copy(const T* first, const T* last, T* dst_first) {
        NOA_PROFILE_FUNCTION();
        std::copy(first, last, dst_first);
    }

    /// Copies all elements in the range [\p src, \p src + \p elements) starting from \p src and proceeding to
    /// \p src + \p elements - 1. The behavior is undefined if \p dst is within the source range.
    /// \tparam T        Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] src   On the \b host. The beginning of the range to copy.
    /// \param[out] dst  On the \b host. The beginning of the destination range.
    /// \param elements  Number of \p T elements to copy.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements) {
        copy(src, src + elements, dst);
    }
}
