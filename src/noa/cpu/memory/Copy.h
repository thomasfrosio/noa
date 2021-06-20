/// \file noa/cpu/memory/Copy.h
/// \brief Copy memory regions.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/Profiler.h"

namespace noa::memory {
    /// Copies all elements in the range [\a first, \a last) starting from \a first and proceeding to \a last - 1.
    /// The behavior is undefined if \a dst_first is within the range [\a first, \a last).
    /// \tparam T                Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] first         The beginning of range to copy.
    /// \param[in] last          The end of range to copy.
    /// \param[out] dst_first    The beginning of the destination range.
    template<typename T>
    NOA_IH void copy(const T* first, const T* last, T* dst_first) {
        NOA_PROFILE_FUNCTION();
        std::copy(first, last, dst_first);
    }

    /// Copies all elements in the range [\a src, \a src + \a elements) starting from \a src and proceeding to
    /// \a src + \a elements - 1. The behavior is undefined if \a dst is within the source range.
    /// \tparam T        Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[in] src   The beginning of the range to copy.
    /// \param[out] dst  The beginning of the destination range.
    /// \param elements  Number of \a T elements to copy.
    template<typename T>
    NOA_IH void copy(const T* src, T* dst, size_t elements) {
        copy(src, src + elements, dst);
    }
}
