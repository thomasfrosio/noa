/// \file noa/cpu/memory/Set.h
/// \brief Set to value.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

namespace noa::memory {
    /// Initializes or sets device memory to a value.
    /// \tparam T           Most types are supported. See https://en.cppreference.com/w/cpp/algorithm/copy
    /// \param[out] first   On the \b host. The beginning of range to set.
    /// \param[out] last    On the \b host. The end of range to set.
    /// \param value        The value to assign.
    template<typename T>
    NOA_IH void set(T* first, T* last, T value) {
        if constexpr (noa::traits::is_scalar_v<T> || noa::traits::is_complex_v<T> ||
                      noa::traits::is_intX_v<T> || noa::traits::is_floatX_v<T>) {
            if (value == static_cast<T>(0))
                // calling memset, https://godbolt.org/z/1zEzTnoTK
                // the cast is not necessary for basic types, but for Complex<>, IntX<> or FloatX<>, it could help...
                std::fill(reinterpret_cast<char*>(first), reinterpret_cast<char*>(last), 0);
            else
                std::fill(first, last, value);
        } else {
            std::fill(first, last, value);
        }
    }

    /// Initializes or sets device memory to a value.
    /// \tparam T           Most types are supported.
    /// \param[out] first   On the \b host. The beginning of range to set.
    /// \param elements     Number of elements to set.
    /// \param value        The value to assign.
    template<typename T>
    NOA_IH void set(T* src, size_t elements, T value) {
        set(src, src + elements, value);
    }
}
