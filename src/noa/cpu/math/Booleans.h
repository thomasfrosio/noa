/// \file noa/cpu/math/Booleans.h
/// \brief Boolean "operators" for arrays.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <algorithm>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"

namespace noa::math {
    /// Computes whether \p input is less than \p threshold, element-wise, i.e. x < threshold.
    /// \tparam T           Any type with `bool operator<(T,T)` defined.
    /// \tparam U           Any type that can be casted from bool. Can be equal to \p T.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param threshold    Value to use as threshold.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements to compute.
    template<typename T, typename U>
    NOA_IH void isLess(const T* input, T threshold, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [threshold](const T& element) -> U { return static_cast<U>(element < threshold); });
    }

    /// Computes whether \p input is greater than \p threshold, element-wise, i.e. threshold < x.
    /// \tparam T           Any type with `bool operator<(T,T)` defined.
    /// \tparam U           Any type that can be casted from bool. Can be equal to \p T.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param threshold    Value to use as threshold.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements to compute.
    template<typename T, typename U>
    NOA_IH void isGreater(const T* input, T threshold, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [threshold](const T& element) -> U { return static_cast<U>(threshold < element); });
    }

    /// Computes whether \p input is within \p low and \p high, element-wise, i.e. x < high && low < x.
    /// \tparam T           Any type with `bool operator<(T,T)` defined.
    /// \tparam U           Any type that can be casted from bool. Can be equal to \p T.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param low          Low threshold.
    /// \param high         High threshold.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements to compute.
    template<typename T, typename U>
    NOA_IH void isWithin(const T* input, T low, T high, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [low, high](const T& element) -> U {
                           return static_cast<U>(element < high && low < element);
                       });
    }

    /// Computes the logical NOT, element-wise, i.e. !x.
    /// \tparam T           Any type with `bool operator!(T)` defined.
    /// \tparam U           Any type that can be casted from bool. Can be equal to \p T.
    /// \param[in] input    On the \b host. Contiguous input array.
    /// \param[out] output  On the \b host. Contiguous output array. Can be equal to \p input.
    /// \param elements     Number of elements to compute.
    template<typename T, typename U, typename = std::enable_if_t<std::is_integral_v<T>>>
    NOA_IH void logicNOT(const T* input, U* output, size_t elements) {
        NOA_PROFILE_FUNCTION();
        std::transform(input, input + elements, output,
                       [](const T& element) -> U { return static_cast<U>(!element); });
    }
}
