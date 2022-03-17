/// \file noa/common/string/Parse.h
/// \brief Parse strings.
/// \author Thomas - ffyr2w
/// \date 10 Jan 2021

#pragma once

#include <string_view>
#include "noa/common/Definitions.h"

namespace noa::string {
    /// Parses a null-terminated string into an integer.
    /// \tparam T Supported integers are: (u)int8_t, (u)short, (u)int, (u)long, (u)long long.
    template<typename T = int>
    NOA_HOST T toInt(const std::string& str);

    /// Parses a null-terminated string into a floating point.
    /// \tparam T Supported floating points are: float, double.
    template<typename T = float>
    NOA_HOST T toFloat(const std::string& str);

    /// Parses a null-terminated string into a bool.
    NOA_HOST bool toBool(const std::string& str);
}
