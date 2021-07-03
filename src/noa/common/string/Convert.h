/// \file noa/common/string/Convert.h
/// \brief Convert strings to other type.
/// \author Thomas - ffyr2w
/// \date 10 Jan 2021

#pragma once

#include <string_view>
#include "noa/common/Definitions.h"

namespace noa::string {
    /// Convert a string into an integer.
    /// \tparam T   Supported integers are: (u)int8_t, (u)short, (u)int, (u)long, (u)long long.
    /// \param str  String to convert to \a T.
    /// \return     Resulting integer.
    /// \throw      Throws a \a noa::Exception if a character couldn't be converted or if a string
    ///             translates to an number that is out of the \a T range
    /// \note       \c errno is reset to 0 before starting the conversion.
    template<typename T = int>
    NOA_HOST T toInt(const std::string& str);

    /// Convert a string into a floating point.
    /// \tparam T   Supported floating points are: float, double.
    /// \param str  String to convert to \a T.
    /// \return     Resulting floating point.
    /// \throw      Throws a \a noa::Exception if a character couldn't be converted or if a string
    ///             translates to an number that is out of the \a T range
    /// \note       \c errno is reset to 0 before starting the conversion.
    template<typename T = float>
    NOA_HOST T toFloat(const std::string& str);

    /// Convert a string into a bool.
    /// \param str  String to convert.
    /// \return     Resulting bool.
    /// \throw      Throws a \a noa::Exception if \a str could not be converted.
    NOA_HOST bool toBool(const std::string& str);
}
