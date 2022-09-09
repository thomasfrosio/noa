/// \file noa/common/string/Parse.h
/// \brief Parse strings.
/// \author Thomas - ffyr2w
/// \date 10 Jan 2021
#pragma once

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

#include "noa/common/Exception.h"
#include "noa/common/string/Format.h"
#include "noa/common/string/Parse.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa::string::details {
    template<typename T>
    constexpr bool can_be_parsed_v = std::bool_constant<traits::is_string_v<T> ||
                                                        traits::is_scalar_v<T> ||
                                                        traits::is_bool_v<T>>::value;
}

namespace noa::string {
    /// Parses a null-terminated string into a \p T. Throws if the parsing fails.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline T parse(const std::string& string);

    /// Parses a null-terminated string into a \p T.
    /// \p error is set to non-zero if the parsing fails, otherwise it is set to 0.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline T parse(const std::string& string, int& error) noexcept;

    /// Parses a vector of null-terminated strings in a vector of \p T. Throws if the parsing fails.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline std::vector<T> parse(const std::vector<std::string>& vector);

    /// Parses a vector of null-terminated strings in a vector of \p T.
    /// \p error is set to non-zero if the parsing fails, otherwise it is set to 0.
    /// If the parsing fails for an element, the function stops and returns the output
    /// vector with the elements that were successfully parsed.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline std::vector<T> parse(const std::vector<std::string>& vector, int& error) noexcept;

    /// Returns an error message given an \p error value.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline std::string parseErrorMessage(const std::string& str, int error);
}

namespace noa::string {
    template<typename T = int, typename = std::enable_if_t<traits::is_int_v<T>>>
    T toInt(const std::string& str);

    template<typename T = int, typename = std::enable_if_t<traits::is_int_v<T>>>
    T toInt(const std::string& str, int& ec) noexcept;

    template<typename T = float, typename = std::enable_if_t<traits::is_float_v<T>>>
    T toFloat(const std::string& str);

    template<typename T = float, typename = std::enable_if_t<traits::is_float_v<T>>>
    T toFloat(const std::string& str, int& ec) noexcept;

    inline bool toBool(const std::string& str);

    inline bool toBool(const std::string& str, int& error) noexcept;
}

#define NOA_STRING_PARSE_
#include "noa/common/string/Parse.inl"
#undef NOA_STRING_PARSE_
