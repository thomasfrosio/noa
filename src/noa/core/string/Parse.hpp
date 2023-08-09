#pragma once

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

#include "noa/core/Exception.hpp"
#include "noa/core/string/Format.hpp"
#include "noa/core/traits/Numerics.hpp"

namespace noa::string::details {
    template<typename T>
    constexpr bool can_be_parsed_v =
            std::bool_constant<nt::is_string_v<T> ||
                               nt::is_numeric_v<T>>::value;

    template<typename T = int, typename = std::enable_if_t<nt::is_int_v<T>>>
    T to_int(const std::string& str);

    template<typename T = int, typename = std::enable_if_t<nt::is_int_v<T>>>
    T to_int(const std::string& str, int& error) noexcept;

    template<typename T = float, typename = std::enable_if_t<nt::is_real_v<T>>>
    T to_real(const std::string& str);

    template<typename T = float, typename = std::enable_if_t<nt::is_real_v<T>>>
    T to_real(const std::string& str, int& error) noexcept;

    inline bool to_bool(const std::string& str);

    inline bool to_bool(const std::string& str, int& error) noexcept;
}

namespace noa::string {
    // Parses a null-terminated string into a T. Throws if the parsing fails.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    T parse(const std::string& string);

    // Parses a null-terminated string into a T.
    // error is set to non-zero if the parsing fails, otherwise it is set to 0.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    T parse(const std::string& string, int& error) noexcept;

    // Parses a vector of null-terminated strings into a vector of T. Throws if the parsing fails.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    std::vector<T> parse(const std::vector<std::string>& vector);

    // Parses a vector of null-terminated strings into a vector of T.
    // error is set to non-zero if the parsing fails, otherwise it is set to 0.
    // If the parsing fails for an element, the function stops and returns the output
    // vector with the elements that were successfully parsed.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    std::vector<T> parse(const std::vector<std::string>& vector, int& error) noexcept;

    // Returns an error message given an error value.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    std::string parse_error_message(const std::string& str, int error);
}

#define NOA_STRING_PARSE_
#include "noa/core/string/Parse.inl"
#undef NOA_STRING_PARSE_
