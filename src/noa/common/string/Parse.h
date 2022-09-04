/// \file noa/common/string/Parse.h
/// \brief Parse strings.
/// \author Thomas - ffyr2w
/// \date 10 Jan 2021

#pragma once

#include <string>

namespace noa::string::details {
    template<typename T>
    constexpr bool can_be_parsed_v = std::bool_constant<traits::is_string_v<T> ||
                                                        traits::is_scalar_v<T> ||
                                                        traits::is_bool_v<T>>::value;
}

namespace noa::string {
    /// Parses a null-terminated string into an integer.
    /// \tparam T Supported integers are: (u)int8_t, (u)short, (u)int, (u)long, (u)long long.
    template<typename T = int>
    T toInt(const std::string& str);

    /// Parses a null-terminated string into a floating point.
    /// \tparam T Supported floating points are: float, double.
    template<typename T = float>
    T toFloat(const std::string& str);

    /// Parses a null-terminated string into a bool.
    bool toBool(const std::string& str);

    /// Parses a null-terminated string into a \p T.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline T parse(const std::string& string) {
        if constexpr (traits::is_string_v<T>) {
            return string;
        } else if constexpr (traits::is_float_v<T>) {
            return toFloat<T>(string);
        } else if constexpr (traits::is_bool_v<T>) {
            return toBool(string);
        } else if constexpr (traits::is_int_v<T>) {
            return toInt<T>(string);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T>>>
    inline std::vector<T> parse(const std::vector<std::string>& vector) {
        if constexpr (traits::is_string_v<T>) {
            return vector;
        } else {
            std::vector<T> output;
            for (const auto& string: vector) {
                if constexpr (traits::is_float_v<T>) {
                    output.emplace_back(toFloat<T>(string));
                } else if constexpr (traits::is_bool_v<T>) {
                    output.emplace_back(toBool(string));
                } else if constexpr (traits::is_int_v<T>) {
                    output.emplace_back(toInt<T>(string));
                } else {
                    static_assert(traits::always_false_v<T>);
                }
            }
            return output;
        }
    }
}
