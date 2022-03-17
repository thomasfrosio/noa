/// \file noa/common/string/Split.h
/// \brief Split strings.
/// \author Thomas - ffyr2w
/// \date 10 Jan 2021

#pragma once

#include <array>
#include <cstddef>
#include <utility>  // std::move
#include <vector>
#include <string>
#include <string_view>
#include <type_traits>

#include "noa/common/Exception.h"
#include "noa/common/string/Parse.h"
#include "noa/common/traits/BaseTypes.h"

namespace noa::string::details {
    template<typename T>
    constexpr bool can_be_parsed_v = std::bool_constant<noa::traits::is_string_v<T> ||
                                                        noa::traits::is_scalar_v<T> ||
                                                        noa::traits::is_bool_v<T>>::value;

    template<typename T>
    NOA_IH auto parse(std::string&& string) { // std::string to make sure it is null terminated
        if constexpr (noa::traits::is_string_v<T>) {
            return std::move(string);
        } else if constexpr (noa::traits::is_float_v<T>) {
            return toFloat<T>(string);
        } else if constexpr (noa::traits::is_bool_v<T>) {
            return toBool(string);
        } else if constexpr (noa::traits::is_int_v<T>) {
            return toInt<T>(string);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }
    }
}

namespace noa::string {
    /// Splits (and parses) \p str.
    /// \details Split a string using commas as delimiters. Split strings are trimmed and empty strings are kept.
    ///          If \p T is an integer, floating-point or boolean, the strings are parsed using toInt(), toFloat()
    ///          or toBool(), respectively. If \p T is a string, the strings are not parsed.
    ///
    /// \tparam T        Parsing type.
    /// \param[in] str   String to parse. Read only.
    /// \return          Output vector.
    /// \throw Exception If the parsing failed.
    ///
    /// \example
    /// \code
    /// std::vector<std::string> vec1 = split<std::string>(" 1, 2,  ,  4 5 "); // {"1", "2", "", "4 5"}
    /// std::vector<float> vec2 = split<float>(" 1, 2,  ,  4 5 "); // throws noa::Exception
    /// std::vector<float> vec3 = split<float>(" 1, 2, 3 ,  4"); // {1.f, 2.f, 3.f, 4.f}
    /// \endcode
    template<typename T>
    NOA_HOST std::vector<T> split(std::string_view str) {
        static_assert(details::can_be_parsed_v<T> && !std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        std::vector<T> out;

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                out.emplace_back(details::parse<T>({str.data() + idx_start, idx_end - idx_start}));
                idx_start = 0;
                idx_end = 0;
                capture = false;
            } else if (!std::isspace(str[i])) {
                if (capture)
                    idx_end = i + 1;
                else {
                    idx_start = i;
                    idx_end = i + 1;
                    capture = true;
                }
            }
        }
        out.emplace_back(details::parse<T>({str.data() + idx_start, idx_end - idx_start}));
        return out;
    }

    /// Splits (and parses) \p str.
    /// This is identical to the overload above, but given a known number of fields.
    ///
    /// \tparam T        Parsing type.
    /// \tparam N        Number of fields to expect.
    /// \param[in] str   String to parse. Read only.
    /// \return          Output array.
    /// \throw Exception If the conversion failed or if the number of fields is not equal to \p N.
    ///
    /// \example
    /// \code
    /// std::array<float, 5> vec1 = parse<float, 5>(" 1, 2,  3,  4, 5 "); // {"1", "2", "3", "4", "5"}
    /// std::array<float, 2> vec2 = parse<float, 2>(" 1, 2, 3 "); // throws noa::Exception
    /// std::array<bool, 2> vec3 = parse<bool, 2>(" 1 "); // throws noa::Exception
    /// \endcode
    template<typename T, uint N>
    NOA_HOST std::array<T, N> split(std::string_view string) {
        static_assert(details::can_be_parsed_v<T> && !std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0}, idx{0};
        bool capture{false};
        std::array<T, N> out;

        for (size_t i{0}; i < string.size(); ++i) {
            if (string[i] == ',') {
                if (idx == N)
                    break;
                out[idx] = details::parse<T>({string.data() + idx_start, idx_end - idx_start});
                ++idx;
                idx_start = 0;
                idx_end = 0;
                capture = false;
            } else if (!std::isspace(string[i])) {
                if (capture)
                    idx_end = i + 1;
                else {
                    idx_start = i;
                    idx_end = i + 1;
                    capture = true;
                }
            }
        }
        if (idx + 1 != N)
            NOA_THROW("The number of value(s) ({}) does not match the number of "
                      "expected value(s) ({}). Input string: \"{}\"", idx + 1, N, string);
        out[idx] = details::parse<T>({string.data() + idx_start, idx_end - idx_start});
        return out;
    }

    /// Splits (and parses) \p str, allowing default values.
    /// \details This is similar to the parsing functions above, except that if one field in \p str is empty or
    ///          contains only whitespaces, it falls back to the corresponding field in \p str_defaults.
    ///
    /// \tparam T               Parsing type.
    /// \param[in] str          String to parse. Read only.
    /// \param[in] str_defaults String containing the default field(s). Read only.
    /// \return                 Output vector. The parsed values are inserted at the end of the vector.
    /// \throw Exception        If the conversion failed or if the number of fields in \p str and
    ///                         \p str_defaults do not match.
    template<typename T>
    std::vector<T> split(std::string_view str, std::string_view str_defaults) {
        static_assert(details::can_be_parsed_v<T> && !std::is_reference_v<T>);
        auto v1 = split<std::string>(str);
        auto v2 = split<std::string>(str_defaults);

        size_t size = v1.size();
        if (size != v2.size())
            NOA_THROW("The input string \"{}\" and default string \"{}\" do not match", str, str_defaults);

        std::vector<T> out;
        for (size_t i{0}; i < size; ++i)
            out.emplace_back(details::parse<T>(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i])));
        return out;
    }

    /// Splits (and parses) \p str, allowing default values.
    /// \details This is similar to the parsing functions above, except that if one field in \p str is empty or
    ///          contains only whitespaces, it falls back to the corresponding field in \p str_defaults.
    ///
    /// \tparam T               Parsing type.
    /// \tparam N               Number of fields to expect.
    /// \param[in] str          String to parse. Read only.
    /// \param[in] str_defaults String containing the default field(s). Read only.
    /// \return                 Output vector. The parsed values are inserted at the end of the vector.
    /// \throw Exception        If the conversion failed or if the number of fields in \p str and
    ///                         \p str_defaults do not match.
    template<typename T, uint N>
    std::array<T, N> split(std::string_view str, std::string_view str_defaults) {
        static_assert(details::can_be_parsed_v<T> && !std::is_reference_v<T>);
        auto v1 = split<std::string>(str);
        auto v2 = split<std::string>(str_defaults);

        if (N != v1.size() || N != v2.size())
            NOA_THROW("The input string \"{}\" and/or default string \"{}\" do not match the expected "
                      "number of field(s) ({})", str, str_defaults, N);

        std::array<T, N> out;
        for (size_t i{0}; i < N; ++i)
            out[i] = details::parse<T>(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i]));
        return out;
    }
}
