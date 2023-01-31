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
#include "noa/common/traits/Numerics.h"

namespace noa::string {
    /// Splits (and parses) \p str.
    /// \details Splits a string using \p separator as delimiter. Then, trim each token (empty strings are kept).
    ///          If \p T is an integer, floating-point or boolean, the strings are parsed.
    ///          If \p T is a string, the strings are not parsed.
    /// \example
    /// \code
    /// std::vector<std::string> vec1 = split<std::string>(" 1, 2,  ,  4 5 "); // {"1", "2", "", "4 5"}
    /// std::vector<float> vec2 = split<float>(" 1, 2,  ,  4 5 "); // throws noa::Exception
    /// std::vector<float> vec3 = split<float>(" 1, 2, 3 ,  4"); // {1.f, 2.f, 3.f, 4.f}
    /// \endcode
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T> && !std::is_reference_v<T>>>
    std::vector<T> split(std::string_view str, char separator = ',') {
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        std::vector<T> out;

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == separator) {
                out.emplace_back(parse<T>({str.data() + idx_start, idx_end - idx_start}));
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
        out.emplace_back(parse<T>({str.data() + idx_start, idx_end - idx_start}));
        return out;
    }

    /// Splits (and parses) \p str.
    /// \details This is identical to the overload returning a vector, except that this overload
    ///          expects a compile time known number of fields.
    /// \example
    /// \code
    /// std::array<float, 5> vec1 = parse<float, 5>(" 1, 2,  3,  4, 5 "); // {"1", "2", "3", "4", "5"}
    /// std::array<float, 2> vec2 = parse<float, 2>(" 1, 2, 3 "); // throws noa::Exception
    /// std::array<bool, 2> vec3 = parse<bool, 2>(" 1 "); // throws noa::Exception
    /// \endcode
    template<typename T, uint N, typename = std::enable_if_t<details::can_be_parsed_v<T> && !std::is_reference_v<T>>>
    std::array<T, N> split(std::string_view string, char separator = ',') {
        size_t idx_start{0}, idx_end{0}, idx{0};
        bool capture{false};
        std::array<T, N> out;

        for (size_t i{0}; i < string.size(); ++i) {
            if (string[i] == separator) {
                if (idx == N)
                    break;
                out[idx] = parse<T>({string.data() + idx_start, idx_end - idx_start});
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
        out[idx] = parse<T>({string.data() + idx_start, idx_end - idx_start});
        return out;
    }

    /// Splits (and parses) \p str, allowing default values.
    /// \details This is similar to the parsing functions above, except that if one field in \p str is empty or
    ///          contains only whitespaces, it falls back to the corresponding field in \p fallback.
    template<typename T, typename = std::enable_if_t<details::can_be_parsed_v<T> && !std::is_reference_v<T>>>
    std::vector<T> split(std::string_view str, std::string_view fallback, char separator = ',') {
        auto v1 = split<std::string>(str, separator);
        auto v2 = split<std::string>(fallback, separator);

        size_t size = v1.size();
        if (size != v2.size())
            NOA_THROW("The input string \"{}\" and fallback string \"{}\" do not match", str, fallback);

        std::vector<T> out;
        for (size_t i{0}; i < size; ++i)
            out.emplace_back(parse<T>(v1[i].empty() ? v2[i] : v1[i]));
        return out;
    }

    /// Splits (and parses) \p str, allowing default values.
    /// \details This is similar to the parsing functions above, except that if one field in \p str is empty or
    ///          contains only whitespaces, it falls back to the corresponding field in \p fallback.
    template<typename T, uint N, typename = std::enable_if_t<details::can_be_parsed_v<T> && !std::is_reference_v<T>>>
    std::array<T, N> split(std::string_view str, std::string_view fallback, char separator = ',') {
        auto v1 = split<std::string>(str, separator);
        auto v2 = split<std::string>(fallback, separator);

        if (N != v1.size() || N != v2.size())
            NOA_THROW("The input string \"{}\" and/or fallback string \"{}\" do not match the expected "
                      "number of field(s) ({})", str, fallback, N);

        std::array<T, N> out;
        for (size_t i{0}; i < N; ++i)
            out[i] = parse<T>(v1[i].empty() ? v2[i] : v1[i]);
        return out;
    }
}
