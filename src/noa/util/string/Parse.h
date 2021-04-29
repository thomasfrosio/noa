/**
 * @file string/Parse.h
 * @brief Parsing string related functions.
 * @author Thomas - ffyr2w
 * @date 10 Jan 2021
 */
#pragma once

#include <array>
#include <cstddef>
#include <utility>  // std::move
#include <vector>
#include <string>
#include <string_view>
#include <type_traits>

#include "noa/Errno.h"
#include "noa/util/FloatX.h"
#include "noa/util/IntX.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Convert.h"

#define IS_CONVERTIBLE(T) (Noa::Traits::is_string_v<T> || Noa::Traits::is_scalar_v<T> || Noa::Traits::is_bool_v<T>)

namespace Noa::String::Details {
    template<typename T>
    NOA_IH auto convert(std::string&& string) {
        if constexpr (Noa::Traits::is_string_v<T>) {
            return std::move(string);
        } else if constexpr (Noa::Traits::is_float_v<T>) {
            return toFloat<T>(string);
        } else if constexpr (Noa::Traits::is_int_v<T>) {
            return toInt<T>(string);
        } else if constexpr (Noa::Traits::is_bool_v<T>) {
            return toBool(string);
        } else {
            static_assert(Noa::Traits::always_false_v<T>);
        }
    }
}

namespace Noa::String {
    /**
     * Parses (and formats) @a str.
     * @details Parses a string using commas as delimiters. Parsed strings are trimmed and empty strings are kept.
     *          If @a T is an integer, floating point or boolean, the strings are converted using toInt(), toFloat()
     *          or toBool(), respectively. If @a T is a string, no conversion is performed.
     *
     * @tparam T        Sets the type of formatting that should be used.
     * @tparam S        @c std::string(_view) by rvalue or lvalue.
     * @param[in] str   String to parse. Read only.
     * @return          Output vector.
     * @throw           Can throw if the conversion failed. See the corresponding @c String::to*() function.
     *
     * @example
     * @code
     * std::vector<std::string> vec1 = parse<std::string>(" 1, 2,  ,  4 5 ", vec1); // {"1", "2", "", "4 5"}
     * std::vector<float> vec2 = parse<float>(" 1, 2,  ,  4 5 ", vec2); // throws Noa::Exception
     * std::vector<float> vec3 = parse<float>(" 1, 2, 3 ,  4", vec3); // {1.f, 2.f, 3.f, 4.f}
     * @endcode
     */
    template<typename T>
    NOA_HOST std::vector<T> parse(std::string_view str) {
        static_assert(IS_CONVERTIBLE(T) && !std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        std::vector<T> out;

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                out.emplace_back(Details::convert<T>({str.data() + idx_start, idx_end - idx_start}));
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
        out.emplace_back(Details::convert<T>({str.data() + idx_start, idx_end - idx_start}));
        return out;
    }

    /**
     * Parses (and formats) @a str.
     * This is identical to the overload above, but given a known number of fields to parse.
     *
     * @tparam T        Sets the type of formatting that should be used.
     * @tparam N        Number of fields to expect.
     * @tparam S        @c std::string(_view) by rvalue or lvalue.
     * @param[in] str   String to parse. Read only.
     * @return          Output array.
     * @throw           Can throw if the conversion failed or if the number of fields is not equal to @a N.
     *                  See the corresponding @c String::to*() function.
     *
     * @example
     * @code
     * std::array<float> vec1 = parse<float, 5>(" 1, 2,  3,  4, 5 ", vec1); // {"1", "2", "3", "4", "5"}
     * std::array<float> vec2 = parse<float, 2>(" 1, 2, 3 ", vec2); // throws Noa::Exception
     * std::array<bool> vec3 = parse<bool, 2>(" 1 ", vec3); // throws Noa::Exception
     * @endcode
     */
    template<typename T, uint N>
    NOA_HOST std::array<T, N> parse(std::string_view string) {
        static_assert(IS_CONVERTIBLE(T) && !std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0}, idx{0};
        bool capture{false};
        std::array<T, N> out;

        for (size_t i{0}; i < string.size(); ++i) {
            if (string[i] == ',') {
                if (idx == N)
                    break;
                out[idx] = Details::convert<T>({string.data() + idx_start, idx_end - idx_start});
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
            NOA_THROW("The number of parsed value(s) ({}) does not match the number of "
                      "expected value(s) ({}). Input string: \"{}\"", idx + 1, N, string);
        out[idx] = Details::convert<T>({string.data() + idx_start, idx_end - idx_start});
        return out;
    }

    /**
     * Parses (and formats) @a str.
     * @details This is similar to the parsing functions above, except that if one string in @a str is empty or
     *          contains only whitespaces, it falls back to the corresponding field in @a str_defaults.
     *
     * @tparam T                Type contained by @c vec. Sets the type of formatting that should be used.
     * @param[in] str           String to parse. Read only.
     * @param[in] str_defaults  String containing the default value(s). Read only.
     * @return                  Output vector. The parsed values are inserted at the end of the vector.
     * @throw                   Can throw if the conversion failed or if the numbers of fields in @a str and
     *                          @a str_defaults do not match. See the corresponding @c String::to*() function.
     */
    template<typename T>
    std::vector<T> parse(std::string_view str, std::string_view str_defaults) {
        static_assert(IS_CONVERTIBLE(T) && !std::is_reference_v<T>);
        auto v1 = parse<std::string>(str);
        auto v2 = parse<std::string>(str_defaults);

        size_t size = v1.size();
        if (size != v2.size())
            NOA_THROW("The input string \"{}\" and default string \"{}\" do not match", str, str_defaults);

        std::vector<T> out;
        for (size_t i{0}; i < size; ++i)
            out.emplace_back(Details::convert<T>(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i])));
        return out;
    }

    /**
     * Parses (and formats) @a str, allowing default values.
     * @details This is similar to the parsing functions above, except that if one string in @a str is empty or
     *          contains only whitespaces, it falls back to the corresponding field in @a str_defaults.
     *
     * @tparam T                Type contained by @c vec. Sets the type of formatting that should be used.
     * @param[in] str           String to parse. Read only.
     * @param[in] str_defaults  String containing the default value(s). Read only.
     * @return                  Output vector. The parsed values are inserted at the end of the vector.
     * @throw                   Can throw if the conversion failed or if the numbers of fields in @a str and
     *                          @a str_defaults do not match. See the corresponding @c String::to*() function.
     */
    template<typename T, uint N>
    std::array<T, N> parse(std::string_view str, std::string_view str_defaults) {
        static_assert(IS_CONVERTIBLE(T) && !std::is_reference_v<T>);
        auto v1 = parse<std::string>(str);
        auto v2 = parse<std::string>(str_defaults);

        if (N != v1.size() || N != v2.size())
            NOA_THROW("The input string \"{}\" and/or default string \"{}\" do not match the expected "
                      "number of field(s) ({})", str, str_defaults, N);

        std::array<T, N> out;
        for (size_t i{0}; i < N; ++i)
            out[i] = Details::convert<T>(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i]));
        return out;
    }
}

#undef IS_CONVERTIBLE
