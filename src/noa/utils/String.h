/**
 * @file String.h
 * @brief String related functions.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/Traits.h"


/// Group of string related functions.
namespace Noa::String {

    /**
     * @brief               Left trim a string in-place.
     * @param[in,out] str   String to left trim (taken as lvalue)
     * @return              Left trimmed string.
     */
    inline std::string& leftTrim(std::string& str) {
        str.erase(str.begin(),
                  std::find_if(str.begin(),
                               str.end(),
                               [](int ch) { return !std::isspace(ch); }));
        return str;
    }


    /**
     * @brief           Left trim a string in-place.
     * @param[in] str   String to left trim (taken as rvalue)
     * @return          Left trimmed string.
     */
    [[nodiscard]] inline std::string leftTrim(std::string&& str) {
        leftTrim(str);
        return std::move(str);
    }


    /**
     * @brief           Left trim a string by copy.
     * @param[in] str   String to left trim (taken as value)
     * @return          Left trimmed string.
     */
    [[nodiscard]] inline std::string leftTrimCopy(std::string str) {
        leftTrim(str);
        return str;
    }


    /**
     * @brief               Right trim a string in-place.
     * @param[in,out] str   String to right trim (taken by lvalue)
     * @return              Right trimmed string.
     */
    inline std::string& rightTrim(std::string& str) {
        str.erase(std::find_if(str.rbegin(),
                               str.rend(),
                               [](int ch) { return !std::isspace(ch); }).base(),
                  str.end());
        return str;
    }


    /**
     * @brief           Right trim a string in-place.
     * @param[in] str   String to right trim (taken by rvalue)
     * @return          Right trimmed string.
     */
    [[nodiscard]] inline std::string rightTrim(std::string&& str) {
        rightTrim(str);
        return std::move(str);
    }


    /**
     * @brief           Right trim a string by copy.
     * @param[in] str   String to right trim (taken by value)
     * @return          Right trimmed string.
     */
    [[nodiscard]] inline std::string rightTrimCopy(std::string str) {
        rightTrim(str);
        return str;
    }


    /**
     * @brief               Trim (left and right) a string in-place.
     * @param[in,out] str   String to trim (taken by lvalue)
     * @return              Trimmed string.
     */
    inline std::string& trim(std::string& str) {
        return leftTrim(rightTrim(str));
    }


    /**
     * @brief           Trim (left and right) a string in-place.
     * @param[in] str   String to trim (taken by rvalue)
     * @return          Trimmed string.
     */
    [[nodiscard]] inline std::string trim(std::string&& str) {
        leftTrim(rightTrim(str));
        return std::move(str);
    }


    /**
     * @brief                   Trim (left and right) string(s) stored in vector.
     * @param[in,out] vec_str   Vector of string(s) to trim (taken by lvalue)
     * @return                  Trimmed string.
     */
    inline std::vector<std::string>& trim(std::vector<std::string>& vec_str) {
        for (auto& str : vec_str)
            trim(str);
        return vec_str;
    }


    /**
     * @brief           Trim (left and right) a string in-place.
     * @param[in] str   String to trim (taken by value)
     * @return          Trimmed string.
     */
    [[nodiscard]] inline std::string trimCopy(std::string str) {
        leftTrim(rightTrim(str));
        return str;
    }


    /**
     * @brief               Trim (left and right) string(s) stored in vector.
     * @param[in] vec_str   Vector of string(s) to trim (taken by value)
     * @return              Trimmed string.
     */
    [[nodiscard]] inline std::vector<std::string> trimCopy(std::vector<std::string> vec_str) {
        for (auto& str : vec_str)
            trim(str);
        return vec_str;
    }


    /**
     * @brief           Split a string(_view) using std::string(_view)::find().
     *
     * @tparam String   std::string(_view), taken by rvalue or lvalue.
     * @param[in] str   String to split.
     * @param[in] delim C-string to use as delimiter.
     * @param[out] vec  Output vector to insert back the output string(s) into.
     *                  The delimiter is not included in the output strings.
     *
     * @example
     * @code
     * std::string str = "  12, 12  , 12, ";
     * std::vector<std::string> vec;
     * split(str, ",", vec);
     * fmt::print(vec);  // {"  12", " 12  ", " 12", " "}
     * @endcode
     */
    template<typename String, typename = std::enable_if_t<::Noa::Traits::is_string_v<String>>>
    void split(String&& str, const char* delim, std::vector<std::string>& vec) {
        size_t inc = strlen(delim), previous = 0, current = str.find(delim, previous);
        while (current != std::string::npos) {
            vec.emplace_back(str.substr(previous, current - previous));
            previous = current + inc;
            current = str.find(delim, previous);
        }
        vec.emplace_back(str.substr(previous, current - previous));
    }


    /**
     * @brief           Split a string(_view) using std::string(_view)::find().
     *
     * @tparam String   std::string(_view), taken by rvalue or lvalue.
     * @param[in] str   String to split.
     * @param[in] delim C-string to use as delimiter.
     * @return          Output vector to insert back the output string(s) into.
     *                  The delimiter is not included in the output strings.
     */
    template<typename String, typename = std::enable_if_t<::Noa::Traits::is_string_v<String>>>
    [[nodiscard]] auto split(String&& a_str, const char* a_delim) {
        std::vector<std::string> o_vec;
        ::Noa::String::split(a_str, a_delim, o_vec);
        return o_vec;
    }


    /**
     * @brief           Split a string(_view) using std::string(_view)::find_first_of().
     *
     * @tparam String   std::string(_view), taken by rvalue or lvalue.
     * @param[in] str   String to split.
     * @param[in] delim Delimiters for the splitting. If more than one character, it is like
     *                  running split() for each character sequentially. If it is a single
     *                  character, it is identical to split().
     * @param[out] vec  Output vector to insert back the output string(s) into.
     *                  The delimiter is not included in the output strings.
     *
     * @example
     * @code
     * std::string str = "  12, 12, 12,";
     * auto vec = splitFirstOf(str, " ,");
     * fmt::print(vec);  // {"", "", "12", "", "12", "", "12", ""}
     * @endcode
     */
    template<typename String, typename = std::enable_if_t<::Noa::Traits::is_string_v<String>>>
    void splitFirstOf(String&& str, const char* delim, std::vector<std::string>& vec) {
        size_t previous = 0;
        size_t current = str.find_first_of(delim);
        while (current != std::string::npos) {
            vec.emplace_back(str.substr(previous, current - previous));
            previous = current + 1;
            current = str.find_first_of(delim, previous);
        }
        vec.emplace_back(str.substr(previous, current - previous));
    }


    /**
     * @brief           Split a string(_view) using std::string(_view)::find_first_of().
     *
     * @tparam String   std::string(_view), taken by rvalue or lvalue.
     * @param[in] str   String to split.
     * @param[in] delim Delimiters for the splitting. If more than one character, it is like
     *                  running split() for each character sequentially. If it is a single
     *                  character, it is identical to split().
     * @return          Output vector to insert back the output string(s) into.
     *                  The delimiter is not included in the output strings.
     */
    template<typename String, typename = std::enable_if_t<::Noa::Traits::is_string_v<String>>>
    std::vector<std::string> splitFirstOf(String&& str, const char* delim) {
        std::vector<std::string> vec;
        ::Noa::String::splitFirstOf(str, delim, vec);
        return vec;
    }

    /**
     * @brief               Parse a string.
     * @details             Parse a string using commas as a separator. Parsed values are trimmed
     *                      and empty strings are kept, similar to `Noa::String::split`.
     *
     * @tparam String       std::string(_view) by rvalue or lvalue.
     * @param[in] str       String to parse.
     * @param[out] vec      Output vector containing the parsed string(s). The new strings are
     *                      inserted at the end of the vector, in order.
     *
     * @note                This is used to parse the command line and parameter file values.
     *
     * @example
     * @code
     * std::vector<std::string> vec;
     * parse(" 1, 2,  ,  4 5 ", vec);
     * fmt::print(vec);  // {"1", "2", "", "4 5"}
     * @endcode
     */
    template<typename String, typename = std::enable_if_t<::Noa::Traits::is_string_v<String>>>
    void parse(String&& str, std::vector<std::string>& vec) {
        size_t idx_start{0}, idx_end{0};
        bool capture{false};

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                vec.emplace_back(str, idx_start, idx_end - idx_start);
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
        vec.emplace_back(str, idx_start, idx_end - idx_start);
    }


    /**
     * @brief           Parse a string.
     * @details         See overload above.
     *
     * @tparam String   std::string(_view) by rvalue or lvalue.
     * @param str       String to parse.
     * @return          Output vector containing the parsed string(s).
     */
    template<typename String, typename = std::enable_if_t<::Noa::Traits::is_string_v<String>>>
    std::vector<std::string> parse(String&& str) {
        std::vector<std::string> vec;
        parse(std::forward<String>(str), vec);
        return vec;
    }


    /**
     * @brief                   Convert a string into an `int` with std::stoi.
     *
     * @param[in] str           String to convert into an `int`.
     * @return                  `int` resulting from the conversion.
     *
     * @warning                 This is using the decimal system (base 10). For different
     *                          bases, use std::stoi directly.
     * @throw Noa::ErrorCore    If str cannot be converted into an `int` or is out of range.
     */
    inline int toInt(const std::string& str) {
        try {
            return std::stoi(str);
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("\"{}\" is out of the int range", str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("\"{}\" cannot be converted into an int", str);
        }
    }


    /**
     * @brief                   Convert a vector of string(s) into integer(s) with std::stoi.
     *
     * @tparam Sequence         Type of the output sequence (vector or array) that will contain
     *                          the formatted integers(s).
     * @param[in] vec_str       Vector containing the string(s) to convert. Can be empty, but
     *                          empty strings are not allowed.
     * @return                  Output sequence with a size equal to the size of the input vector.
     *                          If the input vector is empty, the output sequence will be empty as well.
     *
     * @throw Noa::ErrorCore    If at least one element in the input vector cannot be converted
     *                          into an int or is out of range.
     */
    template<typename Sequence = std::vector<int>,
            typename = std::enable_if_t<::Noa::Traits::is_sequence_of_int_v<Sequence>>>
    auto toInt(const std::vector<std::string>& vec_str) {
        Sequence out_ints;
        try {
            if constexpr(::Noa::Traits::is_array_v<Sequence>) {
                for (size_t i = 0; i < vec_str.size(); ++i)
                    out_ints[i] = std::stoi(vec_str[i]);
            } else if constexpr(::Noa::Traits::is_vector_v<Sequence>) {
                out_ints.reserve(vec_str.size());
                for (const auto& i : vec_str)
                    out_ints.emplace_back(std::stoi(i));
            }
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("at least one element in {} is out of the int range", vec_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("at least one element in {} cannot be converted into an int", vec_str);
        }
        return out_ints;
    }


    /**
     * @brief               Convert a string into a `float` with std::stof.
     *
     * @tparam String       std:string(_view) by lvalue or rvalue
     * @param[in] str       String to convert into a `float`.
     * @return              `float` resulting from the conversion.
     *
     * @throw Noa::Error    If `str` cannot be converted into a `float` or is out of range.
     */
    inline float toFloat(const std::string& str) {
        try {
            return std::stof(str);
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("\"{}\" is out of the float range", str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("\"{}\" cannot be converted into a float", str);
        }
    }


    /**
     * @brief                   Convert a vector of string(s) into float(s) with std::stoi.
     *
     * @tparam Sequence         Type of the output sequence (vector or array) that will contain
     *                          the formatted float(s).
     * @param[in] vec_str       Vector containing the string(s) to convert. Can be empty, but
     *                          empty strings are not allowed.
     * @return                  Output sequence with a size equal to the size of the input vector.
     *                          If the input vector is empty, the output sequence will be empty as well.
     *
     * @throw Noa::ErrorCore    If at least one element in the input vector cannot be converted
     *                          into a float or is out of range.
     */
    template<typename Sequence = std::vector<float>,
            typename = std::enable_if_t<::Noa::Traits::is_sequence_of_float_v<Sequence>>>
    auto toFloat(const std::vector<std::string>& vec_str) {
        Sequence out_floats;
        try {
            if constexpr(Noa::Traits::is_array_v<Sequence>) {
                for (size_t i = 0; i < vec_str.size(); ++i)
                    out_floats[i] = std::stof(vec_str[i]);
            } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
                out_floats.reserve(vec_str.size());
                for (const auto& i : vec_str)
                    out_floats.emplace_back(std::stof(i));
            }
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("at least one element in {} is out of the float range", vec_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("at least one element in {} cannot be converted into a float",
                           vec_str);
        }
        return out_floats;
    }


    /**
     * @brief                   Convert a string into a bool.
     *
     * @param[in] str           String to convert.
     * @return                  bool resulting from the conversion.
     *
     * @throw Noa::ErrorCore    If str cannot be converted into a bool.
     */
    inline bool toBool(const std::string& str) {
        if (str == "1" || str == "true" || str == "True" || str == "y" || str == "yes" ||
            str == "on" || str == "YES" || str == "ON" || str == "TRUE")
            return true;
        else if (str == "0" || str == "false" || str == "False" || str == "n" || str == "no" ||
                 str == "off" || str == "NO" || str == "OFF" || str == "FALSE")
            return false;
        else {
            NOA_CORE_ERROR("\"{}\" cannot be converted into a bool", str);
        }
    }


    /**
     * @short                   Convert a vector of string(s) into a vector of bool(s).
     *
     * @tparam Sequence         A sequence (std::vector|std::array) of bool(s).
     * @param[in] vec_str       Vector containing the strings to convert.
     * @return                  Output sequence with a size equal to the size of the input vector.
     *                          If the input vector is empty, the output sequence will be empty as well.
     *
     * @throw Noa::ErrorCore    If at least one element in vec_str cannot be converted into a bool.
     */
    template<typename Sequence = std::vector<bool>,
            typename = std::enable_if_t<::Noa::Traits::is_sequence_of_bool_v<Sequence>>>
    auto toBool(const std::vector<std::string>& vec_str) {
        Sequence out_booleans;
        if constexpr(Noa::Traits::is_array_v<Sequence>) {
            for (size_t i = 0; i < vec_str.size(); ++i)
                out_booleans[i] = toBool(vec_str[i]);
        } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
            out_booleans.reserve(vec_str.size());
            for (const auto& i : vec_str)
                out_booleans.emplace_back(toBool(i));
        }
        return out_booleans;
    }


    /**
     * Get the index of the first non-space (whitespace, TAB, newline or carriage return) character.
     * @param str   String(_view) to look at.
     * @return      Index of the first non-space character. If str is entirely composed of spaces,
     *              returns std::string::npos.
     */
    template<typename S, typename = std::enable_if_t<::Noa::Traits::is_string_v<S>>>
    inline size_t firstNonSpace(S&& str) noexcept {
        return str.find_first_not_of(" \t\r\n");
    }
}
