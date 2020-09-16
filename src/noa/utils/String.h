/**
 * @file String.h
 * @brief String related functions.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "../Base.h"
#include "Traits.h"


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
    template<typename String>
    void split(String&& str, const char* delim, std::vector<std::string>& vec) {
        static_assert(::Noa::Traits::is_string_v<String>);
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
    template<typename String>
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
    template<typename String>
    void splitFirstOf(String&& str, const char* delim, std::vector<std::string>& vec) {
        static_assert(::Noa::Traits::is_string_v<String>);

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
    template<typename String>
    std::vector<std::string> splitFirstOf(String&& str, const char* delim) {
        std::vector<std::string> vec;
        ::Noa::String::splitFirstOf(str, delim, vec);
        return vec;
    }

    /**
     * @brief               Parse a string using our parsing convention.
     * @details             Parse a string using a whitespace _and_ a comma as a separator,
     *                      which effectively trims the string while splitting it.
     *                      Empty strings are kept, similar to `Noa::String::split`.
     *
     * @tparam String       std::string(_view) by rvalue or lvalue.
     * @param[in] str       String to parse.
     * @param[out] vec      Output vector containing the parsed string(s). The new strings are
     *                      inserted at the end of the vector. It doesn't have to be empty.
     *
     * @note                This is used to parse the command line and parameter file arguments.
     *
     * @example
     * @code
     * std::vector<std::string> vec;
     * parse(" 1, 2,  ,  4 5", vec);
     * fmt::print(vec);  // {"1", "2", "", "4", "5"}
     * @endcode
     */
    template<typename String>
    void parse(String&& str, std::vector<std::string>& vec) {
        static_assert(::Noa::Traits::is_string_v<String>);
        size_t idx_start{0};
        bool flushed{true}, comma{true};

        for (size_t i{0}; i < str.size(); ++i) {
            if (std::isspace(str[i])) {
                if (flushed) {
                    continue;
                } else {
                    vec.emplace_back(str.substr(idx_start, i - idx_start));
                    flushed = true;
                    comma = false;
                }
            } else if (str[i] == ',') {
                if (comma && flushed) {
                    vec.emplace_back("");
                } else if (!flushed) {
                    vec.emplace_back(str.substr(idx_start, i - idx_start));
                    flushed = true;
                }
                comma = true;
            } else {
                if (flushed) {
                    idx_start = i;
                    flushed = false;
                } else
                    continue;
            }
        }
        if (flushed) {
            if (comma)
                vec.emplace_back("");
        } else
            vec.emplace_back(str.substr(idx_start, str.size() - idx_start));
    }


    /**
     * @brief           Parse a string using our parsing convention.
     * @details         See overload above.
     *
     * @tparam String   std::string(_view) by rvalue or lvalue.
     * @param str       String to parse.
     * @return          Output vector containing the parsed string(s).
     */
    template<typename String>
    std::vector<std::string> parse(String&& str) {
        std::vector<std::string> vec;
        parse(std::forward<String>(str), vec);
        return vec;
    }


    /**
     * @brief               Convert a string into an int with std::stoi.
     *
     * @param [in] a_str    String to convert into an int.
     * @return              int resulting from the conversion.
     *
     * @throw Noa::Error    If a_str cannot be converted into an int or is out of range.
     */
    inline int toInt(const std::string& a_str) {
        try {
            return std::stoi(a_str);
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("\"{}\" is out of the int range", a_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("\"{}\" cannot be converted into an int", a_str);
        }
    }


    /**
     * @short                   Convert a vector of string(s) into integer(s) with std::stoi.
     *
     * @tparam [out] Sequence   A sequence (std::vector|std::array) of int(s).
     * @param [in] a_vec_str    Vector containing the strings to convert.
     * @return                  int(s) resulting from the conversion. They are stored
     *                          in a Sequence which has a size equal to the size of
     *                          the input vector.
     *
     * @throw Noa::Error        If at least one element in a_vec_str cannot be converted
     *                          into an int or is out of range.
     */
    template<typename Sequence = std::vector<int>>
    auto toInt(const std::vector<std::string>& a_vec_str) {
        static_assert(::Noa::Traits::is_sequence_of_int_v<Sequence>);
        std::remove_reference_t<Sequence> o_array_int;
        try {
            if constexpr(::Noa::Traits::is_array_v<Sequence>) {
                for (size_t i = 0; i < a_vec_str.size(); ++i)
                    o_array_int[i] = std::stoi(a_vec_str[i]);
            } else if constexpr(::Noa::Traits::is_vector_v<Sequence>) {
                o_array_int.reserve(a_vec_str.size());
                for (const auto& i : a_vec_str)
                    o_array_int.emplace_back(std::stoi(i));
            }
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("at least one element in {} is out of the int range", a_vec_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("at least one element in {} cannot be converted into an int", a_vec_str);
        }
        return o_array_int;
    }


    /**
     * @brief               Convert a string into a float with std::stof.
     *
     * @param [in] a_str    String to convert into a float.
     * @return              float resulting from the conversion.
     *
     * @throw Noa::Error    If a_str cannot be converted into a float or is out of range.
     */
    inline float toFloat(const std::string& a_str) {
        try {
            return std::stof(a_str);
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("\"{}\" is out of the float range", a_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("\"{}\" cannot be converted into a float", a_str);
        }
    }


    /**
     * @short                   Convert a vector of string(s) into float(s) with std::stof.
     *
     * @tparam [out] Sequence   A sequence (std::vector|std::array) of float(s).
     * @param [in] a_vec_str    Vector containing the strings to convert.
     * @return                  float(s) resulting from the conversion. They are stored
     *                          in a Sequence which has a size equal to the size of
     *                          the input vector.
     *
     * @throw Noa::Error        If at least one element in a_vec_str cannot be converted
     *                          into a float or is out of range.
     */
    template<typename Sequence = std::vector<float>>
    auto toFloat(const std::vector<std::string>& a_vec_str) {
        static_assert(Noa::Traits::is_sequence_of_float_v<Sequence>);
        std::remove_reference_t<Sequence> o_array_float;
        try {
            if constexpr(Noa::Traits::is_array_v<Sequence>) {
                for (size_t i = 0; i < a_vec_str.size(); ++i)
                    o_array_float[i] = std::stof(a_vec_str[i]);
            } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
                o_array_float.reserve(a_vec_str.size());
                for (const auto& i : a_vec_str)
                    o_array_float.emplace_back(std::stof(i));
            }
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("at least one element in {} is out of the float range", a_vec_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("at least one element in {} cannot be converted into a float",
                           a_vec_str);
        }
        return o_array_float;
    }


    /**
     * @brief               Convert a string into a bool.
     *
     * @param [in] a_str    String to convert into a bool.
     * @return              bool resulting from the conversion.
     *
     * @throw Noa::Error    If a_str cannot be converted into a bool.
     */
    template<typename String>
    bool toBool(String&& a_str) {
        static_assert(::Noa::Traits::is_string_v<String>);
        if (a_str == "1" || a_str == "true" || a_str == "True" || a_str == "TRUE" ||
            a_str == "on" || a_str == "On" || a_str == "ON")
            return true;
        else if (a_str == "0" || a_str == "false" || a_str == "False" || a_str == "FALSE" ||
                 a_str == "off" || a_str == "Off" || a_str == "OFF")
            return false;
        else {
            NOA_CORE_ERROR("\"{}\" cannot be converted into a bool", a_str);
        }
    }


    /**
     * @short                   Convert a vector of string(s) into bool(s).
     *
     * @tparam [out] Sequence   A sequence (std::vector|std::array) of bool(s).
     * @param [in] a_vec_str    Vector containing the strings to convert.
     * @return                  bool(s) resulting from the conversion. They are stored
     *                          in a Sequence which has a size equal to the size of
     *                          the input vector.
     *
     * @throw Noa::Error        If at least one element in a_vec_str cannot be converted into a bool.
     */
    template<typename Sequence = std::vector<bool>>
    auto toBool(const std::vector<std::string>& a_vec_str) {
        static_assert(Noa::Traits::is_sequence_of_bool_v<Sequence>);
        std::remove_reference_t<Sequence> o_array_bool;
        if constexpr(Noa::Traits::is_array_v<Sequence>) {
            for (size_t i = 0; i < a_vec_str.size(); ++i)
                o_array_bool[i] = toBool(a_vec_str[i]);
        } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
            o_array_bool.reserve(a_vec_str.size());
            for (const auto& i : a_vec_str)
                o_array_bool.emplace_back(toBool(i));
        }
        return o_array_bool;
    }
}
