/**
 * @file String.h
 * @brief String related functions.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "noa/noa.h"
#include "noa/utils/Traits.h"


/// Group of string related functions.
namespace Noa::String {

    /**
     * @brief               Left trim a std::string.
     * @param [in] a_str    String to left trim (taken as lvalue)
     * @return              Left trimmed string.
     */
    inline std::string& leftTrim(std::string& a_str);


    /**
     * @brief               Left trim a std::string.
     * @param [in] a_str    String to left trim (taken as rvalue)
     * @return              Left trimmed string.
     */
    [[nodiscard]] inline std::string leftTrim(std::string&& a_str);


    /**
     * @brief               Right trim a std::string.
     * @param [in] a_str    String to right trim (taken by lvalue ref)
     * @return              Right trimmed string.
     */
    inline std::string& rightTrim(std::string& a_str);


    /**
     * @brief               Right trim a std::string.
     * @param [in] a_str    String to right trim (taken by rvalue ref)
     * @return              Right trimmed string.
     */
    [[nodiscard]] inline std::string rightTrim(std::string&& a_str);


    /**
     * @brief               Trim (left and right) a std::string.
     * @param [in] a_str    String to trim (taken by lvalue ref)
     * @return              Trimmed string.
     */
    inline std::string& trim(std::string& a_str);


    /**
     * @brief               Trim (left and right) a std::string.
     * @param [in] a_str    String to trim (taken by rvalue ref)
     * @return              Trimmed string.
     */
    [[nodiscard]] inline std::string trim(std::string&& a_str);


    /**
     * @brief               Trim (left and right) std::string(s) stored in std::vector.
     * @param [in] a_str    Vector of string(s) to trim (taken by lvalue ref)
     * @return              Trimmed string.
     */
    inline std::vector<std::string>& trim(std::vector<std::string>& a_vec_str);


    /**
     * @brief                   Split a string(_view) using std::string(_view)::find.
     *
     * @tparam [in] String      std::string or std::string_view.
     * @param [in] a_str        String to split.
     * @param [in] a_delim      C-string to use as delimiter.
     * @param [int,out] o_vec   Output vector to store the output string(s) into.
     *                          The delimiter is not included in the output strings.
     *
     * @example
     * @code
     * std::string str = "  12, 12  , 12, ";
     * std::vector<std::string> vec;
     * split(str, ",", vec);
     * // vec = {"  12", " 12  ", " 12", " "}
     * @endcode
     */
    template<typename String>
    void split(String&& a_str, const char* a_delim, std::vector<std::string>& o_vec) {
        static_assert(::Noa::Traits::is_string_v<String>);
        size_t previous = 0, current = a_str.find(a_delim, previous);
        while (current != std::string::npos) {
            o_vec.emplace_back(a_str.substr(previous, current - previous));
            previous = current + 1;
            current = a_str.find(a_delim, previous);
        }
        o_vec.emplace_back(a_str.substr(previous, current - previous));
    }


    /**
     * @brief                   Split a string(_view) using std::string(_view)::find.
     *
     * @tparam [in] String      std::string or std::string_view.
     * @param [in] a_str        String to split.
     * @param [in] a_delim      C-string to use as delimiter.
     * @return                  std::vector containing the output std::string(s).
     *                          The delimiter is not included in the returned strings.
     */
    template<typename String>
    [[nodiscard]] auto split(String&& a_str, const char* a_delim) {
        std::vector<std::string> o_vec;
        ::Noa::String::split(a_str, a_delim, o_vec);
        return o_vec;
    }


    /**
     * @short                   Split a string or string_view using std::string(_view)::find_first_of.
     *                          Most basic splitting.
     *
     * @tparam [in] String      std::string(&) or std::string_view(&).
     * @param [in] a_str        String to split.
     * @param [in] a_delim      C-string to use as delimiter.
     * @param [int,out] o_vec   Output vector to store the strings into.
     *                          The delimiter is not included in the returned strings.
     *
     * @example
     * @code
     * std::string str = "12, 12 12";
     * std::vector<std::string> vec;
     * split(str, ", ", vec);
     * // vec = {"12", "", "12", "12"}
     * @endcode
     */
    template<typename String>
    void splitFirstOf(String&& a_str,
                      const char* a_delim,
                      std::vector<std::string>& o_vec) {
        static_assert(::Noa::Traits::is_string_v<String>);

        size_t previous = 0;
        size_t current = a_str.find_first_of(a_delim);
        while (current != std::string::npos) {
            o_vec.emplace_back(a_str.substr(previous, current - previous));
            previous = current + 1;
            current = a_str.find_first_of(a_delim, previous);
        }
        o_vec.emplace_back(a_str.substr(previous, current - previous));
    }

    template<typename String>
    std::vector<std::string> splitFirstOf(String&& a_str, const char* a_delim) {
        std::vector<std::string> o_vec;
        splitFirstOf(a_str, a_delim, o_vec);
        return o_vec;
    }

    /**
     * @fn parse
     * @short                   Parse a string.
     *
     * @tparam [in] String      std::string(&) or std::string_view(&).
     * @param [in] a_str        String to parse.
     * @param [in,out] o_vec    Output vector containing the parsed string(s).
     *
     * @example
     * @code
     * std::string str = " 12,, 12   12,";
     * std::vector<std::string> vec;
     * split(str, vec);
     * // vec = {"12", "", "12", "12", ""}
     * @endcode
     */
    template<typename String>
    void parse(String&& a_str, std::vector<std::string>& o_vec) {
        static_assert(::Noa::Traits::is_string_v<String>);
        size_t idx_start{0};
        bool flushed{true}, comma{true};

        for (size_t i{0}; i < a_str.size(); ++i) {
            if (std::isspace(a_str[i])) {
                if (flushed) {
                    continue;
                } else {
                    o_vec.emplace_back(a_str.substr(idx_start, i - idx_start));
                    flushed = true;
                    comma = false;
                }
            } else if (a_str[i] == ',') {
                if (comma && flushed) {
                    o_vec.emplace_back("");
                } else if (!flushed) {
                    o_vec.emplace_back(a_str.substr(idx_start, i - idx_start));
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
                o_vec.emplace_back("");
        } else
            o_vec.emplace_back(a_str.substr(idx_start, a_str.size() - idx_start));
    }


    template<typename String>
    std::vector<std::string> parse(String&& a_str) {
        std::vector<std::string> o_vec;
        parse(std::forward<String>(a_str), o_vec);
        return o_vec;
    }


    /**
     * @fn tokenize
     * @short                   Tokenize a string with std::string(_view)::find.
     *
     * @tparam [in] String      std::string(&) or std::string_view(&).
     * @param [in] a_str        String to tokenize.
     * @param [in] a_delim      C-string to use as delimiter.
     * @param [in,out] o_vec    Output vector containing the parsed string(s).
     *
     * @example
     * @code
     * std::string str = "12,, 12   12,";
     * std::vector<std::string> vec;
     * split(str, "," vec);
     * // vec = {"12", " 12   12"}
     * @endcode
     */
    template<typename String>
    void tokenize(String&& a_str, const char* a_delim, std::vector<std::string>& o_vec) {
        static_assert(::Noa::Traits::is_string_v<String>);
        size_t start;
        size_t end = 0;

        while ((start = a_str.find_first_not_of(a_delim, end)) != std::string::npos) {
            end = a_str.find(a_delim, start);
            o_vec.emplace_back(a_str.substr(start, end - start));
        }
    }


    template<typename String>
    std::vector<std::string> tokenize(String&& a_str, const char* a_delim) {
        std::vector<std::string> o_vec;
        tokenize(std::forward<String>(a_str), a_delim, o_vec);
        return o_vec;
    }


    /**
     * @fn tokenizeFirstOf
     * @short                   Tokenize a string with std::string(_view)::find_first_of.
     *
     * @tparam [in] String      std::string(&) or std::string_view(&).
     * @param [in] a_str        String to tokenize.
     * @param [in] a_delim      C-string to use as delimiter.
     * @param [in,out] o_vec    Output vector containing the parsed string(s).
     *
     * @example
     * @code
     * std::string str = "12,, 12   12,";
     * std::vector<std::string> vec;
     * split(str, ", " vec);
     * // vec = {"12", "12", "12"}
     * @endcode
     */
    template<typename String>
    void tokenizeFirstOf(String&& a_str, const char* a_delim, std::vector<std::string>& o_vec) {
        static_assert(std::is_same_v<std::decay_t<String>, std::string> ||
                      std::is_same_v<std::decay_t<String>, std::string_view>);
        size_t start;
        size_t end = 0;

        while ((start = a_str.find_first_not_of(a_delim, end)) != std::string::npos) {
            end = a_str.find_first_of(a_delim, start);
            o_vec.emplace_back(a_str.substr(start, end - start));
        }
    }


    template<typename String>
    std::vector<std::string> tokenizeFirstOf(String&& str, const char* delim) {
        std::vector<std::string> o_vec;
        tokenizeFirstOf(std::forward<String>(str), delim, o_vec);
        return o_vec;
    }


    /**
     * @brief               Convert a string into an int with std::stoi.
     *
     * @param [in] a_str    String to convert into an int.
     * @return              int resulting from the conversion.
     *
     * @throw Noa::Error    If a_str cannot be converted into an int or is out of range.
     */
    inline int toInt(const std::string& a_str);


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
    inline float toFloat(const std::string& a_str);


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
            NOA_CORE_ERROR("at least one element in {} cannot be converted into a float", a_vec_str);
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
