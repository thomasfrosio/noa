/**
 * @file String.h
 * @brief Just a static class to store methods related to strings.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "Core.h"
#include "Assert.h"
#include "Traits.h"

namespace Noa {

    class String {
    public:
        String() = delete;

        ~String() = delete;

        // Left trim a string (lvalue).
        static inline std::string& leftTrim(std::string& a_str) {
            a_str.erase(a_str.begin(),
                        std::find_if(a_str.begin(),
                                     a_str.end(),
                                     [](int ch) { return !std::isspace(ch); }));
            return a_str;
        };

        // Left trim a string (rvalue).
        static inline std::string leftTrim(std::string&& a_str) {
            a_str.erase(a_str.begin(),
                        std::find_if(a_str.begin(),
                                     a_str.end(),
                                     [](int ch) { return !std::isspace(ch); }));
            return std::move(a_str);
        };

        // Right trim a string (lvalue).
        static inline std::string& rightTrim(std::string& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(),
                                     a_str.rend(),
                                     [](int ch) { return !std::isspace(ch); }).base(),
                        a_str.end());
            return a_str;
        };

        // Right trim a string (take rvalue reference, trim, return rvalue).
        static inline std::string rightTrim(std::string&& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(),
                                     a_str.rend(),
                                     [](int ch) { return !std::isspace(ch); }).base(),
                        a_str.end());
            return std::move(a_str);
        };

        // Strip (left and right trim) a string (lvalue).
        static inline std::string& strip(std::string& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(),
                                     a_str.rend(),
                                     [](int ch) { return !std::isspace(ch); }).base(),
                        std::find_if(a_str.begin(),
                                     a_str.end(),
                                     [](int ch) { return !std::isspace(ch); }));
            return a_str;
        };

        // Strip (left and right trim) a string (rvalue).
        static inline std::string strip(std::string&& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(),
                                     a_str.rend(),
                                     [](int ch) { return !std::isspace(ch); }).base(),
                        std::find_if(a_str.begin(),
                                     a_str.end(),
                                     [](int ch) { return !std::isspace(ch); }));
            return std::move(a_str);
        };

        // Strip all the strings of a vector (lvalue).
        static inline std::vector<std::string>& strip(std::vector<std::string>& a_vec_str) {
            for (auto& str : a_vec_str)
                strip(str);
            return a_vec_str;
        };

        // Strip all the strings of a vector (rvalue).
        static inline std::vector<std::string> strip(std::vector<std::string>&& a_vec_str) {
            for (auto& str : a_vec_str)
                strip(str);
            return std::move(a_vec_str);
        };

        /**
         * @short                   Split a string or string_view using std::string(_view)::find.
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
         * std::string str = "  12, 12  , 12, ";
         * std::vector<std::string> vec;
         * split(str, ",", vec);
         * // vec = {"  12", " 12  ", " 12", " "}
         * @endcode
         */
        template<typename String>
        static void split(String&& a_str,
                          const char* a_delim,
                          std::vector<std::string>& o_vec) {
            static_assert(std::is_same_v<std::decay_t<String>, std::string> ||
                          std::is_same_v<std::decay_t<String>, std::string_view>);

            size_t previous = 0;
            size_t current = a_str.find(a_delim, previous);
            while (current != std::string::npos) {
                o_vec.emplace_back(a_str.substr(previous, current - previous));
                previous = current + 1;
                current = a_str.find(a_delim, previous);
            }
            o_vec.emplace_back(a_str.substr(previous, current - previous));
        };

        template<typename String>
        static std::vector<std::string> split(String&& a_str, const char* a_delim) {
            std::vector<std::string> o_vec;
            split(a_str, a_delim, o_vec);
            return o_vec;
        };


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
        static void splitFirstOf(String&& a_str,
                                 const char* a_delim,
                                 std::vector<std::string>& o_vec) {
            static_assert(std::is_same_v<std::decay_t<String>, std::string> ||
                          std::is_same_v<std::decay_t<String>, std::string_view>);

            size_t previous = 0;
            size_t current = a_str.find_first_of(a_delim);
            while (current != std::string::npos) {
                o_vec.emplace_back(a_str.substr(previous, current - previous));
                previous = current + 1;
                current = a_str.find_first_of(a_delim, previous);
            }
            o_vec.emplace_back(a_str.substr(previous, current - previous));
        };

        template<typename String>
        static std::vector<std::string> splitFirstOf(String&& a_str, const char* a_delim) {
            std::vector<std::string> o_vec;
            splitFirstOf(a_str, a_delim, o_vec);
            return o_vec;
        };

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
        static void parse(String&& a_str, std::vector<std::string>& o_vec) {
            static_assert(std::is_same_v<std::decay_t<String>, std::string> ||
                          std::is_same_v<std::decay_t<String>, std::string_view>);
            int pos = -1;
            bool comma = true;

            for (int i{0}; i < a_str.size(); ++i) {
                if (std::isspace(a_str[i])) {
                    if (pos == -1) {
                        continue;
                    } else {
                        o_vec.emplace_back(a_str.substr(pos, i - pos));
                        pos = -1;
                        comma = false;
                    }
                } else if (a_str[i] == ',') {
                    if (comma && pos == -1) {
                        o_vec.emplace_back("");
                    } else if (pos != -1) {
                        o_vec.emplace_back(a_str.substr(pos, i - pos));
                        pos = -1;
                    }
                    comma = true;
                } else {
                    if (pos == -1)
                        pos = i;
                    else
                        continue;
                }
            }
            if (pos == -1) {
                if (comma)
                    o_vec.emplace_back("");
            } else
                o_vec.emplace_back(a_str.substr(pos, a_str.size() - pos));
        }


        template<typename BasicString>
        static std::vector<std::string> parse(BasicString&& a_str) {
            std::vector<std::string> o_vec;
            parse(std::forward<BasicString>(a_str), o_vec);
            return o_vec;
        };


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
            static_assert(std::is_same_v<std::decay_t<String>, std::string> ||
                          std::is_same_v<std::decay_t<String>, std::string_view>);
            size_t start;
            size_t end = 0;

            while ((start = a_str.find_first_not_of(a_delim, end)) != std::string::npos) {
                end = a_str.find(a_delim, start);
                o_vec.emplace_back(a_str.substr(start, end - start));
            }
        }

        template<typename BasicString>
        static std::vector<std::string> tokenize(BasicString&& a_str, const char* a_delim) {
            std::vector<std::string> o_vec;
            tokenize(std::forward<BasicString>(a_str), a_delim, o_vec);
            return o_vec;
        };


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
        static std::vector<std::string> tokenizeFirstOf(String&& str, const char* delim) {
            std::vector<std::string> o_vec;
            tokenizeFirstOf(std::forward<String>(str), delim, o_vec);
            return o_vec;
        };


        /**
         * @fn String::toInteger
         * @short               Convert a string into an int with std::stoi.
         *
         * @param [in] a_str    String to convert into an integer.
         * @return              Integer resulting from the conversion.
         */
        static inline int toInteger(const std::string& a_str) {
            try {
                return std::stoi(a_str);
            } catch (const std::out_of_range& e) {
                NOA_CORE_ERROR("String::toInteger: \"{}\" is out of the int range", a_str);
            } catch (const std::invalid_argument& e) {
                NOA_CORE_ERROR("String::toInteger: \"{}\" cannot be converted into an int", a_str);
            }
        }

        /**
         * @fn String::toInteger
         * @short                   Convert a vector of string(s) into integer(s) with std::stoi.
         *
         * @tparam [out] Sequence   A sequence (std::vector|std::array) of integer.
         * @param [in] a_vec_str    Vector containing the strings to convert.
         *
         * @return                  Integer(s) resulting from the conversion. They are stored
         *                          in a Sequence which has a size equal to the size of
         *                          the input vector.
         */
        template<typename Sequence = std::vector<int>>
        static auto toInteger(const std::vector<std::string>& a_vec_str) {
            static_assert(Noa::Traits::is_sequence_of_int_v<Sequence>);
            try {
                std::remove_reference_t<Sequence> o_array_int;
                if constexpr(Noa::Traits::is_array_v<Sequence>) {
                    for (int i = 0; i < a_vec_str.size(); ++i)
                        o_array_int[i] = std::stoi(a_vec_str[i]);
                } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
                    o_array_int.reserve(a_vec_str.size());
                    for (const auto& i : a_vec_str)
                        o_array_int.emplace_back(std::stoi(i));
                }
                return o_array_int;
            } catch (const std::out_of_range& e) {
                NOA_CORE_ERROR("String::toInteger: at least one element in {} "
                               "is out of the int range",
                               a_vec_str);
            } catch (const std::invalid_argument& e) {
                NOA_CORE_ERROR("String::toInteger: at least one element in {} "
                               "cannot be converted into an int",
                               a_vec_str);
            }
        }


        /**
         * @fn String::toFloat
         * @short               Convert a string into an float with std::stof.
         *
         * @param [in] a_str    String to convert into a float.
         * @return              Float resulting from the conversion.
         */
        static inline float toFloat(const std::string& a_str) {
            try {
                return std::stof(a_str);
            } catch (const std::out_of_range& e) {
                NOA_CORE_ERROR("String::toFloat: \"{}\" is out of the float range", a_str);
            } catch (const std::invalid_argument& e) {
                NOA_CORE_ERROR("String::toFloat: \"{}\" cannot be converted into a float", a_str);
            }
        }

        /**
         * @fn String::toFloat
         * @short                   Convert a vector of string(s) into float(s) with std::stof.
         *
         * @tparam [out] Sequence   A sequence (std::vector|std::array) of float.
         * @param [in] a_vec_str    Vector containing the strings to convert.
         *
         * @return                  Float(s) resulting from the conversion. They are stored
         *                          in a Sequence which has a size equal to the size of
         *                          the input vector.
         */
        template<typename Sequence = std::vector<float>>
        static auto toFloat(const std::vector<std::string>& a_vec_str) {
            static_assert(Noa::Traits::is_sequence_of_float_v<Sequence>);
            try {
                std::remove_reference_t<Sequence> o_array_float;
                if constexpr(Noa::Traits::is_array_v<Sequence>) {
                    for (int i = 0; i < a_vec_str.size(); ++i)
                        o_array_float[i] = std::stof(a_vec_str[i]);
                } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
                    o_array_float.reserve(a_vec_str.size());
                    for (const auto& i : a_vec_str)
                        o_array_float.emplace_back(std::stof(i));
                }
                return o_array_float;
            } catch (const std::out_of_range& e) {
                NOA_CORE_ERROR("String::toFloat: at least one element in {} "
                               "is out of the float range",
                               a_vec_str);
            } catch (const std::invalid_argument& e) {
                NOA_CORE_ERROR("String::toFloat: at least one element in {} "
                               "cannot be converted into a float",
                               a_vec_str);
            }
        }


        /**
         * @fn String::toBool
         * @short               Convert a string into an float with std::stof.
         *
         * @tparam [in]         std::string(&) || std::string_view(&).
         * @param [in] a_str    String to convert into a bool.
         * @return              Bool resulting from the conversion.
         */
        template<typename String>
        static bool toBool(String&& a_str) {
            static_assert(std::is_same_v<std::decay_t<String>, std::string> ||
                          std::is_same_v<std::decay_t<String>, std::string_view>);

            if (a_str == "1" || a_str == "true" || a_str == "True" || a_str == "TRUE" ||
                a_str == "on" || a_str == "On" || a_str == "ON")
                return true;
            else if (a_str == "0" || a_str == "false" || a_str == "False" || a_str == "FALSE" ||
                     a_str == "off" || a_str == "Off" || a_str == "OFF")
                return false;
            else {
                NOA_CORE_ERROR("String::toBool: \"{}\" cannot be converted into a bool", a_str);
            }
        }

        /**
         * @fn String::toBool
         * @short                   Convert a vector of string(s) into bool(s).
         *
         * @tparam [out] Sequence   A sequence (std::vector|std::array) of bool.
         * @param [in] a_vec_str    Vector containing the strings to convert.
         *
         * @return                  Bool(s) resulting from the conversion. They are stored
         *                          in a Sequence which has a size equal to the size of
         *                          the input vector.
         */
        template<typename Sequence = std::vector<bool>>
        static Sequence toBool(const std::vector<std::string>& a_vec_str) {
            static_assert(Noa::Traits::is_sequence_of_bool_v<Sequence>);

            std::remove_reference_t<Sequence> o_array_bool;
            if constexpr(Noa::Traits::is_array_v<Sequence>) {
                for (int i = 0; i < a_vec_str.size(); ++i)
                    o_array_bool[i] = toBool(a_vec_str[i]);
            } else if constexpr(Noa::Traits::is_vector_v<Sequence>) {
                o_array_bool.reserve(a_vec_str.size());
                for (const auto& i : a_vec_str)
                    o_array_bool.emplace_back(toBool(i));
            }
            return o_array_bool;
        }
    };
}
