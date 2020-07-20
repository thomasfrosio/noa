//
// Created by thomas on 15/07/2020.
//

#pragma once

#include "Noa.h"

namespace Noa {

    // For now, just a static class to store methods related to strings
    class String {

    public:
        String() = delete;

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
         * @short               Split a string or string_view using std::string(_view)::find.
         *
         * @tparam BasicString  a_str should be std::string or std::string_view.
         * @param a_str         String to split.
         * @param a_delim       C-string to use as delimiter.
         * @param o_vec         Output vector to store the strings into.
         *                      The delimiter is not included in the returned strings.
         *                      Empty strings are not added to the vector.
         *
         * @code
         *                      std::string str = "abc,,def,  ghijklm,no,";
         *                      std::vector<std::string> vec;
         *                      split(str, ",", vec);
         *                      // vec = {"abc", "def", "  ghijklm", "no"}
         * @endcode
         */
        template<typename BasicString>
        static void
        split(const BasicString& a_str, const char* a_delim, std::vector<std::string>& o_vec) {
            if (a_str.empty())
                return;

            size_t previous = 0;
            size_t current = a_str.find(a_delim, previous);

            while (current != std::string::npos) {
                if (current != previous)
                    o_vec.emplace_back(a_str.substr(previous, current - previous));
                previous = current + 1;
                if (previous == a_str.size())
                    return;
                current = a_str.find(a_delim, previous);
            }
            o_vec.emplace_back(a_str.substr(previous, current - previous));
        };

        template<typename BasicString>
        static std::vector<std::string> split(const BasicString& a_str, const char* a_delim) {
            std::vector<std::string> o_vec;
            split(a_str, a_delim, o_vec);
            return o_vec;
        };


        /**
         * @short               Split a string or string_view using std::string(_view)::find_first_of.
         *
         * @tparam BasicString  a_str should be std::string or std::string_view.
         * @param a_str         String to split.
         * @param a_delim       C-string containing the delimiters.
         * @param o_vec         Output vector to store the strings into.
         *                      The delimiters are not included from the returned strings.
         *                      Empty strings are not added to the vector.
         *
         * @code
         *                      std::string str = "abc, ,def ,ghijklm,  no,  ";
         *                      std::vector<std::string> vec;
         *                      split(str, ", ", vec);
         *                      // vec = {"abc", "def", "ghijklm", "no"}
         * @endcode
         */
        template<typename BasicString>
        static void splitFirstOf(const BasicString& a_str, const char* a_delim,
                                 std::vector<std::string>& o_vec) {
            if (a_str.empty())
                return;

            size_t previous = 0;
            size_t current = a_str.find_first_of(a_delim);

            while (current != std::string::npos) {
                if (current != previous)
                    o_vec.emplace_back(a_str.substr(previous, current - previous));
                previous = current + 1;
                if (previous == a_str.size())
                    return;
                current = a_str.find_first_of(a_delim, previous);
            }
            o_vec.emplace_back(a_str.substr(previous, current - previous));
        };

        template<typename BasicString>
        static std::vector<std::string> splitFirstOf(const BasicString& str, const char* delim) {
            std::vector<std::string> o_vec;
            splitFirstOf(str, delim, o_vec);
            return o_vec;
        };


        /**
         * @short           Convert a string into an int with std::stoi.
         *
         * @param a_str     String to convert into an integer.
         * @return          Integer resulting from the conversion.
         */
        static int toOneInt(const std::string& a_str) {
            return std::stoi(a_str);
        };

        /**
         * @short               Convert a vector of string(s) into integer(s) with std::stoi.
         *
         * @tparam Container    STL Containers, e.g. vector, array, etc.
         * @param a_vec_str     Vector containing at the strings.
         *
         * @return              Integer(s) resulting from the conversion. They are stored
         *                      in the Container which has a size equal to the size of
         *                      the input vector. Therefore, if the input vector is empty
         *                      the output Container will be empty as well.
         */
        template<typename Container>
        static Container toMultipleInt(const std::vector<std::string>& a_vec_str) {
            Container o_array_int;
            if constexpr(std::is_same_v<Container, std::vector<int>>)
                o_array_int.reserve(a_vec_str.size());

            for (int i = 0; i < a_vec_str.size(); ++i) {
                if constexpr(std::is_same_v<Container, std::vector<int>>)
                    o_array_int.emplace_back(std::stoi(a_vec_str[i]));
                else
                    o_array_int[i] = std::stoi(a_vec_str[i]);
            }
            return o_array_int;
        };


        /**
         *
         * @param a_vec_str
         * @return
         */
        static float toOneFloat(const std::string& a_str) {

        };

        template<typename Container>
        static Container toMultipleFloat(const std::vector<std::string>& a_vec_str) {

        };


        /**
         *
         * @param a_vec_str
         * @return
         *
         *
         */
        static bool toOneBool(const std::string& a_str) {
            if (a_str == "1" || a_str == "true" || a_str == "True" || a_str == "TRUE" ||
                a_str == "one" || a_str == "On" || a_str == "ON")
                return true;
            else if (a_str == "0" || a_str == "false" || a_str == "False" || a_str == "FALSE" ||
                     a_str == "off" || a_str == "Off" || a_str == "OFF")
                return false;
            else
                std::cerr << "Error" << std::endl;
        }

        /**
         *
         * @tparam N
         * @param a_vec_str
         * @return
         */
        template<typename Container>
        static Container toMultipleBool(const std::vector<std::string>& a_vec_str) {
            Container o_array_bool;
            if constexpr(std::is_same_v<Container, std::vector<bool>>)
                o_array_bool.reserve(a_vec_str.size());

            for (int i = 0; i < a_vec_str.size(); ++i) {
                if constexpr(std::is_same_v<Container, std::vector<bool>>)
                    o_array_bool.emplace_back(toOneBool(a_vec_str[i]));
                else
                    o_array_bool[i] = toOneBool(a_vec_str[i]);
            }
            return o_array_bool;
        }

        /**
         *
         * @param a_vec_str
         * @return
         */
        static char toOneChar(const std::string& a_str) {
            if (a_str.size() > 1)
                std::cerr << "Error" << std::endl;
            return a_str[0];
        }

        /**
         *
         * @tparam Container
         * @param a_vec_str
         * @return
         */
        template<typename Container>
        static Container toMultipleChar(const std::vector<std::string>& a_vec_str) {
            Container o_array_char{};
            if constexpr(std::is_same_v<Container, std::vector<char>>)
                o_array_char.reserve(a_vec_str.size());

            for (int i = 0; i < a_vec_str.size(); ++i) {
                if (a_vec_str[i].size() > 1)
                    std::cerr << "Error" << std::endl;
                if constexpr(std::is_same_v<Container, std::vector<char>>)
                    o_array_char.emplace_back(a_vec_str[i][0]);
                else
                    o_array_char[i] = a_vec_str[i][0];
            }
            return o_array_char;
        }
    };
}


