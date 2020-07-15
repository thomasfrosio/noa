//
// Created by thomas on 15/07/2020.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <utility>

namespace Noa {

    // For now, just a static class to store methods related to strings
    class String {

    public:
        String() = delete;

        static inline std::string& leftTrim(std::string& a_str);    // Left trim a string (lvalue).
        static inline std::string leftTrim(std::string&& a_str);    // Left trim a string (rvalue).

        static inline std::string& rightTrim(std::string& a_str);   // Right trim a string (lvalue).
        static inline std::string rightTrim(std::string&& a_str);   // Right trim a string (rvalue).

        static inline std::string& strip(std::string& a_str);       // Strip (left and right trim) a string (lvalue).
        static inline std::string strip(std::string&& a_str);       // Strip (left and right trim) a string (rvalue).

        static std::vector<std::string>& strip(std::vector<std::string>& a_vec_str);    // Strip all the strings of a vector (lvalue).
        static std::vector<std::string> strip(std::vector<std::string>&& a_vec_str);    // Strip all the strings of a vector (rvalue).

        /**
         * @short               Split a string or string_view using std::string(_view)::find.
         *
         * @tparam T            a_str can be std::string or std::string_view.
         * @param a_str         String to split.
         * @param a_delimiter   Character to use as delimiter.
         * @param o_vector      Output vector to store the strings into.
         *                      The delimiter is stripped from the returned strings.
         * @return o_vector     Return by reference to allow method chaining.
         *
         * @example             split("1. 2.3.", '.', out); out = {"1", " 2", "3", ""}
         */
        template<typename T>
        static std::vector<std::string>& splitFind(const T& a_str, const char a_delimiter, std::vector<std::string>& o_vector) {
            // Prepare indexes.
            std::size_t current, previous = 0;

            // Pre-allocate, which could save a few copies. Not sure it is worth it.
            o_vector.reserve(std::count(a_str.begin(), a_str.end(), a_delimiter));

            // Look for a_delimiter until end of a_str.
            current = a_str.find(a_delimiter);
            while (current != std::string::npos) {
                o_vector.emplace_back(a_str.substr(previous, current - previous));
                previous = current + 1;
                current = a_str.find(a_delimiter, previous);
            }
            o_vector.emplace_back(a_str.substr(previous, current - previous));
            return o_vector;
        };

        template<typename T>
        static std::vector<std::string> splitFind(const T& a_str, const char a_delimiter) {
            std::vector<std::string> o_vector;
            splitFind(a_str, a_delimiter, o_vector);
            return o_vector;
        }


        /**
         * @short               Split a string or string_view using std::string(_view)::find_first_of.
         *
         * @tparam T            a_str can be std::string or std::string_view
         * @param a_str         String to split.
         * @param a_first_of    Array of characters to use as delimiter. If it is a string literal
         *                      of one character, this function is equivalent to String::splitFind()
         * @param o_vector      Output vector to store the strings into.
         *                      The delimiter is stripped from the returned strings.
         * @return o_vector     Return by reference to allow method chaining.
         *
         * @example             split("1. 2.3.", ".",  out); out = {"1",    " 2", "3", ""}
         * @example             split("1. 2.3.", " .", out); out = {"1", "", "2", "3", ""}
         */
        template<typename T>
        std::vector<std::string>& splitFindFirstOf(const T& a_str, const char* a_first_of, std::vector<std::string>& o_vector) {
            // prepare indexes
            std::size_t current, previous = 0, reserve = 0;

            // Pre-allocate, which could save a few copies. Not sure it is worth it.
            for (int i = 0; a_first_of[i] != 0; ++i)
                reserve += std::count(a_str.begin(), a_str.end(), i);
            o_vector.reserve(reserve);

            // look for a_char until end of a_str
            current = a_str.find_first_of(a_first_of);
            while (current != std::string::npos) {
                o_vector.emplace_back(a_str.substr(previous, current - previous));
                previous = current + 1;
                current = a_str.find_first_of(a_first_of, previous);
            }
            o_vector.emplace_back(a_str.substr(previous, current - previous));
        }

        template<typename T>
        std::vector<std::string> splitFindFirstOf(const T& a_str, const char* a_first_of) {
            std::vector<std::string> o_vector;
            splitFindFirstOf(a_str, a_first_of, o_vector);
            return o_vector;
        }
    };
}


