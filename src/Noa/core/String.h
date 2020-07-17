//
// Created by thomas on 15/07/2020.
//

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>

namespace Noa {

    // For now, just a static class to store methods related to strings
    class String {

    public:
        String() = delete;

        // Left trim a string (lvalue).
        static inline std::string& leftTrim(std::string& a_str) {
            a_str.erase(a_str.begin(), std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
            return a_str;
        };

        // Left trim a string (rvalue).
        static inline std::string leftTrim(std::string&& a_str) {
            a_str.erase(a_str.begin(), std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
            return std::move(a_str);
        };

        // Right trim a string (lvalue).
        static inline std::string& rightTrim(std::string& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(), a_str.end());
            return a_str;
        };

        // Right trim a string (take rvalue reference, trim, return rvalue).
        static inline std::string rightTrim(std::string&& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(), a_str.end());
            return std::move(a_str);
        };

        // Strip (left and right trim) a string (lvalue).
        static inline std::string& strip(std::string& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                        std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
            return a_str;
        };

        // Strip (left and right trim) a string (rvalue).
        static inline std::string strip(std::string&& a_str) {
            a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                        std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
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
                String::strip(str);
            return std::move(a_vec_str);
        };

        /**
         * @short               Split a string or string_view using std::string(_view)::find.
         *
         * @tparam BasicString  a_str should be std::string or std::string_view.
         * @param a_str         String to split.
         * @param a_delimiter   Character to use as delimiter.
         * @param o_vector      Output vector to store the strings into.
         *                      The delimiter is stripped from the returned strings.
         * @return o_vector     Return by reference to allow method chaining.
         *
         * @example             split("1. 2.3.", '.', out); out = {"1", " 2", "3", ""}
         */
        template<typename BasicString>
        static std::vector<std::string>& splitFind(const BasicString& a_str, const char a_delimiter, std::vector<std::string>& o_vector) {
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

        template<typename BasicString>
        static std::vector<std::string> splitFind(const BasicString& a_str, const char a_delimiter) {
            std::vector<std::string> o_vector;
            splitFind(a_str, a_delimiter, o_vector);
            return o_vector;
        }


        /**
         * @short               Split a string or string_view using std::string(_view)::find_first_of.
         *
         * @tparam BasicString  a_str can be std::string or std::string_view
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
        template<typename BasicString>
        std::vector<std::string>& splitFindFirstOf(const BasicString& a_str, const char* a_first_of, std::vector<std::string>& o_vector) {
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

        template<typename BasicString>
        std::vector<std::string> splitFindFirstOf(const BasicString& a_str, const char* a_first_of) {
            std::vector<std::string> o_vector;
            splitFindFirstOf(a_str, a_first_of, o_vector);
            return o_vector;
        }


        /**
         * @short           Convert a string into an int with std::stoi.
         *
         * @param a_str     String to convert into an integer.
         * @return          Integer resulting from the conversion.
         */
        static int toOneInt(const std::string& a_str) {
            try {
                return std::stoi(a_str);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid Argument" << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range" << std::endl;
            }
        };

        /**
         * @short               Convert a vector of string(s) into integer(s) with std::stoi.
         *
         * @tparam Container    STL Containers, e.g. vector, array, etc.
         * @param a_vec_str     Vector containing at least one string. Only the first
         *                      element is converted into an integer
         * @return              Integer(s) resulting from the conversion. They are stored
         *                      int the Container which has a size equal to the size of
         *                      the input vector a_vec_str.
         */
        template<typename Container>
        static Container toMultipleInt(const std::vector<std::string>& a_vec_str) {
            Container array_int{};
            try {
                unsigned int i = 0;
                for (auto& str : a_vec_str) {
                    array_int[i] = std::stoi(str);
                    i++;
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "Invalid Argument" << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "Out of range" << std::endl;
            }
            return array_int;
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
         */
        static bool toOneBool(const std::string& a_str) {

        };

        /**
         *
         * @tparam N
         * @param a_vec_str
         * @return
         */
        template<typename Container>
        static Container toMultipleBool(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @param a_vec_str
         * @return
         */
        static std::string toOneString(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @tparam N
         * @param a_vec_str
         * @return
         */
        template<typename T>
        T toMultipleString(const std::vector<std::string>* a_vec_str) {

        };

    };
}


