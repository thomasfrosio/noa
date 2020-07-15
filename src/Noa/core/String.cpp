//
// Created by thomas on 15/07/2020.
//

#include "String.h"

namespace Noa {

    // Left trim a string (lvalue).
    inline std::string& String::leftTrim(std::string& a_str) {
        a_str.erase(a_str.begin(), std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
    }

    // Left trim a string (rvalue).
    inline std::string String::leftTrim(std::string&& a_str) {
        a_str.erase(a_str.begin(), std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
    }

    // Right trim a string (lvalue).
    inline std::string& String::rightTrim(std::string& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(), a_str.end());
    }

    // Right trim a string (rvalue).
    inline std::string String::rightTrim(std::string&& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(), a_str.end());
        return a_str;
    }

    // Strip (left and right trim) a string (lvalue).
    inline std::string& String::strip(std::string& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
    }

    // Strip (left and right trim) a string (rvalue).
    inline std::string String::strip(std::string&& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(), a_str.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                    std::find_if(a_str.begin(), a_str.end(), [](int ch) { return !std::isspace(ch); }));
    }

    // Strip all the strings of a vector (lvalue).
    std::vector<std::string>& String::strip(std::vector<std::string>& a_vec_str) {
        for (auto& str : a_vec_str)
            String::strip(str);
        return a_vec_str;
    }

    // Strip all the strings of a vector (rvalue).
    std::vector<std::string> String::strip(std::vector<std::string>&& a_vec_str) {
        for (auto& str : a_vec_str)
            String::strip(str);
        return a_vec_str;
    }


}

