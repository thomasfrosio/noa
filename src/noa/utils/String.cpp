/**
 * @file String-inl.h
 * @brief inline header (.cpp-like) of String.h
 * @author Thomas - ffyr2w
 * @date 02 Sep 2020
 */


#include "String.h"


namespace Noa::String {

    inline std::string& leftTrim(std::string& a_str) {
        a_str.erase(a_str.begin(),
                    std::find_if(a_str.begin(),
                                 a_str.end(),
                                 [](int ch) { return !std::isspace(ch); }));
        return a_str;
    }

    [[nodiscard]] inline std::string leftTrim(std::string&& a_str) {
        a_str.erase(a_str.begin(),
                    std::find_if(a_str.begin(),
                                 a_str.end(),
                                 [](int ch) { return !std::isspace(ch); }));
        return std::move(a_str);
    }

    inline std::string& rightTrim(std::string& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(),
                                 a_str.rend(),
                                 [](int ch) { return !std::isspace(ch); }).base(),
                    a_str.end());
        return a_str;
    }

    [[nodiscard]] inline std::string rightTrim(std::string&& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(),
                                 a_str.rend(),
                                 [](int ch) { return !std::isspace(ch); }).base(),
                    a_str.end());
        return std::move(a_str);
    }

    inline std::string& trim(std::string& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(),
                                 a_str.rend(),
                                 [](int ch) { return !std::isspace(ch); }).base(),
                    std::find_if(a_str.begin(),
                                 a_str.end(),
                                 [](int ch) { return !std::isspace(ch); }));
        return a_str;
    }

    [[nodiscard]] inline std::string trim(std::string&& a_str) {
        a_str.erase(std::find_if(a_str.rbegin(),
                                 a_str.rend(),
                                 [](int ch) { return !std::isspace(ch); }).base(),
                    std::find_if(a_str.begin(),
                                 a_str.end(),
                                 [](int ch) { return !std::isspace(ch); }));
        return std::move(a_str);
    }

    inline std::vector<std::string>& trim(std::vector<std::string>& a_vec_str) {
        for (auto& str : a_vec_str)
            ::Noa::String::trim(str);
        return a_vec_str;
    }

    inline int toInt(const std::string& a_str) {
        try {
            return std::stoi(a_str);
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("\"{}\" is out of the int range", a_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("\"{}\" cannot be converted into an int", a_str);
        }
    }

    inline float toFloat(const std::string& a_str) {
        try {
            return std::stof(a_str);
        } catch (const std::out_of_range& e) {
            NOA_CORE_ERROR("\"{}\" is out of the float range", a_str);
        } catch (const std::invalid_argument& e) {
            NOA_CORE_ERROR("\"{}\" cannot be converted into a float", a_str);
        }
    }
}
