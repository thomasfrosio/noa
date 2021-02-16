/**
 * @file string/Format.h
 * @brief Formatting string related functions.
 * @author Thomas - ffyr2w
 * @date 10 Jan 2021
 */
#pragma once

#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include "noa/util/traits/BaseTypes.h"

namespace Noa::String {
    /** Left trim. */
    inline std::string& leftTrim(std::string& str) {
        str.erase(str.begin(),
                  std::find_if(str.begin(),
                               str.end(),
                               [](int ch) { return !std::isspace(ch); }));
        return str;
    }

    [[nodiscard]] inline std::string leftTrim(std::string&& str) {
        leftTrim(str);
        return std::move(str);
    }

    [[nodiscard]] inline std::string leftTrimCopy(std::string str) {
        leftTrim(str);
        return str;
    }

    /** Right trim. */
    inline std::string& rightTrim(std::string& str) {
        str.erase(std::find_if(str.rbegin(), str.rend(), [](int ch) { return !std::isspace(ch); }).base(),
                  str.end());
        return str;
    }

    [[nodiscard]] inline std::string rightTrim(std::string&& str) {
        rightTrim(str);
        return std::move(str);
    }

    [[nodiscard]] inline std::string rightTrimCopy(std::string str) {
        rightTrim(str);
        return str;
    }

    /** Trim (left and right). */
    inline std::string& trim(std::string& str) { return leftTrim(rightTrim(str)); }

    [[nodiscard]] inline std::string trim(std::string&& str) {
        leftTrim(rightTrim(str));
        return std::move(str);
    }

    [[nodiscard]] inline std::string trimCopy(std::string str) {
        leftTrim(rightTrim(str));
        return str;
    }

    /**
     * Convert the string @c str to lowercase.
     * @note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
     */
    inline std::string& toLower(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
        return str;
    }

    inline std::string toLower(std::string&& str) { return std::move(toLower(str)); }
    inline std::string toLowerCopy(std::string str) { return toLower(str); }
    inline std::string toLowerCopy(std::string_view str) { return toLower(std::string(str)); }

    /**
     * Convert the string @c str to uppercase.
     * @note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
     */
    inline std::string& toUpper(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
        return str;
    }

    inline std::string toUpper(std::string&& str) { return std::move(toUpper(str)); }
    inline std::string toUpperCopy(std::string str) { return toUpper(str); }
    inline std::string toUpperCopy(std::string_view str) { return toUpper(std::string(str)); }

    /**
     * Formats a string, using {fmt}.
     * @note This is mostly to NOT have fmt:: everywhere in the code base, given that in the future we might switch to std::format.
     */
    template<typename... Args>
    inline std::string format(Args&& ...args) { return fmt::format(std::forward<Args>(args)...); }

    /** Gets an human-readable type name. Other types can then add specializations. */
    template<typename T>
    NOA_IH const char* typeName() {
        if constexpr (Noa::Traits::is_same_v<float, T>) {
            return "float";
        } else if constexpr (Noa::Traits::is_same_v<double, T>) {
            return "double";

        } else if constexpr (Noa::Traits::is_same_v<uint8_t, T>) {
            return "uint8";
        } else if constexpr (Noa::Traits::is_same_v<uint16_t, T>) {
            return "uint16";
        } else if constexpr (Noa::Traits::is_same_v<uint32_t, T>) {
            return "uint32";
        } else if constexpr (Noa::Traits::is_same_v<uint64_t, T>) {
            return "uint64";
        } else if constexpr (Noa::Traits::is_same_v<int8_t, T>) {
            return "int8";
        } else if constexpr (Noa::Traits::is_same_v<int16_t, T>) {
            return "int16";
        } else if constexpr (Noa::Traits::is_same_v<int32_t, T>) {
            return "int32";
        } else if constexpr (Noa::Traits::is_same_v<int64_t, T>) {
            return "int64";

        } else if constexpr (Noa::Traits::is_bool_v<T>) {
            return "bool";

        } else if constexpr (Noa::Traits::is_same_v<char, T>) {
            return "char";
        } else if constexpr (Noa::Traits::is_same_v<unsigned char, T>) {
            return "uchar";
        } else if constexpr (Noa::Traits::is_same_v<std::byte, T>) {
            return "byte (uchar)";

        } else if constexpr (Noa::Traits::is_same_v<std::complex<float>, T>) {
            return "std::complex64";
        } else if constexpr (Noa::Traits::is_same_v<std::complex<double>, T>) {
            return "std::complex128";

        } else {
            return typeid(T).name(); // implementation defined, no guarantee to be human-readable.
        }
    }
}
