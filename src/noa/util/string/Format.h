/**
 * @file string/Format.h
 * @brief String formatting related functions.
 * @author Thomas - ffyr2w
 * @date 10 Jan 2021
 */
#pragma once

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/bundled/compile.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/bundled/os.h>
#include <spdlog/fmt/bundled/chrono.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/fmt/bundled/ostream.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>

#include "noa/util/traits/BaseTypes.h"

namespace Noa::String {
    /// Left trim.
    NOA_IH std::string& leftTrim(std::string& str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        str.erase(str.begin(), std::find_if(str.begin(), str.end(), is_not_space));
        return str;
    }

    [[nodiscard]] NOA_IH std::string leftTrim(std::string&& str) {
        leftTrim(str);
        return std::move(str);
    }

    [[nodiscard]] NOA_IH std::string leftTrimCopy(std::string str) {
        leftTrim(str);
        return str;
    }

    /// Right trim.
    NOA_IH std::string& rightTrim(std::string& str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        str.erase(std::find_if(str.rbegin(), str.rend(), is_not_space).base(), str.end());
        return str;
    }

    [[nodiscard]] NOA_IH std::string rightTrim(std::string&& str) {
        rightTrim(str);
        return std::move(str);
    }

    [[nodiscard]] NOA_IH std::string rightTrimCopy(std::string str) {
        rightTrim(str);
        return str;
    }

    /// Trim (left and right).
    NOA_IH std::string& trim(std::string& str) { return leftTrim(rightTrim(str)); }

    [[nodiscard]] NOA_IH std::string trim(std::string&& str) {
        leftTrim(rightTrim(str));
        return std::move(str);
    }

    [[nodiscard]] NOA_IH std::string trimCopy(std::string str) {
        leftTrim(rightTrim(str));
        return str;
    }

    /// Convert the string @c str to lowercase.
    /// @note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    NOA_IH std::string& toLower(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
        return str;
    }

    NOA_IH std::string toLower(std::string&& str) { return std::move(toLower(str)); }
    NOA_IH std::string toLowerCopy(std::string str) { return toLower(str); }
    NOA_IH std::string toLowerCopy(std::string_view str) { return toLower(std::string(str)); }

    /// Convert the string @c str to uppercase.
    /// @note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    NOA_IH std::string& toUpper(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
        return str;
    }

    NOA_IH std::string toUpper(std::string&& str) { return std::move(toUpper(str)); }
    NOA_IH std::string toUpperCopy(std::string str) { return toUpper(str); }
    NOA_IH std::string toUpperCopy(std::string_view str) { return toUpper(std::string(str)); }

    /// Formats a string, using {fmt}.
    /// @note This is mostly to not have fmt:: everywhere in the code base, given that in the future we might switch to std::format.
    template<typename... Args>
    NOA_IH std::string format(Args&& ...args) { return fmt::format(std::forward<Args>(args)...); }

    /// Gets an human-readable type name. Other types can then add specializations.
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
            return "byte";

        } else if constexpr (Noa::Traits::is_same_v<std::complex<float>, T>) {
            return "std::complex64";
        } else if constexpr (Noa::Traits::is_same_v<std::complex<double>, T>) {
            return "std::complex128";

        } else {
            return typeid(T).name(); // implementation defined, no guarantee to be human-readable.
        }
    }
}
