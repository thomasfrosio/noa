/// \file noa/common/string/Format.h
/// \brief String formatting related functions.
/// \author Thomas - ffyr2w
/// \date 10 Jan 2021

#pragma once

#include "noa/common/Definitions.h"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wformat-nonliteral"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/bundled/compile.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/bundled/os.h>
#include <spdlog/fmt/bundled/chrono.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/fmt/bundled/ostream.h>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <typeinfo>
#include <builtin_types.h>

#include "noa/common/Traits.h"

namespace noa::string {
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

    /// Convert the string \c str to lowercase.
    /// \note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    NOA_IH std::string& toLower(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
        return str;
    }

    NOA_IH std::string toLower(std::string&& str) { return std::move(toLower(str)); }
    NOA_IH std::string toLowerCopy(std::string str) { return toLower(str); }
    NOA_IH std::string toLowerCopy(std::string_view str) { return toLower(std::string(str)); }

    /// Convert the string \c str to uppercase.
    /// \note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    NOA_IH std::string& toUpper(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
        return str;
    }

    NOA_IH std::string toUpper(std::string&& str) { return std::move(toUpper(str)); }
    NOA_IH std::string toUpperCopy(std::string str) { return toUpper(str); }
    NOA_IH std::string toUpperCopy(std::string_view str) { return toUpper(std::string(str)); }

    /// Formats a string, using {fmt}. Equivalent to fmt::format().
    template<typename... Args>
    NOA_IH std::string format(Args&& ...args) { return fmt::format(std::forward<Args>(args)...); }

    /// Gets an human-readable type name. Other types can then add specializations.
    template<typename T>
    NOA_IH std::string typeName() {
        std::string out;
        if constexpr (noa::traits::is_same_v<float, T>) {
            out = "float";
        } else if constexpr (noa::traits::is_same_v<double, T>) {
            out = "double";

        } else if constexpr (noa::traits::is_same_v<uint8_t, T>) {
            out = "uint8";
        } else if constexpr (noa::traits::is_same_v<unsigned short, T>) {
            out = "ushort";
        } else if constexpr (noa::traits::is_same_v<unsigned int, T>) {
            out = "uint";
        } else if constexpr (noa::traits::is_same_v<unsigned long, T>) {
            out = "ulong";
        } else if constexpr (noa::traits::is_same_v<unsigned long long, T>) {
            out = "ulonglong";
        } else if constexpr (noa::traits::is_same_v<int8_t, T>) {
            out = "int8";
        } else if constexpr (noa::traits::is_same_v<short, T>) {
            out = "short";
        } else if constexpr (noa::traits::is_same_v<int, T>) {
            out = "int";
        } else if constexpr (noa::traits::is_same_v<long, T>) {
            out = "long";
        } else if constexpr (noa::traits::is_same_v<long long, T>) {
            out = "longlong";

        } else if constexpr (noa::traits::is_bool_v<T>) {
            out = "bool";
        } else if constexpr (noa::traits::is_same_v<char, T>) {
            out = "char";
        } else if constexpr (noa::traits::is_same_v<unsigned char, T>) {
            out = "uchar";
        } else if constexpr (noa::traits::is_same_v<std::byte, T>) {
            out = "byte";

        } else if constexpr (noa::traits::is_same_v<std::complex<float>, T>) {
            out = "std::complex<float>";
        } else if constexpr (noa::traits::is_same_v<std::complex<double>, T>) {
            out = "std::complex<double>";

        } else if constexpr (noa::traits::is_std_vector_v<T>) {
            out = format("std::vector<{}>", typeName<T::value_type>());
        } else if constexpr (noa::traits::is_std_array_v<T>) {
            out = format("std::array<{},{}>", typeName<T::value_type>(), std::tuple_size_v<T>);

        } else {
            out = typeid(T).name(); // implementation defined, no guarantee to be human-readable.
        }
        return out;
    }
}
