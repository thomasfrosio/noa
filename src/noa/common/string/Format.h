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
    [[nodiscard]] NOA_IH std::string_view leftTrim(std::string_view str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        const char* start = std::find_if(str.begin(), str.end(), is_not_space);
        return std::string_view{start, static_cast<size_t>(str.end() - start)};
    }

    /// Right trim.
    [[nodiscard]] NOA_IH std::string_view rightTrim(std::string_view str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        const char* end = std::find_if(str.rbegin(), str.rend(), is_not_space).base();
        return std::string_view{str.begin(), static_cast<size_t>(end - str.begin())};
    }

    /// Trim (left and right).
    [[nodiscard]] NOA_IH std::string_view trim(std::string_view str) {
        return leftTrim(rightTrim(str));
    }

    /// Converts the string \c str, in-place, to lowercase.
    /// \note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    NOA_IH std::string lower(std::string&& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
        return str;
    }

    /// Converts to lowercase.
    NOA_IH std::string lower(std::string_view str) {
        return lower(std::string(str));
    }

    /// Converts the string \p str, in-place, to uppercase.
    /// \note Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    NOA_IH std::string& upper(std::string&& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
        return str;
    }

    /// Converts to uppercase.
    NOA_IH std::string upper(std::string_view str) {
        return upper(std::string(str));
    }

    /// Reverse a string, in-place.
    NOA_IH std::string reverse(std::string&& str) {
        std::reverse(str.begin(), str.end());
        return str;
    }

    /// Reverse a string, out-of-place.
    NOA_IH std::string reverse(std::string_view str) {
        return reverse(std::string(str));
    }

    NOA_IH bool startsWith(std::string_view str, std::string_view prefix) {
        return str.rfind(prefix, 0) == 0;
    }

    /// Formats a string, using {fmt}. Equivalent to fmt::format().
    template<typename... Args>
    NOA_IH std::string format(Args&& ...args) { return fmt::format(std::forward<Args>(args)...); }

    /// Gets an human-readable type name. Other types can then add their specializations.
    template<typename T>
    NOA_IH std::string human() {
        if constexpr (noa::traits::is_same_v<float, T>) {
            return "float";
        } else if constexpr (noa::traits::is_same_v<double, T>) {
            return "double";

        } else if constexpr (noa::traits::is_same_v<uint8_t, T>) {
            return "uint8";
        } else if constexpr (noa::traits::is_same_v<unsigned short, T>) {
            return "uint16";
        } else if constexpr (noa::traits::is_same_v<unsigned int, T>) {
            return "uint32";
        } else if constexpr (noa::traits::is_same_v<unsigned long, T>) {
            return sizeof(unsigned long) == 4 ? "uint32" : "uint64";
        } else if constexpr (noa::traits::is_same_v<unsigned long long, T>) {
            return "uint64";
        } else if constexpr (noa::traits::is_same_v<int8_t, T>) {
            return "int8";
        } else if constexpr (noa::traits::is_same_v<short, T>) {
            return "int16";
        } else if constexpr (noa::traits::is_same_v<int, T>) {
            return "int32";
        } else if constexpr (noa::traits::is_same_v<long, T>) {
            return sizeof(unsigned long) == 4 ? "int32" : "int64";
        } else if constexpr (noa::traits::is_same_v<long long, T>) {
            return "int64";

        } else if constexpr (noa::traits::is_bool_v<T>) {
            return "bool";
        } else if constexpr (noa::traits::is_same_v<char, T>) {
            return "char";
        } else if constexpr (noa::traits::is_same_v<unsigned char, T>) {
            return "uchar";
        } else if constexpr (noa::traits::is_same_v<std::byte, T>) {
            return "std::byte";

        } else if constexpr (noa::traits::is_same_v<std::complex<float>, T>) {
            return "std::complex<float>";
        } else if constexpr (noa::traits::is_same_v<std::complex<double>, T>) {
            return "std::complex<double>";

        } else if constexpr (noa::traits::is_std_vector_v<T>) {
            return format("std::vector<{}>", human<T::value_type>());
        } else if constexpr (noa::traits::is_std_array_v<T>) {
            return format("std::array<{},{}>", human<T::value_type>(), std::tuple_size_v<T>);

        } else {
            return typeid(T).name(); // implementation defined, no guarantee to be human-readable.
        }
    }
}
