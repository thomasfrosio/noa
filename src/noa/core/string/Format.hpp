#pragma once

#include "noa/core/Definitions.hpp"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wformat-nonliteral"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#else
#include <spdlog/fmt/fmt.h>
#include <spdlog/fmt/bundled/chrono.h>
#include <spdlog/fmt/bundled/color.h>
#include <spdlog/fmt/bundled/compile.h>
#include <spdlog/fmt/bundled/os.h>
#include <spdlog/fmt/bundled/ostream.h>
#include <spdlog/fmt/bundled/ranges.h>
#endif

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

#include "noa/core/Traits.hpp"

namespace noa::string {
    // Left trims str.
    [[nodiscard]] inline std::string_view trim_left(std::string_view str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        const char* start = std::find_if(str.begin(), str.end(), is_not_space);
        return std::string_view{start, static_cast<size_t>(str.end() - start)};
    }

    // Right trims str.
    [[nodiscard]] inline std::string_view trim_right(std::string_view str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        const char* end = std::find_if(str.rbegin(), str.rend(), is_not_space).base();
        return std::string_view{str.begin(), static_cast<size_t>(end - str.begin())};
    }

    // Trims (left and right) str.
    [[nodiscard]] inline std::string_view trim(std::string_view str) {
        return trim_left(trim_right(str));
    }

    // Converts the string str, in-place, to lowercase.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline void lower_(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    // Returns the lowercase version of str.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline std::string lower(std::string_view str) {
        std::string out(str);
        std::transform(str.begin(), str.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
        return out;
    }

    // Converts the string str, in-place, to uppercase.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline void upper_(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
    }

    // Returns the uppercase version of str.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline std::string upper(std::string_view str) {
        std::string out(str);
        std::transform(str.begin(), str.end(), out.begin(), [](unsigned char c) { return std::toupper(c); });
        return out;
    }

    // Reverses a string, in-place.
    inline void reverse_(std::string& str) {
        std::reverse(str.begin(), str.end());
    }

    // Returns the reverse of str.
    inline std::string reverse(std::string_view str) {
        std::string out(str);
        reverse_(out);
        return out;
    }

    // Whether str starts with prefix.
    inline bool starts_with(std::string_view str, std::string_view prefix) {
        return str.rfind(prefix, 0) == 0;
    }

    // Formats a string, using {fmt}. Equivalent to ::fmt::format().
    template<typename... Args>
    inline std::string format(Args&& ...args) { return ::fmt::format(std::forward<Args>(args)...); }

    // Gets a human-readable type name. Other types can then add their specializations.
    template<typename T>
    inline std::string human() {
        if constexpr (traits::is_almost_same_v<float, T>) {
            return "f32";
        } else if constexpr (traits::is_almost_same_v<double, T>) {
            return "f64";

        } else if constexpr (traits::is_almost_same_v<uint8_t, T>) {
            return "u8";
        } else if constexpr (traits::is_almost_same_v<unsigned short, T>) {
            return "u16";
        } else if constexpr (traits::is_almost_same_v<unsigned int, T>) {
            return "u32";
        } else if constexpr (traits::is_almost_same_v<unsigned long, T>) {
            return sizeof(unsigned long) == 4 ? "u32" : "u64";
        } else if constexpr (traits::is_almost_same_v<unsigned long long, T>) {
            return "u64";
        } else if constexpr (traits::is_almost_same_v<int8_t, T>) {
            return "i8";
        } else if constexpr (traits::is_almost_same_v<short, T>) {
            return "i16";
        } else if constexpr (traits::is_almost_same_v<int, T>) {
            return "i32";
        } else if constexpr (traits::is_almost_same_v<long, T>) {
            return sizeof(unsigned long) == 4 ? "i32" : "i64";
        } else if constexpr (traits::is_almost_same_v<long long, T>) {
            return "i64";

        } else if constexpr (traits::is_bool_v<T>) {
            return "bool";
        } else if constexpr (traits::is_almost_same_v<char, T>) {
            return "char";
        } else if constexpr (traits::is_almost_same_v<unsigned char, T>) {
            return "uchar";
        } else if constexpr (traits::is_almost_same_v<std::byte, T>) {
            return "byte";

        } else if constexpr (traits::is_almost_same_v<std::complex<float>, T>) {
            return "std::complex<f32>";
        } else if constexpr (traits::is_almost_same_v<std::complex<double>, T>) {
            return "std::complex<f64>";

        } else if constexpr (traits::has_name_v<T>) {
            return T::name();
        } else {
            return typeid(T).name(); // implementation defined, no guarantee to be human-readable.
        }
        return ""; // unreachable
    }
}
