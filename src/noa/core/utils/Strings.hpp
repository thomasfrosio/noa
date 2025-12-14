#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

// Suppress fmt warnings...
#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wsign-conversion"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wformat-nonliteral"
#   pragma GCC diagnostic ignored "-Wtautological-compare"
#   if defined(NOA_COMPILER_GCC)
#       pragma GCC diagnostic ignored "-Wstringop-overflow"
#   endif
#elif defined(NOA_COMPILER_MSVC)
#   pragma warning(push, 0)
#endif

#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#   pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
#   pragma warning(pop)
#endif

#include <algorithm>
#include <charconv>
#include <complex>
#include <string>
#include <string_view>
#include <typeinfo>

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(string);
    template<> struct proclaim_is_string<std::string> : std::true_type {};
    template<> struct proclaim_is_string<std::string_view> : std::true_type {};
}

namespace noa::details {
    // Left trims str.
    [[nodiscard]] inline std::string_view trim_left(std::string_view str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        const char* start = std::find_if(str.begin(), str.end(), is_not_space);
        return std::string_view{start, static_cast<usize>(str.end() - start)};
    }

    // Right trims str.
    [[nodiscard]] inline std::string_view trim_right(std::string_view str) {
        auto is_not_space = [](int ch) { return !std::isspace(ch); };
        const char* end = std::find_if(str.rbegin(), str.rend(), is_not_space).base();
        return std::string_view{str.begin(), static_cast<usize>(end - str.begin())};
    }

    // Trims (left and right) str.
    [[nodiscard]] inline std::string_view trim(std::string_view str) {
        return trim_left(trim_right(str));
    }

    // Converts the string str, in-place, to lowercase.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline void to_lower_(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
    }

    // Returns the lowercase version of str.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline std::string to_lower(std::string_view str) {
        std::string out(str);
        std::transform(str.begin(), str.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
        return out;
    }

    // Converts the string str, in-place, to uppercase.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline void to_upper_(std::string& str) {
        std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::toupper(c); });
    }

    // Returns the uppercase version of str.
    // Undefined behavior if the characters are neither representable as unsigned char nor equal to EOF.
    inline std::string to_upper(std::string_view str) {
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

    inline std::string_view offset_by(std::string_view str, usize count) {
        const usize offset = std::min(str.size(), count);
        return {str.data() + offset, str.size() - offset};
    }

    /// Entrypoint to the stringify<T> function.
    template<typename T>
    struct Stringify {
        static auto get() -> std::string {
            if constexpr (nt::is_almost_same_v<float, T>) {
                return "f32";
            } else if constexpr (nt::is_almost_same_v<double, T>) {
                return "f64";
            } else if constexpr (nt::is_almost_same_v<uint8_t, T>) {
                return "u8";
            } else if constexpr (nt::is_almost_same_v<unsigned short, T>) {
                return "u16";
            } else if constexpr (nt::is_almost_same_v<unsigned int, T>) {
                return "u32";
            } else if constexpr (nt::is_almost_same_v<unsigned long, T>) {
                return sizeof(unsigned long) == 4 ? "u32" : "u64";
            } else if constexpr (nt::is_almost_same_v<unsigned long long, T>) {
                return "u64";
            } else if constexpr (nt::is_almost_same_v<int8_t, T>) {
                return "i8";
            } else if constexpr (nt::is_almost_same_v<short, T>) {
                return "i16";
            } else if constexpr (nt::is_almost_same_v<int, T>) {
                return "i32";
            } else if constexpr (nt::is_almost_same_v<long, T>) {
                return sizeof(unsigned long) == 4 ? "i32" : "i64";
            } else if constexpr (nt::is_almost_same_v<long long, T>) {
                return "i64";
            } else if constexpr (nt::is_boolean_v<T>) {
                return "bool";
            } else if constexpr (nt::is_almost_same_v<char, T>) {
                return "char";
            } else if constexpr (nt::is_almost_same_v<unsigned char, T>) {
                return "uchar";
            } else if constexpr (nt::is_almost_same_v<std::byte, T>) {
                return "byte";
            } else if constexpr (nt::is_almost_same_v<std::complex<float>, T>) {
                return "std::complex<f32>";
            } else if constexpr (nt::is_almost_same_v<std::complex<double>, T>) {
                return "std::complex<f64>";
            } else {
                return typeid(T).name(); // implementation defined, no guarantee to be human-readable.
            }
        }
    };

    /// Gets a human-readable type name. Other types can then add their specializations via Stringify.
    template<typename T>
    auto stringify() -> std::string {
        return Stringify<T>::get();
    }

    /// Parses a string into a T.
    /// \tparam T   integer: similar to from_chars (+ plus-sign support) with base=10, \p fmt is ignored.
    ///             bool: same as integer, plus recognizes true={"y", "yes", "true"} and false={"n", "no", "false"}
    ///                   as valid matches (ignoring trailing whitespaces and case-insensitive).
    ///             floating-point: use std::strto(f|d).
    ///             std::string: remove trailing whitespaces and convert to lowercase.
    template<typename T>
    auto parse(std::string_view string) noexcept -> std::optional<T> {
        if constexpr (std::is_same_v<T, bool>) {
            auto equal_case_insensitive = [](std::string_view lhs, std::string_view rhs) {
                return lhs.size() == rhs.size() and
                       std::equal(lhs.begin(), lhs.end(), rhs.begin(), [](char a, char b) {
                           return std::tolower(static_cast<unsigned char>(a)) ==
                                  std::tolower(static_cast<unsigned char>(b));
                       });
            };
            string = trim(string);
            for (std::string_view match: {"1", "y", "yes", "true"})
                if (equal_case_insensitive(string, match))
                    return true;
            for (std::string_view match: {"0", "n", "no", "false"})
                if (equal_case_insensitive(string, match))
                    return false;
            return std::nullopt;

        } else if constexpr (std::is_integral_v<T>) {
            string = trim_left(string);
            T output{};
            const bool has_plus = string.size() > 1 and string[0] == '+';
            if (std::from_chars(string.begin() + has_plus, string.end(), output).ec == std::errc{})
                return output;
            return std::nullopt;

        } else if constexpr (nt::is_real_v<T>) {
            // std::from_chars for floating-point isn't available in libc++?
            T output{};
            char* ending;
            errno = 0; // check for ERANGE
            if constexpr (nt::is_almost_same_v<T, f64>) {
                output = std::strtod(string.data(), &ending);
            } else {
                output = static_cast<T>(std::strtof(string.data(), &ending));
            }
            if (errno != 0 or string.data() == ending)
                return std::nullopt;
            return output;

        } else if constexpr (std::is_same_v<T, std::string>) {
            return to_lower(trim(string));

        } else {
            static_assert(nt::always_false<T>);
        }
    }
}
