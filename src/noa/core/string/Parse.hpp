#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/string/Format.hpp"

#ifdef NOA_IS_OFFLINE
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

namespace noa::string {
    namespace guts {
        template<typename T>
        constexpr bool is_parsable_from_string_v = std::bool_constant<nt::is_string_v<T> || nt::is_numeric_v<T>>::value;
    }

    // Returns an error message given an error value.
    template<typename T, typename nt::enable_if_bool_t<guts::is_parsable_from_string_v<T>> = true>
    std::string parse_error_message(const std::string& str, int error) {
        switch (error) {
            case 1:
                return fmt::format("Failed to convert \"{}\" to {}", str, ns::to_human_readable<T>());
            case 2:
                return fmt::format("Out of range. \"{}\" is out of {} range", str, ns::to_human_readable<T>());
            case 3:
                return fmt::format("Out of range. Cannot convert negative number \"{}\" to {}", str, ns::to_human_readable<T>());
            default:
                return "";
        }
    }
}

namespace noa::string::guts {
    template<typename T = int, typename nt::enable_if_bool_t<nt::is_int_v<T>> = true>
    T to_int(const std::string& str, int& error) noexcept {
        errno = 0;
        error = 0;
        char* end{};
        T out;

        if constexpr (nt::is_uint_v<T>) {
            // Shortcut: empty string or negative number.
            size_t idx = str.find_first_not_of(" \t");
            if (idx == std::string::npos) {
                error = 1;
                return {};
            } else if (str[idx] == '-') {
                error = 3;
                return {};
            }
            if constexpr (std::is_same_v<T, unsigned long long>) {
                out = std::strtoull(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<T, unsigned long>) {
                out = std::strtoul(str.data(), &end, 10);
            } else {
                unsigned long tmp = std::strtoul(str.data(), &end, 10);
                if (tmp > std::numeric_limits<T>::max() || tmp < std::numeric_limits<T>::min()) {
                    error = 2;
                    return {};
                }
                out = static_cast<T>(tmp);
            }
        } else if constexpr (nt::is_int_v<T>) {
            if constexpr (std::is_same_v<T, long long>) {
                out = std::strtoll(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<T, long>) {
                out = std::strtol(str.data(), &end, 10);
            } else {
                long tmp = std::strtol(str.data(), &end, 10);
                if (tmp > std::numeric_limits<T>::max() || tmp < std::numeric_limits<T>::min()){
                    error = 2;
                    return {};
                }
                out = static_cast<T>(tmp);
            }
        } else {
            static_assert(nt::always_false_v<T>, "DEV: this should not be possible");
        }

        if (end == str.data())
            error = 1;
        else if (errno == ERANGE)
            error = 2;
        return out;
    }

    template<typename T = int, typename nt::enable_if_bool_t<nt::is_int_v<T>> = true>
    T to_int(const std::string& str) {
        int error{};
        T out = to_int<T>(str, error);
        check_runtime(!error, parse_error_message<T>(str, error));
        return out;
    }

    template<typename T = int, typename nt::enable_if_bool_t<nt::is_real_v<T>> = true>
    T to_real(const std::string& str, int& error) noexcept {
        errno = 0;
        error = 0;
        char* end{};
        T out;
        if constexpr (nt::is_almost_same_v<T, float>) {
            out = std::strtof(str.data(), &end);
        } else if constexpr (nt::is_almost_same_v<T, double>) {
            out = std::strtod(str.data(), &end);
        } else {
            static_assert(nt::always_false_v<T>);
        }

        if (end == str.data())
            error = 1;
        else if (errno == ERANGE)
            error = 2;
        return out;
    }

    template<typename T = int, typename nt::enable_if_bool_t<nt::is_real_v<T>> = true>
    T to_real(const std::string& str) {
        int error{};
        T out = to_real<T>(str, error);
        check_runtime(!error, parse_error_message<T>(str, error));
        return out;
    }

    inline bool to_bool(const std::string& str, int& error) noexcept {
        error = 0;
        if (str == "1" || str == "true")
            return true;
        else if (str == "0" || str == "false")
            return false;

        const std::string str_up = to_upper(str);
        if (str_up == "TRUE" || str_up == "Y" || str_up == "YES" || str_up == "ON") {
            return true;
        } else if (str_up == "FALSE" || str_up == "N" || str_up == "NO" || str_up == "OFF") {
            return false;
        } else {
            error = 1;
            return false;
        }
    }

    inline bool to_bool(const std::string& str) {
        int error{};
        bool out = to_bool(str, error);
        check_runtime(!error, parse_error_message<bool>(str, error));
        return out;
    }
}

namespace noa::string {
    // Parses a null-terminated string into a T.
    // error is set to non-zero if the parsing fails, otherwise it is set to 0.
    template<typename T, typename nt::enable_if_bool_t<guts::is_parsable_from_string_v<T>> = true>
    T parse(const std::string& string, int& error) noexcept {
        if constexpr (nt::is_string_v<T>) {
            error = 0;
            return string;
        } else if constexpr (nt::is_real_v<T>) {
            return guts::to_real<T>(string, error);
        } else if constexpr (nt::is_bool_v<T>) {
            return guts::to_bool(string, error);
        } else if constexpr (nt::is_int_v<T>) {
            return guts::to_int<T>(string, error);
        } else {
            static_assert(nt::always_false_v<T>);
        }
    }

    // Parses a null-terminated string into a T. Throws if the parsing fails.
    template<typename T, typename nt::enable_if_bool_t<guts::is_parsable_from_string_v<T>> = true>
    T parse(const std::string& string) {
        int error{};
        T out = parse<T>(string, error);
        check_runtime(!error, parse_error_message<T>(string, error));
        return out;
    }

    // Parses a vector of null-terminated strings into a vector of T.
    // error is set to non-zero if the parsing fails, otherwise it is set to 0.
    // If the parsing fails for an element, the function stops and returns the output
    // vector with the elements that were successfully parsed.
    template<typename T, typename nt::enable_if_bool_t<guts::is_parsable_from_string_v<T>> = true>
    std::vector<T> parse(const std::vector<std::string>& vector, int& error) noexcept {
        error = 0;
        if constexpr (nt::is_string_v<T>) {
            return vector;
        } else {
            std::vector<T> output;
            for (const auto& string: vector) {
                T value;
                if constexpr (nt::is_real_v<T>) {
                    value = guts::to_real<T>(string, error);
                } else if constexpr (nt::is_bool_v<T>) {
                    value = guts::to_bool(string, error);
                } else if constexpr (nt::is_int_v<T>) {
                    value = guts::to_int<T>(string, error);
                } else {
                    static_assert(nt::always_false_v<T>);
                }
                if (error)
                    break;
                output.emplace_back(value);
            }
            return output;
        }
    }

    // Parses a vector of null-terminated strings into a vector of T. Throws if the parsing fails.
    template<typename T, typename nt::enable_if_bool_t<guts::is_parsable_from_string_v<T>> = true>
    std::vector<T> parse(const std::vector<std::string>& vector) {
        int error{};
        std::vector<T> out = parse<T>(vector, error);
        check_runtime(!error, parse_error_message<T>(vector[out.size()], error));
        return out;
    }
}
#endif
