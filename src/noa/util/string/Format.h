/**
 * @file string/Format.h
 * @brief Formatting string related functions.
 * @author Thomas - ffyr2w
 * @date 10 Jan 2021
 */
#pragma once

#include <string>
#include <string_view>
#include <spdlog/fmt/fmt.h>

#include "noa/API.h"
#include "noa/util/traits/Base.h"
#include "noa/util/Errno.h"


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
        str.erase(std::find_if(str.rbegin(),
                               str.rend(),
                               [](int ch) { return !std::isspace(ch); }).base(),
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
    inline std::string& trim(std::string& str) {
        return leftTrim(rightTrim(str));
    }


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
        std::transform(str.begin(), str.end(), str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
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
        std::transform(str.begin(), str.end(), str.begin(),
                       [](unsigned char c) { return std::toupper(c); });
        return str;
    }

    inline std::string toUpper(std::string&& str) { return std::move(toUpper(str)); }

    inline std::string toUpperCopy(std::string str) { return toUpper(str); }

    inline std::string toUpperCopy(std::string_view str) { return toUpper(std::string(str)); }


    /**
     * Convert a string into an integer.
     * @tparam Int      Supported integers are: (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t.
     * @tparam S        @c std::string(_view) by lvalue or rvalue. Read only.
     * @param[in] str   String to convert into @a T.
     * @param[out] err  Status to update. Is set to:
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range
     *                  Unchanged otherwise.
     * @return          Resulting integer.
     *
     * @note            @c errno is reset to 0 before starting the conversion.
     */
    template<typename Int = int32_t, typename S = std::string_view,
            typename = std::enable_if_t<Noa::Traits::is_int_v<Int> && Noa::Traits::is_string_v<S>>>
    inline auto toInt(S&& str, Noa::Flag<Errno>& err) noexcept {
        using int_t = Noa::Traits::remove_ref_cv_t<Int>;
        errno = 0;
        char* end;
        int_t out{0};

        if constexpr (Noa::Traits::is_uint_v<int_t>) {
            size_t idx = str.find_first_not_of(" \t");
            if (idx == std::string::npos) {
                err = Errno::invalid_argument;
                return out;
            } else if (str[idx] == '-') {
                err = (str.size() >= idx + 1 && str[idx + 1] > 47 && str[idx + 1] < 58) ?
                      Errno::out_of_range : Errno::invalid_argument;
                return out;
            }
            if constexpr (std::is_same_v<int_t, uint64_t>) {
                out = std::strtoull(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<int_t, uint32_t> ||
                                 std::is_same_v<int_t, uint16_t> ||
                                 std::is_same_v<int_t, uint8_t>) {
                if constexpr (std::is_same_v<int_t, uint32_t> && std::is_same_v<long, int32_t>) {
                    out = std::strtoul(str.data(), &end, 10);
                } else /* long == uint64_t */ {
                    unsigned long tmp = std::strtoul(str.data(), &end, 10);
                    if (tmp > std::numeric_limits<int_t>::max() ||
                        tmp < std::numeric_limits<int_t>::min())
                        err = Errno::out_of_range;
                    out = static_cast<int_t>(tmp);
                }
            }
        } else if (Noa::Traits::is_int_v<int_t>) /* signed */ {
            if constexpr (std::is_same_v<int_t, int64_t>) {
                out = std::strtoll(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<int_t, int32_t> ||
                                 std::is_same_v<int_t, int16_t> ||
                                 std::is_same_v<int_t, int8_t>) {
                if constexpr (std::is_same_v<int_t, int32_t> && std::is_same_v<long, int32_t>) {
                    out = std::strtol(str.data(), &end, 10);
                } else /* long == int64_t */ {
                    long tmp = std::strtol(str.data(), &end, 10);
                    if (tmp > std::numeric_limits<int_t>::max() ||
                        tmp < std::numeric_limits<int_t>::min())
                        err = Errno::out_of_range;
                    out = static_cast<int_t>(tmp);
                }
            }
        }

        if (end == str.data())
            err = Errno::invalid_argument;
        else if (errno == ERANGE)
            err = Errno::out_of_range;
        return out;
    }


    /**
     * Convert a string into a floating point.
     * @tparam Float    Supported floating points are: float, double and long double.
     * @tparam S        @c std::string(_view) by lvalue or rvalue. Read only.
     * @param[in] str   String to convert into @a T.
     * @param[out] err  Status to update. Is set to:
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range
     *                  Unchanged otherwise.
     * @return          Resulting floating point.
     * @note            @c errno is reset to 0 before starting the conversion.
     */
    template<typename Float = float, typename S = std::string_view,
            typename = std::enable_if_t<Noa::Traits::is_float_v<Float> &&
                                        Noa::Traits::is_string_v<S>>>
    inline auto toFloat(S&& str, Noa::Flag<Errno>& err) noexcept {
        errno = 0;
        char* end;
        Noa::Traits::remove_ref_cv_t<Float> out{};
        if constexpr (Noa::Traits::is_same_v<Float, float>)
            out = std::strtof(str.data(), &end);
        else if constexpr (Noa::Traits::is_same_v<Float, double>)
            out = std::strtod(str.data(), &end);
        else if constexpr (Noa::Traits::is_same_v<Float, long double>)
            out = std::strtold(str.data(), &end);
        else {
            static_assert(Noa::Traits::always_false_v<Float>);
        }
        if (end == str.data()) {
            err = Errno::invalid_argument;
        } else if (errno == ERANGE) {
            err = Errno::out_of_range;
        }
        return out;
    }


    /**
     * Convert a string into a bool.
     * @tparam S        @c std::string(_view) by lvalue or rvalue.
     * @param[in] str   String to convert. Read only.
     * @param[out] err  Status to update. Is set to:
     *                  @c Errno::invalid_argument, if @a str couldn't be converted.
     *                  Unchanged otherwise.
     * @return          Resulting bool.
     */
    template<typename S, typename = std::enable_if_t<Noa::Traits::is_string_v<S>>>
    inline bool toBool(S&& str, Noa::Flag<Errno>& err) {
        if (str == "1" || str == "true")
            return true;
        else if (str == "0" || str == "false")
            return false;

        // Some rare cases
        std::string str_up = toUpperCopy(str);
        if (str_up == "TRUE" || str_up == "Y" || str_up == "YES" || str_up == "ON") {
            return true;
        } else if (str_up == "FALSE" || str_up == "N" || str_up == "NO" || str_up == "OFF") {
            return false;
        } else {
            err = Errno::invalid_argument;
            return false;
        }
    }


    /**
     * Formats a string, using {fmt}.
     * @note    This is mostly to NOT have fmt:: everywhere in the code base, given that in the
     *          future we might switch to std::format.
     */
    template<typename... Args>
    inline std::string format(Args&& ... args) { return fmt::format(std::forward<Args>(args)...); }


    template<typename T>
    std::string typeName() {
        if constexpr (Noa::Traits::is_same_v<float, T>) {
            return "float";
        } else if constexpr (Noa::Traits::is_same_v<double, T>) {
            return "double";

        } else if constexpr (Noa::Traits::is_int_v<T>) {
            if constexpr (Noa::Traits::is_same_v<uint32_t, T>)
                return "uint32";
            else if constexpr (Noa::Traits::is_same_v<uint64_t, T>)
                return "uint64";
            else if constexpr (Noa::Traits::is_same_v<uint16_t, T>)
                return "uint16";
            else if constexpr (Noa::Traits::is_same_v<uint8_t, T>)
                return "uint8";
            else if constexpr (Noa::Traits::is_same_v<int32_t, T>)
                return "int32";
            else if constexpr (Noa::Traits::is_same_v<int64_t, T>)
                return "int64";
            else if constexpr (Noa::Traits::is_same_v<int16_t, T>)
                return "int16";
            else if constexpr (Noa::Traits::is_same_v<int8_t, T>)
                return "int8";


        } else if constexpr (Noa::Traits::is_bool_v<T>) {
            return "bool";

        } else if constexpr (Noa::Traits::is_same_v<std::complex<float>, T>) {
            return "complex64";
        } else if constexpr (Noa::Traits::is_same_v<std::complex<double>, T>) {
            return "complex128";

        } else {
            return typeid(T).name(); // implementation defined, no guarantee to be human-readable.
        }
    }
}
