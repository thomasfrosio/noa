/**
 * @file string/Convert.h
 * @brief Convert strings to other type.
 * @author Thomas - ffyr2w
 * @date 10 Jan 2021
 */
#pragma once

#include <cstdlib>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <limits>
#include <cerrno>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Errno.h"
#include "noa/util/string/Format.h"     // toUpperCopy
#include "noa/util/traits/BaseTypes.h"

namespace Noa::String {
    /**
     * Convert a string into an integer.
     * @tparam Int      Supported integers are: (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t.
     * @tparam S        @c std::string(_view) by lvalue or rvalue. Read only.
     * @param[in] str   String to convert into @a T.
     * @param[out] err  Status to update. Is set to:
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range.
     *                  Unchanged otherwise.
     * @return          Resulting integer.
     *
     * @note            @c errno is reset to 0 before starting the conversion.
     */
    template<typename Int = int32_t, typename S = std::string_view,
             typename = std::enable_if_t<Noa::Traits::is_int_v<Int> && Noa::Traits::is_string_v<S>>>
    inline auto toInt(S&& str, Errno& err) noexcept {
        using int_t = Noa::Traits::remove_ref_cv_t<Int>;
        int_t out{0};
        errno = 0;
        char* end;

        if constexpr (Noa::Traits::is_uint_v<int_t>) {
            // Shortcut: empty string or negative number.
            size_t idx = str.find_first_not_of(" \t");
            if (idx == std::string::npos) {
                err = Errno::invalid_argument;
                return out;
            } else if (str[idx] == '-') {
                err = (str.size() >= idx + 1 &&
                       str[idx + 1] > 47 &&
                       str[idx + 1] < 58) ? Errno::out_of_range : Errno::invalid_argument;
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
                    if (tmp > std::numeric_limits<int_t>::max() || tmp < std::numeric_limits<int_t>::min())
                        err = Errno::out_of_range;
                    out = static_cast<int_t>(tmp);
                }
            }

        } else if constexpr (Noa::Traits::is_int_v<int_t>) {
            if constexpr (std::is_same_v<int_t, int64_t>) {
                out = std::strtoll(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<int_t, int32_t> ||
                                 std::is_same_v<int_t, int16_t> ||
                                 std::is_same_v<int_t, int8_t>) {
                if constexpr (std::is_same_v<int_t, int32_t> && std::is_same_v<long, int32_t>) {
                    out = std::strtol(str.data(), &end, 10);
                } else /* long == int64_t */ {
                    long tmp = std::strtol(str.data(), &end, 10);
                    if (tmp > std::numeric_limits<int_t>::max() || tmp < std::numeric_limits<int_t>::min())
                        err = Errno::out_of_range;
                    out = static_cast<int_t>(tmp);
                }
            }

        } else {
            static_assert(Noa::Traits::always_false_v<int_t>, "this should not be possible");
        }

        if (end == str.data())
            err = Errno::invalid_argument;
        else if (errno == ERANGE)
            err = Errno::out_of_range;
        return out;
    }

    template<typename Int = int32_t, typename S = std::string_view,
             typename = std::enable_if_t<Noa::Traits::is_int_v<Int> && Noa::Traits::is_string_v<S>>>
    inline auto toInt(S&& str) {
        using int_t = Noa::Traits::remove_ref_cv_t<Int>;
        int_t out{0};
        errno = 0;
        char* end;

        if constexpr (Noa::Traits::is_uint_v<int_t>) {
            // Shortcut: empty string or negative number.
            size_t idx = str.find_first_not_of(" \t");
            if (idx == std::string::npos) {
                NOA_THROW("Cannot convert an empty string to an unsigned integer");
            } else if (str[idx] == '-') {
                if (str.size() >= idx + 1 && str[idx + 1] > 47 && str[idx + 1] < 58)
                    NOA_THROW("Wrap-around. Cannot convert \"{}\" to an unsigned integer.", str);
                else
                    NOA_THROW("Cannot convert \"{}\" to an unsigned integer", str);
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
                    if (tmp > std::numeric_limits<int_t>::max() || tmp < std::numeric_limits<int_t>::min())
                        NOA_THROW("Out of range. \"{}\" is out of {} range", str, String::typeName<int_t>());
                    out = static_cast<int_t>(tmp);
                }
            }
        } else if constexpr (Noa::Traits::is_int_v<int_t>) {
            if constexpr (std::is_same_v<int_t, int64_t>) {
                out = std::strtoll(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<int_t, int32_t> ||
                                 std::is_same_v<int_t, int16_t> ||
                                 std::is_same_v<int_t, int8_t>) {
                if constexpr (std::is_same_v<int_t, int32_t> && std::is_same_v<long, int32_t>) {
                    out = std::strtol(str.data(), &end, 10);
                } else /* long == int64_t */ {
                    long tmp = std::strtol(str.data(), &end, 10);
                    if (tmp > std::numeric_limits<int_t>::max() || tmp < std::numeric_limits<int_t>::min())
                        NOA_THROW("Out of range. \"{}\" is out of {} range", str, String::typeName<int_t>());
                    out = static_cast<int_t>(tmp);
                }
            }
        } else {
            static_assert(Noa::Traits::always_false_v<int_t>, "this should not be possible");
        }

        if (end == str.data())
            NOA_THROW("Cannot convert \"{}\" to {}", str, String::typeName<int_t>());
        else if (errno == ERANGE)
            NOA_THROW("Out of range. \"{}\" is out of {} range", str, String::typeName<int_t>());
        return out;
    }

    /**
     * Convert a string into a floating point.
     * @tparam Float    Supported floating points are: float, double and long double.
     * @tparam S        @c std::string(_view) by lvalue or rvalue. Read only.
     * @param[in] str   String to convert into @a T.
     * @param[out] err  Status to update. Is set to:
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range.
     *                  Unchanged otherwise.
     * @return          Resulting floating point.
     * @note            @c errno is reset to 0 before starting the conversion.
     */
    template<typename Float = float, typename S = std::string_view,
             typename = std::enable_if_t<Noa::Traits::is_float_v<Float> && Noa::Traits::is_string_v<S>>>
    inline auto toFloat(S&& str, Errno& err) noexcept {
        errno = 0;
        char* end;

        Noa::Traits::remove_ref_cv_t<Float> out{};
        if constexpr (Noa::Traits::is_same_v<Float, float>) {
            out = std::strtof(str.data(), &end);
        } else if constexpr (Noa::Traits::is_same_v<Float, double>) {
            out = std::strtod(str.data(), &end);
        } else if constexpr (Noa::Traits::is_same_v<Float, long double>) {
            out = std::strtold(str.data(), &end);
        } else {
            static_assert(Noa::Traits::always_false_v<Float>);
        }

        if (end == str.data())
            err = Errno::invalid_argument;
        else if (errno == ERANGE)
            err = Errno::out_of_range;
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
    inline bool toBool(S&& str, Errno& err) {
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
}
