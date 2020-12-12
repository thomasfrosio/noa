/**
 * @file String.h
 * @brief String related functions.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"
#include "noa/util/Vectors.h"


/** Gathers a bunch of string related functions. */
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
     * @tparam T        Supported integers are: (u)int8_t, (u)int16_t, (u)int32_t, (u)int64_t.
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
    template<typename T = int32_t, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_int_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toInt(S&& str, errno_t& err) noexcept {
        using int_t = Traits::remove_ref_cv_t<T>;
        errno = 0;
        char* end;
        int_t out{0};

        if constexpr (Traits::is_uint_v<int_t>) {
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
        } else if (Traits::is_int_v<int_t>) /* signed */ {
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
     * @tparam T        Supported floating points are: float, double and long double.
     * @tparam S        @c std::string(_view) by lvalue or rvalue. Read only.
     * @param[in] str   String to convert into @a T.
     * @param[out] err  Status to update. Is set to:
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range
     *                  Unchanged otherwise.
     * @return          Resulting floating point.
     * @note            @c errno is reset to 0 before starting the conversion.
     */
    template<typename T = float, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_float_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toFloat(S&& str, errno_t& err) noexcept {
        errno = 0;
        char* end;
        Traits::remove_ref_cv_t<T> out{};
        if constexpr (Traits::is_same_v<T, float>)
            out = std::strtof(str.data(), &end);
        else if constexpr (Traits::is_same_v<T, double>)
            out = std::strtod(str.data(), &end);
        else if constexpr (Traits::is_same_v<T, long double>)
            out = std::strtold(str.data(), &end);
        else {
            static_assert(Traits::always_false_v<T>);
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
    template<typename S, typename = std::enable_if_t<Traits::is_string_v<S>>>
    inline bool toBool(S&& str, errno_t& err) {
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


    /** Format a string and emplace_back the output into a std::vector. */
    template<typename T, typename = std::enable_if_t<Traits::is_string_v<T> ||
                                                     Traits::is_scalar_v<T> ||
                                                     Traits::is_bool_v<T>>>
    inline errno_t formatAndEmplaceBack(std::string&& string, std::vector<T>& vector) {
        if constexpr (Traits::is_string_v<T>) {
            vector.emplace_back(std::move(string));
            return Errno::good;
        }

        errno_t err{Errno::good};
        if constexpr (Traits::is_float_v<T>)
            vector.emplace_back(toFloat<T>(string, err));
        else if constexpr (Traits::is_int_v<T>)
            vector.emplace_back(toInt<T>(string, err));
        else if constexpr (Traits::is_bool_v<T>)
            vector.emplace_back(toBool(string, err));
        return err;
    }


    /** Format a string and assign the output into the array at desired index. */
    template<typename T, typename = std::enable_if_t<Traits::is_string_v<T> ||
                                                     Traits::is_scalar_v<T> ||
                                                     Traits::is_bool_v<T>>>
    inline errno_t formatAndAssign(std::string&& string, T* ptr) {
        if constexpr (Noa::Traits::is_string_v<T>) {
            *ptr = std::move(string);
            return Errno::good;
        }

        errno_t err{Errno::good};
        if constexpr (Noa::Traits::is_float_v<T>)
            *ptr = toFloat<T>(string, err);
        else if constexpr (Noa::Traits::is_int_v<T>)
            *ptr = toInt<T>(string, err);
        else if constexpr (Noa::Traits::is_bool_v<T>)
            *ptr = toBool(string, err);
        return err;
    }


    /**
     * Parse @a str and emplace back the (formatted) output value(s) into @a vec.
     * @details         Parse a string using commas as delimiters. Parsed strings are trimmed
     *                  and empty strings are kept. If @a vec is a vector of integers, floating points
     *                  or booleans, the strings are converted using @c toInt(), @c toFloat()
     *                  or @c toBool(), respectively. If @a vec is a vector of strings, no conversion
     *                  is performed.
     *
     * @tparam S        @c std::string(_view) by rvalue or lvalue.
     * @tparam T        Type contained by @a vec. Sets the type of formatting that should be used.
     * @param[in] str   String to parse. Read only.
     * @param[out] vec  Output vector. The parsed values are inserted at the end of the vector.
     * @return          Whether or not an error occurred. See @c String::to*() functions.
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range.
     *                  @c Errno::good, otherwise.
     * @example
     * @code
     * std::vector<std::string> vec1;
     * std::vector<float> vec2;
     * parse(" 1, 2,  ,  4 5 ", vec1); // {"1", "2", "", "4 5"}
     * parse(" 1, 2,  ,  4 5 ", vec2); // {1.f, 2.f}
     * @endcode
     */
    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& str, std::vector<T>& vec) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        errno_t err;

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                err = formatAndEmplaceBack({str.data() + idx_start, idx_end - idx_start}, vec);
                if (err)
                    return err;
                idx_start = 0;
                idx_end = 0;
                capture = false;
            } else if (!std::isspace(str[i])) {
                if (capture)
                    idx_end = i + 1;
                else {
                    idx_start = i;
                    idx_end = i + 1;
                    capture = true;
                }
            }
        }
        return formatAndEmplaceBack({str.data() + idx_start, idx_end - idx_start}, vec);
    }


    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& str, std::vector<T>& vec, size_t size) {
        size_t size_before = vec.size();
        errno_t err = String::parse(str, vec);
        return (!err && vec.size() - size_before != size) ? Errno::invalid_size : err;
    }


    /**
     * Parse @a str1 and store the (formatted) output value(s) into @a vec.
     * @details         This is similar to the parsing functions above, except that if one string
     *                  in @a str1 is empty or only whitespaces, it falls back to the corresponding
     *                  value of @a str2.
     *
     * @tparam T        Type contained by @c vec. Sets the type of formatting that should be used.
     * @param[in] str1  String to parse. Read only.
     * @param[in] str2  String containing the default value(s). Read only.
     * @param[out] vec  Output vector. The parsed values are inserted at the end of the vector.
     * @return          Whether or not an error occurred.
     *                  @c Errno::invalid_argument, if one of the values couldn't be converted.
     *                  @c Errno::invalid_size, if @c str1 and @c str2 couldn't be parsed into the same number of values.
     *                  @c Errno::out_of_range, if one of the values was out of the @c T range.
     *                  @c Errno::good, otherwise.
     *
     * TODO: maybe optimize this, but not sure if it is really worth it.
     */
    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& string, S&& string_backup, std::vector<T>& vector) {
        std::vector<std::string> v1;
        std::vector<std::string> v2;
        parse(string, v1);
        parse(string_backup, v2);

        size_t size = v1.size();
        if (size != v2.size())
            return Errno::invalid_size;

        errno_t err;
        for (size_t i{0}; i < size; ++i) {
            err = formatAndEmplaceBack(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i]), vector);
            if (err)
                break;
        }
        return err;
    }


    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& string, S&& string_backup, std::vector<T>& vector, size_t size) {
        size_t size_before = vector.size();
        errno_t err = String::parse(string, string_backup, vector);
        return (!err && vector.size() - size_before != size) ? Errno::invalid_size : err;
    }


    /**
     * Parse @a str and store the (formatted) output value(s) into @a arr.
     * @details         This is identical to the overload above, except that the (formatted) output
     *                  values are stored into @c arr, starting at the 0 index and going forward.
     *
     * @tparam S        @c std::string(_view) by rvalue or lvalue.
     * @tparam T        Type contained by @c arr. Sets the type of formatting that should be used.
     * @tparam N        Size of the array @c arr.
     * @param[in] str   String to parse. Read only.
     * @param[out] arr  Output array. The parsed values are stored in the array, starting at the
     *                  begging of the array and going forward.
     * @return          Whether or not an error occurred. See @c String::to*() functions.
     *                  @c Errno::invalid_argument, if a character couldn't be converted.
     *                  @c Errno::invalid_size, if @a str cannot be parsed into exactly @c N elements.
     *                  @c Errno::out_of_range, if a string translates to an number that is out of the @a T range.
     *                  @c Errno::good, otherwise. In this case, all of the items in the array have been updated.
     */
    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& string, T* ptr, size_t size) {
        size_t idx_start{0}, idx_end{0}, idx{0};
        bool capture{false};
        errno_t err;

        for (size_t i{0}; i < string.size(); ++i) {
            if (string[i] == ',') {
                if (idx == size)
                    return Errno::invalid_size;
                err = formatAndAssign({string.data() + idx_start, idx_end - idx_start}, ptr + idx);
                if (err)
                    return err;
                ++idx;
                idx_start = 0;
                idx_end = 0;
                capture = false;
            } else if (!std::isspace(string[i])) {
                if (capture)
                    idx_end = i + 1;
                else {
                    idx_start = i;
                    idx_end = i + 1;
                    capture = true;
                }
            }
        }
        if (idx + 1 != size)
            return Errno::invalid_size;
        return formatAndAssign({string.data() + idx_start, idx_end - idx_start}, ptr + idx);
    }


    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_vector_v<T> ||
                                                                   (Traits::is_std_array_v<T> &&
                                                                    !Traits::is_std_array_complex_v<T>))>>
    inline errno_t parse(S&& string, T& static_array) {
        return parse(string, static_array.data(), static_array.size());
    }


    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_vector_v<T> ||
                                                                   (Traits::is_std_array_v<T> &&
                                                                    !Traits::is_std_array_complex_v<T>))>>
    inline errno_t parse(S&& string, T& static_array, size_t size) {
        return parse(string, static_array.data(), size);
    }


    /**
     * Parse @a str1 and store the (formatted) output value(s) into @a arr.
     * @details         This is similar to the parsing functions above, except that if one string
     *                  in @a str1 is empty or only whitespaces, it falls back to the corresponding
     *                  value of @a str2.
     *
     * @tparam T        Type contained by @c arr. Sets the type of formatting that should be used.
     * @tparam N        Size of the array @c arr.
     * @param[in] str1  String to parse. Read only.
     * @param[in] str2  String containing the default value(s). Read only.
     * @param[out] arr  Output array. The parsed values are stored in the array, starting at the
     *                  begging of the array and going forward.
     * @return          Whether or not an error occurred.
     *                  @c Errno::invalid_argument, if one of the values couldn't be converted.
     *                  @c Errno::invalid_size, if @c str1 and @c str2 cannot be parsed into exactly @c N elements.
     *                  @c Errno::out_of_range, if one of the values was out of the @c T range.
     *                  @c Errno::good, otherwise.
     *
     * TODO: maybe optimize this, but not sure if it is really worth it.
     */
    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& string, S&& string_backup, T* ptr, size_t size) {
        std::vector<std::string> v1, v2;
        parse(string, v1);
        parse(string_backup, v2);

        if (size != v1.size() || size != v2.size())
            return Errno::invalid_size;

        errno_t err;
        size_t idx{0};

        for (size_t i{0}; i < size; ++i) {
            if (idx == size)
                return Errno::invalid_size;
            err = formatAndAssign(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i]), ptr + idx);
            if (err)
                return err;
            ++idx;
        }
        return idx != size ? Errno::invalid_size : Errno::good;
    }


    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_vector_v<T> ||
                                                                   (Traits::is_std_array_v<T> &&
                                                                    !Traits::is_std_array_complex_v<T>))>>
    inline errno_t parse(S&& string, S&& string_backup, T& static_array) {
        return parse(string, string_backup, static_array.data(), static_array.size());
    }


    template<typename S, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_vector_v<T> ||
                                                                   (Traits::is_std_array_v<T> &&
                                                                    !Traits::is_std_array_complex_v<T>))>>
    inline errno_t parse(S&& string, S&& string_backup, T& static_array, size_t size) {
        return parse(string, string_backup, static_array.data(), size);
    }
}
