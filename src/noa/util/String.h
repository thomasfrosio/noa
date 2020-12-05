/**
 * @file String.h
 * @brief String related functions.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/Traits.h"


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


    inline std::vector<std::string>& trim(std::vector<std::string>& vec_str) {
        for (auto& str : vec_str)
            trim(str);
        return vec_str;
    }


    [[nodiscard]] inline std::string trimCopy(std::string str) {
        leftTrim(rightTrim(str));
        return str;
    }


    [[nodiscard]] inline std::vector<std::string> trimCopy(std::vector<std::string> vec_str) {
        for (auto& str : vec_str)
            trim(str);
        return vec_str;
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
     * @tparam T        Supported integers are: short, int, long, long long, int8_t and all corresponding unsigned versions.
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
    template<typename T = int, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_int_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toInt(S&& str, errno_t& err) noexcept {
        using Tv = Traits::remove_ref_cv_t<T>;
        errno = 0;
        char* end;
        Tv out{0};

        if constexpr (std::is_same_v<Tv, long>) {
            out = std::strtol(str.data(), &end, 10);

        } else if constexpr (std::is_same_v<Tv, long long>) {
            out = std::strtoll(str.data(), &end, 10);

        } else if constexpr (std::is_same_v<Tv, int> ||
                             std::is_same_v<Tv, short> ||
                             std::is_same_v<Tv, int8_t>) {
            long tmp = std::strtol(str.data(), &end, 10);
            if (tmp > std::numeric_limits<Tv>::max() || tmp < std::numeric_limits<Tv>::min())
                err = Errno::out_of_range;
            out = static_cast<Tv>(tmp);

        } else /* unsigned */ {
            size_t idx = str.find_first_not_of(" \t");
            if (idx == std::string::npos) {
                err = Errno::invalid_argument;
                return out;
            } else if (str[idx] == '-') {
                if (str.size() >= idx + 1 && str[idx + 1] > 47 && str[idx + 1] < 58)
                    err = Errno::out_of_range;
                else
                    err = Errno::invalid_argument;
                return out;
            }
            if constexpr (std::is_same_v<Tv, unsigned int> ||
                          std::is_same_v<Tv, unsigned short> ||
                          std::is_same_v<Tv, uint8_t>) {
                unsigned long tmp = std::strtoul(str.data(), &end, 10);
                if (tmp > std::numeric_limits<Tv>::max() || tmp < std::numeric_limits<Tv>::min())
                    err = Errno::out_of_range;
                out = static_cast<Tv>(tmp);
            } else if constexpr (std::is_same_v<Tv, unsigned long>) {
                out = std::strtoul(str.data(), &end, 10);
            } else if constexpr (std::is_same_v<Tv, unsigned long long>) {
                out = std::strtoull(str.data(), &end, 10);
            } else {
                static_assert(Traits::always_false_v<Tv>);
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
    template<typename S = std::string_view, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& str, std::vector<T>& vec) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        errno_t err{Errno::good};

        auto add = [&vec, &err](std::string&& buffer) -> bool {
            if constexpr (Traits::is_float_v<T>) {
                vec.emplace_back(toFloat<T>(buffer, err));
            } else if constexpr (Traits::is_int_v<T>) {
                vec.emplace_back(toInt<T>(buffer, err));
            } else if constexpr (Traits::is_bool_v<T>) {
                vec.emplace_back(toBool(buffer, err));
            } else if constexpr (Traits::is_string_v<T>) {
                vec.emplace_back(std::move(buffer));
                return true;
            }
            if (err) {
                vec.pop_back();
                return false;
            }
            return true;
        };

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                if (!add({str.data() + idx_start, idx_end - idx_start}))
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
        add({str.data() + idx_start, idx_end - idx_start});
        return err;
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
    template<typename S = std::string_view, typename T, size_t N,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    errno_t parse(S&& str, std::array<T, N>& arr) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0}, count{0};
        bool capture{false};
        errno_t err{Errno::good};

        auto add = [&arr, &err, &count](std::string&& buffer) -> bool {
            if (count > N) {
                err = Errno::invalid_size;
                return false;
            }
            if constexpr (Noa::Traits::is_float_v<T>) {
                arr[count] = toFloat<T>(buffer, err);
            } else if constexpr (Noa::Traits::is_int_v<T>) {
                arr[count] = toInt<T>(buffer, err);
            } else if constexpr (Noa::Traits::is_bool_v<T>) {
                arr[count] = toBool(buffer, err);
            } else if constexpr (Noa::Traits::is_string_v<T>) {
                arr[count] = std::move(buffer);
            }
            ++count;
            return err == Errno::good;
        };

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                if (!add({str.data() + idx_start, idx_end - idx_start}))
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
        add({str.data() + idx_start, idx_end - idx_start});
        return (!err && count != N) ? Errno::invalid_size : err;
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
    template<typename T, typename = std::enable_if_t<Traits::is_string_v<T> ||
                                                     Traits::is_scalar_v<T> ||
                                                     Traits::is_bool_v<T>>>
    errno_t parse(const std::string& str1, const std::string& str2, std::vector<T>& vec) {
        std::vector<std::string> v_str1;
        std::vector<std::string> v_str2;
        parse(str1, v_str1);
        parse(str2, v_str2);

        size_t size = v_str1.size();
        if (size != v_str2.size())
            return Errno::invalid_size;

        errno_t err{Errno::good};

        auto add = [&vec, &err](std::string& str) {
            if constexpr (Traits::is_float_v<T>) {
                vec.emplace_back(toFloat<T>(str, err));
            } else if constexpr (Traits::is_int_v<T>) {
                vec.emplace_back(toInt<T>(str, err));
            } else if constexpr (Traits::is_bool_v<T>) {
                vec.emplace_back(toBool(str, err));
            } else if constexpr (Traits::is_string_v<T>) {
                vec.emplace_back(std::move(str));
                return;
            }
            if (err)
                vec.pop_back();
        };

        vec.reserve(size);
        for (size_t i{0}; i < size; ++i) {
            add(v_str1[i].empty() ? v_str2[i] : v_str1[i]);
            if (err)
                break;
        }
        return err;
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
    template<typename T, size_t N, typename = std::enable_if_t<Traits::is_string_v<T> ||
                                                               Traits::is_scalar_v<T> ||
                                                               Traits::is_bool_v<T>>>
    errno_t parse(const std::string& str1, const std::string& str2, std::array<T, N>& arr) {
        std::array<std::string, N> a_str1;
        std::array<std::string, N> a_str2;
        if (parse(str1, a_str1) || parse(str2, a_str2))
            return Errno::invalid_size;

        errno_t err{Errno::good};
        size_t count{0};

        auto add = [&arr, &err, &count](std::string& str) {
            if (count > arr.size()) {
                err = Errno::invalid_size;
                return false;
            }
            if constexpr (Noa::Traits::is_float_v<T>)
                arr[count] = toFloat<T>(str, err);
            else if constexpr (Noa::Traits::is_int_v<T>)
                arr[count] = toInt<T>(str, err);
            else if constexpr (Noa::Traits::is_bool_v<T>)
                arr[count] = toBool(str, err);
            else if constexpr (Noa::Traits::is_string_v<T>)
                arr[count] = std::move(str);
            ++count;
            return err == 0;
        };

        for (size_t i{0}; i < N; ++i) {
            add(a_str1[i].empty() ? a_str2[i] : a_str1[i]);
            if (err)
                break;
        }
        return (!err && count != N) ? Errno::invalid_size : err;
    }
}
