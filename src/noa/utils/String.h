/**
 * @file String.h
 * @brief String related functions.
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/Traits.h"


/// Group of string related functions.
namespace Noa::String {

    /**
     * @brief               Left trim a string in-place.
     * @param[in,out] str   String to left trim (taken as lvalue)
     * @return              Left trimmed string.
     */
    inline std::string& leftTrim(std::string& str) {
        str.erase(str.begin(),
                  std::find_if(str.begin(),
                               str.end(),
                               [](int ch) { return !std::isspace(ch); }));
        return str;
    }


    /**
     * @brief           Left trim a string in-place.
     * @param[in] str   String to left trim (taken as rvalue)
     * @return          Left trimmed string.
     */
    [[nodiscard]] inline std::string leftTrim(std::string&& str) {
        leftTrim(str);
        return std::move(str);
    }


    /**
     * @brief           Left trim a string by copy.
     * @param[in] str   String to left trim (taken as value)
     * @return          Left trimmed string.
     */
    [[nodiscard]] inline std::string leftTrimCopy(std::string str) {
        leftTrim(str);
        return str;
    }


    /**
     * @brief               Right trim a string in-place.
     * @param[in,out] str   String to right trim (taken by lvalue)
     * @return              Right trimmed string.
     */
    inline std::string& rightTrim(std::string& str) {
        str.erase(std::find_if(str.rbegin(),
                               str.rend(),
                               [](int ch) { return !std::isspace(ch); }).base(),
                  str.end());
        return str;
    }


    /**
     * @brief           Right trim a string in-place.
     * @param[in] str   String to right trim (taken by rvalue)
     * @return          Right trimmed string.
     */
    [[nodiscard]] inline std::string rightTrim(std::string&& str) {
        rightTrim(str);
        return std::move(str);
    }


    /**
     * @brief           Right trim a string by copy.
     * @param[in] str   String to right trim (taken by value)
     * @return          Right trimmed string.
     */
    [[nodiscard]] inline std::string rightTrimCopy(std::string str) {
        rightTrim(str);
        return str;
    }


    /**
     * @brief               Trim (left and right) a string in-place.
     * @param[in,out] str   String to trim (taken by lvalue)
     * @return              Trimmed string.
     */
    inline std::string& trim(std::string& str) {
        return leftTrim(rightTrim(str));
    }


    /**
     * @brief           Trim (left and right) a string in-place.
     * @param[in] str   String to trim (taken by rvalue)
     * @return          Trimmed string.
     */
    [[nodiscard]] inline std::string trim(std::string&& str) {
        leftTrim(rightTrim(str));
        return std::move(str);
    }


    /**
     * @brief                   Trim (left and right) string(s) stored in vector.
     * @param[in,out] vec_str   Vector of string(s) to trim (taken by lvalue)
     * @return                  Trimmed string.
     */
    inline std::vector<std::string>& trim(std::vector<std::string>& vec_str) {
        for (auto& str : vec_str)
            trim(str);
        return vec_str;
    }


    /**
     * @brief           Trim (left and right) a string in-place.
     * @param[in] str   String to trim (taken by value)
     * @return          Trimmed string.
     */
    [[nodiscard]] inline std::string trimCopy(std::string str) {
        leftTrim(rightTrim(str));
        return str;
    }


    /**
     * @brief               Trim (left and right) string(s) stored in vector.
     * @param[in] vec_str   Vector of string(s) to trim (taken by value)
     * @return              Trimmed string.
     */
    [[nodiscard]] inline std::vector<std::string> trimCopy(std::vector<std::string> vec_str) {
        for (auto& str : vec_str)
            trim(str);
        return vec_str;
    }


    /**
     * Convert a string into an integer.
     * @tparam T        Supported integers are: short, int, long, long long, int8_t and all
     *                  corresponding unsigned versions.
     * @tparam S        @c std::string(_view) by lvalue or rvalue.
     * @param[in] str   String to convert into @c T.
     * @param[out] err  Status to update. If no errors, it is left unchanged, otherwise can be
     *                  set to @c Errno::invalid_argument or @c Errno::out_of_range.
     * @return          Resulting integer of type @c T.
     *
     * @note            @c errno is reset to @c 0 before starting the conversion.
     */
    template<typename T = int, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_int_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toInt(S&& str, uint8_t& err) noexcept {
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

        // Check for invalid argument or out of range.
        if (end == str.data())
            err = Errno::invalid_argument;
        else if (errno == ERANGE)
            err = Errno::out_of_range;
        return out;
    }


    /**
     * Convert a string into a floating point.
     * @tparam T        Supported floating points are: float, double and long double.
     * @tparam S        @c std::string(_view) by lvalue or rvalue.
     * @param[in] str   String to convert into @c T.
     * @param[out] err  Status to update. If no errors, it is left unchanged, otherwise can be
     *                  set to @c Errno::invalid_argument or @c Errno::out_of_range.
     * @return          Resulting floating point of type @c T.
     * @note            @c errno is reset to @c 0 before starting the conversion.
     */
    template<typename T = float, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_float_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toFloat(S&& str, uint8_t& err) noexcept {
        errno = 0;
        char* end;
        Traits::remove_ref_cv_t<T> out{};
        if constexpr (Traits::is_same_v<T, float>)
            out = std::strtof(str.data(), &end);
        else if constexpr (Traits::is_same_v<T, double>)
            out = std::strtod(str.data(), &end);
        else if constexpr (Traits::is_same_v<T, long double>)
            out = std::strtold(str.data(), &end);
        else
                static_assert(Traits::always_false_v<T>);

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
     * @param[in] str   String to convert.
     * @param[out] err  Status to update. If no errors, it is left unchanged, otherwise can be
     *                  set to @c Errno::invalid_argument.
     * @return          bool resulting from the conversion.
     */
    template<typename S, typename = std::enable_if_t<Traits::is_string_v<S>>>
    inline bool toBool(S&& str, uint8_t& err) {
        if (str == "1" || str == "true")
            return true;
        else if (str == "0" || str == "false")
            return false;

        if /* some rare cases */ (str == "TRUE" || str == "y" || str == "Y" ||
                                  str == "yes" || str == "YES" || str == "on" || str == "ON")
            return true;
        else if /* some rare cases */ (str == "FALSE" || str == "n" || str == "no" ||
                                       str == "off" || str == "NO" || str == "OFF")
            return false;
        else {
            err = Errno::invalid_argument;
            return false;
        }
    }


    /**
     * Parse a string and emplace back the (formatted) output value(s) into a vector.
     * @details         Parse a string using commas as a separator. Parsed values are trimmed
     *                  and empty strings are kept. These values are then @c emplaced_back()
     *                  into the vector @c vec. If @c vec is a vector of integers, floating points
     *                  or booleans, the parsed values are converted using @c toInt(), @c toFloat()
     *                  or @c toBool(), respectively. If one of the values cannot be converted,
     *                  the value is not added to @c vec, the parsing stops and the @c status
     *                  is set to @c Errno::invalid_argument or Errno::out_of_range.
     *
     * @tparam S        @c std::string(_view) by rvalue or lvalue. It is not modified.
     * @tparam T        Type of output vector @c vec. Set the type of formatting that should be used.
     * @param[in] str   String to parse.
     * @param[out] vec  Output vector. The parsed values are inserted at the end of the vector.
     * @return          Whether or not an error occurred. This corresponds to @c err of the
     *                  @c String::to*() functions.
     * @example
     * @code
     * std::vector<std::string> vec1;
     * std::vector<float> vec2;
     * parse(" 1, 2,  ,  4 5 ", vec1);
     * fmt::print(vec1);  // {"1", "2", "", "4 5"}
     * parse(" 1, 2,  ,  4 5 ", vec2);
     * fmt::print(vec2);  // {1.f, 2.f}
     * @endcode
     */
    template<typename S = std::string_view, typename T,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    uint8_t parse(S&& str, std::vector<T>& vec) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        uint8_t err{0};

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
     * Parse a string and store the (formatted) output value(s) into an array.
     * @details         Parse a string using commas as a separator. Parsed values are trimmed
     *                  and empty strings are kept. These values are then stored into the array
     *                  @c arr, starting at the 0 index and going forward. If @c arr is an array of
     *                  integers, floating points or booleans, the parsed values are converted using
     *                  @c toInt(), @c toFloat() or @c toBool(), respectively. If one of the values
     *                  cannot be converted, the value is _still_ placed into @c arr, but the
     *                  parsing stops and the @c status is set to @c Errno::invalid_argument or
     *                  @c Errno::out_of_range. If the input string @c str cannot be parsed into
     *                  @c N elements, the status is set to @c Errno::size.
     *
     * @tparam S        @c std::string(_view) by rvalue or lvalue. It is not modified.
     * @tparam T        Type of output array @c arr. Sets the type of formatting that should be used.
     * @tparam N        Size of the array @c arr.
     * @param[in] str   String to parse.
     * @param[out] arr  Output array. The parsed values are stored in the array, starting at the
     *                  begging of the array and going forward. If the function returns 0, it
     *                  guarantees that all of the items in the array have been modified.
     * @return          Whether or not an error occurred. This corresponds to @c err of the
     *                  @c String::to*() functions.
     * @example
     * @code
     * std::array<size_t, 4> vec;
     * parse(" 1, 2, 3 ,  4 5 ", vec);
     * fmt::print(vec);  // {1,2,3,4}
     * @endcode
     */
    template<typename S = std::string_view, typename T, size_t N,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    uint8_t parse(S&& str, std::array<T, N>& arr) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0}, count{0};
        bool capture{false};
        uint8_t err{0};

        auto add = [&arr, &err, &count](std::string&& buffer) -> bool {
            if (count > N) {
                err = Errno::size;
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
            return err == 0;
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
        return (!err && count != N) ? Errno::size : err;
    }


    /**
     * Parse a string with default values and emplace back the output values into a vector.
     * @details         This is similar to the parsing function above, except that if one value
     *                  of the first string is empty, it falls back to the corresponding value
     *                  of the second string. This is mainly used to allow positional default
     *                  values, like with user inputs (see @c Manager::Input).
     *
     * @tparam T        Type of output vector @c vec. Set the type of formatting that should be used.
     * @param[in] str1  String to parse.
     * @param[in] str2  String containing the default value(s).
     * @param[out] vec  Output vector. The parsed values are inserted at the end of the vector.
     * @return          Whether or not an error occurred. Returns:
     *                  @c Errno::invalid_argument, if one of the values couldn't be converted.
     *                  @c Errno::out_of_range, if one of the values was out of the @c T range.
     *                  @c Errno::size, if @c str1 and @c str2 cannot be parsed into the same number
     *                  of values.
     *
     * TODO: maybe optimize this, but not sure if it is really worth it.
     */
    template<typename T, typename = std::enable_if_t<Traits::is_string_v<T> ||
                                                     Traits::is_scalar_v<T> ||
                                                     Traits::is_bool_v<T>>>
    uint8_t parse(const std::string& str1, const std::string& str2, std::vector<T>& vec) {
        std::vector<std::string> v_str1;
        std::vector<std::string> v_str2;
        parse(str1, v_str1);
        parse(str2, v_str2);

        size_t size = v_str1.size();
        if (size != v_str2.size())
            return 1;

        uint8_t err{0};

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
            if (v_str1[i].empty())
                add(v_str2[i]);
            else
                add(v_str1[i]);
            if (err)
                break;
        }
        return err;
    }


    /**
     * Parse a string with default values and store the output values into an array.
     * @details         This is similar to the parsing function above, except that this function
     *                  uses a second string to extract and format values from if one of the values
     *                  from the first string is empty. This is mainly used to allow positional
     *                  default values, like with user inputs (see @c Manager::Input).
     *
     * @tparam T        Type of output array @c arr. Set the type of formatting that should be used.
     * @param[in] str1  String to parse.
     * @param[in] str2  String containing the default value(s).
     * @param[out] arr  Output array. The parsed values are stored in the array, starting at the
     *                  begging of the array and going forward. If the function returns 0, it
     *                  guarantees that all of the items in the array have been modified.
     * @return          Whether or not an error occurred. Returns:
     *                  @c Errno::invalid_argument, if one of the value couldn't be converted.
     *                  @c Errno::out_of_range, if one of the value was out of the @c T range.
     *                  @c Errno::size, if @c str1 and @c str2 cannot be parsed into @c N values.
     *
     * TODO: maybe optimize this, but not sure if it is really worth it.
     */
    template<typename T, size_t N, typename = std::enable_if_t<Traits::is_string_v<T> ||
                                                               Traits::is_scalar_v<T> ||
                                                               Traits::is_bool_v<T>>>
    uint8_t parse(const std::string& str1, const std::string& str2, std::array<T, N>& arr) {
        std::array<std::string, N> a_str1;
        std::array<std::string, N> a_str2;
        if (parse(str1, a_str1) || parse(str2, a_str2))
            return Errno::size;

        uint8_t err{0};
        size_t count{0};

        auto add = [&arr, &err, &count](std::string& str) {
            if (count > arr.size()) {
                err = Errno::size;
                return false;
            }
            if constexpr (Noa::Traits::is_float_v<T>) {
                arr[count] = toFloat<T>(str, err);
            } else if constexpr (Noa::Traits::is_int_v<T>) {
                arr[count] = toInt<T>(str, err);
            } else if constexpr (Noa::Traits::is_bool_v<T>) {
                arr[count] = toBool(str, err);
            } else if constexpr (Noa::Traits::is_string_v<T>) {
                arr[count] = std::move(str);
            }
            ++count;
            return err == 0;
        };

        for (size_t i{0}; i < N; ++i) {
            if (a_str1[i].empty())
                add(a_str2[i]);
            else
                add(a_str1[i]);
            if (err)
                break;
        }
        return (!err && count != N) ? Errno::size : err;
    }
}
