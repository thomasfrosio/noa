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
     * @tparam T            Supported integers are: short, int, long, long long, int8_t and all
     *                      corresponding unsigned versions.
     * @tparam S            @c std::string(_view) by lvalue or rvalue.
     * @param[in] str       String to convert into @c T. If it starts with a minus sign and @c T is
     *                      unsigned, @c caught is set to @c Errno::out_of_range.
     * @param[out] caught   Status to update. If no errors, it is left unchanged, otherwise can be
     *                      set to @c Errno::invalid_argument or @c Errno::out_of_range.
     * @return              Resulting integer of type @c T.
     *
     * @note                @c errno is reset to @c 0 before starting the conversion.
     */
    template<typename T = int, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_int_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toInt(S&& str, uint8_t& caught) noexcept {
        using Tv = Traits::remove_ref_cv_t<T>;
        errno = 0;
        char* end;
        Tv out;

        if constexpr (std::is_same_v<Tv, int>) {
            out = static_cast<Tv>(std::strtol(str.data(), &end, 10));
        } else if constexpr (std::is_same_v<Tv, long>) {
            out = std::strtol(str.data(), &end, 10);
        } else if constexpr (std::is_same_v<Tv, long long>) {
            out = std::strtoll(str.data(), &end, 10);
        } else if constexpr (std::is_same_v<Tv, int8_t> ||
                             std::is_same_v<Tv, uint8_t> ||
                             std::is_same_v<Tv, short>) {
            long tmp = std::strtol(str.data(), &end, 10);
            if (tmp > std::numeric_limits<Tv>::max() || tmp < std::numeric_limits<Tv>::min()) {
                caught = Errno::out_of_range;
                return out;
            } else {
                out = static_cast<Tv>(tmp);
            }
        } else /* unsigned */ {
            size_t idx = str.find_first_of(" \t");
            if (idx == std::string::npos) {
                caught = Errno::invalid_argument;
                return out;
            } else if (idx == '-') {
                caught = Errno::out_of_range;
                return out;
            }
            if constexpr (std::is_same_v<Tv, unsigned long>)
                out = std::strtoul(str.data(), &end, 10);
            else if constexpr (std::is_same_v<Tv, unsigned long long>)
                out = std::strtoll(str.data(), &end, 10);
        }

        // Check for invalid argument or out of range.
        if (end == str.data()) {
            caught = Errno::invalid_argument;
        } else if (errno == ERANGE) {
            caught = Errno::out_of_range;
        }
        return out;
    }


    /**
     * Convert a string into a floating point.
     * @tparam T            Supported floating points are: float, double and long double.
     * @tparam S            @c std::string(_view) by lvalue or rvalue.
     * @param[in] str       String to convert into @c T.
     * @param[out] caught   Status to update. If no errors, it is left unchanged, otherwise can be
     *                      set to @c Errno::invalid_argument or @c Errno::out_of_range.
     * @return              Resulting floating point of type @c T.
     * @note                @c errno is reset to @c 0 before starting the conversion.
     */
    template<typename T = float, typename S = std::string_view,
            typename = std::enable_if_t<Traits::is_float_v<T> && Traits::is_string_v<S>>>
    inline Traits::remove_ref_cv_t<T> toFloat(S&& str, uint8_t& caught) noexcept {
        errno = 0;
        char* end;
        Traits::remove_ref_cv_t<T> out;
        if constexpr (Traits::is_same_v<T, float>)
            out = std::strtof(str.data(), &end);
        if constexpr (Traits::is_same_v<T, double>)
            out = std::strtod(str.data(), &end);
        if constexpr (Traits::is_same_v<T, long double>)
            out = std::strtold(str.data(), &end);

        if (end == str.data()) {
            caught = Errno::invalid_argument;
        } else if (errno == ERANGE) {
            caught = Errno::out_of_range;
        }
        return out;
    }


    /**
     * Convert a string into a bool.
     * @tparam S            @c std::string(_view) by lvalue or rvalue.
     * @param[in] str       String to convert.
     * @param[out] caught   Status to update. If no errors, it is left unchanged, otherwise can be
     *                      set to @c Errno::invalid_argument.
     * @return              bool resulting from the conversion.
     */
    template<typename S, typename = std::enable_if_t<Traits::is_string_v<S>>>
    inline bool toBool(S&& str, uint8_t& caught) {
        if (str == "1" || str == "true")
            return true;
        else if (str == "0" || str == "false")
            return false;

        if /* some rare cases */ (str == "TRUE" || str == "y" || str == "Y" ||
                                  str == "yes" || str == "YES" || str == "on" || str == "ON")
            return true;
        else if /* some rare cases */ (str == "FALSE" || str == "n" || str == "no" ||
                                       str == "off" || str == "NO" || str == "OFF" ||
                                       str == "FALSE")
            return false;
        else {
            caught = Errno::invalid_argument;
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
     * @return          The status of the parsing. This corresponds to @c status of the
     *                  @c String::to*() functions.
     * @example
     * @code
     * std::vector<std::string> vec;
     * std::vector<float> vec;
     * parse(" 1, 2,  ,  4 5 ", vec);
     * fmt::print(vec);  // {1.f, 2.f}
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
        uint8_t caught{0};

        auto add = [&vec, &caught](const std::string_view str_view) -> bool {
            if constexpr (Traits::is_float_v<T>) {
                vec.emplace_back(toFloat<T>(str_view, caught));
            } else if constexpr (Traits::is_int_v<T>) {
                vec.emplace_back(toInt<T>(str_view, caught));
            } else if constexpr (Traits::is_bool_v<T>) {
                vec.emplace_back(toBool(str_view, caught));
            } else if constexpr (Traits::is_string_v<T>) {
                vec.emplace_back(str_view.data());
                return true;
            }
            if (caught) {
                vec.pop_back();
                return false;
            }
            return true;
        };

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                if (!add({str.data() + idx_start, idx_end - idx_start}))
                    return caught;
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
        return caught;
    }


    /**
     * Parse a string and store the (formatted) output value(s) into an array.
     * @details         Parse a string using commas as a separator. Parsed values are trimmed
     *                  and empty strings are kept. These values are then stored into the array
     *                  @c arr, starting at the 0 index and going forward. If @c arr is an array of
     *                  integers, floating points or booleans, the parsed values are converted using
     *                  @c toInt(), @c toFloat() or @c toBool(), respectively. If one of the values
     *                  cannot be converted, the value is _still_ placed into @c arr, the parsing
     *                  stops and the @c status is set to @c Errno::invalid_argument or
     *                  Errno::out_of_range. If there's more values to place than there's space in
     *                  @c arr, the parsing stop and the @c status is set to @c Errno::invalid_argument.
     *
     * @tparam S        @c std::string(_view) by rvalue or lvalue. It is not modified.
     * @tparam T        Type of output array @c arr. Set the type of formatting that should be used.
     * @tparam N        Size of the array @c arr.
     * @param[in] str   String to parse.
     * @param[out] arr  Output array. The parsed values are stored in the array, starting at the
     *                  begging of the array and going forward.
     * @return          First:  The status of the parsing. This corresponds to @c status of the
     *                          @c String::to*() functions.
     *                  Second: Index where the parsing stops. If @c 0, only one value was parsed
     *                          and stored into @c arr[0]. It cannot be larger than @c N.
     * @example
     * @code
     * std::vector<std::string> vec;
     * std::vector<float> vec;
     * parse(" 1, 2,  ,  4 5 ", vec);
     * fmt::print(vec);  // {1.f, 2.f}
     * @endcode
     */
    template<typename S = std::string_view, typename T, size_t N,
            typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_string_v<T> ||
                                                                   Traits::is_scalar_v<T> ||
                                                                   Traits::is_bool_v<T>)>>
    std::pair<uint8_t, size_t> parse(S&& str, std::array<T, N>& vec) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0}, count{0};
        bool capture{false};
        uint8_t caught{0};

        auto add = [&vec, &caught, &count](const std::string_view str_view) -> bool {
            if (count > vec.size()) {
                caught = 1;
                return false;
            }
            if constexpr (Noa::Traits::is_float_v<T>) {
                vec[count] = toFloat<T>(str_view, caught);
            } else if constexpr (Noa::Traits::is_int_v<T>) {
                vec[count] = toInt<T>(str_view, caught);
            } else if constexpr (Noa::Traits::is_bool_v<T>) {
                vec[count] = toBool(str_view, caught);
            } else if constexpr (Noa::Traits::is_string_v<T>) {
                vec[count] = str_view.data();
            }
            ++count;
            return caught == 0;
        };

        for (size_t i{0}; i < str.size(); ++i) {
            if (str[i] == ',') {
                if (!add({str.data() + idx_start, idx_end - idx_start}))
                    return {caught, count};
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
        return {caught, count};
    }
}
