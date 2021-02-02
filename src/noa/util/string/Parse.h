/**
 * @file string/Parse.h
 * @brief Parsing string related functions.
 * @author Thomas - ffyr2w
 * @date 10 Jan 2021
 */
#pragma once

#include <array>
#include <cstddef>
#include <utility>  // std::move
#include <vector>
#include <string>
#include <string_view>
#include <type_traits>

#include "noa/Errno.h"
#include "noa/util/FloatX.h"
#include "noa/util/IntX.h"
#include "noa/util/traits/BaseTypes.h"
#include "noa/util/string/Convert.h"

#define IS_CONVERTIBLE(T) (Noa::Traits::is_string_v<T> || Noa::Traits::is_scalar_v<T> || Noa::Traits::is_bool_v<T>)

namespace Noa::String {
    /** Format a string and emplace_back the output into a std::vector. */
    template<typename T, typename = std::enable_if_t<IS_CONVERTIBLE(T)>>
    inline Errno formatAndEmplaceBack(std::string&& string, std::vector<T>& vector) {
        if constexpr (Traits::is_string_v<T>) {
            vector.emplace_back(std::move(string));
            return Errno::good;
        }

        Errno err;
        if constexpr (Traits::is_float_v<T>)
            vector.emplace_back(toFloat<T>(string, err));
        else if constexpr (Traits::is_int_v<T>)
            vector.emplace_back(toInt<T>(string, err));
        else if constexpr (Traits::is_bool_v<T>)
            vector.emplace_back(toBool(string, err));
        return err;
    }

    /** Format a string and assign the output into the array at desired index. */
    template<typename T, typename = std::enable_if_t<IS_CONVERTIBLE(T)>>
    inline Errno formatAndAssign(std::string&& string, T* ptr) {
        if constexpr (Noa::Traits::is_string_v<T>) {
            *ptr = std::move(string);
            return Errno::good;
        }

        Errno err;
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
    template<typename S, typename T, typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    Errno parse(S&& str, std::vector<T>& vec) {
        static_assert(!std::is_reference_v<T>);
        size_t idx_start{0}, idx_end{0};
        bool capture{false};
        Errno err;

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

    template<typename S, typename T, typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    Errno parse(S&& str, std::vector<T>& vec, size_t size) {
        size_t size_before = vec.size();
        Errno err = String::parse(str, vec);
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
    template<typename S, typename T, typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    Errno parse(S&& string, S&& string_backup, std::vector<T>& vector) {
        std::vector<std::string> v1;
        std::vector<std::string> v2;
        parse(string, v1);
        parse(string_backup, v2);

        size_t size = v1.size();
        if (size != v2.size())
            return Errno::invalid_size;

        Errno err;
        for (size_t i{0}; i < size; ++i) {
            err = formatAndEmplaceBack(v1[i].empty() ? std::move(v2[i]) : std::move(v1[i]), vector);
            if (err)
                break;
        }
        return err;
    }

    template<typename S, typename T, typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    Errno parse(S&& string, S&& string_backup, std::vector<T>& vector, size_t size) {
        size_t size_before = vector.size();
        Errno err = String::parse(string, string_backup, vector);
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
    template<typename S, typename T, typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    Errno parse(S&& string, T* ptr, size_t size) {
        size_t idx_start{0}, idx_end{0}, idx{0};
        bool capture{false};
        Errno err;

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
             typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_intX_v<T> || Traits::is_floatX_v<T>)>>
    inline Errno parse(S&& string, T& vector) {
        auto array = vector.toArray(); // make sure the data is contiguous
        Errno err = parse(string, array.data(), array.size());
        vector = array.data();
        return err;
    }

    template<typename S, typename T, size_t n,
             typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    inline Errno parse(S&& string, std::array<T, n>& array) {
        return parse(string, array.data(), n);
    }

    template<typename S, typename T,
             typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_intX_v<T> || Traits::is_floatX_v<T>)>>
    inline Errno parse(S&& string, T& vector, size_t size) {
        auto array = vector.toArray(); // make sure the data is contiguous
        Errno err = parse(string, array.data(), size);
        vector = array.data();
        return err;
    }

    template<typename S, typename T, size_t n,
             typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    inline Errno parse(S&& string, std::array<T, n>& array, size_t size) {
        return parse(string, array.data(), size);
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
    template<typename S, typename T, typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    Errno parse(S&& string, S&& string_backup, T* ptr, size_t size) {
        std::vector<std::string> v1, v2;
        parse(string, v1);
        parse(string_backup, v2);

        if (size != v1.size() || size != v2.size())
            return Errno::invalid_size;

        Errno err;
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
             typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_intX_v<T> || Traits::is_floatX_v<T>)>>
    inline Errno parse(S&& string, S&& string_backup, T& vector) {
        auto array = vector.toArray(); // make sure the data is contiguous
        Errno err = parse(string, string_backup, array.data(), array.size());
        vector = array.data();
        return err;
    }

    template<typename S, typename T, size_t n,
             typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    inline Errno parse(S&& string, S&& string_backup, std::array<T, n>& array) {
        return parse(string, string_backup, array.data(), n);
    }

    template<typename S, typename T,
             typename = std::enable_if_t<Traits::is_string_v<S> && (Traits::is_intX_v<T> || Traits::is_floatX_v<T>)>>
    inline Errno parse(S&& string, S&& string_backup, T& vector, size_t size) {
        auto array = vector.toArray(); // make sure the data is contiguous
        Errno err = parse(string, string_backup, array.data(), size);
        vector = array.data();
        return err;
    }

    template<typename S, typename T, size_t n,
             typename = std::enable_if_t<Traits::is_string_v<S> && IS_CONVERTIBLE(T)>>
    inline Errno parse(S&& string, S&& string_backup, std::array<T, n>& array, size_t size) {
        return parse(string, string_backup, array.data(), size);
    }
}

#undef IS_CONVERTIBLE
