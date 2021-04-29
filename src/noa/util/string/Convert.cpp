#include "noa/util/string/Convert.h"

#include <cstdlib>
#include <type_traits>
#include <limits>
#include <cerrno>

#include "noa/Exception.h"
#include "noa/util/string/Format.h"     // toUpperCopy
#include "noa/util/traits/BaseTypes.h"

using namespace Noa;

template<typename T>
T String::toInt(const std::string& str) {
    errno = 0;
    char* end;
    T out;

    if constexpr (Noa::Traits::is_uint_v<T>) {
        // Shortcut: empty string or negative number.
        size_t idx = str.find_first_not_of(" \t");
        if (idx == std::string::npos) {
            NOA_THROW("Cannot convert an empty string to an unsigned integer");
        } else if (str[idx] == '-') {
            if (str.size() >= idx + 1 && str[idx + 1] > 47 && str[idx + 1] < 58)
                NOA_THROW("Out of range. Cannot convert \"{}\" to an unsigned integer.", str);
            else
                NOA_THROW("Cannot convert \"{}\" to an unsigned integer", str);
        }
        if constexpr (std::is_same_v<T, unsigned long long>) {
            out = std::strtoull(str.data(), &end, 10);
        } else if constexpr (std::is_same_v<T, unsigned long>) {
            out = std::strtoul(str.data(), &end, 10);
        } else {
            unsigned long tmp = std::strtoul(str.data(), &end, 10);
            if (tmp > std::numeric_limits<T>::max() || tmp < std::numeric_limits<T>::min())
                NOA_THROW("Out of range. \"{}\" is out of {} range", str, String::typeName<T>());
            out = static_cast<T>(tmp);
        }
    } else if constexpr (Noa::Traits::is_int_v<T>) {
        if constexpr (std::is_same_v<T, long long>) {
            out = std::strtoll(str.data(), &end, 10);
        } else if constexpr (std::is_same_v<T, long>) {
            out = std::strtol(str.data(), &end, 10);
        } else {
            long tmp = std::strtol(str.data(), &end, 10);
            if (tmp > std::numeric_limits<T>::max() || tmp < std::numeric_limits<T>::min())
                NOA_THROW("Out of range. \"{}\" is out of {} range", str, String::typeName<T>());
            out = static_cast<T>(tmp);
        }
    } else {
        static_assert(Noa::Traits::always_false_v<T>, "DEV: this should not be possible");
    }

    if (end == str.data())
        NOA_THROW("Cannot convert \"{}\" to {}", str, String::typeName<T>());
    else if (errno == ERANGE)
        NOA_THROW("Out of range. \"{}\" is out of {} range", str, String::typeName<T>());
    return out;
}

template int8_t String::toInt<int8_t>(const std::string& str);
template uint8_t String::toInt<uint8_t>(const std::string& str);
template short String::toInt<short>(const std::string& str);
template unsigned short String::toInt<unsigned short>(const std::string& str);
template int String::toInt<int>(const std::string& str);
template unsigned int String::toInt<unsigned int>(const std::string& str);
template long String::toInt<long>(const std::string& str);
template unsigned long String::toInt<unsigned long>(const std::string& str);
template long long String::toInt<long long>(const std::string& str);
template unsigned long long String::toInt<unsigned long long>(const std::string& str);

template<typename T>
T String::toFloat(const std::string& str) {
    errno = 0;
    char* end;
    T out;
    if constexpr (Noa::Traits::is_same_v<T, float>) {
        out = std::strtof(str.data(), &end);
    } else if constexpr (Noa::Traits::is_same_v<T, double>) {
        out = std::strtod(str.data(), &end);
    } else {
        static_assert(Noa::Traits::always_false_v<T>);
    }

    if (end == str.data())
        NOA_THROW("Invalid data. Cannot convert \"{}\" to {}", str, String::typeName<T>());
    else if (errno == ERANGE)
        NOA_THROW("Out of range. Cannot convert \"{}\" to {}", str, String::typeName<T>());
    return out;
}
template float String::toFloat<float>(const std::string& str);
template double String::toFloat<double>(const std::string& str);

bool String::toBool(const std::string& str) {
    if (str == "1" || str == "true")
        return true;
    else if (str == "0" || str == "false")
        return false;

    std::string str_up = toUpperCopy(str);
    if (str_up == "TRUE" || str_up == "Y" || str_up == "YES" || str_up == "ON") {
        return true;
    } else if (str_up == "FALSE" || str_up == "N" || str_up == "NO" || str_up == "OFF") {
        return false;
    } else {
        NOA_THROW("Cannot convert \"{}\" to bool", str);
    }
}
