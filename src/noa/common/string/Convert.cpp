#include <cstdlib>
#include <type_traits>
#include <limits>
#include <cerrno>

#include "noa/common/Exception.h"
#include "noa/common/string/Format.h" // toUpperCopy
#include "noa/common/string/Convert.h"
#include "noa/common/traits/BaseTypes.h"

using namespace noa;

template<typename T>
T string::toInt(const std::string& str) {
    errno = 0;
    char* end;
    T out;

    if constexpr (noa::traits::is_uint_v<T>) {
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
                NOA_THROW("Out of range. \"{}\" is out of {} range", str, string::typeName<T>());
            out = static_cast<T>(tmp);
        }
    } else if constexpr (noa::traits::is_int_v<T>) {
        if constexpr (std::is_same_v<T, long long>) {
            out = std::strtoll(str.data(), &end, 10);
        } else if constexpr (std::is_same_v<T, long>) {
            out = std::strtol(str.data(), &end, 10);
        } else {
            long tmp = std::strtol(str.data(), &end, 10);
            if (tmp > std::numeric_limits<T>::max() || tmp < std::numeric_limits<T>::min())
                NOA_THROW("Out of range. \"{}\" is out of {} range", str, string::typeName<T>());
            out = static_cast<T>(tmp);
        }
    } else {
        static_assert(noa::traits::always_false_v<T>, "DEV: this should not be possible");
    }

    if (end == str.data())
        NOA_THROW("Cannot convert \"{}\" to {}", str, string::typeName<T>());
    else if (errno == ERANGE)
        NOA_THROW("Out of range. \"{}\" is out of {} range", str, string::typeName<T>());
    return out;
}

template int8_t string::toInt<int8_t>(const std::string& str);
template uint8_t string::toInt<uint8_t>(const std::string& str);
template short string::toInt<short>(const std::string& str);
template unsigned short string::toInt<unsigned short>(const std::string& str);
template int string::toInt<int>(const std::string& str);
template unsigned int string::toInt<unsigned int>(const std::string& str);
template long string::toInt<long>(const std::string& str);
template unsigned long string::toInt<unsigned long>(const std::string& str);
template long long string::toInt<long long>(const std::string& str);
template unsigned long long string::toInt<unsigned long long>(const std::string& str);

template<typename T>
T string::toFloat(const std::string& str) {
    errno = 0;
    char* end;
    T out;
    if constexpr (noa::traits::is_same_v<T, float>) {
        out = std::strtof(str.data(), &end);
    } else if constexpr (noa::traits::is_same_v<T, double>) {
        out = std::strtod(str.data(), &end);
    } else {
        static_assert(noa::traits::always_false_v<T>);
    }

    if (end == str.data())
        NOA_THROW("Invalid data. Cannot convert \"{}\" to {}", str, string::typeName<T>());
    else if (errno == ERANGE)
        NOA_THROW("Out of range. Cannot convert \"{}\" to {}", str, string::typeName<T>());
    return out;
}
template float string::toFloat<float>(const std::string& str);
template double string::toFloat<double>(const std::string& str);

bool string::toBool(const std::string& str) {
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
