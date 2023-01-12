#pragma once
#ifndef NOA_STRING_PARSE_
#error "Cannot include this private header"
#endif

namespace noa::string {
    template<typename T, typename>
    T parse(const std::string& string, int& error) noexcept {
        if constexpr (traits::is_string_v<T>) {
            error = 0;
            return string;
        } else if constexpr (traits::is_float_v<T>) {
            return toFloat<T>(string, error);
        } else if constexpr (traits::is_bool_v<T>) {
            return toBool(string, error);
        } else if constexpr (traits::is_int_v<T>) {
            return toInt<T>(string, error);
        } else {
            static_assert(traits::always_false_v<T>);
        }
    }

    template<typename T, typename>
    T parse(const std::string& string) {
        int error;
        T out = parse<T>(string, error);
        if (error)
            NOA_THROW(parseErrorMessage<T>(string, error));
        return out;
    }

    template<typename T, typename>
    std::vector<T> parse(const std::vector<std::string>& vector, int& error) noexcept {
        error = 0;
        if constexpr (traits::is_string_v<T>) {
            return vector;
        } else {
            std::vector<T> output;
            for (const auto& string: vector) {
                T value;
                if constexpr (traits::is_float_v<T>) {
                    value = toFloat<T>(string, error);
                } else if constexpr (traits::is_bool_v<T>) {
                    value = toBool(string, error);
                } else if constexpr (traits::is_int_v<T>) {
                    value = toInt<T>(string, error);
                } else {
                    static_assert(traits::always_false_v<T>);
                }
                if (error)
                    break;
                output.emplace_back(value);
            }
            return output;
        }
    }

    template<typename T, typename>
    std::vector<T> parse(const std::vector<std::string>& vector) {
        int error;
        std::vector<T> out = parse<T>(vector, error);
        if (error)
            NOA_THROW(parseErrorMessage<T>(vector[out.size()], error));
        return out;
    }

    template<typename T, typename>
    std::string parseErrorMessage(const std::string& str, int error) {
        switch (error) {
            case 1:
                return format("Failed to convert \"{}\" to {}", str, human<T>());
            case 2:
                return format("Out of range. \"{}\" is out of {} range", str, human<T>());
            case 3:
                return format("Out of range. Cannot convert negative number \"{}\" to {}", str, human<T>());
            default:
                return "";
        }
    }
}

namespace noa::string {
    template<typename T, typename>
    T toInt(const std::string& str, int& error) noexcept {
        errno = 0;
        error = 0;
        char* end;
        T out;

        if constexpr (traits::is_uint_v<T>) {
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
        } else if constexpr (traits::is_int_v<T>) {
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
            static_assert(traits::always_false_v<T>, "DEV: this should not be possible");
        }

        if (end == str.data())
            error = 1;
        else if (errno == ERANGE)
            error = 2;
        return out;
    }

    template<typename T, typename>
    T toInt(const std::string& str) {
        int error;
        T out = toInt<T>(str, error);
        if (error)
            NOA_THROW(parseErrorMessage<T>(str, error));
        return out;
    }

    template<typename T, typename>
    T toFloat(const std::string& str, int& error) noexcept {
        errno = 0;
        error = 0;
        char* end;
        T out;
        if constexpr (traits::is_almost_same_v<T, float>) {
            out = std::strtof(str.data(), &end);
        } else if constexpr (traits::is_almost_same_v<T, double>) {
            out = std::strtod(str.data(), &end);
        } else {
            static_assert(traits::always_false_v<T>);
        }

        if (end == str.data())
            error = 1;
        else if (errno == ERANGE)
            error = 2;
        return out;
    }

    template<typename T, typename>
    T toFloat(const std::string& str) {
        int error;
        T out = toFloat<T>(str, error);
        if (error)
            NOA_THROW(parseErrorMessage<T>(str, error));
        return out;
    }

    bool toBool(const std::string& str, int& error) noexcept {
        error = 0;
        if (str == "1" || str == "true")
            return true;
        else if (str == "0" || str == "false")
            return false;

        std::string str_up = upper(str);
        if (str_up == "TRUE" || str_up == "Y" || str_up == "YES" || str_up == "ON") {
            return true;
        } else if (str_up == "FALSE" || str_up == "N" || str_up == "NO" || str_up == "OFF") {
            return false;
        } else {
            error = 1;
            return false;
        }
    }

    bool toBool(const std::string& str) {
        int error;
        bool out = toBool(str, error);
        if (error)
            NOA_THROW(parseErrorMessage<bool>(str, error));
        return out;
    }
}
