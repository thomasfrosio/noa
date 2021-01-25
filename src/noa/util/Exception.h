/**
 * @file Exception.h
 * @brief Various exceptions and error handling things.
 * @author Thomas - ffyr2w
 * @date 14 Sep 2020
 */
#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include "noa/util/string/Format.h"

namespace Noa {
    /** Main exception thrown by the noa. Usually caught in main(). */
    class Exception : public std::exception {
    protected:
        std::string m_buffer{};

    public:
        /**
         * Format the error message, which is then accessible with what() or print().
         * @tparam Args         Any types supported by @c fmt::format.
         * @param[in] file      File name.
         * @param[in] function  Function name.
         * @param[in] line      Line number.
         * @param[in] args      Error message to format.
         *
         * @note "Zero" overhead: https://godbolt.org/z/v43Pzq
         */
        template<typename... Args>
        inline Exception(const char* file, const char* function, const int line, Args&& ... args) {
            namespace fs = std::filesystem;
            m_buffer = String::format("{}:{}:{}: ", fs::path(file).filename().string(), function, line) +
                       String::format(args...);
        }

        [[nodiscard]] inline const char* what() const noexcept override { return m_buffer.data(); }
    };
}

#define NOA_ERROR(...) std::throw_with_nested(Noa::Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))
#define NOA_ERROR_FUNC(func, ...) std::throw_with_nested(Noa::Exception(__FILE__, func, __LINE__, __VA_ARGS__))
