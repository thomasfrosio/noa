/**
 * @file Exception.h
 * @brief Various exceptions and error handling things.
 * @author Thomas - ffyr2w
 * @date 14 Sep 2020
 */
#pragma once

#include <string>
#include <exception>

#include "noa/util/string/Format.h"

namespace Noa {
    /** Main exception thrown by the noa. Usually caught in main(). */
    class NOA_API Exception : public std::exception {
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
         */
        template<typename... Args>
        inline Exception(const char* file, const char* function, const int line, Args&& ... args) {
            m_buffer = String::format("{}:{}:{}:\n", file, function, line) + String::format(args...);
        }

        [[nodiscard]] inline const char* what() const noexcept override { return m_buffer.data(); }
    };
}
