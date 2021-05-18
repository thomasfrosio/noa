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

#include "noa/Definitions.h"
#include "noa/util/string/Format.h"

namespace Noa {
    /** Global (within ::Noa) exception. Usually caught in main(). */
    class Exception : public std::exception {
    protected:
        std::string m_buffer{};

        static std::string format_(const char* file, const char* function, const int line, const std::string& message) {
            namespace fs = std::filesystem;
            size_t idx = std::string(file).rfind(std::string("noa") + fs::path::preferred_separator);
            return String::format("{}:{}:{}: {}",
                                  idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                  function, line, message);
        }

    public:
        NOA_HOST Exception(const char* file, const char* function, const int line, const std::string& message) {
            m_buffer = format_(file, function, line, message);
        }

        template<typename T>
        NOA_HOST Exception(const char* file, const char* function, const int line, T arg) {
            m_buffer = format_(file, function, line, String::format("{}", arg));
        }

        /**
         * Format the error message, which is then accessible with what().
         * @tparam Args         Any types supported by String::format.
         * @param[in] file      File name.
         * @param[in] function  Function name.
         * @param[in] line      Line number.
         * @param[in] args      Error message or arguments to format.
         * @note "Zero" try-catch overhead: https://godbolt.org/z/v43Pzq
         */
        template<typename... Args>
        NOA_HOST Exception(const char* file, const char* function, const int line, Args&& ... args) {
            m_buffer = format_(file, function, line, String::format(args...));
        }

        [[nodiscard]] NOA_HOST const char* what() const noexcept override { return m_buffer.data(); }
    };

    /**
     * Throw a nested Noa::Exception if result evaluates to false.
     * @note    This is often used via the macro NOA_THROW_IF.
     * @note    This template function is specialized, e.g. cudaError_t.
     */
    template<typename T>
    NOA_IH void throwIf(T result, const char* file, const char* function, const int line) {
        if (result)
            std::throw_with_nested(Noa::Exception(file, function, line, result));
    }
}

/// Throws a nested Noa::Exception.
#define NOA_THROW(...) std::throw_with_nested(::Noa::Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))

/// Throws a nested Noa::Exception. Allows to modify the function name.
#define NOA_THROW_FUNC(func, ...) std::throw_with_nested(::Noa::Exception(__FILE__, func, __LINE__, __VA_ARGS__))

/// Throws a nested exception if @a result is an error. @a see Noa::throwIf.
#define NOA_THROW_IF(result) ::Noa::throwIf(result, __FILE__, __FUNCTION__, __LINE__)
