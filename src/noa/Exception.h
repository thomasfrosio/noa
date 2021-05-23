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

        static std::string format_(const char* file, const char* function, int line, const std::string& message) {
            namespace fs = std::filesystem;
            size_t idx = std::string(file).rfind(std::string("noa") + fs::path::preferred_separator);
            return String::format("{}:{}:{}: {}",
                                  idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                  function, line, message);
        }

    public:
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
        NOA_HOST Exception(const char* file, const char* function, int line, Args&& ... args) {
            m_buffer = format_(file, function, line, String::format(args...));
        }

        [[nodiscard]] NOA_HOST const char* what() const noexcept override { return m_buffer.data(); }
    };

    /// Throw a nested Noa::Exception if result evaluates to false.
    template<typename T>
    NOA_IH void throwIf(T&& result, const char* file, const char* function, int line) {
        if (result)
            std::throw_with_nested(Noa::Exception(file, function, line, String::format("{}", std::forward<T>(result))));
    }
}

/// Throws a nested Noa::Exception.
#define NOA_THROW(...) std::throw_with_nested(::Noa::Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))

/// Throws a nested Noa::Exception. Allows to modify the function name.
#define NOA_THROW_FUNC(func, ...) std::throw_with_nested(::Noa::Exception(__FILE__, func, __LINE__, __VA_ARGS__))

/// Throws a nested exception if @a result is an error. @see the throwIf() overload for the current namespace.
#define NOA_THROW_IF(result) throwIf(result, __FILE__, __FUNCTION__, __LINE__)
