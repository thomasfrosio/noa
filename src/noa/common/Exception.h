/// \file noa/common/Exception.h
/// \brief Various exceptions and error handling things.
/// \author Thomas - ffyr2w
/// \date 14 Sep 2020

#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"

namespace noa {
    /// Global (within ::noa) exception. Usually caught in main().
    class Exception : public std::exception {
    private:
        static thread_local std::string s_message;

    protected:
        std::string m_buffer{};

        NOA_HOST static std::string format_(const char* file, const char* function, int line,
                                            const std::string& message) {
            namespace fs = std::filesystem;
            size_t idx = std::string(file).rfind(std::string("noa") + fs::path::preferred_separator);
            return string::format("ERROR:{}:{}:{}: {}",
                                  idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                  function, line, message);
        }

        static void backtrace_(std::string& message,
                               const std::exception_ptr& exception_ptr = std::current_exception(),
                               size_t level = 0);

    public:
        /// Format the error message, which is then accessible with what().
        /// \tparam Args         Any types supported by string::format.
        /// \param[in] file      File name.
        /// \param[in] function  Function name.
        /// \param[in] line      Line number.
        /// \param[in] args      Error message or arguments to format.
        /// \note "Zero" try-catch overhead: https://godbolt.org/z/v43Pzq
        template<typename... Args>
        Exception(const char* file, const char* function, int line, Args&& ... args) {
            m_buffer = format_(file, function, line, string::format(args...));
        }

        [[nodiscard]] const char* what() const noexcept override {
            s_message.clear();
            backtrace_(s_message);
            return s_message.data();
        }
    };

    /// Throw a nested noa::Exception if result evaluates to false.
    template<typename T>
    NOA_IH void throwIf(T&& result, const char* file, const char* function, int line) {
        if (result)
            std::throw_with_nested(noa::Exception(file, function, line, string::format("{}", std::forward<T>(result))));
    }
}

/// Throws a nested noa::Exception.
#define NOA_THROW(...) std::throw_with_nested(::noa::Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))

/// Throws a nested noa::Exception.
#if defined(NOA_DEBUG) || defined(NOA_ENABLE_CHECKS_RELEASE)
#define NOA_CHECK(cond, ...) if (!(cond)) NOA_THROW(__VA_ARGS__)
#else
#define NOA_CHECK(cond, ...)
#endif

/// Throws a nested noa::Exception. Allows to modify the function name.
#define NOA_THROW_FUNC(func, ...) std::throw_with_nested(::noa::Exception(__FILE__, func, __LINE__, __VA_ARGS__))

/// Throws a nested exception if \a result is an error. \see the throwIf() overload for the current namespace.
#define NOA_THROW_IF(result) throwIf(result, __FILE__, __FUNCTION__, __LINE__)
