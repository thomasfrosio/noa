#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include "noa/common/Definitions.h"
#include "noa/common/string/Format.h"

namespace noa {
    /// Global (within ::noa) exception. Usually caught in main().
    class Exception : public std::exception {
    public:
        /// Format the error message, which is then accessible with what().
        /// \tparam Args         Any types supported by string::format.
        /// \param[in] file      File name.
        /// \param[in] function  Function name.
        /// \param[in] line      Line number.
        /// \param[in] args      Error message or arguments to format.
        /// \note "Zero" try-catch overhead: https://godbolt.org/z/v43Pzq
        template<typename... Args>
        Exception(const char* file, const char* function, int line, Args&& ... args)
                : m_buffer(format_(file, function, line, string::format(args...))) {}

        /// Returns the formatted error message of this (and only this) exception.
        [[nodiscard]] const char* what() const noexcept override {
            return m_buffer.data();
        }

        /// Returns the message of all of the nested exceptions, from the newest to the oldest exception.
        /// \details This gets the current exception and gets its message using what(). Then if the exception is
        ///          a std::nested_exception, i.e. it was thrown using std::throw_with_nested, it gets the nested
        ///          exceptions' messages until it reaches the last exception. These exceptions should inherit from
        ///          std::exception, otherwise we have no way to retrieve its message and a generic message is
        ///          returned instead saying that an unknown exception was thrown and the backtrace stops.
        [[nodiscard]] static std::vector<std::string> backtrace() noexcept {
            std::vector<std::string> message;
            backtrace_(message);
            return message;
        }

    protected:
        static std::string format_(const char* file, const char* function, int line,
                                   const std::string& message) {
            namespace fs = std::filesystem;
            size_t idx = std::string(file).rfind(std::string("noa") + fs::path::preferred_separator);
            return string::format("ERROR:{}:{}:{}: {}",
                                  idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                  function, line, message);
        }

        static void backtrace_(std::vector<std::string>& message,
                               const std::exception_ptr& exception_ptr = std::current_exception());

    protected:
        std::string m_buffer{};
    };

    /// Throw a nested noa::Exception if result evaluates to false.
    template<typename T>
    void throwIf(T&& result, const char* file, const char* function, int line) {
        if (result)
            std::throw_with_nested(noa::Exception(file, function, line, string::format("{}", std::forward<T>(result))));
    }
}

/// Throws a nested noa::Exception.
#define NOA_THROW(...) std::throw_with_nested(::noa::Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))

/// Throws a nested noa::Exception. Allows to modify the function name.
#define NOA_THROW_FUNC(func, ...) std::throw_with_nested(::noa::Exception(__FILE__, func, __LINE__, __VA_ARGS__))

/// Throws a nested exception if \a result is an error. \see the throwIf() overload for the current namespace.
#define NOA_THROW_IF(result) throwIf(result, __FILE__, __FUNCTION__, __LINE__)

#if defined(NOA_DEBUG) || defined(NOA_ENABLE_CHECKS_RELEASE)
/// Checks vs assertions:\n
/// - \b Assertions called via \e NOA_ASSERT are turned off when NOA_ENABLE_ASSERTS is not defined (see Assert.h).
///   They are using the C assert macro and calls abort() when the condition is not satisfied. They can be used
///   in noexcept(true) contexts.\n
/// - \b "Checks" are throwing noa::Exception when the condition is not satisfied. As such, the \e NOA_CHECK macro
///   should not be used in noexcept(true) contexts. They indicate an error and should really be seen as a way to
///   enforce/check the pre- or post-conditions of a function. The caller then has some flexibility on how to react
///   to this error, including 1) not catching the exception and let the program terminate, 2) log the exception and
///   the state of the program before exiting, or in the very rare case 3) catch the exception and try to recover
///   from it... These checks can be very useful even in Release builds, hence the CMake option and macro
///   NOA_ENABLE_CHECKS_RELEASE. Since they add a "throw" statement, they cannot be used in "device" code and is best
///   to not use them in performance critical scopes (e.g. hot loop).
#define NOA_CHECK(cond, ...) if (!(cond)) NOA_THROW(__VA_ARGS__)
#define NOA_CHECK_FUNC(func, cond, ...) if (!(cond)) NOA_THROW_FUNC(func, __VA_ARGS__)
#else
#define NOA_CHECK(cond, ...)
#define NOA_CHECK_FUNC(func, cond, ...)
#endif
