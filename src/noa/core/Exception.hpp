#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/utils/Strings.hpp"

#ifdef NOA_IS_OFFLINE
#include <exception>
#include <filesystem>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace noa {
    /// Exception type used in ::noa.
    class Exception : public std::exception {
    public:
        /// Returns the message of all of the nested exceptions, from the newest to the oldest exception.
        /// \details This gets the current exception and gets its message using what(). Then if the exception is
        ///          a std::nested_exception, i.e. it was thrown using std::throw_with_nested, it gets the nested
        ///          exceptions' messages until it reaches the last exception. These exceptions should inherit from
        ///          std::exception, otherwise we have no way to retrieve its message and a generic message is
        ///          returned instead saying that an unknown exception was thrown and the backtrace stops.
        /// \example
        /// \code
        /// const std::vector<std::string> backtrace_vector = noa::Exception::backtrace();
        /// std::string backtrace_message;
        /// for (i64 i{}; auto& message: backtrace_vector)
        ///     backtrace_message += fmt::format("[{}]: {}\n", i++, message);
        /// \endcode
        [[nodiscard]] static auto backtrace() noexcept -> std::vector<std::string> {
            std::vector<std::string> message;
            backtrace_(message);
            return message;
        }

    public:
        /// Format the error message, which is then accessible with what().
        /// \param[in] file      File name.
        /// \param[in] function  Function name.
        /// \param[in] line      Line number.
        /// \param[in] message   Error message.
        Exception(const char* file, const char* function, std::uint_least32_t line, std::string_view message) :
            m_buffer(format_(file, function, line, message)) {}

        /// Returns the formatted error message of this exception.
        [[nodiscard]] auto what() const noexcept -> const char* override{
            return m_buffer.data();
        }

    protected:
        static auto format_(
            const char* file,
            const char* function,
            std::uint_least32_t line,
            const std::string_view& message
        ) -> std::string;

        static void backtrace_(
            std::vector<std::string>& message,
            const std::exception_ptr& exception_ptr = std::current_exception()
        );

    protected:
        std::string m_buffer{};
    };

    namespace guts {
        template<typename... Ts>
        struct FormatWithLocation {
            fmt::format_string<Ts...> fmt;
            std::source_location location;

            template<typename T>
            consteval /*implicit*/ FormatWithLocation(
                const T& s,
                const std::source_location& l = std::source_location::current()
            ) : fmt(s), location(l) {} // fmt checks at compile time that "s" is compatible with "Ts"

            /*implicit*/ FormatWithLocation(
                const fmt::runtime_format_string<char>& s,
                const std::source_location& l = std::source_location::current()
            ) : fmt(s), location(l) {} // no checks
        };
    }

    /// Throws an Exception with an error message and a specific, i.e. non-defaulted, source location.
    /// The format string is checked at compile time by default, except if a fmt::basic_runtime
    /// (the return type of fmt::runtime) is passed.
    template<typename... Ts>
    [[noreturn]] constexpr void panic_at_location(
        const std::source_location& location,
        fmt::format_string<Ts...> fmt,
        Ts&&... args
    ) {
        std::throw_with_nested(
                Exception(location.file_name(), location.function_name(), location.line(),
                          fmt::format(fmt::runtime(fmt), std::forward<Ts>(args)...))
        );
    }

    /// Throws an Exception with an error message already formatted.
    /// This is be equivalent to panic(fmt::runtime(message));
    [[noreturn]] inline void panic_runtime(
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) {
        panic_at_location(location, fmt::runtime(message));
    }

    /// Throws an Exception with no error message other than the source location.
    [[noreturn]] inline void panic(const std::source_location& location = std::source_location::current()) {
        panic_at_location(location, "");
    }

    /// Throws an Exception with an error message and the current source location.
    template<typename... Ts>
    [[noreturn]] constexpr void panic(guts::FormatWithLocation<std::type_identity_t<Ts>...> fmt, Ts&&... args) {
        panic_at_location(fmt.location, fmt.fmt, std::forward<Ts>(args)...);
    }

    template<typename... Ts>
    constexpr void check_at_location(
        const std::source_location& location,
        auto&& expression,
        fmt::format_string<Ts...> fmt,
        Ts&&... args
    ) {
        if (expression) {
            /*do nothing*/
        } else {
            panic_at_location(location, fmt, std::forward<Ts>(args)...);
        }
    }

    /// Throws an Exception with an error message already formatted.
    /// This is be equivalent to panic(fmt::runtime(message));
    void check_runtime(
        auto&& expression,
        std::string_view message,
        const std::source_location& location = std::source_location::current()
    ) {
        if (expression) {
            /*do nothing*/
        } else {
            panic_at_location(location, fmt::runtime(message));
        }
    }

    /// If the expression evaluates to false, throw an exception with no error message. Otherwise, do nothing.
    constexpr void check(auto&& expression, const std::source_location& location = std::source_location::current()) {
        if (expression) {
            /*do nothing*/
        } else {
            panic_at_location(location, "");
        }
    }

    /// Throws an Exception if the expression evaluates to false.
    template<typename... Ts>
    constexpr void check(auto&& expression, guts::FormatWithLocation<std::type_identity_t<Ts>...> fmt, Ts&&... args) {
        if (expression) {
            /*do nothing*/
        } else {
            panic_at_location(fmt.location, fmt.fmt, std::forward<Ts>(args)...);
        }
    }

    [[noreturn]] inline void panic_no_gpu_backend(const std::source_location& location = std::source_location::current()) {
        // FIXME Should we just std::terminate?
        panic_at_location(location, "Built without GPU support");
    }
}

#endif
