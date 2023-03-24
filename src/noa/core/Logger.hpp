#pragma once

#include "noa/core/Definitions.hpp"

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wshadow"
    #pragma GCC diagnostic ignored "-Wformat-nonliteral"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

#include <string>
#include <cstdint>
#include <memory>
#include <vector>
#include <exception>

namespace noa {
    /// By default, Logger contains two thread safe sinks, one controls the console, the other controls the log file.
    /// \details Sinks have specific settings, such as log formatting and log level, and can be can be shared with
    ///          other loggers. Loggers are not registered to some static registry, however, in most cases, functions
    ///          should be able to access a pre-existing logger without having to take it as input argument. To fill
    ///          gap, one can create a Logger static (e.g. the static noa::Session::logger used by the functions in
    ///          the noa namespace) or one can use the \a spdlog register (i.e. see \c spdlog::register_logger).
    /// \see Logger() to initialize the logger.
    /// \see Logger::get() to get the underlying logger.
    /// \see Logger::sinks() to get the underlying sinks.
    /// \see Logger::setLevel() to set the level of the console sink.
    /// \see Logger::trace|info|warn|debug|error() to log messages at specific log levels.
    class Logger {
    private:
        std::shared_ptr<spdlog::logger> m_logger;

    public:
        /// Log levels:
        ///   - \a VERBOSE: activates the error, warn, info, debug and trace levels.
        ///   - \a BASIC:   activates the error, warn and info levels.
        ///   - \a ALERT:   activates the error and warn levels.
        ///   - \a SILENT:  deactivates all logging.
        enum Level : uint { SILENT, ALERT, BASIC, VERBOSE };

    public: // static functions
        /// Creates a logger connected to existing sinks.
        /// \param name      Name of the logger.
        /// \param sinks     Sinks to link to the logger
        /// \return          The owning pointer of the created logger, which also owns the \a sinks.
        static std::shared_ptr<spdlog::logger> create(std::string_view name,
                                                      const std::vector<spdlog::sink_ptr>& sinks);

        /// Creates a logger with a thread-safe console sink and, optionally, a thread-safe basic file sink.
        /// \param name         Name of the logger. The file sink prefixes all entry with this name.
        /// \param filename     Filename used by the file sink. If empty, the file sink is not created.
        ///                     If the file exists, logging messages will be appended.
        /// \param verbosity    Level of verbosity of the console sink. The file sink is set to Level::VERBOSE.
        /// \returns            The owning pointer of the created logger, which also owns the created sink(s).
        /// \note The console sink is the first sink. The file sink, if any, comes next.
        static std::shared_ptr<spdlog::logger> create(std::string_view name,
                                                      std::string_view filename,
                                                      Level verbosity_console);

        /// Sets the level of \a verbosity of a \a sink.
        static void setLevel(spdlog::sink_ptr& sink, Level verbosity);

    public:
        /// Creates an empty instance.
        Logger() = default;

        /// Creates a new logger. \see Logger::create for more details.
        Logger(std::string_view name, std::string_view filename, Level verbosity)
                : m_logger(create(name, filename, verbosity)) {}

        /// Creates a new logger using existing sinks. \see Logger::create for more details.
        Logger(std::string_view name, const std::vector<spdlog::sink_ptr>& sinks)
                : m_logger(create(name, sinks)) {}

        /// Returns a reference of the underlying \a spdlog logger.
        std::shared_ptr<spdlog::logger>& get() { return m_logger; }

        /// Returns a reference of the underlying \a spdlog sinks.
        std::vector<spdlog::sink_ptr>& sinks() { return m_logger->sinks(); }

        template<typename... Args>
        void trace(Args&& ... args) { m_logger->trace(std::forward<Args>(args)...); }

        template<typename... Args>
        void info(Args&& ... args) { m_logger->info(std::forward<Args>(args)...); }

        template<typename... Args>
        void warn(Args&& ... args) { m_logger->warn(std::forward<Args>(args)...); }

        template<typename... Args>
        void error(Args&& ... args) { m_logger->error(std::forward<Args>(args)...); }

        template<typename... Args>
        [[maybe_unused]] void debug([[maybe_unused]] Args&& ... args) {
        #ifdef NOA_DEBUG
            m_logger->debug(std::forward<Args>(args)...);
        #endif
        }
    };
}
