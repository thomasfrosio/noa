/// \file noa/common/Logger.h
/// \brief The default logger used by the core.
/// \author Thomas - ffyr2w
/// \date 25 Jul 2020

#pragma once

#include "noa/common/Definitions.h"

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
    /// A Logger contains two thread safe sinks, one controls the console, the other controls the log file.
    /// \details Sinks have specific settings, such as log formatting and log level, and can be can be shared with
    ///          other loggers. Loggers are not registered to some static registry, however, in most cases, functions
    ///          should be able to access a pre-existing logger without having to take it as input argument. To fill
    ///          gap, one can make a Logger static (e.g. the static noa::Session::logger used by the functions in
    ///          the noa namespace) or one can use the \a spdlog register (i.e. see \c spdlog::register_logger).
    /// \see Logger() to initialize the logger.
    /// \see Logger::get() to get the underlying logger.
    /// \see Logger::getSinks() to get the underlying sinks.
    /// \see Logger::setConsoleLevel() to set the level of the console sink.
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
        enum : uint { SILENT, ALERT, BASIC, VERBOSE };

    public: // static functions
        /// Creates a logger connected to existing sinks.
        /// \param name      Name of the logger.
        /// \param sinks     Sinks to link to the logger
        /// \return          The owning pointer of the created logger, which also owns the \a sinks.
        NOA_HOST static std::shared_ptr<spdlog::logger> create(const std::string& name,
                                                               const std::vector<spdlog::sink_ptr>& sinks);

        /// Creates a logger with a thread-safe console sink and a thread-safe basic file sink.
        /// \param name          Name of the logger. The file sink prefixes all entry by this name.
        /// \param filename      Filename used by the file sink. If the file exists, it is not overwritten.
        /// \param verbosity     Level of verbosity for the console sink. The file sink is set to \c Logger::VERBOSE.
        /// \returns             The owning pointer of the created logger, which also owns the created sinks.
        NOA_HOST static std::shared_ptr<spdlog::logger> create(const std::string& name,
                                                               const std::string& filename,
                                                               uint verbosity);

        /// Sets the level of \a verbosity of a \a sink.
        NOA_HOST static void setSinkLevel(spdlog::sink_ptr& sink, uint verbosity);

    public:
        /// Creates an empty instance.
        Logger() = default;

        /// Creates a new logger. \see Logger::create for more details.
        NOA_HOST Logger(const std::string& name, const std::string& filename, uint verbosity)
                : m_logger(create(name, filename, verbosity)) {}

        /// Creates a new logger using existing sinks. \see Logger::create for more details.
        NOA_HOST Logger(const std::string& name, const std::vector<spdlog::sink_ptr>& sinks)
                : m_logger(create(name, sinks)) {}

        /// (Re)Sets the underlying logger with a new logger. \see Logger::create for more details.
        NOA_HOST void set(const std::string& name, const std::string& filename, uint verbosity) {
            m_logger = create(name, filename, verbosity);
        }

        /// (Re)Sets the underlying logger with a new logger. \see Logger::create for more details.
        NOA_HOST void set(const std::string& name, const std::vector<spdlog::sink_ptr>& sinks) {
            m_logger = create(name, sinks);
        }

        /// Returns a reference of the underlying \a spdlog logger.
        NOA_HOST std::shared_ptr<spdlog::logger>& get() { return m_logger; }

        /// Returns a reference of the underlying \a spdlog sinks.
        NOA_HOST std::vector<spdlog::sink_ptr>& getSinks() { return m_logger->sinks(); }

        template<typename... Args>
        NOA_HOST void trace(Args&& ... args) { m_logger->trace(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_HOST void info(Args&& ... args) { m_logger->info(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_HOST void warn(Args&& ... args) { m_logger->warn(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_HOST void error(Args&& ... args) { m_logger->error(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_HOST [[maybe_unused]] void debug(Args&& ... args) {
#ifdef NOA_DEBUG
            m_logger->debug(std::forward<Args>(args)...);
#endif
        }

        /// Sets the verbosity of the console sink (the file sink is not affected).
        /// \note It is assumed that the console sink is at getSinks()[0].
        NOA_HOST void setLevelConsole(uint verbosity) {
            setSinkLevel(getSinks()[0], verbosity);
        }

        /// Sets the verbosity of the console sink (the file sink is not affected).
        /// \note It is assumed that the console sink is at getSinks()[0].
        NOA_HOST void setLevelFile(uint verbosity) {
            setSinkLevel(getSinks()[1], verbosity);
        }

        /// Unwind all the nested exceptions that were thrown and caught.
        /// \note    This function is meant to be called from the catch scope of main() before exiting the program.
        /// \note    It is meant to be called using the default values.
        NOA_HOST void backtrace(const std::exception_ptr& exception_ptr = std::current_exception(), size_t level = 0);
    };
}
