/**
 * @file Log.h
 * @brief Logger used throughout the core.
 * @author Thomas - ffyr2w
 * @date 25 Jul 2020
 */
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <string>
#include <cstdint>
#include <memory>
#include <vector>
#include <exception>

#include "noa/Definitions.h"

namespace Noa {
    /**
     * Interface with the static loggers.
     * @see Log::init() to initialize the loggers. Must be called before anything.
     * @see Log::get() to get the logger and log messages.
     * @see Log::setLevel() to set the level of the stdout sink.
     * @see Log::trace|info|warn|debug|error() to log messages at specific log levels.
     */
    class Log {
    public:
        /**
         * Log levels compared to the @c spdlog levels:
         *   - @c verbose: off, error, warn, info, debug, trace
         *   - @c basic:   off, error, warn, info
         *   - @c alert:   off, error, warn
         *   - @c silent:  off
         * @note This is not an enum class, since the verbosity is often an user input and it is
         *       much easier to get an uint from the Inputs.
         */
        enum Level : uint { SILENT, ALERT, BASIC, VERBOSE };

    private:
        static std::shared_ptr<spdlog::logger> s_logger;

    public:
        /**
         * @param[in] filename  Log filename used by all file sinks.
         * @param[in] verbosity Level of verbosity for the stdout sink. The log file isn't affected
         *                      and is always set to level::verbose.
         * @note One must initialize the loggers before using anything in the Noa namespace.
         */
        NOA_HOST static void init(const std::string& filename, uint verbosity);
        NOA_IH static void init(const std::string& filename) { init(filename, Level::BASIC); }

        NOA_IH static std::shared_ptr<spdlog::logger>& get() { return s_logger; }
        NOA_IH static spdlog::sink_ptr& getFile() { return s_logger->sinks()[0]; }
        NOA_IH static spdlog::sink_ptr& getCmdLine() { return s_logger->sinks()[1]; }

        template<typename... Args>
        NOA_IH static void trace(Args&& ... args) { s_logger->trace(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_IH static void info(Args&& ... args) { s_logger->info(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_IH static void warn(Args&& ... args) { s_logger->warn(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_IH static void error(Args&& ... args) { s_logger->error(std::forward<Args>(args)...); }

        template<typename... Args>
        NOA_IH static void debug(Args&& ... args) {
#ifdef NOA_DEBUG
            s_logger->debug(std::forward<Args>(args)...);
#endif
        }

        /** Sets the level of verbosity for the stdout sink (file sink is not affected). */
        NOA_HOST static void setLevel(uint verbosity);

        /**
         * Unwind all the nested exceptions that were thrown and caught.
         * @note    This function is meant to be called from the catch scope of main() before exiting the program.
         * @note    It is meant to be called using the default values, i.e. Noa::Log::backtrace();
         */
        NOA_HOST static void backtrace(const std::exception_ptr& exception_ptr = std::current_exception(),
                                       size_t level = 0);
    };
}
