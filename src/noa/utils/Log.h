/**
 * @file Log.h
 * @brief Logging system used throughout the core and the programs.
 * @author Thomas - ffyr2w
 * @date 25 Jul 2020
 */
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

// Add some fmt functionalities that spdlog doesn't import.
#include <spdlog/fmt/bundled/compile.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/bundled/os.h>
#include <spdlog/fmt/bundled/chrono.h>
#include <spdlog/fmt/bundled/color.h>

#include "noa/API.h"


/// Base level namespace of the entire core.
namespace Noa {

    /**
     * @brief   Logging system of noa core. Also contains an additional logger for the app using noa.
     * @details Static class containing the logger for the core (noa) and the app (akira, etc.).
     *          Initialize it with `::Noa::Log::Init(...)` before using the `::Noa::Log::getCoreLogger(...)`
     *          and `::Noa::Log::getAppLogger(...)` static methods.
     */
    class NOA_API Log {
    public:

        /**
         * verbose: off, error, warn, info, debug, trace
         * basic:   off, error, warn, info
         * alert:   off, error, warn
         * silent:  off
         */
        enum class level : uint8_t {
            silent, alert, basic, verbose
        };

    private:
        static std::shared_ptr<spdlog::logger> s_core_logger;
        static std::shared_ptr<spdlog::logger> s_app_logger;

    public:
        /**
         * Initialize the core and the app logger.
         * @param[in] filename  Log filename of the core and app logger.
         * @param[in] prefix    Prefix displayed before each logging entry of the app logger.
         *                      This _cannot_ be "NOA", since it is already used by the core logger.
         * @param[in] verbosity Level of verbosity for the stdout sink. The logfile isn't affected
         *                      and is always set to level::verbose.
         * @note                One must initialize the loggers via this function _before_ using
         *                      anything in the Noa namespace.
         */
        static void Init(const char* filename,
                         const char* prefix,
                         level verbosity = level::verbose);

        /** Get the app logger prefix */
         static inline const std::string& prefix() {
             return s_app_logger->name();
         }

        /**
         * Set the log level of the stdout sink. The log file isn't affected.
         * @param verbosity     Level of verbosity for the stdout sink. The logfile isn't affected
         *                      and is always set to level::verbose.
         */
        static inline void setLevel(level verbosity) {
            setSinkLevel(s_core_logger->sinks()[1], verbosity);
        }


        /**
         *  Set the log level of the stdout sink. The log file isn't affected.
         * @param verbosity     Level of verbosity for the stdout sink. The logfile isn't affected
         *                      and is always set to level::verbose.
         *                      - 0: silent
         *                      - 1: alert
         *                      - 2: basic
         *                      - 3: verbose
         * @return              Whether or not the verbosity was set.
         */
        static inline bool setLevel(int verbosity) {
            switch (verbosity) {
                case 0:
                    setSinkLevel(s_core_logger->sinks()[1], level::silent);
                case 1:
                    setSinkLevel(s_core_logger->sinks()[1], level::alert);
                case 2:
                    setSinkLevel(s_core_logger->sinks()[1], level::basic);
                case 3:
                    setSinkLevel(s_core_logger->sinks()[1], level::verbose);
                default:
                    return false;
            }
        }


        /**
         * @brief               Get the core logger.
         * @details             The logger have the following methods to output something:
         *                      log(...), debug(...), trace(...), info(...), warn(...) and error(...).
         * @return              The shared pointer of the core logger.
         *
         * @note                Usually called using the `NOA_CORE_*` definitions.
         */
        static inline std::shared_ptr<spdlog::logger>& getCoreLogger() { return s_core_logger; }

        /**
         * @brief               Get the app logger. See `::Noa::Log::getCoreLogger()->trace` for more details.
         * @return              The shared pointer of the app logger.
         *
         * @note                Usually called using the `NOA_LOG_*` definitions.
         */
        static inline std::shared_ptr<spdlog::logger>& getAppLogger() { return s_app_logger; }

    private:
        static inline void setSinkLevel(spdlog::sink_ptr& sink, level verbosity) {
            switch (verbosity) {
                case level::verbose:
                    sink->set_level(spdlog::level::trace);
                    break;
                case level::basic:
                    sink->set_level(spdlog::level::info);
                    break;
                case level::alert:
                    sink->set_level(spdlog::level::warn);
                    break;
                case level::silent:
                    sink->set_level(spdlog::level::off);
                    break;
            }
        }
    };
}
