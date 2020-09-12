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

#include "noa/api.h"


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
         * @brief               Initialize the core and the app logging system.
         * @param[in] filename  Log filename of the core and app logger. Default=`"noa.log"`
         * @param[in] prefix    Prefix displayed before each logging entry of the app logger. Default=`"APP"`
         *
         * @note                One must initialize the loggers via this function _before_ using noa.
         */
        static void Init(const char* filename = "noa.log", const char* prefix = "NOA");

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
        static std::shared_ptr<spdlog::logger> s_core_logger;
        static std::shared_ptr<spdlog::logger> s_app_logger;
    };


    /**
     * @brief   Main exception thrown by noa. Usually caught in the main().
     */
    class NOA_API ErrorCore : public std::exception {
    public:

        /**
         * @brief                   Output an error message using the core logger.
         * @details                 The error message is formatted as followed:
         *                          <file_name>:<function_name>:<line_nb>: <message>
         * @tparam[in] Args         Any types supported by fmt:format.
         * @param[in] file_name     File name.
         * @param[in] function_name Function name.
         * @param[in] line_nb       Line number.
         * @param[in] message       Error message.
         *
         * @note                    Usually called via the `NOA_CORE_ERROR` definition.
         */
        template<typename... Args>
        ErrorCore(const char* file_name,
                  const char* function_name,
                  const int line_nb,
                  Args&& ... message) {
            Noa::Log::getCoreLogger()->error(
                    fmt::format("{}:{}:{}: \n", file_name, function_name, line_nb) +
                    fmt::format(message...)
            );
        }
    };

    /**
     * @brief   Main exception thrown by the applications (akira, etc.).  Usually caught in the main().
     */
    class NOA_API ErrorApp : public std::exception {
    public:

        /**
         * @brief                   Output an error message using the app logger.
         * @details                 The error message is formatted as followed:
         *                          <file_name>:<function_name>:<line_nb>: <message>
         * @tparam[in] Args         Any types supported by fmt:format.
         * @param[in] file_name     File name.
         * @param[in] function_name Function name.
         * @param[in] line_nb       Line number.
         * @param[in] message       Error message.
         *
         * @note                    Usually called via the `NOA_APP_ERROR` definition.
         */
        template<typename... Args>
        ErrorApp(const char* file_name,
                 const char* function_name,
                 const int line_nb,
                 Args&& ... args) {
            Noa::Log::getAppLogger()->error(
                    fmt::format("{}:{}:{}: \n", file_name, function_name, line_nb) +
                    fmt::format(args...)
            );
        }
    };
}

/**
 * @defgroup NOA_LOGGING Logging macros
 * @{
 */

#if NOA_DEBUG
/** Log a debug message using the core logger. Ignored for Release build types */
#define NOA_CORE_DEBUG(...) Noa::Log::getCoreLogger()->debug(__VA_ARGS__)

/** Log a debug message using the app logger. Ignored for Release build types */
#define NOA_APP_DEBUG(...) Noa::Log::getCoreLogger()->debug(__VA_ARGS__)

#else
/** Log a debug message using the core logger. Ignored for Release build types */
#define NOA_CORE_DEBUG

/** Log a debug message using the app logger. Ignored for Release build types */
#define NOA_APP_DEBUG
#endif


/** Log a trace message using the core logger. */
#define NOA_CORE_TRACE(...) Noa::Log::getCoreLogger()->trace(__VA_ARGS__)

/** Log a info message using the core logger. */
#define NOA_CORE_INFO(...)  Noa::Log::getCoreLogger()->info(__VA_ARGS__)

/** Log a warning using the core logger. */
#define NOA_CORE_WARN(...)  Noa::Log::getCoreLogger()->warn(__VA_ARGS__)

/** Log a error using the core logger and throw a `::Noa::ErrorCore` exception. */
#define NOA_CORE_ERROR(...) throw Noa::ErrorCore(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)


/** Log a trace message using the app logger. */
#define NOA_APP_TRACE(...) Noa::Log::getAppLogger()->trace(__VA_ARGS__)

/** Log a info message using the app logger. */
#define NOA_APP_INFO(...)  Noa::Log::getAppLogger()->info(__VA_ARGS__)

/** Log a warning using the app logger. */
#define NOA_APP_WARN(...)  Noa::Log::getAppLogger()->warn(__VA_ARGS__)

/** Log a error using the app logger and throw a `::Noa::ErrorApp` exception. */
#define NOA_APP_ERROR(...) throw Noa::ErrorApp(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

/** @} */
