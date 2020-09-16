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
         * @brief               Initialize the core and the app logging system.
         * @param[in] filename  Log filename of the core and app logger.
         * @param[in] prefix    Prefix displayed before each logging entry of the app logger.
         *                      This _cannot_ be "NOA", since it is already used by the core logger.
         * @param[in] silent    Whether or not the logging system should stdout/stderr. If true,
         *                      there will be no output on the terminal. In any case, everything
         *                      is logged into the input file.
         * @note                One must initialize the loggers via this function _before_ using noa.
         */
        static void Init(const char* filename, const char* prefix, bool silent = false);

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
}
