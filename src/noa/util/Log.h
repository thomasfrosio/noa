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

// Add some {fmt} functionalities that spdlog doesn't import by default.
#include <spdlog/fmt/bundled/compile.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/bundled/os.h>
#include <spdlog/fmt/bundled/chrono.h>
#include <spdlog/fmt/bundled/color.h>


namespace Noa {
    /**
     * Interface with the static loggers.
     * @see Log::init() to initialize the loggers. Must be called before anything.
     * @see Log::get() to get and log messages.
     * @see Log::setLevel() to set the level of the stdout sink.
     */
    class NOA_API Log {
    public:
        /**
         * Log levels compared to the @c spdlog levels:
         *   - @c verbose: off, error, warn, info, debug, trace
         *   - @c basic:   off, error, warn, info
         *   - @c alert:   off, error, warn
         *   - @c silent:  off
         */
         struct Level {
             static constexpr uint8_t silent = 0U;
             static constexpr uint8_t alert = 1U;
             static constexpr uint8_t basic = 2U;
             static constexpr uint8_t verbose = 3U;
         };

    private:
        static std::shared_ptr<spdlog::logger> s_logger;

    public:
        /**
         * @param[in] filename  Log filename used by all file sinks.
         * @param[in] verbosity Level of verbosity for the stdout sink. The log file isn't affected
         *                      and is always set to level::verbose.
         * @note One must initialize the loggers before using anything in the Noa namespace.
         */
        static void init(const char* filename = "noa.log", uint8_t verbosity = Level::verbose);


        /**
         * @return  The shared pointer of the core logger.
         * @note    Usually called using the @c NOA_CORE_* macros.
         * @see     https://github.com/gabime/spdlog/wiki
         */
        static inline std::shared_ptr<spdlog::logger>& get() { return s_logger; }


        /**
         * @param verbosity     Level of verbosity for the stdout sink. The log file isn't affected
         *                      and is always set to level::verbose.
         */
        static inline errno_t setLevel(uint8_t verbosity) {
            if (verbosity == Level::verbose)
                s_logger->sinks()[1]->set_level(spdlog::level::trace);
            else if (verbosity == Level::basic)
                s_logger->sinks()[1]->set_level(spdlog::level::info);
            else if (verbosity == Level::alert)
                s_logger->sinks()[1]->set_level(spdlog::level::warn);
            else if (verbosity == Level::silent)
                s_logger->sinks()[1]->set_level(spdlog::level::off);
            else
                return Errno::invalid_argument;
            return Errno::good;
        }
    };
}