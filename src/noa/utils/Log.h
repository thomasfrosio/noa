/**
 * @file Log.h
 * @brief Logging system used throughout the core and programs.
 * @author Thomas - ffyr2w
 * @date 25 Jul 2020
 */
#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

// Add some functionalities that spdlog doesn't import.
#include <spdlog/fmt/bundled/compile.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/bundled/os.h>
#include <spdlog/fmt/bundled/chrono.h>

#include "noa/api.h"


namespace Noa {

    class NOA_API Log {
    public:
        /**
         *
         */
        static void Init();

        /**
         *
         * @return
         */
        static inline std::shared_ptr<spdlog::logger>& getCoreLogger() { return s_logger; }

    private:
        static std::shared_ptr<spdlog::logger> s_logger;
    };

    // Initialize the static member of the Log class
    std::shared_ptr<spdlog::logger> Noa::Log::s_logger;


    // Class used as a custom exception and caught in main().
    class NOA_API Error : public std::exception {
    public:
        template<typename... Args>
        Error(const char* a_file, const char* a_function, const int a_line, Args&& ... args) {
            Noa::Log::getCoreLogger()->error(
                    fmt::format("{}:{}:{}: \n", a_file, a_function, a_line) + fmt::format(args...)
            );
        }
    };
}

// Core log macros
#if NOA_DEBUG
#define NOA_LOG_DEBUG(...) Noa::Log::getCoreLogger()->debug(__VA_ARGS__)
#else
#define NOA_LOG_DEBUG
#endif

#define NOA_LOG_TRACE(...) Noa::Log::getCoreLogger()->trace(__VA_ARGS__)
#define NOA_LOG_INFO(...)  Noa::Log::getCoreLogger()->info(__VA_ARGS__)
#define NOA_LOG_WARN(...)  Noa::Log::getCoreLogger()->warn(__VA_ARGS__)
#define NOA_LOG_ERROR(...) throw Noa::Error(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
