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
#include <fmt/compile.h>
#include <fmt/ranges.h>
#include <fmt/os.h>
#include <fmt/chrono.h>

#include "Exception.h"


namespace Noa {

    class Log {
    public:
        static void Init() {
            std::vector<spdlog::sink_ptr> log_sinks;
            log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("noa.log"));

            log_sinks[0]->set_pattern("%^[%T] %n: %v%$");
            log_sinks[1]->set_pattern("[%T] [%l]: %v");

            s_logger = std::make_shared<spdlog::logger>("NOA", begin(log_sinks), end(log_sinks));
            spdlog::register_logger(s_logger);
            s_logger->set_level(spdlog::level::trace);
            s_logger->flush_on(spdlog::level::err);
        }

        static inline std::shared_ptr<spdlog::logger>& getCoreLogger() { return s_logger; }

    private:
        static std::shared_ptr<spdlog::logger> s_logger;
    };

    // Initialize the loggers.
    std::shared_ptr<spdlog::logger> Noa::Log::s_logger;
}

// Core log macros
#define NOA_DEBUG(...) Noa::Log::getCoreLogger()->debug(__VA_ARGS__)
#define NOA_TRACE(...) Noa::Log::getCoreLogger()->trace(__VA_ARGS__)
#define NOA_INFO(...)  Noa::Log::getCoreLogger()->info(__VA_ARGS__)
#define NOA_WARN(...)  Noa::Log::getCoreLogger()->warn(__VA_ARGS__)
#define NOA_ERROR(...) throw Noa::Error(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
