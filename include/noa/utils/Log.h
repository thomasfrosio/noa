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


namespace Noa {

    // Class used as custom exception catch in main().
    class Error : public std::exception {
    };

    class Log {
    public:
        static void Init() {
            std::vector<spdlog::sink_ptr> log_sinks;
            log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>("noa.log"));

            log_sinks[0]->set_pattern("%^[%T] %n: %v%$");
            log_sinks[1]->set_pattern("[%T] [%l]: %v");

            s_core_logger = std::make_shared<spdlog::logger>("NOA",
                                                             begin(log_sinks),
                                                             end(log_sinks));
            spdlog::register_logger(s_core_logger);
            s_core_logger->set_level(spdlog::level::trace);
            s_core_logger->flush_on(spdlog::level::err);
        }

        static inline std::shared_ptr<spdlog::logger>& getCoreLogger() { return s_core_logger; }

        template<typename... Args>
        static void throwError(const char* a_file,
                               const char* a_function,
                               const int a_line,
                               Args&& ... args) {
            Noa::Log::getCoreLogger()->error(
                    fmt::format("{}:{}:{}: \n", a_file, a_function, a_line) +
                    fmt::format(args...));
            throw Noa::Error();
        }

    private:
        static std::shared_ptr<spdlog::logger> s_core_logger;
    };

    // Initialize the loggers.
    std::shared_ptr<spdlog::logger> Noa::Log::s_core_logger;
}

// Core log macros
#define NOA_CORE_DEBUG(...) Noa::Log::getCoreLogger()->debug(__VA_ARGS__)
#define NOA_CORE_TRACE(...) Noa::Log::getCoreLogger()->trace(__VA_ARGS__)
#define NOA_CORE_INFO(...)  Noa::Log::getCoreLogger()->info(__VA_ARGS__)
#define NOA_CORE_WARN(...)  Noa::Log::getCoreLogger()->warn(__VA_ARGS__)
#define NOA_CORE_ERROR(...) Noa::Log::throwError(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
