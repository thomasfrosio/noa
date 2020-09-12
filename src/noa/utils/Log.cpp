/**
 * @file Log.cpp
 * @brief Logging system used throughout the core and the programs.
 * @author Thomas - ffyr2w
 * @date 02 Sep 2020
 */


#include "Log.h"


namespace Noa {

    // Initialize the static members of the Log class
    std::shared_ptr<spdlog::logger> Noa::Log::s_core_logger;
    std::shared_ptr<spdlog::logger> Noa::Log::s_app_logger;

    void Log::Init(const char* filename, const char* prefix) {
        std::vector<spdlog::sink_ptr> log_sinks;
        log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename));

        log_sinks[0]->set_pattern("%^[%T] %n: %v%$");
        log_sinks[1]->set_pattern("[%T] [%l]: %v");

        s_core_logger = std::make_shared<spdlog::logger>("NOA", begin(log_sinks), end(log_sinks));
        spdlog::register_logger(s_core_logger);
        s_core_logger->set_level(spdlog::level::trace);
        s_core_logger->flush_on(spdlog::level::err);

        s_app_logger = std::make_shared<spdlog::logger>(prefix, begin(log_sinks), end(log_sinks));
        spdlog::register_logger(s_app_logger);
        s_app_logger->set_level(spdlog::level::trace);
        s_app_logger->flush_on(spdlog::level::err);
    }
}
