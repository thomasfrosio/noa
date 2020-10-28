#include "Log.h"


namespace Noa {

    // Initialize the static members
    std::shared_ptr<spdlog::logger> Noa::Log::s_core_logger;
    std::shared_ptr<spdlog::logger> Noa::Log::s_app_logger;

    void Log::Init(const char* filename, const char* prefix, level verbosity) {
        std::vector<spdlog::sink_ptr> log_sinks;

        log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename));
        log_sinks[0]->set_pattern("[%T] [%l]: %v");
        log_sinks[0]->set_level(spdlog::level::trace);

        log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
        log_sinks[1]->set_pattern("%^[%T] %n: %v%$");
        setSinkLevel_(log_sinks[1], verbosity);

        s_core_logger = std::make_shared<spdlog::logger>("CORE", begin(log_sinks), end(log_sinks));
        spdlog::register_logger(s_core_logger);
        s_core_logger->set_level(spdlog::level::trace);
        s_core_logger->flush_on(spdlog::level::err);

        s_app_logger = std::make_shared<spdlog::logger>(prefix, begin(log_sinks), end(log_sinks));
        spdlog::register_logger(s_app_logger);
        s_core_logger->set_level(spdlog::level::trace);
        s_app_logger->flush_on(spdlog::level::err);
    }
}
