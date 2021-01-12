#include "noa/util/Log.h"

std::shared_ptr<spdlog::logger> Noa::Log::s_logger;

void Noa::Log::init(const std::string& filename, uint32_t verbosity) {
    std::vector<spdlog::sink_ptr> log_sinks;

    log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename));
    log_sinks[0]->set_pattern("[%T] [%l]: %v");
    log_sinks[0]->set_level(spdlog::level::trace);

    log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    log_sinks[1]->set_pattern("%^[%T] %n: %v%$");

    s_logger = std::make_shared<spdlog::logger>("NOA", begin(log_sinks), end(log_sinks));
    spdlog::register_logger(s_logger);
    setLevel(verbosity);
    s_logger->set_level(spdlog::level::trace);
    s_logger->flush_on(spdlog::level::err);
}

