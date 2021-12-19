#include "noa/common/Logger.h"
#include "noa/common/Exception.h"

using namespace ::noa;

std::shared_ptr<spdlog::logger> Logger::create(std::string_view name,
                                               const std::vector<spdlog::sink_ptr>& sinks) {
    auto logger = std::make_shared<spdlog::logger>(name.data(), std::begin(sinks), std::end(sinks));
    logger->set_level(spdlog::level::trace); // be limited by the sink levels.
    logger->flush_on(spdlog::level::err);
    return logger;
}

std::shared_ptr<spdlog::logger> Logger::create(std::string_view name,
                                               std::string_view filename,
                                               Level verbosity_console) {
    // The sinks are transferred to the logger, so they'll continue to live through
    // the logger even after the end of this scope.
    std::vector<spdlog::sink_ptr> log_sinks;

    log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    log_sinks[0]->set_pattern("%^%v%$");
    setLevel(log_sinks[0], verbosity_console);

    if (!filename.empty()) {
        log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename.data()));
        log_sinks[1]->set_pattern("[%T] [%n::%l]: %v");
        setLevel(log_sinks[1], Level::VERBOSE);
    }
    return create(name, log_sinks);
}

void Logger::setLevel(spdlog::sink_ptr& sink, Level verbosity) {
    switch (verbosity) {
        case Logger::VERBOSE:
            sink->set_level(spdlog::level::trace);
            break;
        case Logger::BASIC:
            sink->set_level(spdlog::level::info);
            break;
        case Logger::ALERT:
            sink->set_level(spdlog::level::warn);
            break;
        case Logger::SILENT:
            sink->set_level(spdlog::level::off);
            break;
        default:
            NOA_THROW("Sink level should be 0 (SILENT), 1 (ALERT), 2 (BASIC), or 3 (VERBOSE), got {}", verbosity);
    }
}
