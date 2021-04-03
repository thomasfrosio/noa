#include "noa/Logger.h"
#include "noa/Exception.h"

using namespace ::Noa;

std::shared_ptr<spdlog::logger> Logger::create(const std::string& name, const std::vector<spdlog::sink_ptr>& sinks) {
    std::shared_ptr<spdlog::logger> logger =
            std::make_shared<spdlog::logger>(name, std::begin(sinks), std::end(sinks));
    logger->set_level(spdlog::level::trace);
    logger->flush_on(spdlog::level::err);
    return logger;
}

std::shared_ptr<spdlog::logger> Logger::create(const std::string& name, const std::string& filename, uint verbosity) {
    // The sinks are transferred to the logger, so they'll continue to live through
    // the logger even after the end of this scope.
    std::vector<spdlog::sink_ptr> log_sinks;

    log_sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    log_sinks[0]->set_pattern("%^%v%$");
    setSinkLevel(log_sinks[0], verbosity);

    log_sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename));
    log_sinks[1]->set_pattern("[%T] [%n::%l]: %v");
    log_sinks[1]->set_level(spdlog::level::trace);

    return create(name, log_sinks);
}

void Logger::setSinkLevel(spdlog::sink_ptr& sink, uint verbosity) {
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

void Logger::backtrace(const std::exception_ptr& exception_ptr, size_t level) {
    static auto get_nested = [](auto& e) -> std::exception_ptr {
        try {
            return dynamic_cast<const std::nested_exception&>(e).nested_ptr();
        } catch (const std::bad_cast&) {
            return nullptr;
        }
    };

    try {
        if (exception_ptr)
            std::rethrow_exception(exception_ptr);
    } catch (const std::exception& e) {
        error(fmt::format("{} {}\n", std::string(level + 1, '>'), e.what()));
        backtrace(get_nested(e), level + 1);
    }
}
