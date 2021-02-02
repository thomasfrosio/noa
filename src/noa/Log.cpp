#include "Log.h"

using namespace ::Noa;

std::shared_ptr<spdlog::logger> Noa::Log::s_logger;

void Log::init(const std::string& filename, int verbosity) {
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

Errno Log::setLevel(int verbosity) {
    switch (verbosity) {
        case Level::verbose: {
            getCmdLine()->set_level(spdlog::level::trace);
            break;
        }
        case Level::basic: {
            getCmdLine()->set_level(spdlog::level::info);
            break;
        }
        case Level::alert: {
            getCmdLine()->set_level(spdlog::level::warn);
            break;
        }
        case Level::silent: {
            getCmdLine()->set_level(spdlog::level::off);
            break;
        }
        default: {
            return Errno::invalid_argument;
        }
    }
    return Errno::good;
}

void Log::backtrace(const std::exception_ptr& exception_ptr, size_t level) {
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
        Log::error(fmt::format("{} {}\n", std::string(level + 1, '>'), e.what()));
        backtrace(get_nested(e), level + 1);
    }
}
