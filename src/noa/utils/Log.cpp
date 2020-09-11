/**
 * @file Log-inl.h
 * @brief inline header (.cpp-like) of Log.h
 * @author Thomas - ffyr2w
 * @date 02 Sep 2020
 */


#include "Log.h"


namespace Noa {
    void Log::Init() {
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
}
