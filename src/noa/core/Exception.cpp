#include "noa/core/Exception.hpp"

namespace noa {
    auto Exception::format_(
        const char* file,
        const char* function,
        std::uint_least32_t line,
        const std::string_view& message
    ) -> std::string {
        namespace fs = std::filesystem;
        size_t idx = std::string(file).rfind(std::string("noa") + fs::path::preferred_separator);
        return fmt::format("ERROR:{}:{}:{}: {}",
                           idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                           function, line, message);
    }

    void Exception::backtrace_(
        std::vector<std::string>& message,
        const std::exception_ptr& exception_ptr
    ) {
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
            message.emplace_back(e.what());
            backtrace_(message, get_nested(e));
        } catch (...) {
            message.emplace_back("ERROR: Unknown exception type. Stopping the backtrace\n");
        }
    }
}
