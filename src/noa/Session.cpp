#include "noa/Session.h"

::noa::Logger noa::Session::logger;

void noa::Session::backtrace(const std::exception_ptr& exception_ptr, size_t level) {
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
        logger.error(string::format("[{}] {}\n", level, e.what()));
        backtrace(get_nested(e), level + 1);
    }
}
