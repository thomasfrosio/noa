#include "noa/common/Exception.h"

thread_local std::string noa::Exception::s_message{};

void noa::Exception::backtrace_(std::string& message, const std::exception_ptr& exception_ptr, size_t level) {
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
    } catch (const noa::Exception& e) {
        // Don't call what(), otherwise we'll have an infinite recursion resulting in stack overflow.
        message += fmt::format("[{}] {}\n", level, e.m_buffer);
        backtrace_(message, get_nested(e), level + 1);
    } catch (const std::exception& e) {
        message += fmt::format("[{}] {}\n", level, e.what());
        backtrace_(message, get_nested(e), level + 1);
    }
}

