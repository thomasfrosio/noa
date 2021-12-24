#ifdef NOA_ENABLE_OPENMP
#include <omp.h>
#endif

#include "noa/Session.h"
#include "noa/common/string/Convert.h"

::noa::Logger noa::Session::logger;
size_t noa::Session::m_threads = 0;

void noa::Session::threads(size_t threads) {
    if (threads) {
        m_threads = threads;
    } else {
        uint max_threads;
        try {
            const char* str = std::getenv("NOA_THREADS");
            if (str) {
                max_threads = std::max(noa::string::toInt<uint>(str), 1u);
            } else {
                #ifdef NOA_ENABLE_OPENMP
                max_threads = static_cast<uint>(omp_get_max_threads());
                #else
                max_threads = math::max(std::thread::hardware_concurrency(), 1u);
                #endif
            }
        } catch (...) {
            max_threads = 1;
        }
        m_threads = max_threads;
    }
    #ifdef NOA_ENABLE_OPENMP
    omp_set_num_threads(static_cast<int>(m_threads));
    #endif
}

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
