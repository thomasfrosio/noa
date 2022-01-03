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
        const char* str;
        try {
            str = std::getenv("NOA_THREADS");
            if (str) {
                max_threads = noa::string::toInt<uint>(str);
            } else {
                #ifdef NOA_ENABLE_OPENMP
                str = std::getenv("OMP_NUM_THREADS");
                if (str)
                    max_threads = noa::string::toInt<uint>(str);
                else
                    max_threads = static_cast<uint>(omp_get_max_threads());
                #else
                max_threads = std::thread::hardware_concurrency();
                #endif
            }
        } catch (...) {
            max_threads = 1;
        }
        m_threads = std::max(max_threads, 1U);
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
