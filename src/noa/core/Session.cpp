#ifdef NOA_ENABLE_OPENMP
#include <omp.h>
#endif

#include "noa/core/Session.hpp"
#include "noa/core/string/Parse.hpp"

::noa::Logger noa::Session::logger;
int64_t noa::Session::m_threads = 0;

void noa::Session::set_threads(int64_t threads) {
    if (threads) {
        m_threads = threads;
    } else {
        int64_t max_threads{};
        const char* str{};
        try {
            str = std::getenv("NOA_THREADS");
            if (str) {
                max_threads = noa::string::parse<int64_t>(str);
            } else {
                #ifdef NOA_ENABLE_OPENMP
                str = std::getenv("OMP_NUM_THREADS");
                if (str)
                    max_threads = noa::string::parse<int64_t>(str);
                else
                    max_threads = static_cast<int64_t>(omp_get_max_threads());
                #else
                max_threads = std::thread::hardware_concurrency();
                #endif
            }
        } catch (...) {
            max_threads = 1;
        }
        m_threads = std::max(max_threads, int64_t{1});
    }
}
