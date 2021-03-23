#pragma once

#include <cstdlib>
#include <thread>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/string/Convert.h"

namespace Noa {
    /** Returns the maximum number of threads to use. At least 1. */
    NOA_IH uint maxThreads() {
        uint max_threads;
        try {
            const char* str = std::getenv("NOA_MAX_THREADS");
            max_threads = str ?
                          Math::max(String::toInt<uint>(std::string_view{str}), 1u) :
                          Math::max(std::thread::hardware_concurrency(), 1u);
        } catch (...) {
            NOA_THROW("Failed to deduce the maximum number of threads");
        }
        return max_threads;
    }
}
