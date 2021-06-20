/// \file noa/Environment.h
/// \brief Deals with environmental variables.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cstdlib>
#include <thread>

#include "noa/Definitions.h"
#include "noa/Math.h"
#include "noa/util/string/Convert.h"

namespace noa {
    /// Returns the maximum number of threads to use. At least 1.
    NOA_IH uint maxThreads() {
        uint max_threads;
        try {
            const char* str = std::getenv("NOA_MAX_THREADS");
            max_threads = str ?
                          math::max(string::toInt<uint>(str), 1u) :
                          math::max(std::thread::hardware_concurrency(), 1u);
        } catch (...) {
            NOA_THROW("Failed to deduce the maximum number of threads");
        }
        return max_threads;
    }
}
