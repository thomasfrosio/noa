#pragma once

#include "noa/Logger.h"
#include "noa/Profiler.h"
#include "noa/util/string/Format.h"

namespace Noa {
    /// Creates and holds the static data necessary to run. There can be only one session at a time.
    class Session {
    public:
        static Logger logger;

        NOA_HOST Session(const std::string& name, const std::string& filename, uint log_level) {
            logger.set(name, filename, log_level);
            NOA_PROFILE_BEGIN_SESSION(String::format("profile_{}.json", name));
        }

        NOA_HOST ~Session() {
            NOA_PROFILE_END_SESSION();
        }
    };
}
