/// \file noa/Session.h
/// \brief The base session.
/// \author Thomas - ffyr2w
/// \date 18/06/2021

#pragma once

#include "noa/Version.h"
#include "noa/common/Logger.h"
#include "noa/common/Profiler.h"
#include "noa/common/string/Format.h"

namespace noa {
    /// Creates and holds the static data necessary to run noa. There can be only one session at a time.
    class Session {
    public:
        static Logger logger;

        NOA_HOST Session(const std::string& name, const std::string& filename, uint log_level) {
            logger.set(name, filename, log_level);
            NOA_PROFILE_BEGIN_SESSION(string::format("profile_{}.json", name));
        }

        NOA_HOST ~Session() {
            NOA_PROFILE_END_SESSION();
        }

    public:
        NOA_HOST static std::string getVersion() { return NOA_VERSION; }
        NOA_HOST static std::string getURL() { return NOA_URL; }
    };
}
