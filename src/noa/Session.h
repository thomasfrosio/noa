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
    /// Creates and holds the static data necessary to run noa.
    /// There should only be one session at a given time.
    class Session {
    public:
        static Logger logger;

        NOA_HOST Session(const std::string& name, const std::string& filename, uint log_level) {
            logger.set(name, filename, log_level);
            NOA_PROFILE_BEGIN_SESSION(string::format("profile_{}.json", name));
        }

        NOA_HOST explicit Session(const std::string& name) {
            logger.set(name, name + ".log", Logger::BASIC);
            NOA_PROFILE_BEGIN_SESSION(string::format("profile_{}.json", name));
        }

        NOA_HOST ~Session() {
            NOA_PROFILE_END_SESSION();
        }

    public:
        NOA_HOST static std::string version() { return NOA_VERSION; }
        NOA_HOST static std::string url() { return NOA_URL; }
    };
}
