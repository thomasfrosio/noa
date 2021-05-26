#pragma once

#include "noa/Definitions.h"

#ifdef NOA_ENABLE_ASSERTS

#include <string>
#include <filesystem>
#include "noa/Session.h"

namespace Noa::Details {
    NOA_IH void logAssert(const char* file, const char* function, int line) {
        namespace fs = std::filesystem;
        size_t idx = std::string(file).rfind(std::string("noa") + fs::path::preferred_separator);
        Noa::Session::logger.error("{}:{}:{}: Assertion failed.",
                                   idx == std::string::npos ? fs::path(file).filename().string() : file + idx,
                                   function, line);
    }
}

#define NOA_ASSERT(check) do { if(!(check)) { Noa::Details::logAssert(__FILE__, __FUNCTION__, __LINE__); NOA_DEBUG_BREAK(); } } while (false)

#else
    #define NOA_ASSERT(check)
#endif
