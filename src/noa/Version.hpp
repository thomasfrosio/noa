#pragma once

// major * 10000 + minor * 100 + patch.
#define NOA_VERSION 001000
#define NOA_VERSION_STRING "001000"

#define NOA_URL "https://github.com/thomasfrosio/noa"

namespace noa {
    constexpr const char* VERSION = NOA_VERSION_STRING;
    constexpr const char* URL = NOA_URL;
}
