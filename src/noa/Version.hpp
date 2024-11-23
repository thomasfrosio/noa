#pragma once

#define NOA_VERSION_MAJOR 0
#define NOA_VERSION_MINOR 1
#define NOA_VERSION_PATCH 0
#define NOA_URL "https://github.com/thomasfrosio/noa"

#define NOA_STRINGIFY(x) #x
#define NOA_VERSION_STRING(major, minor, patch) \
    NOA_STRINGIFY(major) "." NOA_STRINGIFY(minor) "." NOA_STRINGIFY(patch)

#define NOA_VERSION NOA_VERSION_STRING(NOA_VERSION_MAJOR, NOA_VERSION_MINOR, NOA_VERSION_PATCH)

#define NOA_VERSION_AT_LEAST(x, y, z) \
    (NOA_VERSION_MAJOR > x ||         \
     (NOA_VERSION_MAJOR >= x && (NOA_VERSION_MINOR > y || (NOA_VERSION_MINOR >= y && NOA_VERSION_PATCH >= z))))

namespace noa {
    constexpr const char* VERSION = NOA_VERSION;
    constexpr const char* URL = NOA_URL;
}
