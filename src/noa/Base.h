/**
 * @file Base.h
 * @brief Contain the minimum files to include for the core.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */
#pragma once

// Basics:
#include <iostream>
#include <fstream>
#include <string>
#include <string_view>
#include <cctype>
#include <cstring>  // std::strerror
#include <cerrno>   // errno
#include <cmath>
#include <filesystem>
#include <system_error>
#include <thread>
#include <utility>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <complex>

// Containers:
#include <map>
#include <unordered_map>
#include <vector>
#include <array>
#include <tuple>

// noa commons:
#include "noa/API.h"
#include "noa/Version.h"
#include "noa/util/Log.h"
#include "noa/util/Exception.h"

// Set some useful aliases
namespace Noa {
    namespace fs = std::filesystem;
}

/**
 * @defgroup NOA_LOGGING Logging macros
 * @{
 */

#if NOA_DEBUG
/** Log a debug message using the noa logger. Ignored for Release build types */
#define NOA_CORE_DEBUG(...) ::Noa::Log::getCoreLogger()->debug(__VA_ARGS__)

/** Log a debug message using the app logger. Ignored for Release build types */
#define NOA_APP_DEBUG(...) ::Noa::Log::getCoreLogger()->debug(__VA_ARGS__)
#else
#define NOA_CORE_DEBUG
#define NOA_APP_DEBUG
#endif

/** Log a trace message using the core logger. */
#define NOA_CORE_TRACE(...) ::Noa::Log::getCoreLogger()->trace(__VA_ARGS__)

/** Log an info message using the core logger. */
#define NOA_CORE_INFO(...)  ::Noa::Log::getCoreLogger()->info(__VA_ARGS__)

/** Log a warning using the core logger. */
#define NOA_CORE_WARN(...)  ::Noa::Log::getCoreLogger()->warn(__VA_ARGS__)

/** Log an error using the core logger and throw ::Noa::ErrorCore(). */
#define NOA_CORE_ERROR(...) throw ::Noa::ErrorCore(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

/** Log an error inside lambdas using the core logger and throw ::Noa::ErrorCore(). */
#define NOA_CORE_ERROR_LAMBDA(func, ...) throw ::Noa::ErrorCore(__FILE__, func, __LINE__, __VA_ARGS__)

/** Log a trace message using the app logger. */
#define NOA_APP_TRACE(...) ::Noa::Log::getAppLogger()->trace(__VA_ARGS__)

/** Log a info message using the app logger. */
#define NOA_APP_INFO(...)  ::Noa::Log::getAppLogger()->info(__VA_ARGS__)

/** Log a warning using the app logger. */
#define NOA_APP_WARN(...)  ::Noa::Log::getAppLogger()->warn(__VA_ARGS__)

/** Log an error using the app logger and throw Noa::ErrorApp(). */
#define NOA_APP_ERROR(...) throw ::Noa::ErrorApp(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

/** Log an error inside lambdas using the app logger and throw Noa::ErrorCore(). */
#define NOA_APP_ERROR_LAMBDA(func, ...) throw ::Noa::ErrorCore(__FILE__, func, __LINE__, __VA_ARGS__)
/** @} */

