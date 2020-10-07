/**
 * @file Core.h
 * @brief Contain the minimum files to include for the noa core.
 * @author Thomas - ffyr2w
 * @date 20 Jul 2020
 */

#pragma once

// Basics:
#include <iostream>
#include <fstream>
#include <string>
#include <cctype>

// Containers:
#include <unordered_map>
#include <vector>
#include <array>
#include <tuple>

// Others STL:
#include <utility>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <complex>

// noa commons:
#include "noa/API.h"
#include "noa/Version.h"
#include "noa/utils/Log.h"
#include "noa/utils/Exception.h"


/**
 * @defgroup NOA_LOGGING Logging macros
 * @{
 */

#if NOA_DEBUG
/** Log a debug message using the core logger. Ignored for Release build types */
#define NOA_CORE_DEBUG(...) ::Noa::Log::getCoreLogger()->debug(__VA_ARGS__)

/** Log a debug message using the app logger. Ignored for Release build types */
#define NOA_APP_DEBUG(...) ::Noa::Log::getCoreLogger()->debug(__VA_ARGS__)

#else
/** Log a debug message using the core logger. Ignored for Release build types */
#define NOA_CORE_DEBUG

/** Log a debug message using the app logger. Ignored for Release build types */
#define NOA_APP_DEBUG
#endif


/** Log a trace message using the core logger. */
#define NOA_CORE_TRACE(...) ::Noa::Log::getCoreLogger()->trace(__VA_ARGS__)

/** Log an info message using the core logger. */
#define NOA_CORE_INFO(...)  ::Noa::Log::getCoreLogger()->info(__VA_ARGS__)

/** Log a warning using the core logger. */
#define NOA_CORE_WARN(...)  ::Noa::Log::getCoreLogger()->warn(__VA_ARGS__)

/** Log an error using the core logger and throw a `::::Noa::ErrorCore` exception. */
#define NOA_CORE_ERROR(...) throw ::Noa::ErrorCore(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

/** Log an error inside lambdas using the core logger and throw a `::::Noa::ErrorCore` exception. */
#define NOA_CORE_ERROR_LAMBDA(func, ...) throw ::Noa::ErrorCore(__FILE__, func, __LINE__, __VA_ARGS__)

/** Log a trace message using the app logger. */
#define NOA_APP_TRACE(...) ::Noa::Log::getAppLogger()->trace(__VA_ARGS__)

/** Log a info message using the app logger. */
#define NOA_APP_INFO(...)  ::Noa::Log::getAppLogger()->info(__VA_ARGS__)

/** Log a warning using the app logger. */
#define NOA_APP_WARN(...)  ::Noa::Log::getAppLogger()->warn(__VA_ARGS__)

/** Log a error using the app logger and throw a `::::Noa::ErrorApp` exception. */
#define NOA_APP_ERROR(...) throw ::Noa::ErrorApp(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

/** Log an error inside lambdas using the core logger and throw a `::::Noa::ErrorCore` exception. */
#define NOA_APP_ERROR_LAMBDA(func, ...) throw ::Noa::ErrorCore(__FILE__, func, __LINE__, __VA_ARGS__)
/** @} */

