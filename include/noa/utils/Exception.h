/**
 * @file Exception.h
 * @brief Exceptions thrown by noa.
 * @author Thomas - ffyr2w
 * @date 09 Aug 2020
 */
#pragma once

#include "Log.h"


namespace Noa {

    // Class used as a custom exception and caught in main().
    class Error : public std::exception {
    public:
        template<typename... Args>
        Error(const char* a_file, const char* a_function, const int a_line, Args&& ... args) {
            Noa::Log::getCoreLogger()->error(
                    fmt::format("{}:{}:{}: \n", a_file, a_function, a_line) +
                    fmt::format(args...)
            );
        }
    };
}
