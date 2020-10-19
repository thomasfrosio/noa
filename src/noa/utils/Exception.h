/**
 * @file Exception.h
 * @brief Various exceptions.
 * @author Thomas - ffyr2w
 * @date 14 Sep 2020
 */
#pragma once

#include "noa/utils/Log.h"


namespace Noa {
    /** Error numbers used throughout the @c Noa namespace. */
    struct NOA_API Errno {
        // 0 is reserved to signal no errors
        static constexpr uint8_t fail{1U};
        static constexpr uint8_t invalid_argument{2U};
        static constexpr uint8_t invalid_size{3U};
        static constexpr uint8_t out_of_range{4U};
    };


    class NOA_API Error : public std::exception {
    };


    /**
     * @brief   Main exception thrown by noa. Usually caught in the main().
     */
    class NOA_API ErrorCore : public ::Noa::Error {
    public:

        /**
         * @brief                   Output an error message using the core logger.
         * @details                 The error message is formatted as followed:
         *                          <file_name>:<function_name>:<line_nb>: <message>
         * @tparam[in] Args         Any types supported by fmt:format.
         * @param[in] file_name     File name.
         * @param[in] function_name Function name.
         * @param[in] line_nb       Line number.
         * @param[in] message       Error message.
         *
         * @note                    Usually called via the `NOA_CORE_ERROR` definition.
         */
        template<typename... Args>
        ErrorCore(const char* file_name, const char* function_name,
                  const int line_nb, Args&& ... message) {
            Noa::Log::getCoreLogger()->error(
                    fmt::format("{}:{}:{}: \n", file_name, function_name, line_nb) +
                    fmt::format(message...)
            );
        }
    };


    /**
     * @brief   Main exception thrown by the applications (akira, etc.).  Usually caught in the main().
     */
    class NOA_API ErrorApp : public ::Noa::Error {
    public:

        /**
         * @brief                   Output an error message using the app logger.
         * @details                 The error message is formatted as followed:
         *                          <file_name>:<function_name>:<line_nb>: <message>
         * @tparam[in] Args         Any types supported by fmt:format.
         * @param[in] file_name     File name.
         * @param[in] function_name Function name.
         * @param[in] line_nb       Line number.
         * @param[in] message       Error message.
         *
         * @note                    Usually called via the `NOA_APP_ERROR` definition.
         */
        template<typename... Args>
        ErrorApp(const char* file_name, const char* function_name,
                 const int line_nb, Args&& ... args) {
            Noa::Log::getAppLogger()->error(
                    fmt::format("{}:{}:{}: \n", file_name, function_name, line_nb) +
                    fmt::format(args...)
            );
        }
    };
}
