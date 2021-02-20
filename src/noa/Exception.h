/**
 * @file Exception.h
 * @brief Various exceptions and error handling things.
 * @author Thomas - ffyr2w
 * @date 14 Sep 2020
 */
#pragma once

#include <string>
#include <exception>
#include <filesystem>

#include "noa/Definitions.h"
#include "noa/Errno.h"
#include "noa/util/string/Format.h"

namespace Noa {
    /** Main exception thrown by the noa. Usually caught in main(). */
    class Exception : public std::exception {
    protected:
        std::string m_buffer{};

    public:
        /**
         * Format the error message, which is then accessible with what() or print().
         * @tparam Args         Any types supported by @c fmt::format.
         * @param[in] file      File name.
         * @param[in] function  Function name.
         * @param[in] line      Line number.
         * @param[in] args      Error message to format.
         *
         * @note "Zero" overhead: https://godbolt.org/z/v43Pzq
         */
        template<typename... Args>
        NOA_IH Exception(const char* file, const char* function, const int line, Args&& ... args) {
            namespace fs = std::filesystem;
            m_buffer = String::format("{}:{}:{}: ", fs::path(file).filename().string(), function, line) +
                       String::format(args...);
        }

        [[nodiscard]] NOA_IH const char* what() const noexcept override { return m_buffer.data(); }
    };

    /**
     * Throw a nested @c Noa::Exception if Errno != Errno::good.
     * @note    As the result of this function being defined in the Noa namespace, the macro NOA_THROW_IF
     *          defined below will now call this function when used within Noa. Other deeper namespace may
     *          add their own throwIf function.
     */
    NOA_IH void throwIf(Errno error, const char* file, const char* function, const int line) {
        if (error)
            std::throw_with_nested(Noa::Exception(file, function, line, toString(error)));
    }
}

/** Throw a nested exception. Should be called from within the Noa namespace. */
#define NOA_THROW(...) std::throw_with_nested(Exception(__FILE__, __FUNCTION__, __LINE__, __VA_ARGS__))

/** Throw a nested exception. Should be called from within the Noa namespace. */
#define NOA_THROW_FUNC(func, ...) std::throw_with_nested(Exception(__FILE__, func, __LINE__, __VA_ARGS__))

/** Throw a nested exception if @a call returns an error. @c throwIf might be specific to a namespace. */
#define NOA_THROW_IF(call) throwIf(call, __FILE__, __FUNCTION__, __LINE__)
