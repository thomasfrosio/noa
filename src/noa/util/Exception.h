/**
 * @file Exception.h
 * @brief Various exceptions and error handling things.
 * @author Thomas - ffyr2w
 * @date 14 Sep 2020
 */
#pragma once

#include "noa/util/Log.h"


namespace Noa {
    /**
     * Error numbers used throughout the @a Noa namespace.
     * Errno should evaluate to @c false if no errors (@c Errno::good),
     * and @c true for errors. Code should first check whether or not there's
     * an error before looking for specific errors.
     */
    struct NOA_API Errno {
        static constexpr errno_t good{0U}; // this one should not change !
        static constexpr errno_t fail{1U};

        static constexpr errno_t invalid_argument{2U};
        static constexpr errno_t invalid_size{3U};
        static constexpr errno_t invalid_data{4U};
        static constexpr errno_t out_of_range{5U};
        static constexpr errno_t not_supported{6U};

        // I/O
        static constexpr errno_t fail_close{10U};
        static constexpr errno_t fail_open{11U};
        static constexpr errno_t fail_read{12U};
        static constexpr errno_t fail_write{13U};

        // OS
        static constexpr errno_t out_of_memory{20U};
        static constexpr errno_t fail_os{21U};
    };


    /** Base class for the exceptions thrown in the @c Noa namespace. */
    class NOA_API Error : public std::exception {
    protected:
        std::string m_message{};

    public:
        [[nodiscard]] const char* what() const noexcept override {
            return m_message.data();
        }

        virtual void print() const {
            fmt::print(m_message);
        }
    };


    /** Main exception thrown by the noa. Usually caught in main(). */
    class NOA_API ErrorCore : public ::Noa::Error {
    public:

        /**
         * Format the error message, which is then accessible with what() or print().
         * @tparam Args             Any types supported by @c fmt::format.
         * @param[in] file_name     File name.
         * @param[in] function_name Function name.
         * @param[in] line_nb       Line number.
         * @param[in] args          Error message to format.
         *
         * @note                    Usually called via the @c NOA_CORE_ERROR definition.
         */
        template<typename... Args>
        ErrorCore(const char* file_name, const char* function_name,
                  const int line_number, Args&& ... args) {
            m_message = fmt::format("{}:{}:{}:\n", file_name, function_name, line_number) +
                        fmt::format(args...);
        }


        /** Log the error message that was thrown using the core logger. */
        void print() const override {
            Noa::Log::getCoreLogger()->error(m_message);
        }
    };


    /** Main exception thrown by the app. Usually caught in the main(). */
    class NOA_API ErrorApp : public ::Noa::Error {
    public:

        /**
         * Format the error message, which is then accessible with what() or print().
         * @tparam[in] Args         Any types supported by @c fmt::format.
         * @param[in] file_name     File name.
         * @param[in] function_name Function name.
         * @param[in] line_nb       Line number.
         * @param[in] args          Error message to format.
         *
         * @note                    Usually called via the @c NOA_APP_ERROR definition.
         */
        template<typename... Args>
        ErrorApp(const char* file_name, const char* function_name,
                 const int line_number, Args&& ... args) {
            m_message = fmt::format("{}:{}:{}: \n", file_name, function_name, line_number) +
                        fmt::format(args...);
        }


        /** Log the error message that was thrown using the app logger. */
        void print() const override {
            Noa::Log::getAppLogger()->error(m_message);
        }
    };
}
