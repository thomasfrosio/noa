/**
 * @file Exception.h
 * @brief Various exceptions and error handling things.
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


    /** Main exception thrown by the core. Usually caught in the @c main(). */
    class NOA_API ErrorCore : public ::Noa::Error {
    public:

        /**
         * Format the error message, which is then accessible with @c what() or @c print().
         * @tparam[in] Args         Any types supported by @c fmt::format.
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


    /** Main exception thrown by the app. Usually caught in the @c main(). */
    class NOA_API ErrorApp : public ::Noa::Error {
    public:

        /**
         * Format the error message, which is then accessible with @c what() or @c print().
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
