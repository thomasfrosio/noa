//
// Created by thomas on 12/07/2020.
//

#pragma once

#include <array>

#include "Parser.h"

namespace Noa {

    // Interface for the options of the Noa programs
    struct Options {
    private:
        Parser* m_parser;

    public:

        explicit Options(Parser* a_parser);

        void saveOptions(const std::string& a_logfile) const;

        void printOptions() const;

    private:
        /**
         * @short                   Get the formatted values stored in m_parser for a given option.
         *
         * @tparam T                Return type. It should always be known by the compiler, no need for explicit typing.
         * @param a_long_name       Long name of the option, e.g --Input.
         * @param a_short_name      Short name of the option, e.g. -i.
         * @param a_type            Option type.
         * @param a_default_value   Default value to use if m_parser doesn't contain the option or
         *                          if the option value is empty.
         */
        template<typename T>
        T getValues(const char* a_long_name, const char* a_short_name, const char* a_type, const char* a_default_value) {

        };

        /**
         * @short                   Get the formatted values stored in m_parser for a given option.
         *
         * @tparam T                Return type. It should always be known by the compiler, no need for explicit typing.
         * @param a_long_name       Long name of the option, e.g --Input.
         * @param a_short_name      Short name of the option, e.g. -i.
         * @param a_type            Option type.
         *
         * @note                    If m_parser doesn't contain the option or if the option value
         *                          is empty, an error will be raised.
         */
        template<typename T>
        T getValues(const char* a_long_name, const char* a_short_name, const char* a_type) {

        };

        /**
         *
         * @param a_vec_str
         * @return
         */
        int formatOneInt(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @tparam N
         * @param a_vec_str
         * @return
         */
        template<unsigned int N>
        std::array<int, N> formatMultipleInt(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @param a_vec_str
         * @return
         */
        float formatOneFloat(const std::vector<std::string>& a_vec_str) {

        };

        template<unsigned int N>
        std::array<float, N> formatMultipleFloat(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @param a_vec_str
         * @return
         */
        bool formatOneBool(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @tparam N
         * @param a_vec_str
         * @return
         */
        template<unsigned int N>
        std::array<bool, N> formatMultipleBool(const std::vector<std::string>& a_vec_str) {

        };

        /**
         *
         * @param a_vec_str
         * @return
         */
        int formatOneString(const std::vector<std::string>& a_vec_str) {

        };

    };
}
