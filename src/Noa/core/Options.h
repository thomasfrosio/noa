//
// Created by thomas on 12/07/2020.
//

#pragma once

#include <array>
#include <charconv>
#include <utility>

#include "Parser.h"

namespace Noa {

    // Interface for the options of the Noa programs
    struct Options {
    private:
        Parser* m_parser;
        std::string opt1;
        std::vector<int> opt2;
        bool opt3;
        std::vector<std::string> opt4;

    public:

        explicit Options(Parser* a_parser);

        void saveOptions(const std::string& a_logfile) const;

        void printOptions() const;

    private:
        /**
         * @short                   Get the formatted values stored in m_parser for a given option.
         *
         * @tparam T                Returned/Desired type, which can be any default type.
         * @param a_long_name       Long name of the option, e.g --Input.
         * @param a_short_name      Short name of the option, e.g. -i.
         * @param a_type            Option type.
         * @param a_default_value   Default value to use if m_parser doesn't contain the option or
         *                          if the option value is empty.
         *
         * @note                    The command line options have the priority, then the parameter file,
         *                          then the default value.
         * @note                    If T is std::vector<std::string> (e.g. a_value_nb=4, a_value_type='S')
         *                          no conversion is performed since the parser already stores a vector
         *                          of stripped and split strings.
         */
        template<typename T>
        T getValues(const char* a_long_name,
                    const char* a_short_name,
                    const unsigned int a_value_nb,
                    char a_value_type,
                    T&& a_default_value) {

            // First, get the value from the parser.
            std::vector<std::string>* value;

            if (m_parser->m_options_cmdline.count(a_long_name)) {
                if (m_parser->m_options_cmdline.count(a_short_name))
                    std::cerr << "Option specified twice using long name and short name" << std::endl;
                value = &m_parser->m_options_cmdline.at(a_long_name);
            } else if (m_parser->m_options_cmdline.count(a_short_name)) {
                value = &m_parser->m_options_cmdline.at(a_short_name);
            } else if (m_parser->m_options_parameter_file.count(a_long_name)) {
                value = &m_parser->m_options_parameter_file.at(a_long_name);
            } else {
                return std::forward<T>(a_default_value);
            }

            // The strings are stripped and split, so we except one value per string in the vector.
            if (value->size() == a_value_nb)
                std::cerr << "Error" << std::endl;

            // Then, format these strings according to the type.
            if (a_value_nb == 1) {
                if (a_value_type == 'I') return String::toOneInt((*value)[0]);
                if (a_value_type == 'F') return String::toOneFloat((*value)[0]);
                if (a_value_type == 'B') return String::toOneBool((*value)[0]);
                if (a_value_type == 'S') return (*value)[0];
            } else {
                if (a_value_type == 'I') return String::toMultipleInt<T>(*value);
                if (a_value_type == 'F') return String::toMultipleFloat<T>(*value);
                if (a_value_type == 'B') return String::toMultipleBool<T>(*value);
                if (a_value_type == 'S') return *value;
            }
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
    };
}
