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
    class Options {
    public:

        /**
         * @short       The goal of the subclasses constructors is to get and format all of their
         *              attributes. This should be done with the protected getValues() methods,
         *              which rely on the options stored in the Noa::Parser. Usually, attributes
         *              can and should be defined with a member initializer list, which is more
         *              efficient.
         *
         * @code        class YourOptions : Noa::Options {
         *              public:
         *                  // define your attributes...
         *                  std::string filename;
         *                  std::array<int, 3> size;
         *                  bool fast;
         *                  Example(Parser* a_parser) :
         *                      filename(getValues<std::string, 1, 'S'>("Input", "i")),
         *                      size(getValues<std::array<int, 3>, 3, 'I'>("Size", "s", {100,100,100})),
         *                      fast(getValues<bool, 1, true>("Fast", "f")) {};
         *              }
         * @endcode
         */
        Options() = default;

        void print() const {

        };

    protected:
        /**
         * @short                   Get the formatted values stored in the Noa::Parser, for a given option.
         *
         * @tparam t_type_output    Returned/Desired type. All the default types should be supported,
         *                          plus std::vectors and std::array.
         * @tparam t_number         How many values are excepted. A positive int or -1. 0 is not allowed.
         *                          If -1, expect a range of values. In this case t_type_output should be
         *                          a vector.
         * @tparam t_type_value     "Display" type. It should corresponds to t_type_output.
         *                              'I': int
         *                              'U': unsigned int
         *                              'F': float
         *                              'B': bool
         *                              'S': std::string        /!\ See note2
         *                              'C': char
         * @param a_long_name       Long name of the option, e.g --Input.
         * @param a_short_name      Short name of the option, e.g. -i.
         * @param a_default_value   Default value to use if the Noa::Parser doesn't contain the option or
         *                          if the option value is empty. The default is perfectly forwarded, so
         *                          no copy/move should happen.
         *
         * @note1                   The command line options have the priority, then the parameter file,
         *                          then the default value. This is defined by Noa::Parser::get().
         * @note2                   If t_type_value is 'S', no formatting is performed because the parser
         *                          already stores the values as std::string. This is also true when more
         *                          than one string is expected (t_number > 1). In this case, t_type_output
         *                          has to be a vector of strings.
         */
        template<typename t_type_output, int t_number, char t_type_value>
        t_type_output getValues(const char* a_long_name,
                                const char* a_short_name,
                                t_type_output&& a_default_value) {

            // First, get the value from the parser.
            std::vector<std::string>* value = m_parser->getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type_output>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            if constexpr(t_number == 1) {
                if constexpr(t_type_value == 'I')
                    return String::toOneInt((*value)[0]);
                else if constexpr(t_type_value == 'U')
                    return String::toOneUInt((*value)[0]);
                else if constexpr(t_type_value == 'F')
                    return String::toOneFloat((*value)[0]);
                else if constexpr(t_type_value == 'B')
                    return String::toOneBool((*value)[0]);
                else if constexpr(t_type_value == 'C')
                    return String::toOneChar((*value)[0]);
                else if constexpr(t_type_value == 'S')
                    return (*value)[0];
                else
                    std::cerr << "Invalid type" << std::endl;
            } else {

                if constexpr(t_type_value == 'I')
                    return String::toMultipleInt<t_type_output>(*value);
                else if constexpr(t_type_value == 'U')
                    return String::toMultipleUInt<t_type_output>(*value);
                else if constexpr(t_type_value == 'F')
                    return String::toMultipleFloat<t_type_output>(*value);
                else if constexpr(t_type_value == 'B')
                    return String::toMultipleBool<t_type_output>(*value);
                else if constexpr(t_type_value == 'C')
                    return String::toMultipleChar<t_type_output>(*value);
                else if constexpr(t_type_value == 'S')
                    return *value;
                else
                    std::cerr << "Invalid type" << std::endl;
            }
        }

        // Version without a default value - raise an error if no value has been found.
        template<typename t_type_output, int t_number, char t_type_value>
        t_type_output getValues(const char* a_long_name, const char* a_short_name) {

            std::vector<std::string>* value = m_parser->getOption(a_long_name, a_short_name);

            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Format.
            if constexpr(t_number == 1) {
                if constexpr(t_type_value == 'I')
                    return String::toOneInt((*value)[0]);
                else if constexpr(t_type_value == 'U')
                    return String::toOneUInt((*value)[0]);
                else if constexpr(t_type_value == 'F')
                    return String::toOneFloat((*value)[0]);
                else if constexpr(t_type_value == 'B')
                    return String::toOneBool((*value)[0]);
                else if constexpr(t_type_value == 'C')
                    return String::toOneChar((*value)[0]);
                else if constexpr(t_type_value == 'S')
                    return (*value)[0];
                else
                    std::cerr << "Invalid type" << std::endl;
            } else {

                if constexpr(t_type_value == 'I')
                    return String::toMultipleInt<t_type_output>(*value);
                else if constexpr(t_type_value == 'U')
                    return String::toMultipleUInt<t_type_output>(*value);
                else if constexpr(t_type_value == 'F')
                    return String::toMultipleFloat<t_type_output>(*value);
                else if constexpr(t_type_value == 'B')
                    return String::toMultipleBool<t_type_output>(*value);
                else if constexpr(t_type_value == 'C')
                    return String::toMultipleChar<t_type_output>(*value);
                else if constexpr(t_type_value == 'S')
                    return *value;
                else
                    std::cerr << "Invalid type" << std::endl;
            }
        }
    };
}
