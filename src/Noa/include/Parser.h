#pragma once

#include "Noa.h"
#include "String.h"
#include "Assert.h"

namespace Noa {

    /**
     * @class       Parser
     * @brief       Input manager for the noa programs.
     * @details     Parse the command line and the parameter file (if any) and makes
     *              the input easily accessible for other programs.
     *
     * @see         Parser::Parser() to parse the user inputs.
     * @see         Parser::get() to retrieve the formatted inputs.
     */
    class Parser {
    public:
        std::string program;
        std::string parameter_file;
        bool has_parameter_file{false};
        bool has_asked_help{false};

    private:
        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline;
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file;

    public:
        /**
         * @fn          Parser::Parser() - default constructor.
         * @short       Parses the command line and the parameter file, if it exists.
         *
         * @details
         * Supported scenarios:
         *      1) [./noa]
         *      2) [./noa] [-h]
         *      3) [./noa] [program] [-h]
         *      4) [./noa] [program] [--option]
         *      5) [./noa] [program] [file] [--option]
         *
         * Scenario descriptions:
         * - In scenario 1 or 2, this->program is set to "-h".
         * - In scenario 3, this->program is correctly set to the [program],
         *   this->has_asked_help is set to true and the [--option] is NOT parsed.
         *   It is up to your program to check whether or not a help was asked.
         * - In scenario 4, this->program is correctly set and the [--option]
         *   is parsed and accessible using the this->get(...) method. If a [-h]
         *   is found within [--option], this->has_asked_help is set to true and
         *   the parsing stops.
         * - In scenario 5, [file] is parsed and its options can be accessed with
         *   this->get(...). Otherwise, it is like scenario 4.
         *
         * Arguments:
         * - [-h]: can be any of the the following: "--help", "-help", "help", "-h", "h".
         * - [program]: should be the second argument specified in the command line.
         *              This is saved in this->program.
         * - [--option]: Sequence of #options and #values.
         *               #options should be prefixed with either "--" or '-',
         *               e.g. "--Input" or "-i". They can be (optionally) followed by
         *               one or multiple #values. #values are arguments that are not
         *               prefixed with "--" or '-', but are prefixed with an #option.
         *               To specify multiple #values for one #option, use a comma or
         *               a whitespace, e.g. --size 100,100,100. Successive commas or
         *               whitespaces are collapsed, so "100,,100, 100" is parsed as
         *               the previous example.
         * - [file]:    Parameter/option file. #options should start at the beginning
         *              of a line and be prefixed by "noa_". The #values should be
         *              specified after an '=', i.e. #option=#value. Whitespaces are
         *              ignored before and after #option, '=' and #value. Multiple
         *              #values can be specified as in [--option]. Inline comments
         *              are allowed and should start with '#'.
         *
         * @note1   The arguments are positional arguments, except [-h].
         * @note2   Single "--" or '-' arguments are ignored.
         * @note3   You can find more information on the priority between [--option]
         *          and [file], as well as long-name and short-name options, in
         *          this->get(...).
         *
         * @param argc  How many c-strings are contained in argv.
         * @param argv  Contains the stripped and split c-strings. See formatting below.
         *
         * @example     Here is an example on how to start your program using the Noa::Parser:
         * @code        int main(int argc, char* argv) {
         *                  // Parse cmd line and parameter file.
         *                  Noa::Parser parser(argc, argv);
         *                  switch (parser.program) {
         *                      case "program1":
         *                          start_program1();
         *                      // ...
         *                      case "-h":
         *                          print_global_help();
         *                      default:
         *                          printf("Unknown program");
         *                          print_global_help();
         *                  }
         *              }
         * @endcode
         */
        explicit Parser(int argc, char* argv[]) {
            // Parse the command line.
            parseCommandLine(argc, argv);

            // Parse the parameter file.
            parseParameterFile();
        }

        /**
         * @short                       Get the formatted values stored in the Noa::Parser, for a given option.
         *
         * @tparam [in, out] t_type     Returned/Desired type. All the default types should be supported,
         *                              plus std::vectors and std::array.
         * @tparam [in] t_format        "Display" type. It should corresponds to t_type_output.
         *                                  'I': int
         *                                  'F': float
         *                                  'B': bool
         *                                  'S': std::string        /!\ See note2
         *                                  'C': char
         *                                  'N': filename
         *                                  'X': symmetry
         * @tparam [in] t_number        How many values are excepted. A positive int or -1. 0 is not allowed.
         *                              If -1, expect a range of values. In this case t_type_output should be
         *                              a vector.
         * @tparam [in] t_range
         * @param [in] a_long_name      Long name of the option, e.g --Input.
         * @param [in] a_short_name     Short name of the option, e.g. -i.
         * @param [in] a_default_value  Default value to use if the Noa::Parser doesn't contain the option or
         *                              if the option value is empty. The default is perfectly forwarded, so
         *                              no copy/move should happen.
         * @param [in] a_range_min      Minimum value allowed. This is only used with t_format == 'I' or 'F'.
         * @param [in] a_range_max      Maximum value allowed. This is only used with t_format == 'I' or 'F'.
         *
         * @note1                       The command line options have the priority, then the parameter file,
         *                              then the default value. This is defined by Noa::Parser::get().
         * @note2                       If t_type_value is 'S', no formatting is performed because the parser
         *                                      already stores the values as std::string. This is also true when more
         *                                      than one string is expected (t_number > 1). In this case, t_type_output
         *                                      has to be a vector of strings.
         */

        // INT - DEFAULT - RANGE
        template<typename t_type = int, int t_number = 1>
        t_type getInteger(const char* a_long_name,
                          const char* a_short_name,
                          const t_type& a_default_value,
                          int a_range_min,
                          int a_range_max) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, int>) {
                    return String::toOneInt((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<int>> ||
                                    std::is_same_v<t_type, std::array<int, t_number>>) {
                    return String::toMultipleInt<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };

            checkRange<t_type, t_number>(output, a_range_min, a_range_max);
            return output;
        }

        // INT - DEFAULT
        template<typename t_type = int, int t_number = 1>
        t_type getInteger(const char* a_long_name,
                          const char* a_short_name,
                          const t_type& a_default_value) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, int>) {
                    return String::toOneInt((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<int>> ||
                                    std::is_same_v<t_type, std::array<int, t_number>>) {
                    return String::toMultipleInt<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        // INT  - RANGE
        template<typename t_type = int, int t_number = 1>
        t_type getInteger(const char* a_long_name,
                          const char* a_short_name,
                          int a_range_min,
                          int a_range_max) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, int>) {
                    return String::toOneInt((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<int>> ||
                                    std::is_same_v<t_type, std::array<int, t_number>>) {
                    return String::toMultipleInt<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };

            checkRange<t_type, t_number>(output, a_range_min, a_range_max);
            return output;
        }

        // INT
        template<typename t_type = int, int t_number = 1>
        t_type getInteger(const char* a_long_name, const char* a_short_name) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, int>) {
                    return String::toOneInt((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<int>> ||
                                    std::is_same_v<t_type, std::array<int, t_number>>) {
                    return String::toMultipleInt<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        /**
         *
         * @tparam t_type
         * @tparam t_number
         * @param a_long_name
         * @param a_short_name
         * @param a_default_value
         * @param a_range_min
         * @param a_range_max
         * @return
         */
        // FLOAT - DEFAULT - RANGE
        template<typename t_type = float, int t_number = 1>
        t_type getFloat(const char* a_long_name,
                        const char* a_short_name,
                        const t_type& a_default_value,
                        float a_range_min,
                        float a_range_max) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, float>) {
                    return String::toOneFloat((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<float>> ||
                                    std::is_same_v<t_type, std::array<float, t_number>>) {
                    return String::toMultipleFloat<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };

            checkRange<t_type, t_number>(output, a_range_min, a_range_max);
            return output;
        }

        // FLOAT - DEFAULT
        template<typename t_type = float, int t_number = 1>
        t_type getFloat(const char* a_long_name,
                        const char* a_short_name,
                        const t_type& a_default_value) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, float>) {
                    return String::toOneFloat((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<float>> ||
                                    std::is_same_v<t_type, std::array<float, t_number>>) {
                    return String::toMultipleFloat<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        // FLOAT  - RANGE
        template<typename t_type = float, int t_number = 1>
        t_type getFloat(const char* a_long_name,
                        const char* a_short_name,
                        float a_range_min,
                        float a_range_max) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, float>) {
                    return String::toOneFloat((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<float>> ||
                                    std::is_same_v<t_type, std::array<float, t_number>>) {
                    return String::toMultipleFloat<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };

            checkRange<t_type, t_number>(output, a_range_min, a_range_max);
            return output;
        }

        // FLOAT
        template<typename t_type = float, int t_number = 1>
        t_type getFloat(const char* a_long_name, const char* a_short_name) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, float>) {
                    return String::toOneFloat((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<float>> ||
                                    std::is_same_v<t_type, std::array<float, t_number>>) {
                    return String::toMultipleFloat<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }


        /**
         *
         * @tparam t_type
         * @tparam t_number
         * @param a_long_name
         * @param a_short_name
         * @param a_default_value
         * @return
         */
        // BOOL - DEFAULT
        template<typename t_type = bool, int t_number = 1>
        t_type getBool(const char* a_long_name,
                       const char* a_short_name,
                       const t_type& a_default_value) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, bool>) {
                    return String::toOneBool((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<bool>> ||
                                    std::is_same_v<t_type, std::array<bool, t_number>>) {
                    return String::toMultipleBool<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        // BOOL
        template<typename t_type = bool, int t_number = 1>
        t_type getBool(const char* a_long_name, const char* a_short_name) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, bool>) {
                    return String::toOneBool((*value)[0]);
                } else if constexpr(std::is_same_v<t_type, std::vector<bool>> ||
                                    std::is_same_v<t_type, std::array<bool, t_number>>) {
                    return String::toMultipleBool<t_type>();
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        /**
         *
         * @tparam t_type
         * @tparam t_number
         * @param a_long_name
         * @param a_short_name
         * @param a_default_value
         * @return
         */
        // STRING - DEFAULT
        template<typename t_type = std::string, int t_number = 1>
        t_type getString(const char* a_long_name,
                         const char* a_short_name,
                         const t_type& a_default_value) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, std::string>) {
                    return (*value)[0];
                } else if constexpr(std::is_same_v<t_type, std::vector<std::string>>) {
                    return *value;
                } else if constexpr(std::is_same_v<t_type, std::array<std::string, t_number>>) {
                    std::cerr << "Error - Use a vector..." << std::endl;
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        // STRING
        template<typename t_type = std::string, int t_number = 1>
        t_type getString(const char* a_long_name, const char* a_short_name) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            return [value]() {
                if constexpr(std::is_same_v<t_type, std::string>) {
                    return (*value)[0];
                } else if constexpr(std::is_same_v<t_type, std::vector<std::string>>) {
                    return *value;
                } else if constexpr(std::is_same_v<t_type, std::array<std::string, t_number>>) {
                    std::cerr << "Error - Use a vector..." << std::endl;
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
        }

        /**
         *
         * @tparam t_type
         * @tparam t_number
         * @param a_long_name
         * @param a_short_name
         * @param a_default_value
         * @return
         */
        // FILENAME - DEFAULT
        template<typename t_type = std::string, int t_number = 1>
        t_type getFilename(const char* a_long_name,
                           const char* a_short_name,
                           const t_type& a_default_value,
                           bool exist = true) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, std::string>) {
                    return (*value)[0];
                } else if constexpr(std::is_same_v<t_type, std::vector<std::string>>) {
                    return *value;
                } else if constexpr(std::is_same_v<t_type, std::array<std::string, t_number>>) {
                    std::cerr << "Error - Use a vector..." << std::endl;
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
            checkFileExists(output);
            return output;
        }

        // FILENAME
        template<typename t_type = std::string, int t_number = 1>
        t_type getFilename(const char* a_long_name,
                           const char* a_short_name,
                           bool exist = true) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, std::string>) {
                    return (*value)[0];
                } else if constexpr(std::is_same_v<t_type, std::vector<std::string>>) {
                    return *value;
                } else if constexpr(std::is_same_v<t_type, std::array<std::string, t_number>>) {
                    std::cerr << "Error - Use a vector..." << std::endl;
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
            checkFileExists(output);
            return output;
        }

        /**
         *
         * @tparam t_type
         * @tparam t_number
         * @param a_long_name
         * @param a_short_name
         * @param a_default_value
         * @return
         */
        // SYMMETRY - DEFAULT
        template<typename t_type = std::string, int t_number = 1>
        t_type getSymmetry(const char* a_long_name,
                           const char* a_short_name,
                           const t_type& a_default_value) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty())
                return std::forward<t_type>(a_default_value);
            else if (value->size() != t_number && t_number != -1)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, std::string>) {
                    return (*value)[0];
                } else if constexpr(std::is_same_v<t_type, std::vector<std::string>>) {
                    return *value;
                } else if constexpr(std::is_same_v<t_type, std::array<std::string, t_number>>) {
                    std::cerr << "Error - Use a vector..." << std::endl;
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
            checkSymmetry(output);
            return output;
        }

        // SYMMETRY
        template<typename t_type = std::string, int t_number = 1>
        t_type getSymmetry(const char* a_long_name,
                           const char* a_short_name) {
            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Second, make sure it has the excepted number of values.
            if (!value || value->empty() || value->size() == t_number)
                std::cerr << "Error" << std::endl;

            // Then, format them into to the desired type.
            t_type output = [value]() {
                if constexpr(std::is_same_v<t_type, std::string>) {
                    return (*value)[0];
                } else if constexpr(std::is_same_v<t_type, std::vector<std::string>>) {
                    return *value;
                } else if constexpr(std::is_same_v<t_type, std::array<std::string, t_number>>) {
                    std::cerr << "Error - Use a vector..." << std::endl;
                } else {
                    std::cerr << "Error" << std::endl;
                }
            };
            checkSymmetry(output);
            return output;
        }


    private:
        void parseCommandLine(int argc, char* argv[]) {
            if (argc < 2) {
                program = "-h";
                return;
            } else if (argc == 2) {
                program = argv[1];
                has_asked_help = true;
                return;
            } else {
                program = argv[1];
            }

            std::string_view tmp_string;
            unsigned int idx_option = 0;
            for (int i = 0; i < argc - 2; ++i) {  // exclude executable and program name
                tmp_string = argv[i + 2];

                // is it --help? if so, no need to continue the parsing
                if (tmp_string == "--help" || tmp_string == "-help" || tmp_string == "help" ||
                    tmp_string == "-h" || tmp_string == "h") {
                    has_asked_help = true;
                    return;
                }

                // check that it is not a single - or --. If so, ignore it.
                if (tmp_string == "--" || tmp_string == "-")
                    continue;

                // is it an option? if so, save the index. -x: longname=false, --x: longname=true
                if ((tmp_string.size() > 1) &&
                    (tmp_string[0] == '-' && !std::isdigit(tmp_string[1])) ||
                    (tmp_string.rfind("--", 1) == 0 && !std::isdigit(tmp_string[2]))) {
                    idx_option = i + 2;
                    if (m_options_cmdline.count(tmp_string.data()))
                        std::cerr << "Same option specified twice" << std::endl;
                    m_options_cmdline[argv[idx_option]];
                    continue;
                }

                // If the first argument isn't an option, it should be a parameter file.
                // If no options where found at the second iteration, it is not a valid
                // syntax (only one parameter file allowed).
                if (idx_option == 0 && i == 0) {
                    parameter_file = tmp_string;
                    continue;
                } else if (idx_option == 0 && i == 1) {
                    has_asked_help = true;
                    return;
                }

                // If ',' within the string_view, split and add the elements in m_cmd_options.
                String::split(tmp_string, ",", m_options_cmdline[argv[idx_option]]);
            }
        }

        /**
         * @short           Parse the parameter file if there is one registered into this->parameter_file.
         *
         * @see             Parser::Parser() for more details about the format of the parameter file.
         *
         * @param a_path    Path of the parameter file.
         */
        void parseParameterFile() {
            if (parameter_file.empty())
                return;

            std::ifstream file(parameter_file);
            if (file.is_open()) {
                std::string line;
                while (std::getline(file, line)) {
                    // The line should at least contain "noa_".
                    if (line.size() < 5)
                        continue;

                    // Increment up to the first non-space character.
                    size_t idx_start = 0;
                    for (int i = 0; i < line.size(); ++i) {
                        if (std::isspace(line[i])) {
                            continue;
                        } else {
                            idx_start = i;
                            break;
                        }
                    }

                    // If it doesn't start with "noa_", skip this line.
                    if (!line.rfind("noa_", idx_start + 3))
                        continue;

                    // Get idx in-line comment and equal sign. If no "=", skip this line.
                    size_t idx_end = line.find('#', idx_start + 4);
                    size_t idx_equal = line.find('=', idx_start + 4);
                    if (idx_equal == std::string::npos || idx_start + 4 == idx_equal ||
                        idx_equal > idx_end)
                        continue;

                    // Get the [key, value], of the line.
                    m_options_parameter_file.emplace(
                            String::rightTrim(line.substr(idx_start + 3, idx_equal)),
                            String::splitFirstOf<std::string_view>(
                                    {line.data() + idx_equal + 1, idx_end - idx_equal + 1}, ", "));
                }
                file.close();
            } else {
                std::cout << '"' << parameter_file
                          << "\": parameter file does not exist or you don't the permission to read it\n";
            }
        };

        /**
         *
         * @param a_longname
         * @param a_shortname
         * @return
         */
        std::vector<std::string>*
        getOption(const std::string& a_longname, const std::string& a_shortname) {
            if (m_options_cmdline.count(a_longname)) {
                if (m_options_cmdline.count(a_shortname))
                    std::cerr << "Option specified twice using long name and short name"
                              << std::endl;
                return &m_options_cmdline.at(a_longname);

            } else if (m_options_cmdline.count(a_shortname)) {
                return &m_options_cmdline.at(a_shortname);

            } else if (m_options_parameter_file.count(a_longname)) {
                if (m_options_parameter_file.count(a_shortname))
                    std::cerr << "Option specified twice using long name and short name"
                              << std::endl;
                return &m_options_parameter_file.at(a_longname);

            } else if (m_options_parameter_file.count(a_shortname)) {
                return &m_options_parameter_file.at(a_shortname);

            } else {
                return nullptr;
            }
        }
    };
}
