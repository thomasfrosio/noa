/**
 * @file Parser.h
 * @brief Parser of the user inputs (command line and parameter file).
 * @author Thomas - ffyr2w
 * @date 15 Jul 2020
 */
#pragma once

#include "Noa.h"
#include "String.h"
#include "Assert.h"
#include "Helper.h"


namespace Noa {

    /**
     * @class       Parser
     * @brief       Input manager for the noa programs.
     * @details     Parse the command line and the parameter file (if any) and makes
     *              the inputs accessible for other programs.
     *
     * @see         Parser::Parser() to parse the user inputs.
     * @see         Parser::get...() to retrieve the formatted inputs.
     */
    class Parser {
    public:
        std::string program;
        std::string parameter_file;
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
         * - Scenario 1 or 2, this->program is set to "-h".
         * - Scenario 3, this->program is set to the [program], this->has_asked_help is set to true
         *   and the [--option] is NOT parsed. It is up to your program to check whether or not a
         *   help was asked.
         * - Scenario 4, this->program is set to the [program] and the [--option] is parsed and
         *   accessible using the this->get...(...) methods. If a [-h] is found within [--option],
         *   this->has_asked_help is set to true and the parsing stops.
         * - Scenario 5, [file] is parsed and its options can be accessed using the this->get...(...)
         *   methods. Otherwise, it is like scenario 4.
         *
         * Arguments:
         * - [-h]: can be any of the the following: "--help", "-help", "help", "-h", "h".
         * - [program]: should be the second argument specified in the command line.
         *              This is saved in this->program.
         * - [--option]: Sequence of #options and #values.
         *               #options:
         *               When entered at the command line, #options must be preceded by one or
         *               two dashes (- or --) and must be followed by a space then a value. One
         *               dash specifies an option with a short-name and two dashes specify an option
         *               with a long-name. Options cannot be concatenated the way single letter
         *               options in Unix programs often can be.
         *               #values:
         *               They are arguments that are not prefixed with "--" or '-', but are prefixed
         *               with an #option. If the value contains embedded blanks it must be enclosed
         *               in quotes. To specify multiple #values for one #option, use a comma or
         *               a whitespace, e.g. --size 100,100,100. Commas without values indicates that
         *               the default value for that position should be taken. For example, "12,,15,"
         *               takes the default for the second and fourth values. Defaults can be used
         *               only when a fixed number of values are expected.
         * - [file]: Parameter/option file. #options should start at the beginning of a line and
         *           be prefixed by "noa_". The #values should be specified after an '=', i.e.
         *           #option=#value. Whitespaces are ignored before and after #option, '=' and
         *           #value. Multiple #values can be specified as in [--option]. Inline comments
         *           are allowed and should start with '#'.
         *
         * @note1   The arguments are positional arguments, except [-h].
         * @note2   Single "--" or '-' are ignored.
         * @note3   If an option is both specified in the command line and in the parameter file,
         *          the command line has the priority.
         * @note4   Both long-names and short-names are allowed in the command line and in parameter
         *          file.
         *
         * @param argc  How many C-strings are contained in argv.
         * @param argv  Contains the stripped and split C-strings. See formatting below.
         *
         * @example     Here is an example on how to start your program using the this parser:
         * @code        int main(int argc, char* argv) {
         *                  // Parse cmd line and parameter file.
         *                  noa::Parser parser(argc, argv);
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
            parseCommandLine(argc, argv);
            parseParameterFile();
        }

        /**
         * @fn                          getInteger(...)
         * @short                       For a given option, get a corresponding integer(s)
         *                              stored in the noa::Parser.
         *
         * @tparam [in,out] T           Returned type. A sequence of integers or an integer.
         * @tparam [in] N               Number of expected values. It should be a positive int, or
         *                              -1 which would indicates that an unknown range of integers
         *                              are to be expected. 0 is not allowed.
         * @param [in] a_long_name      Long name of the option (without the two dashes), e.g "Input".
         * @param [in] a_short_name     Short name of the option (without the dash), e.g. "i".
         * @param [in, out] a_value     Default value(s) to use if the option is unknown, empty or
         *                              if one of the value is not specified.
         * @param [in] a_range_min      Minimum value allowed. Using noa::Assert::range().
         * @param [in] a_range_max      Maximum value allowed. Using noa::Assert::range().
         *
         * @note1                       The command line options have the priority, then the
         *                              parameter file, then the default value.
         */
        template<typename T = int, int N = 1>
        auto getInteger(const char* a_long_name,
                        const char* a_short_name,
                        T&& a_value,
                        int a_range_min,
                        int a_range_max) {
            static_assert((Traits::is_sequence_of_int_v<T> && (N == -1 || N > 0)) ||
                          (Traits::is_int_v<T> && N == 1));

            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            // Check default value has correct size.
            if constexpr(Traits::is_sequence_of_int_v<T>) {
                if (N != -1 && a_value.size() != N)
                    throw std::invalid_argument("default value should have the expected size");
            }

            // Shortcut - Unknown option or empty. Take the default value.
            if (!value || (*value).empty()) {
                Assert::range(a_value, a_range_min, a_range_max);
                // print key: value
                return std::forward<T>(a_value);
            }

            std::remove_reference_t<T> output;
            if constexpr(N == -1) {
                // When an unknown number of value is expected, values cannot be defaulted
                // based on their position. Thus, empty strings are not allowed here.
                // String::toInteger will raise an error if it is empty, so leave it be.
                output = String::toInteger<T>(*value);

            } else if constexpr(N == 1) {
                // If empty or empty string, take default. Otherwise try to convert to int.
                if ((*value).size() == 1) {
                    if ((*value)[0].empty())
                        output = a_value;
                    else
                        output = String::toInteger((*value)[0]);
                } else {
                    throw std::out_of_range("Only one value is expected.");
                }
            } else {
                // Fixed range.
                if ((*value).size() != N)
                    throw std::invalid_argument("Error");

                if constexpr(Traits::is_vector_v<T>)
                    output.reserve(N);
                unsigned int i{0};
                for (auto& element : *value) {
                    if (element.empty())
                        Helper::sequenceAssign(output, a_value[i], i);
                    else
                        Helper::sequenceAssign(output, String::toInteger(element), i);
                }
            }
            Assert::range(output, a_range_min, a_range_max);
            // print key: value
            return output;
        }

        // Same as above, but without default values.
        template<typename T = int, int N = 1>
        auto getInteger(const char* a_long_name,
                        const char* a_short_name,
                        int a_range_min,
                        int a_range_max) {
            static_assert((Traits::is_sequence_of_int_v<T> && (N == -1 || N > 0)) ||
                          (Traits::is_int_v<T> && N == 1));

            // First, get the value from the parser.
            std::vector<std::string>* value = getOption(a_long_name, a_short_name);

            if (!value || (*value).empty())
                throw std::invalid_argument("");

            std::remove_reference_t<T> output;
            if constexpr(N == -1) {
                // When an unknown number of value is expected, values cannot be defaulted
                // based on their position. Thus, empty strings are not allowed here.
                // String::toInteger will raise an error if it is empty, so leave it be.
                output = String::toInteger<T>(*value);

            } else if constexpr(N == 1) {
                // If empty or empty string, take default. Otherwise try to convert to int.
                if ((*value).size() == 1) {
                    output = String::toInteger((*value)[0]);
                } else {
                    throw std::out_of_range("Only one value is expected.");
                }
            } else {
                // Fixed range.
                if ((*value).size() != N)
                    throw std::invalid_argument("Error");
                output = String::toInteger<T>(*value);
            }
            Assert::range(output, a_range_min, a_range_max);
            // print key: value
            return output;
        }


    private:
        /**
         *
         * @param argc
         * @param argv
         */
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


                if ((tmp_string.size() > 1) &&
                    (tmp_string[0] == '-' && !std::isdigit(tmp_string[1]))) {
                    // Option - short-name
                    idx_option = i + 2;
                    if (m_options_cmdline.count(argv[idx_option] + 1))
                        std::cerr << "Same option specified twice" << std::endl;
                    m_options_cmdline[argv[idx_option] + 1];
                } else if ((tmp_string.size() > 2) &&
                           (tmp_string.rfind("--", 1) == 0 && !std::isdigit(tmp_string[2]))) {
                    // Option - long-name
                    idx_option = i + 2;
                    if (m_options_cmdline.count(argv[idx_option] + 2))
                        std::cerr << "Same option specified twice" << std::endl;
                    m_options_cmdline[argv[idx_option] + 2];
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
                String::parse(tmp_string, m_options_cmdline[argv[idx_option]]);
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
                    if (idx_equal == std::string::npos ||
                        idx_start + 4 == idx_equal ||
                        idx_equal > idx_end)
                        continue;

                    // Get the [key, value], of the line.
                    m_options_parameter_file.emplace(
                            String::rightTrim(line.substr(idx_start + 3, idx_equal)),
                            String::parse<std::string_view>(
                                    {line.data() + idx_equal + 1, idx_end - idx_equal + 1}));
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
        std::vector<std::string>* getOption(const std::string& a_longname,
                                            const std::string& a_shortname) {
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
