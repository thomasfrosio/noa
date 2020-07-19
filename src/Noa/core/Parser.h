#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cctype>
#include <unordered_map>
#include <memory>

#include "String.h"

namespace Noa {

    class Parser {
    public:
        /**
         * @short       Singleton that parse and manages the command line and the
         *              parameter file, if it exists.
         *
         * @param argc  How many c-strings are contained in argv.
         * @param argv  Contains the stripped and split c-strings. See formatting below.
         *
         * @example     Here is an example on how to start your program using the Noa::Parser:
         * @code        int main(int argc, char* argv) {
         *                  // Parse cmd line and parameter file.
         *                  Noa::Parser::instance():parse(argc, argv);
         *                  switch (Noa::Parser::instance().program) {
         *                      case "program1": start_program1();
         *                      // ...
         *                      case "-h": print_global_help();
         *                      default:
         *                          printf("Unknown program");
         *                          print_global_help();
         *                  }
         *              }
         * @endcode
         *
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
         */
        explicit Parser(int argc, char* argv[]) {
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
            unsigned int current_option_idx = 0;
            for (int i = 0; i < argc - 2; ++i) {  // exclude executable name and program name

                // create a string_view of the c-string
                tmp_string = argv[i + 2];

                // is it --help? if so, no need to continue the parsing
                if (tmp_string == "--help" ||
                    tmp_string == "-help" ||
                    tmp_string == "help" ||
                    tmp_string == "-h") {
                    has_asked_help = true;
                    return;
                }

                // check that it is not a single '.', '-', ',' or a "--". If so, ignore it.
                if (tmp_string == "--" || tmp_string.size() == 1 && (tmp_string[0] == ',' ||
                                                                     tmp_string[0] == '.' ||
                                                                     tmp_string[0] == '-'))
                    continue;

                // is it an option? if so, save the index. -x: longname=false, --x: longname=true
                if ((tmp_string[0] == '-' && !isdigit(tmp_string[1])) ||
                    (tmp_string.rfind("--", 1) == 0 && !isdigit(tmp_string[2]))) {
                    current_option_idx = i + 2;
                    if (m_options_cmdline.count(argv[current_option_idx]))
                        std::cerr << "Same option specified twice" << std::endl;
                    m_options_cmdline[argv[current_option_idx]];
                    continue;
                }

                // If the first argument isn't an option, it should be a parameter file.
                // If no options where found at the second iteration, it is not a valid
                // syntax (only one parameter file allowed).
                if (current_option_idx == 0 && i == 0) {
                    has_parameter_file = true;
                    continue;
                } else if (current_option_idx == 0 && i == 1) {
                    has_asked_help = true;
                    return;
                }

                // If ',' within the string_view, split and add the elements in m_cmd_options.
                String::split(tmp_string, ",", m_options_cmdline[argv[current_option_idx]]);
            }

            // Once the command line is parsed, parse the parameter file if it exists.
            if (has_parameter_file)
                parseParameterFile(argv[2]);
        };

        /**
         * Parse the parameter file if there is one.
         *
         * The options within the parameter file should be prefixed with "noa_".
         * Comments are defined using the # prefix. Inline comments are allowed.
         *
         * @param a_path: path of the parameter file.
         */
        void parseParameterFile(const char* a_path) {
            std::ifstream file(a_path);
            if (file.is_open()) {
                std::string line;
                std::vector<std::string> tmp_line;

                while (std::getline(file, line)) {
                    // If doesn't start with "noa_", skip this line.
                    String::leftTrim(line);
                    if (!line.rfind("noa_", 3))
                        continue;

                    // Get idx in-line comment and equal sign. If no "=", skip this line.
                    size_t idx_end = line.find_first_of('#', 4);
                    size_t idx_equal = line.find_first_of("=", 4, idx_end);
                    if (idx_equal == std::string::npos)
                        continue;

                    // Get the [key, value], of the line.
                    m_options_parameter_file.emplace(
                            String::rightTrim(line.substr(4, idx_equal)),
                            String::splitFirstOf<std::string_view>(line.substr(idx_equal + 1, idx_end), ", "));
                }
                file.close();
            } else {
                std::cout << '"' << a_path << "\": parameter file does not exist or you don't the permission to read it\n";
            }
        };

        /**
         *
         * @param a_longname
         * @param a_shortname
         * @return
         */
        std::vector<std::string>* get(const std::string& a_longname, const std::string& a_shortname) {
            if (m_options_cmdline.count(a_longname)) {
                if (m_options_cmdline.count(a_shortname))
                    std::cerr << "Option specified twice using long name and short name" << std::endl;
                return &m_options_cmdline.at(a_longname);

            } else if (m_options_cmdline.count(a_shortname)) {
                return &m_options_cmdline.at(a_shortname);

            } else if (m_options_parameter_file.count(a_longname)) {
                return &m_options_parameter_file.at(a_shortname);

            } else {
                return nullptr;
            }
        };

    private:
        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline;
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file;

    public:
        std::string program;
        bool has_parameter_file{false};
        bool has_asked_help{false};
    };

}
