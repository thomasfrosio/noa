//
// Created by thomas on 05/07/2020.
//



#include "Parser.h"

namespace Noa {

    /**
     * @short       Parse the command line, which is an array of c-strings.
     *
     * @param argc  How many c-strings are contained in argv. It must be >0.
     * @param argv  Contains the stripped c-strings.
     */
    Parser::Parser(int argc, char* argv[]) {
        // no arguments: ask for global help
        // one argument: ask for program help
        if (argc == 1) {
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
            String::splitFind(tmp_string, ',', m_options_cmdline[argv[current_option_idx]]);
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
    void Parser::parseParameterFile(const char* a_path) {
        std::ifstream file(a_path);
        if (file.is_open()) {
            std::string line;
            std::vector<std::string> tmp_line;

            while (std::getline(file, line)) {
                // if doesn't start with "noa_", skip this line
                String::leftTrim(line);
                if (!line.rfind("noa_", 3))
                    continue;

                // Get idx in-line comment and equal sign.
                size_t idx_end = line.find_first_of('#', 4);
                size_t idx_equal = line.find_first_of("=", 4, idx_end - 4);

                // Get the [key, value], of the line.
                m_options_parameter_file[String::rightTrim(line.substr(4, idx_equal))] =
                        String::strip(String::splitFind(line.substr(idx_equal + 1, idx_end), ','));
            }
            file.close();
        } else {
            std::cout << '"' << a_path << "\": parameter file does not exist or you don't the permission to read it\n";
        }
    }

    /**
     * // Parse line from a parameter file: noa_* = value1, value2, etc...
     *
     *
     *
     * @param a_str: Line from the parameter file. To be
     * @return
     */
    std::pair<std::string, std::vector<std::string>> Parser::parseParameterLine(const std::string& a_str) {


        return std::pair<std::string, std::vector<std::string>>();
    }

}
