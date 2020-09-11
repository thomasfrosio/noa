///**
// * @file Parser.h
// * @brief Parser of the user inputs (command line and parameter file).
// * @author Thomas - ffyr2w
// * @date 15 Jul 2020
// */
//#pragma once
//
//#include "Core.h"
//#include "String.h"
//#include "Assert.h"
//#include "Helper.h"
//
//
//namespace Noa {
//
//    /**
//     * @class       Parser
//     * @brief       Input manager for the noa programs.
//     * @details     Parse the command line and the parameter file (if any) and makes
//     *              the inputs accessible for other programs.
//     *
//     * @see         Parser::Parser() to parse the user inputs.
//     * @see         Parser::get...() to retrieve the formatted inputs.
//     */
//    class Parser {
//    private:
//        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline;
//        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file;
//        std::vector<std::string> m_usage{};
//
//    public:
//        std::string program;
//        std::string parameter_file;
//        bool has_asked_help{false};
//
//
//    public:
//        /**
//         * @fn          Parser::Parser() - default constructor.
//         * @short       Parses the command line and the parameter file, if it exists.
//         *
//         * @details
//         * Supported scenarios:
//         *      1) [./noa]
//         *      2) [./noa] [-h]
//         *      3) [./noa] [program] [-h]
//         *      4) [./noa] [program] [--option]
//         *      5) [./noa] [program] [file] [--option]
//         *
//         * - Scenario 1 or 2, this->program is set to "-h".
//         * - Scenario 3, this->program is set to the [program], this->has_asked_help is set to true
//         *   and the [--option] is NOT parsed. It is up to your program to check whether or not a
//         *   help was asked.
//         * - Scenario 4, this->program is set to the [program] and the [--option] is parsed and
//         *   accessible using the this->get...(...) methods. If a [-h] is found within [--option],
//         *   this->has_asked_help is set to true and the parsing stops.
//         * - Scenario 5, [file] is parsed and its options can be accessed using the this->get...(...)
//         *   methods. Otherwise, it is like scenario 4.
//         *
//         * Arguments:
//         * - [-h]: can be any of the the following: "--help", "-help", "help", "-h", "h".
//         * - [program]: should be the second argument specified in the command line.
//         *              This is saved in this->program.
//         * - [--option]: Sequence of #options and #values.
//         *               #options:
//         *               When entered at the command line, #options must be preceded by one or
//         *               two dashes (- or --) and must be followed by a space then a value. One
//         *               dash specifies an option with a short-name and two dashes specify an option
//         *               with a long-name. Options cannot be concatenated the way single letter
//         *               options in Unix programs often can be.
//         *               #values:
//         *               They are arguments that are not prefixed with "--" or '-', but are prefixed
//         *               with an #option. If the value contains embedded blanks it must be enclosed
//         *               in quotes. To specify multiple #values for one #option, use a comma or
//         *               a whitespace, e.g. --size 100,100,100. Commas without values indicates that
//         *               the default value for that position should be taken. For example, "12,,15,"
//         *               takes the default for the second and fourth values. Defaults can be used
//         *               only when a fixed number of values are expected.
//         * - [file]: Parameter/option file. #options should start at the beginning of a line and
//         *           be prefixed by "noa_". The #values should be specified after an '=', i.e.
//         *           #option=#value. Whitespaces are ignored before and after #option, '=' and
//         *           #value. Multiple #values can be specified as in [--option]. Inline comments
//         *           are allowed and should start with '#'.
//         *
//         * @note1   The arguments are positional arguments, except [-h].
//         * @note2   Single "--" or '-' are ignored.
//         * @note3   If an option is both specified in the command line and in the parameter file,
//         *          the command line has the priority.
//         * @note4   Both long-names and short-names are allowed in the command line and in parameter
//         *          file.
//         *
//         * @param argc  How many C-strings are contained in argv.
//         * @param argv  Contains the stripped and split C-strings. See formatting below.
//         *
//         * @example     Here is an example on how to start your program using the this parser:
//         * @code        int main(int argc, char* argv) {
//         *                  // Parse cmd line and parameter file.
//         *                  noa::Parser parser(argc, argv);
//         *                  switch (parser.program) {
//         *                      case "program1":
//         *                          start_program1();
//         *                      // ...
//         *                      case "-h":
//         *                          print_global_help();
//         *                      default:
//         *                          printf("Unknown program");
//         *                          print_global_help();
//         *                  }
//         *              }
//         * @endcode
//         */
//        explicit Parser(int argc, char* argv[]) {
//            parseCommandLine(argc, argv);
//            parseParameterFile();
//        }
//
//        /**
//         *
//         * @param a_vec
//         *
//         * Usage = longname, shortname, type, default_value, help_string.
//         */
//        constexpr void setUsage(std::initializer_list<std::string> a_vec) {
//            if (a_vec.size() % 5) {
//                NOA_CORE_ERROR("Parser::setUsage: the size of the usage vector should be a "
//                               "multiple of 5, got {}", a_vec.size());
//            }
//            m_usage = a_vec;
//        }
//
//
//        /**
//         * @fn                          get(...)
//         * @short                       For a given option, get a corresponding integer(s)
//         *                              stored in the noa::Parser.
//         *
//         * @tparam [in,out] T           Returned type.
//         * @tparam [in] N               Number of expected values. It should be a positive int, or
//         *                              -1 which would indicates that an unknown range of integers
//         *                              are to be expected. 0 is not allowed.
//         * @param [in] a_long_name      Long name of the option (without the two dashes), e.g "size".
//         *
//         * @note1                       The command line options have the priority, then the
//         *                              parameter file, then the default value.
//         */
//        template<typename T, int N = 1>
//        auto get(const std::string& a_long_name) {
//            NOA_CORE_DEBUG(__PRETTY_FUNCTION__);
//            static_assert(Traits::is_sequence_v<T> || (Traits::is_int_v<T> && N == 1));
//
//            // Get usage and the value(s).
//            const size_t usage_idx = getUsage(a_long_name);
//            const std::string& usage_short = getUsageShort(usage_idx);
//            const std::string& usage_type = getUsageType(usage_idx);
//            const std::string& usage_value = getUsageDefault(usage_idx);
//            std::vector<std::string>* value = getOption(a_long_name, usage_short);
//
//            assertUsageType<T, N>(usage_type);
//
//            // Parse the default value.
//            std::vector<std::string> default_value = String::parse(usage_value);
//            if (N != -1 && default_value.size() != N) {
//                NOA_CORE_ERROR("Parser::get: Number of default value(s) ({}) doesn't match "
//                               "the desired number of value(s) ({})",
//                               default_value.size(), N);
//            }
//
//            // If option not registered or left empty, replace with the default.
//            if (!value || value->empty()) {
//                if (usage_value.empty()) {
//                    NOA_CORE_ERROR("Parser::get: No value available for option {} ({})",
//                                   a_long_name, usage_short);
//                }
//                value = &default_value;
//            }
//
//            std::remove_reference_t<T> output;
//            if constexpr(N == -1) {
//                // When an unknown number of value is expected, values cannot be defaulted
//                // based on their position. Thus, empty strings are not allowed here.
//                if constexpr (Traits::is_sequence_of_bool_v<T>) {
//                    output = String::toBool<T>(*value);
//                } else if constexpr (Traits::is_sequence_of_int_v<T>) {
//                    output = String::toInteger<T>(*value);
//                } else if constexpr (Traits::is_sequence_of_float_v<T>) {
//                    output = String::toFloat<T>(*value);
//                } else if constexpr (Traits::is_sequence_of_string_v<T>) {
//                    output = *value;
//                }
//            } else if constexpr (N == 1) {
//                // If empty or empty string, take default. Otherwise try to convert.
//                if (value->size() != 1) {
//                    NOA_CORE_ERROR("Parser::get: {} ({}): only 1 value is expected, got {}",
//                                   a_long_name, usage_short, value->size());
//                }
//                auto& chosen_value = ((*value)[0].empty()) ?
//                                     default_value[0] : (*value)[0];
//                if constexpr (Traits::is_bool_v<T>) {
//                    output = String::toBool(chosen_value);
//                } else if constexpr (Traits::is_int_v<T>) {
//                    output = String::toInteger(chosen_value);
//                } else if constexpr (Traits::is_float_v<T>) {
//                    output = String::toFloat(chosen_value);
//                } else if constexpr (Traits::is_string_v<T>) {
//                    output = chosen_value;
//                }
//            } else {
//                // Fixed range.
//                if (value->size() != N) {
//                    NOA_CORE_ERROR("Parser::get: {} ({}): {} values are expected, got {}",
//                                   a_long_name, usage_short, N, value->size());
//                }
//
//                if constexpr (Traits::is_vector_v<T>)
//                    output.reserve(value->size());
//                for (size_t i{0}; i < value->size(); ++i) {
//                    auto& chosen_value = ((*value)[i].empty()) ?
//                                         default_value[i] : (*value)[i];
//                    if constexpr (Traits::is_sequence_of_bool_v<T>) {
//                        Helper::sequenceAssign(output, String::toBool(chosen_value), i);
//                    } else if constexpr (Traits::is_sequence_of_int_v<T>) {
//                        Helper::sequenceAssign(output, String::toInteger(chosen_value), i);
//                    } else if constexpr (Traits::is_sequence_of_float_v<T>) {
//                        Helper::sequenceAssign(output, String::toFloat(chosen_value), i);
//                    } else if constexpr (Traits::is_sequence_of_string_v<T>) {
//                        Helper::sequenceAssign(output, chosen_value, i);
//                    }
//                }
//            }
//
//            NOA_TRACE("{} ({}): {}", a_long_name, usage_short, output);
//            return output;
//        }
//
//    private:
//        /**
//         *
//         * @param argc
//         * @param argv
//         */
//        void parseCommandLine(int argc, char* argv[]) {
//            if (argc < 2) {
//                program = "-h";
//                return;
//            } else if (argc == 2) {
//                program = argv[1];
//                has_asked_help = true;
//                return;
//            } else {
//                program = argv[1];
//            }
//
//            std::string_view tmp_string;
//            const char* tmp_option = nullptr;
//            for (int i = 0; i < argc - 2; ++i) {  // exclude executable and program name
//                tmp_string = argv[i + 2];
//
//                // is it --help? if so, no need to continue the parsing
//                if (tmp_string == "--help" || tmp_string == "-help" || tmp_string == "help" ||
//                    tmp_string == "-h" || tmp_string == "h") {
//                    has_asked_help = true;
//                    return;
//                }
//
//                // check that it is not a single - or --. If so, ignore it.
//                if (tmp_string == "--" || tmp_string == "-")
//                    continue;
//
//                if (tmp_string.size() > 2 &&
//                    tmp_string.rfind("--", 1) == 0) {
//                    // Option - long-name
//                    tmp_option = argv[i + 2] + 2; // remove the --
//                    if (m_options_cmdline.count(tmp_option)) {
//                        NOA_CORE_ERROR("Parser::parseCommandLine: option \"{}\" is specified twice",
//                                       tmp_option);
//                    }
//                    m_options_cmdline[tmp_option];
//                    continue;
//                } else if (tmp_string.size() > 1 &&
//                           tmp_string[0] == '-' &&
//                           !std::isdigit(tmp_string[1])) {
//                    // Option - short-name
//                    tmp_option = argv[i + 2] + 1; // remove the --
//                    if (m_options_cmdline.count(tmp_option)) {
//                        NOA_CORE_ERROR("Parser::parseCommandLine: option \"{}\" is specified twice",
//                                       tmp_option);
//                    }
//                    m_options_cmdline[tmp_option];
//                }
//
//                // If the first argument isn't an option, it should be a parameter file.
//                // If no options where found at the second iteration, it is not a valid
//                // syntax (only one parameter file allowed).
//                if (!tmp_option && i == 0) {
//                    parameter_file = tmp_string;
//                    continue;
//                } else if (!tmp_option && i == 1) {
//                    has_asked_help = true;
//                    return;
//                }
//
//                // Parse the value.
//                String::parse(tmp_string, m_options_cmdline.at(tmp_option));
//            }
//        }
//
//        /**
//         * @short           Parse the parameter file if there is one registered into this->parameter_file.
//         *
//         * @see             Parser::Parser() for more details about the format of the parameter file.
//         *
//         * @param a_path    Path of the parameter file.
//         */
//        void parseParameterFile() {
//            if (parameter_file.empty())
//                return;
//
//            std::ifstream file(parameter_file);
//            if (file.is_open()) {
//                std::string line;
//                while (std::getline(file, line)) {
//                    // The line should at least contain "noa_".
//                    if (line.size() < 5)
//                        continue;
//
//                    // Increment up to the first non-space character.
//                    size_t idx_start = 0;
//                    for (int i = 0; i < line.size(); ++i) {
//                        if (std::isspace(line[i])) {
//                            continue;
//                        } else {
//                            idx_start = i;
//                            break;
//                        }
//                    }
//
//                    // If it doesn't start with "noa_", skip this line.
//                    if (!line.rfind("noa_", idx_start + 3))
//                        continue;
//
//                    // Get idx in-line comment and equal sign. If no "=", skip this line.
//                    size_t idx_end = line.find('#', idx_start + 4);
//                    size_t idx_equal = line.find('=', idx_start + 4);
//                    if (idx_equal == std::string::npos ||
//                        idx_start + 4 == idx_equal ||
//                        idx_equal > idx_end)
//                        continue;
//
//                    // Get the [key, value], of the line.
//                    m_options_parameter_file.emplace(
//                            String::rightTrim(line.substr(idx_start + 3, idx_equal)),
//                            String::parse<std::string_view>(
//                                    {line.data() + idx_equal + 1, idx_end - idx_equal + 1}));
//                }
//                file.close();
//            } else {
//                NOA_CORE_ERROR("Parser::parseParameterFile: \"{}\" does not exist or you don't "
//                               "have the permission to read it", parameter_file);
//            }
//        }
//
//        /**
//         *
//         * @param a_longname
//         * @param a_shortname
//         * @return
//         */
//        std::vector<std::string>* getOption(const std::string& a_longname,
//                                            const std::string& a_shortname) {
//            if (m_options_cmdline.count(a_longname)) {
//                if (m_options_cmdline.count(a_shortname)) {
//                    NOA_CORE_ERROR("Parser::getOption: \"{}\" (long-name) and \"{}\" (short-name) "
//                                   "are linked to the same option, thus cannot be both specified "
//                                   "in the command line", a_longname, a_shortname);
//                }
//                return &m_options_cmdline.at(a_longname);
//
//            } else if (m_options_cmdline.count(a_shortname)) {
//                return &m_options_cmdline.at(a_shortname);
//
//            } else if (m_options_parameter_file.count(a_longname)) {
//                if (m_options_parameter_file.count(a_shortname)) {
//                    NOA_CORE_ERROR("Parser::getOption: \"{}\" (long-name) and \"{}\" (short-name) "
//                                   "are linked to the same option, thus cannot be both specified "
//                                   "in the parameter file", a_longname, a_shortname);
//                }
//                return &m_options_parameter_file.at(a_longname);
//
//            } else if (m_options_parameter_file.count(a_shortname)) {
//                return &m_options_parameter_file.at(a_shortname);
//
//            } else {
//                return nullptr;
//            }
//        }
//
//        constexpr size_t getUsage(const std::string& a_longname) const {
//            if (m_usage.empty()) {
//                NOA_CORE_ERROR("Parser::getUsage: usage is not set. Set it with Parser::setUsage");
//            }
//            for (size_t i{0}; i < m_usage.size(); i += 5) {
//                if (m_usage[i] == a_longname)
//                    return i;
//            }
//            NOA_CORE_ERROR("Parser::getUsage: the \"{}\" option is not registered in the usage. "
//                           "Did you give the longname?", a_longname);
//        }
//
//        inline const std::string& getUsageShort(size_t a_idx) const {
//            return m_usage[a_idx + 1];
//        }
//
//        inline const std::string& getUsageType(size_t a_idx) const {
//            return m_usage[a_idx + 2];
//        }
//
//        inline const std::string& getUsageDefault(size_t a_idx) const {
//            return m_usage[a_idx + 3];
//        }
//
//        inline const std::string& getUsageHelp(size_t a_idx) const {
//            return m_usage[a_idx + 4];
//        }
//
//        /**
//         *
//         * @tparam T
//         * @tparam N
//         * @param a_usage_type
//         */
//        template<typename T, int N>
//        static void assertUsageType(const std::string& a_usage_type) {
//            static_assert(N != 0);
//
//            // Number of values.
//            if constexpr(N == -1) {
//                if (a_usage_type[0] != 'A') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected number of values (array)",
//                                   a_usage_type);
//                }
//            } else if constexpr(N == 1) {
//                if (a_usage_type[0] != 'S') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected number of value (1)",
//                                   a_usage_type);
//                }
//            } else if constexpr(N == 2) {
//                if (a_usage_type[0] != 'P') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected number of values (2)",
//                                   a_usage_type);
//                }
//            } else if constexpr(N == 3) {
//                if (a_usage_type[0] != 'T') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected number of values (3)",
//                                   a_usage_type);
//                }
//            }
//
//            // Types.
//            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
//                if (a_usage_type[1] != 'F') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected type (floating point)",
//                                   a_usage_type);
//                }
//            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
//                if (a_usage_type[1] != 'I') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected type (integer)",
//                                   a_usage_type);
//                }
//            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
//                if (a_usage_type[1] != 'B') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected type (boolean)",
//                                   a_usage_type);
//                }
//            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
//                if (a_usage_type[1] != 'S') {
//                    NOA_CORE_ERROR("Parser::assertUsageType: the usage type ({}) does not "
//                                   "correspond to the expected type (string)",
//                                   a_usage_type);
//                }
//            }
//        }
//    };
//}
