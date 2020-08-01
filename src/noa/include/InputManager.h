/**
 * @file InputManager.h
 * @brief Input manager - Manages all user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */

#pragma once

#include "Core.h"
#include "String.h"
#include "Assert.h"
#include "Helper.h"

namespace Noa {
    class InputManager {
    public:
        bool has_asked_help{false};

    private:
        const int m_argc;
        const char** m_argv;

        std::string program{};
        std::string parameter_file{};

        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline{};
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file{};
        std::vector<std::string> m_usage{};
        std::vector<std::string> m_available{};

        enum Usage : unsigned int {
            long_name, short_name, type, default_value, help
        };

        const std::string m_usage_header = fmt::format(
                FMT_STRING("Welcome to noa.\n"
                           "Version {}\n"
                           "Website: {}\n\n"
                           "Usage:\n"
                           "     noa [global options]\n"
                           "     noa program [program options...]\n"
                           "     noa program parameter_file [program options...]\n\n"),
                NOA_VERSION_LONG,
                NOA_WEBSITE);

        const std::string m_usage_footer = fmt::format(
                FMT_STRING("\nGlobal options:\n"
                           "   --help, -h      Show global help.\n"
                           "   --version, -v   Show the version.\n"));

        std::string m_usage_programs = "Programs:\n";


    public:
        InputManager(const int argc, const char** argv) : m_argc(argc), m_argv(argv) {}

        void setAvailable(const std::vector<std::string>& a_programs) {
            size_t size = a_programs.size();
            if (size % 2) {
                NOA_CORE_ERROR("InputManager::setAvailable: the size of the program vector should "
                               "be a multiple of 2, got {}", size);
            }
            m_available.reserve(size / 2);
            for (size_t i{0}; i < size; i += 2) {
                m_available.emplace_back(a_programs[i]);
                m_usage_programs += fmt::format("     {:<{}} {}\n",
                                                a_programs[i], 15, a_programs[i + 1]);
            }
        }

        void printAvailable() const {
            if (m_available.empty()) {
                NOA_CORE_ERROR("InputManager::printAvailable: the available programs are not set. "
                               "Set them with InputManager::setAvailable");
            }
            fmt::print(m_usage_header);
            fmt::print(m_usage_programs);
            fmt::print(m_usage_footer);
        }

        const std::string& setProgram() {
            if (m_argc < 2)
                program = "--help";
            else if (std::find(m_available.begin(),
                               m_available.end(),
                               m_argv[1]) == m_available.end()) {
                const std::string_view argv1 = m_argv[1];
                if (argv1 == "-h" || argv1 == "--help" || argv1 == "help" ||
                    argv1 == "h" || argv1 == "-help")
                    program = "--help";
                else if (argv1 == "-v" || argv1 == "--version" || argv1 == "version" ||
                         argv1 == "v" || argv1 == "-version")
                    program = "--version";
                else {
                    NOA_CORE_ERROR("InputManager::setProgram: \"{}\" is not registered as an "
                                   "available program. Add it with InputManager::setAvailable",
                                   argv1);
                }
            } else program = m_argv[1];
            return program;
        }

        template<typename Sequence>
        void setUsage(Sequence&& a_usage) {
            static_assert(Traits::is_sequence_of_string_v<Sequence>);
            if (a_usage.size() % 5) {
                NOA_CORE_ERROR("InputManager::setUsage: the size of the usage vector should be a "
                               "multiple of 5, got {}", a_usage.size());
            }
            m_usage = std::forward<Sequence>(a_usage);
        }

        void printUsage() const {
            if (program.empty()) {
                NOA_CORE_ERROR("InputManager::printUsage: program is not set. "
                               "Set it first with InputManager::parse or InputManager::setProgram");
            } else if (m_usage.empty()) {
                NOA_CORE_ERROR("InputManager::printUsage: usage is not set. "
                               "Set it first with InputManager::setUsage");
            }
            fmt::print(m_usage_header);
            fmt::print("{} options:\n", program);

            // Get the first necessary padding.
            size_t option_names_padding{0};
            for (unsigned int i = 0; i < m_usage.size(); i += 5) {
                size_t current_size = (m_usage[i + Usage::long_name].size() +
                                       m_usage[i + Usage::short_name].size());
                if (current_size > option_names_padding)
                    option_names_padding = current_size;
            }
            option_names_padding += 10;

            std::string type;
            for (unsigned int i = 0; i < m_usage.size(); i += 5) {
                std::string option_names = fmt::format("   --{}, -{}",
                                                       m_usage[i + Usage::long_name],
                                                       m_usage[i + Usage::short_name]);
                if (m_usage[i + 3].empty())
                    type = fmt::format("({})", formatType(m_usage[i + Usage::type]));
                else
                    type = fmt::format("({} = {})",
                                       formatType(m_usage[i + Usage::type]),
                                       m_usage[i + Usage::default_value]);

                fmt::print("{:<{}} {:<{}} {}\n",
                           option_names, option_names_padding,
                           type, 25, m_usage[i + Usage::help]);
            }
            fmt::print(m_usage_footer);
        }

        void parseInput(int argc, char* argv[]) {
            parseCommandLine(argc, argv);
            parseParameterFile();
        }

        template<typename T, int N = 1>
        auto getInput(const std::string& a_long_name) {
            NOA_CORE_DEBUG(__PRETTY_FUNCTION__);
            static_assert(Traits::is_sequence_v<T> || (Traits::is_int_v<T> && N == 1));

            // Get usage and the value(s).
            auto[usage_short, usage_type, usage_value] = getUsage(a_long_name);
            assertType<T, N>(usage_type);
            std::vector<std::string>* value = getParsedValue(a_long_name, usage_short);

            // Parse the default value.
            std::vector<std::string> default_value = String::parse(usage_value);
            if (N != -1 && default_value.size() != N) {
                NOA_CORE_ERROR("Parser::get: Number of default value(s) ({}) doesn't match "
                               "the desired number of value(s) ({})",
                               default_value.size(), N);
            }

            // If option not registered or left empty, replace with the default.
            if (!value || value->empty()) {
                if (usage_value.empty()) {
                    NOA_CORE_ERROR("Parser::get: No value available for option {} ({})",
                                   a_long_name, usage_short);
                }
                value = &default_value;
            }

            std::remove_reference_t<T> output;
            if constexpr(N == -1) {
                // When an unknown number of value is expected, values cannot be defaulted
                // based on their position. Thus, empty strings are not allowed here.
                if constexpr (Traits::is_sequence_of_bool_v<T>) {
                    output = String::toBool<T>(*value);
                } else if constexpr (Traits::is_sequence_of_int_v<T>) {
                    output = String::toInteger<T>(*value);
                } else if constexpr (Traits::is_sequence_of_float_v<T>) {
                    output = String::toFloat<T>(*value);
                } else if constexpr (Traits::is_sequence_of_string_v<T>) {
                    output = *value;
                }
            } else if constexpr (N == 1) {
                // If empty or empty string, take default. Otherwise try to convert.
                if (value->size() != 1) {
                    NOA_CORE_ERROR("Parser::get: {} ({}): only 1 value is expected, got {}",
                                   a_long_name, usage_short, value->size());
                }
                auto& chosen_value = ((*value)[0].empty()) ?
                                     default_value[0] : (*value)[0];
                if constexpr (Traits::is_bool_v<T>) {
                    output = String::toBool(chosen_value);
                } else if constexpr (Traits::is_int_v<T>) {
                    output = String::toInteger(chosen_value);
                } else if constexpr (Traits::is_float_v<T>) {
                    output = String::toFloat(chosen_value);
                } else if constexpr (Traits::is_string_v<T>) {
                    output = chosen_value;
                }
            } else {
                // Fixed range.
                if (value->size() != N) {
                    NOA_CORE_ERROR("Parser::get: {} ({}): {} values are expected, got {}",
                                   a_long_name, usage_short, N, value->size());
                }

                if constexpr (Traits::is_vector_v<T>)
                    output.reserve(value->size());
                for (size_t i{0}; i < value->size(); ++i) {
                    auto& chosen_value = ((*value)[i].empty()) ?
                                         default_value[i] : (*value)[i];
                    if constexpr (Traits::is_sequence_of_bool_v<T>) {
                        Helper::sequenceAssign(output, String::toBool(chosen_value), i);
                    } else if constexpr (Traits::is_sequence_of_int_v<T>) {
                        Helper::sequenceAssign(output, String::toInteger(chosen_value), i);
                    } else if constexpr (Traits::is_sequence_of_float_v<T>) {
                        Helper::sequenceAssign(output, String::toFloat(chosen_value), i);
                    } else if constexpr (Traits::is_sequence_of_string_v<T>) {
                        Helper::sequenceAssign(output, chosen_value, i);
                    }
                }
            }

            NOA_TRACE("{} ({}): {}", a_long_name, usage_short, output);
            return output;
        }

    private:
        static std::string formatType(const std::string& a_type) {
            auto getType = [&]() {
                switch (a_type[1]) {
                    case 'I':
                        return "integer";
                    case 'F':
                        return "float";
                    case 'S':
                        return "string";
                    case 'B':
                        return "bool";
                    default: {
                        NOA_CORE_ERROR(
                                "InputManager::formatType: usage type ({}) not recognized",
                                a_type);
                    }
                }
            };

            switch (a_type[0]) {
                case 'S':
                    return fmt::format("1 {}", getType());
                case 'P':
                    return fmt::format("2 {}s", getType());
                case 'T':
                    return fmt::format("3 {}s", getType());
                case 'A':
                    return fmt::format("n {}(s)", getType());
                default: {
                    NOA_CORE_ERROR("InputManager::formatType: usage type ({}) not recognized",
                                   a_type);
                }
            }
        }

        template<typename T, int N>
        static void assertType(const std::string& a_usage_type) {
            static_assert(N != 0);

            // Number of values.
            if constexpr(N == -1) {
                if (a_usage_type[0] != 'A') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected number of values (array)",
                                   a_usage_type);
                }
            } else if constexpr(N == 1) {
                if (a_usage_type[0] != 'S') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected number of value (1)",
                                   a_usage_type);
                }
            } else if constexpr(N == 2) {
                if (a_usage_type[0] != 'P') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected number of values (2)",
                                   a_usage_type);
                }
            } else if constexpr(N == 3) {
                if (a_usage_type[0] != 'T') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected number of values (3)",
                                   a_usage_type);
                }
            }

            // Types.
            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
                if (a_usage_type[1] != 'F') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected type (floating point)",
                                   a_usage_type);
                }
            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
                if (a_usage_type[1] != 'I') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected type (integer)",
                                   a_usage_type);
                }
            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
                if (a_usage_type[1] != 'B') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected type (boolean)",
                                   a_usage_type);
                }
            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
                if (a_usage_type[1] != 'S') {
                    NOA_CORE_ERROR("InputManager::assertType: the usage type ({}) does not "
                                   "correspond to the expected type (string)",
                                   a_usage_type);
                }
            }
        }

        void parseCommandLine(int argc, char* argv[]) {
            std::string_view tmp_string;
            const char* tmp_option = nullptr;
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

                if (tmp_string.size() > 2 &&
                    tmp_string.rfind("--", 1) == 0) {
                    // Option - long-name
                    tmp_option = argv[i + 2] + 2; // remove the --
                    if (m_options_cmdline.count(tmp_option)) {
                        NOA_CORE_ERROR("InputManager::parseCommandLine: option \"{}\" is "
                                       "specified twice", tmp_option);
                    }
                    m_options_cmdline[tmp_option];
                    continue;
                } else if (tmp_string.size() > 1 &&
                           tmp_string[0] == '-' &&
                           !std::isdigit(tmp_string[1])) {
                    // Option - short-name
                    tmp_option = argv[i + 2] + 1; // remove the --
                    if (m_options_cmdline.count(tmp_option)) {
                        NOA_CORE_ERROR("InputManager::parseCommandLine: option \"{}\" is "
                                       "specified twice", tmp_option);
                    }
                    m_options_cmdline[tmp_option];
                }

                // If the first argument isn't an option, it should be a parameter file.
                // If no options where found at the second iteration, it is not a valid
                // syntax (only one parameter file allowed).
                if (!tmp_option && i == 0) {
                    parameter_file = tmp_string;
                    continue;
                } else if (!tmp_option && i == 1) {
                    has_asked_help = true;
                    return;
                }

                // Parse the value.
                String::parse(tmp_string, m_options_cmdline.at(tmp_option));
            }
        }

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
                NOA_CORE_ERROR("InputManager::parseParameterFile: \"{}\" does not exist or "
                               "you don't have the permission to read it", parameter_file);
            }
        }

        /**
         *
         * @param a_longname
         * @param a_shortname
         * @return
         */
        std::vector<std::string>* getParsedValue(const std::string& a_longname,
                                                 const std::string& a_shortname) {
            if (m_options_cmdline.count(a_longname)) {
                if (m_options_cmdline.count(a_shortname)) {
                    NOA_CORE_ERROR("InputManager::getParsedValue: \"{}\" (long-name) and \"{}\""
                                   "(short-name) are linked to the same option, thus cannot "
                                   "be both specified in the command line",
                                   a_longname, a_shortname);
                }
                return &m_options_cmdline.at(a_longname);

            } else if (m_options_cmdline.count(a_shortname)) {
                return &m_options_cmdline.at(a_shortname);

            } else if (m_options_parameter_file.count(a_longname)) {
                if (m_options_parameter_file.count(a_shortname)) {
                    NOA_CORE_ERROR("InputManager::getParsedValue: \"{}\" (long-name) and \"{}\""
                                   "(short-name) are linked to the same option, thus cannot "
                                   "be both specified in the parameter file",
                                   a_longname, a_shortname);
                }
                return &m_options_parameter_file.at(a_longname);

            } else if (m_options_parameter_file.count(a_shortname)) {
                return &m_options_parameter_file.at(a_shortname);

            } else {
                return nullptr;
            }
        }

        std::tuple<const std::string&, const std::string&, const std::string&>
        getUsage(const std::string& a_longname) const {
            if (m_usage.empty()) {
                NOA_CORE_ERROR("InputManager::getUsage: usage is not set. "
                               "Set it first with InputManager::setUsage");
            }
            for (size_t i{0}; i < m_usage.size(); i += 5) {
                if (m_usage[i] == a_longname)
                    return {m_usage[i + Usage::short_name],
                            m_usage[i + Usage::type],
                            m_usage[i + Usage::default_value]};
            }
            NOA_CORE_ERROR("InputManager::getUsage: the \"{}\" option is not registered in "
                           "the usage. Did you give the longname?", a_longname);
        }
    };
}
