/**
 * @file InputManager.h
 * @brief Input manager - Manages all user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */

#pragma once

#include "../utils/Core.h"
#include "../utils/String.h"
#include "../utils/Assert.h"
#include "../utils/Helper.h"

#include <fmt/chrono.h>

namespace Noa {
    class InputManager {
    private:
        const int m_argc;
        const char** m_argv;

        std::vector<std::string> m_available_commands{};
        std::vector<std::string> m_available_options{};

        std::string command{};
        std::string parameter_file{};

        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline{};
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file{};

        bool is_parsed{false};

        enum Usage : unsigned int {
            u_long_name, u_short_name, u_type, u_default_value, u_help
        };

        const std::string m_usage_header = fmt::format(
                FMT_COMPILE("Welcome to noa.\n"
                            "Version {}\n"
                            "Website: {}\n\n"
                            "Usage:\n"
                            "     noa [global options]\n"
                            "     noa command [command options...]\n"
                            "     noa command parameter_file [command options...]\n\n"),
                getVersion(),
                NOA_WEBSITE);

        const std::string m_usage_footer = fmt::format("\nGlobal options:\n"
                                                       "   --help, -h      Show global help.\n"
                                                       "   --version, -v   Show the version.\n");

    public:
        InputManager(const int argc, const char** argv) : m_argc(argc), m_argv(argv) {}

        const std::string& setCommand(const std::vector<std::string>& a_programs) {
            if (a_programs.size() % 2) {
                NOA_CORE_ERROR("the size of the command vector should "
                               "be a multiple of 2, got {} element(s)", a_programs.size());
            }
            m_available_commands = a_programs;
            parseCommand();
            return command;
        }

        const std::string& setCommand(std::vector<std::string>&& a_programs) {
            if (a_programs.size() % 2) {
                NOA_CORE_ERROR("the size of the command vector should "
                               "be a multiple of 2, got {} element(s)", a_programs.size());
            }
            m_available_commands = std::move(a_programs);
            parseCommand();
            return command;
        }

        void printCommand() const {
            if (m_available_commands.empty()) {
                NOA_CORE_ERROR("the available commands are not set. "
                               "Set them with InputManager::setCommand");
            }
            fmt::print(m_usage_header);
            fmt::print("Commands:\n");
            for (size_t i{0}; i < m_available_commands.size(); i += 2) {
                fmt::print("     {:<{}} {}\n",
                           m_available_commands[i], 15, m_available_commands[i + 1]);
            }
            fmt::print(m_usage_footer);
        }

        static inline std::string getVersion() {
            return fmt::format(FMT_COMPILE("{}.{}.{} - compiled on {}\n"),
                               NOA_VERSION_MAJOR,
                               NOA_VERSION_MINOR,
                               NOA_VERSION_PATCH,
                               __DATE__);
        }

        static inline void printVersion() {
            fmt::print(getVersion());
        }

        void setOption(const std::vector<std::string>& a_option) {
            if (command.empty()) {
                NOA_CORE_ERROR("the command is not set. "
                               "Set it first with InputManager::setCommand");
            } else if (a_option.size() % 5) {
                NOA_CORE_ERROR("the size of the option vector should be a "
                               "multiple of 5, got {} element(s)", a_option.size());
            }
            m_available_options = a_option;
        }

        void setOption(std::vector<std::string>&& a_option) {
            if (command.empty()) {
                NOA_CORE_ERROR("the command is not set. "
                               "Set it first with InputManager::setCommand");
            } else if (a_option.size() % 5) {
                NOA_CORE_ERROR("the size of the option vector should be a "
                               "multiple of 5, got {} element(s)", a_option.size());
            }
            m_available_options = std::move(a_option);
        }

        void printOption() const {
            if (m_available_options.empty()) {
                NOA_CORE_ERROR("the options are not set. "
                               "Set them first with InputManager::setOption");
            }
            fmt::print(m_usage_header);
            fmt::print("{} options:\n", command);

            // Get the first necessary padding.
            size_t option_names_padding{0};
            for (unsigned int i = 0; i < m_available_options.size(); i += 5) {
                size_t current_size = (m_available_options[i + Usage::u_long_name].size() +
                                       m_available_options[i + Usage::u_short_name].size());
                if (current_size > option_names_padding)
                    option_names_padding = current_size;
            }
            option_names_padding += 10;

            std::string type;
            for (unsigned int i = 0; i < m_available_options.size(); i += 5) {
                std::string option_names = fmt::format(
                        "   --{}, -{}",
                        m_available_options[i + Usage::u_long_name],
                        m_available_options[i + Usage::u_short_name]
                );
                if (m_available_options[i + 3].empty())
                    type = fmt::format("({})", formatType(m_available_options[i + Usage::u_type]));
                else
                    type = fmt::format("({} = {})",
                                       formatType(m_available_options[i + Usage::u_type]),
                                       m_available_options[i + Usage::u_default_value]);

                fmt::print("{:<{}} {:<{}} {}\n",
                           option_names, option_names_padding,
                           type, 25, m_available_options[i + Usage::u_help]);
            }
            fmt::print(m_usage_footer);
        }

        [[nodiscard]] bool parse() {
            if (m_available_options.empty()) {
                NOA_CORE_ERROR("the options are not set. "
                               "Set them first with InputManager::setOption");
            }
            bool asked_for_help = parseCommandLine();
            if (asked_for_help)
                return asked_for_help;
            parseParameterFile();
            is_parsed = true;
            return asked_for_help; // false
        }

        template<typename T, int N = 1>
        auto get(const std::string& a_long_name) {
            NOA_CORE_DEBUG(__PRETTY_FUNCTION__);
            static_assert(Traits::is_sequence_v<T> || (Traits::is_int_v<T> && N == 1));

            if (!is_parsed) {
                NOA_CORE_ERROR("the inputs are not parsed yet. Parse them "
                               "by calling InputManager::parse()");
            }

            // Get usage and the value(s).
            auto[usage_short, usage_type, usage_value] = getOption(a_long_name);
            assertType<T, N>(usage_type);
            std::vector<std::string>* value = getParsedValue(a_long_name, usage_short);

            // Parse the default value.
            std::vector<std::string> default_value = String::parse(usage_value);
            if (N != -1 && default_value.size() != N) {
                NOA_CORE_ERROR("Number of default value(s) ({}) doesn't match "
                               "the desired number of value(s) ({})",
                               default_value.size(), N);
            }

            // If option not registered or left empty, replace with the default.
            if (!value || value->empty()) {
                if (usage_value.empty()) {
                    NOA_CORE_ERROR("No value available for option {} ({})",
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
                    NOA_CORE_ERROR("{} ({}): only 1 value is expected, got {}",
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
                    NOA_CORE_ERROR("{} ({}): {} values are expected, got {}",
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
                        NOA_CORE_ERROR("usage type ({}) not recognized", a_type);
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
                    NOA_CORE_ERROR("usage type ({}) not recognized", a_type);
                }
            }
        }

        template<typename T, int N>
        static void assertType(const std::string& a_usage_type) {
            static_assert(N != 0);

            // Number of values.
            if constexpr(N == -1) {
                if (a_usage_type[0] != 'A') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "number of values (array)",
                                   a_usage_type);
                }
            } else if constexpr(N == 1) {
                if (a_usage_type[0] != 'S') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "number of value (1)",
                                   a_usage_type);
                }
            } else if constexpr(N == 2) {
                if (a_usage_type[0] != 'P') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "number of values (2)",
                                   a_usage_type);
                }
            } else if constexpr(N == 3) {
                if (a_usage_type[0] != 'T') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "number of values (3)",
                                   a_usage_type);
                }
            }

            // Types.
            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
                if (a_usage_type[1] != 'F') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "type (floating point)",
                                   a_usage_type);
                }
            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
                if (a_usage_type[1] != 'I') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "type (integer)",
                                   a_usage_type);
                }
            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
                if (a_usage_type[1] != 'B') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "type (boolean)",
                                   a_usage_type);
                }
            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
                if (a_usage_type[1] != 'S') {
                    NOA_CORE_ERROR("the usage type ({}) does not correspond to the expected "
                                   "type (string)",
                                   a_usage_type);
                }
            }
        }

        void parseCommand() {
            if (m_argc < 2)
                command = "--help";
            else if (std::find(m_available_commands.begin(),
                               m_available_commands.end(),
                               m_argv[1]) == m_available_commands.end()) {
                const std::string_view argv1 = m_argv[1];
                if (argv1 == "-h" || argv1 == "--help" || argv1 == "help" ||
                    argv1 == "h" || argv1 == "-help" || argv1 == "--h")
                    command = "--help";
                else if (argv1 == "-v" || argv1 == "--version" || argv1 == "version" ||
                         argv1 == "v" || argv1 == "-version" || argv1 == "--v")
                    command = "--version";
                else {
                    NOA_CORE_ERROR("\"{}\" is not registered as an available command. "
                                   "Add it with InputManager::setCommand", argv1);
                }
            } else command = m_argv[1];
        }

        bool parseCommandLine() {


            std::string_view tmp_string;
            const char* tmp_option = nullptr;
            for (int i = 0; i < m_argc - 2; ++i) {  // exclude executable and program name
                tmp_string = m_argv[i + 2];

                // is it --help? if so, no need to continue the parsing
                if (tmp_string == "--help" || tmp_string == "-help" || tmp_string == "help" ||
                    tmp_string == "-h" || tmp_string == "h") {
                    return true;
                }

                // check that it is not a single - or --. If so, ignore it.
                if (tmp_string == "--" || tmp_string == "-")
                    continue;

                if (tmp_string.size() > 2 &&
                    tmp_string.rfind("--", 1) == 0) {
                    // Option - long-name
                    tmp_option = m_argv[i + 2] + 2; // remove the --
                    if (m_options_cmdline.count(tmp_option)) {
                        NOA_CORE_ERROR("option \"{}\" is specified twice", tmp_option);
                    }
                    m_options_cmdline[tmp_option];
                    continue;
                } else if (tmp_string.size() > 1 &&
                           tmp_string[0] == '-' &&
                           !std::isdigit(tmp_string[1])) {
                    // Option - short-name
                    tmp_option = m_argv[i + 2] + 1; // remove the --
                    if (m_options_cmdline.count(tmp_option)) {
                        NOA_CORE_ERROR("option \"{}\" is specified twice", tmp_option);
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
                    return true;
                }

                // Parse the value.
                String::parse(tmp_string, m_options_cmdline.at(tmp_option));
            }
            return false;
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
                    for (size_t i = 0; i < line.size(); ++i) {
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
                NOA_CORE_ERROR("\"{}\" does not exist or you don't have the permission to read it",
                               parameter_file);
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
                    NOA_CORE_ERROR("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                                   "same option, thus cannot be both specified in the command line",
                                   a_longname, a_shortname);
                }
                return &m_options_cmdline.at(a_longname);

            } else if (m_options_cmdline.count(a_shortname)) {
                return &m_options_cmdline.at(a_shortname);

            } else if (m_options_parameter_file.count(a_longname)) {
                if (m_options_parameter_file.count(a_shortname)) {
                    NOA_CORE_ERROR("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                                   "same option, thus cannot be both specified in the parameter file",
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
        getOption(const std::string& a_longname) const {
            for (size_t i{0}; i < m_available_options.size(); i += 5) {
                if (m_available_options[i] == a_longname)
                    return {m_available_options[i + Usage::u_short_name],
                            m_available_options[i + Usage::u_type],
                            m_available_options[i + Usage::u_default_value]};
            }
            NOA_CORE_ERROR("the \"{}\" option is not known. Did you give the longname?",
                           a_longname);
        }
    };
}
