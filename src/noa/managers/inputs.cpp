/**
 * @file inputs.cpp
 * @brief  Input manager
 * @author Thomas - ffyr2w
 * @date 02 Sep 2020
 */


#include "noa/managers/inputs.h"


namespace Noa {

    void InputManager::printCommand() const {
        if (m_available_commands.empty()) {
            NOA_CORE_ERROR("the available commands are not set. Set them with"
                           "::Noa::InputManager::setCommand");
        }
        fmt::print(m_usage_header);
        fmt::print("Commands:\n");
        for (size_t i{0}; i < m_available_commands.size(); i += 2) {
            fmt::print("     {:<{}} {}\n",
                       m_available_commands[i], 15, m_available_commands[i + 1]);
        }
        fmt::print(m_usage_footer);
    }

    void InputManager::printOption() const {
        if (m_available_options.empty()) {
            NOA_CORE_ERROR("the options are not set. "
                          "Set them first with ::Noa::InputManager::setOption");
        }
        fmt::print(m_usage_header);
        fmt::print("{} options:\n", m_command);

        // Get the first necessary padding.
        size_t option_names_padding{0};
        for (unsigned int i = 0; i < m_available_options.size(); i += 5) {
            size_t current_size = (m_available_options[i + m_option_usage::e_long_name].size() +
                                   m_available_options[i + m_option_usage::e_short_name].size());
            if (current_size > option_names_padding)
                option_names_padding = current_size;
        }
        option_names_padding += 10;

        std::string type;
        for (unsigned int i = 0; i < m_available_options.size(); i += 5) {
            std::string option_names = fmt::format(
                    "   --{}, -{}",
                    m_available_options[i + m_option_usage::e_long_name],
                    m_available_options[i + m_option_usage::e_short_name]
            );
            if (m_available_options[i + m_option_usage::e_default_value].empty())
                type = fmt::format("({})",
                                   formatType(m_available_options[i + m_option_usage::e_type]));
            else
                type = fmt::format("({} = {})",
                                   formatType(m_available_options[i + m_option_usage::e_type]),
                                   m_available_options[i + m_option_usage::e_default_value]);

            fmt::print("{:<{}} {:<{}} {}\n",
                       option_names, option_names_padding,
                       type, 25, m_available_options[i + m_option_usage::e_help]);
        }
        fmt::print(m_usage_footer);
    }

    [[nodiscard]] bool InputManager::parse() {
        if (m_available_options.empty()) {
            NOA_CORE_ERROR("the options are not set. "
                          "Set them first with InputManager::setOption");
        }
        if (parseCommandLine())
            return true;
        parseParameterFile();
        m_is_parsed = true;
        return false;
    }

    std::string InputManager::formatType(const std::string& usage_type) {
        if (usage_type.size() != 2) {
            NOA_CORE_ERROR("usage type ({}) not recognized. It should be a string with 2 characters",
                           usage_type);
        }

        const char* type_name;
        switch (usage_type[1]) {
            case 'I':
                type_name = "integer";
                break;
            case 'F':
                type_name = "float";
                break;
            case 'S':
                type_name = "string";
                break;
            case 'B':
                type_name = "bool";
                break;
            default: {
                NOA_CORE_ERROR("usage type ({}) not recognized. The second character should be "
                              "I, F, S or B (all in upper case)", usage_type);
            }
        }

        switch (usage_type[0]) {
            case 'S':
                return fmt::format("1 {}", type_name);
            case 'P':
                return fmt::format("2 {}s", type_name);
            case 'T':
                return fmt::format("3 {}s", type_name);
            case 'A':
                return fmt::format("n {}(s)", type_name);
            default: {
                NOA_CORE_ERROR("type usage ({}) not recognized", usage_type);
            }
        }
    }

    void InputManager::parseCommand() {
        if (m_argc < 2)
            m_command = "--help";
        else if (std::find(m_available_commands.begin(),
                           m_available_commands.end(),
                           m_argv[1]) == m_available_commands.end()) {
            const std::string_view argv1 = m_argv[1];
            if (argv1 == "-h" || argv1 == "--help" || argv1 == "help" ||
                argv1 == "h" || argv1 == "-help" || argv1 == "--h")
                m_command = "--help";
            else if (argv1 == "-v" || argv1 == "--version" || argv1 == "version" ||
                     argv1 == "v" || argv1 == "-version" || argv1 == "--v")
                m_command = "--version";
            else {
                NOA_CORE_ERROR("\"{}\" is not a registered command. "
                              "Add it with ::Noa::InputManager::setCommand", argv1);
            }
        } else m_command = m_argv[1];
    }


    bool InputManager::parseCommandLine() {
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
                m_parameter_file = tmp_string;
                continue;
            } else if (!tmp_option && i == 1) {
                return true;
            }

            // Parse the value.
            String::parse(tmp_string, m_options_cmdline.at(tmp_option));
        }
        return false;
    }


    void InputManager::parseParameterFile() {
        if (m_parameter_file.empty())
            return;

        std::ifstream file(m_parameter_file);
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
                           m_parameter_file);
        }
    }


    std::vector<std::string>* InputManager::getParsedValue(const std::string& long_name,
                                                           const std::string& short_name) {
        if (m_options_cmdline.count(long_name)) {
            if (m_options_cmdline.count(short_name)) {
                NOA_CORE_ERROR("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                              "same option, thus cannot be both specified in the command line",
                               long_name, short_name);
            }
            return &m_options_cmdline.at(long_name);

        } else if (m_options_cmdline.count(short_name)) {
            return &m_options_cmdline.at(short_name);

        } else if (m_options_parameter_file.count(long_name)) {
            if (m_options_parameter_file.count(short_name)) {
                NOA_CORE_ERROR("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                              "same option, thus cannot be both specified in the parameter file",
                               long_name, short_name);
            }
            return &m_options_parameter_file.at(long_name);

        } else if (m_options_parameter_file.count(short_name)) {
            return &m_options_parameter_file.at(short_name);

        } else {
            return nullptr;
        }
    }


    std::tuple<const std::string&, const std::string&, const std::string&>
    InputManager::getOption(const std::string& a_longname) const {
        for (size_t i{0}; i < m_available_options.size(); i += 5) {
            if (m_available_options[i] == a_longname)
                return {m_available_options[i + m_option_usage::e_short_name],
                        m_available_options[i + m_option_usage::e_type],
                        m_available_options[i + m_option_usage::e_default_value]};
        }
        NOA_CORE_ERROR("the \"{}\" option is not known. Did you give the longname?", a_longname);
    }
}
