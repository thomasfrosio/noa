/**
 * @file inputs.cpp
 * @brief  Input manager
 * @author Thomas - ffyr2w
 * @date 02 Sep 2020
 */


#include "noa/managers/Inputs.h"


namespace Noa {

    void InputManager::printCommand() const {
        auto it = m_registered_commands.cbegin(), end = m_registered_commands.cend();
        if (it == end) {
            NOA_CORE_ERROR("the available commands are not set. Set them with"
                           "::Noa::InputManager::setCommand");
        }
        fmt::print("{}\n\nCommands:\n", m_usage_header);
        for (; it < end; it += 2)
            fmt::print("     {:<{}} {}\n", *it, 15, *(it + 1));
        fmt::print(m_usage_footer);
    }


    void InputManager::printOption() const {
        auto it = m_registered_options.cbegin(), end = m_registered_options.cend();
        if (it == end) {
            NOA_CORE_ERROR("the options are not set. "
                           "Set them first with ::Noa::InputManager::setOption");
        }

        // Get the first necessary padding.
        size_t option_names_padding{0};
        for (; it < end; it += 5) {
            size_t current_size = ((it + OptionUsage::long_name)->size() +
                                   (it + OptionUsage::short_name)->size());
            if (current_size > option_names_padding)
                option_names_padding = current_size;
        }
        option_names_padding += 10;

        fmt::print("{}\n\n\"{}\" options:\n", m_usage_header, m_command);

        std::string type;
        for (it = m_registered_options.cbegin(); it < end; it += 5) {
            std::string option_names = fmt::format("   --{}, -{}",
                                                   *(it + OptionUsage::long_name),
                                                   *(it + OptionUsage::short_name)
            );
            if ((it + OptionUsage::default_value)->empty())
                type = formatType(*(it + OptionUsage::type));
            else
                type = fmt::format("{} = {}",
                                   formatType(*(it + OptionUsage::type)),
                                   *(it + OptionUsage::default_value));

            fmt::print("{:<{}} ({:<{}}) {}\n",
                       option_names, option_names_padding,
                       type, 25,
                       *(it + OptionUsage::help));
        }
        fmt::print(m_usage_footer);
    }


    [[nodiscard]] bool InputManager::parse() {
        if (m_registered_options.empty()) {
            NOA_CORE_ERROR("the options are not set. "
                           "Set them first with ::Noa::InputManager::setOption");
        }
        parseCommandLine();
        if (!m_parsing_is_complete)
            return m_parsing_is_complete;
        parseParameterFile();

        return m_parsing_is_complete;
    }

    std::string InputManager::formatType(const std::string& usage_type) {
        if (usage_type.size() != 2) {
            NOA_CORE_ERROR(
                    "usage type ({}) not recognized. It should be a string with 2 characters",
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
                               "I, F, S or B (in upper case)", usage_type);
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
                NOA_CORE_ERROR("usage type ({}) not recognized. The first character should be "
                               "S, P, T or A (in upper case)", usage_type);
            }
        }
    }

    void InputManager::parseCommand() {
        if (m_cmdline.size() < 2)
            m_command = "help";
        else if (std::find(m_registered_commands.begin(), m_registered_commands.end(), m_cmdline[1])
                 == m_registered_commands.end()) {
            const std::string& argv1 = m_cmdline[1];
            if (argv1 == "-h" || argv1 == "--help" || argv1 == "help" ||
                argv1 == "h" || argv1 == "-help" || argv1 == "--h")
                m_command = "help";
            else if (argv1 == "-v" || argv1 == "--version" || argv1 == "version" ||
                     argv1 == "v" || argv1 == "-version" || argv1 == "--v")
                m_command = "version";
            else {
                NOA_CORE_ERROR("\"{}\" is not a registered command. "
                               "Add it with ::Noa::InputManager::setCommand", argv1);
            }
        } else {
            m_command = m_cmdline[1];
        }
    }


    void InputManager::parseCommandLine() {
        std::string opt, value;
        for (size_t i{2}; i < m_cmdline.size(); ++i) /* exclude executable and cmd */ {
            std::string& str = m_cmdline[i];

            // check that it is not a single - or --. If so, ignore it.
            if (str == "--" || str == "-")
                continue;

            // is it --help? if so, no need to continue the parsing
            if (str == "--help" || str == "-h" || str == "-help" ||
                str == "help" || str == "--h" || str == "h") {
                m_parsing_is_complete = false;
                return;
            }

            // if it is an option
            if (str.rfind("--", 0) == 0) /* long_name */ {
                if (!opt.empty()) {
                    auto[p, ok] = m_options_cmdline.emplace(std::move(opt), String::parse(value));
                    if (!ok) {
                        NOA_CORE_ERROR("\"{}\" is entered twice in the command line", p->first);
                    }
                }
                opt = m_cmdline[i].data() + 2; // remove the --
                value.clear();
                continue;
            } else if (str[0] == '-' && !std::isdigit(str[1])) /* short_name */ {
                if (!opt.empty()) {
                    auto[p, ok] = m_options_cmdline.emplace(std::move(opt), String::parse(value));
                    if (!ok) {
                        NOA_CORE_ERROR("\"{}\" is entered twice in the command line", p->first);
                    }
                }
                opt = m_cmdline[i].data() + 1; // remove the -
                value.clear();
                continue;
            }

            // at this point str is either a value or a parameter file
            if (!opt.empty()) {
                if (value.empty())
                    value = std::move(str);
                else if (value[value.size() - 1] == ',' || str[0] == ',')
                    value += str;
                else {
                    value += str.insert(0, 1, ',');
                }
            } else {
                if (i == 2)
                    m_parameter_filename = &str;
                else {
                    NOA_CORE_ERROR("only one parameter file is allowed");
                }
            }
        }
        if (!opt.empty()) {
            auto[p, ok] = m_options_cmdline.emplace(std::move(opt), String::parse(value));
            if (!ok) {
                NOA_CORE_ERROR("\"{}\" is entered twice in the command line", p->first);
            }
        }
        m_parsing_is_complete = true;
    }


    void InputManager::parseParameterFile() {
        if (!m_parameter_filename)
            return;

        std::ifstream file(*m_parameter_filename);
        if (!file.is_open()) {
            NOA_CORE_ERROR("error while opening the parameter file \"{}\": {}",
                           *m_parameter_filename, std::strerror(errno));
        }

        std::string line;
        size_t prefix_size = m_prefix.size();
        while (std::getline(file, line)) {
            if (line.size() <= prefix_size)
                continue;

            size_t idx_inc = String::firstNonSpace(line);
            if (idx_inc == std::string::npos)
                continue;

            // If it doesn't start with the prefix, skip this line.
            if (line.rfind(m_prefix, idx_inc) != idx_inc)
                continue;

            // Get idx range of the right side of the equal sign.
            size_t idx_start = idx_inc + prefix_size;
            size_t idx_end = line.find('#', idx_start);
            size_t idx_equal = line.find('=', idx_start);
            if (idx_equal == std::string::npos || idx_equal + 1 >= idx_end ||
                idx_start == idx_equal || std::isspace(line[idx_start]))
                continue;

            // Make sure the value to be parsed isn't only whitespaces.
            std::string_view value{line.data() + idx_equal + 1, idx_end - idx_equal + 1};
            if (String::firstNonSpace(value) == std::string::npos)
                continue;

            // Get the [key, value].
            if (!m_options_parameter_file.emplace(
                    String::rightTrim(line.substr(idx_start, idx_equal - 1)),
                    String::parse(value)).second) {
                NOA_CORE_ERROR("option \"{}\" is specified twice in the parameter file",
                               String::rightTrim(line.substr(idx_start, idx_equal - 1)));
            }
        }
        if (file.bad()) {
            NOA_CORE_ERROR("error while reading the parameter file \"{}\": {}",
                           *m_parameter_filename, std::strerror(errno));
        }
        file.close();
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


    std::tuple<const std::string*, const std::string*, const std::string*>
    InputManager::getOption(const std::string& a_longname) const {
        for (size_t i{0}; i < m_registered_options.size(); i += 5) {
            if (m_registered_options[i] == a_longname)
                return {&m_registered_options[i + OptionUsage::short_name],
                        &m_registered_options[i + OptionUsage::type],
                        &m_registered_options[i + OptionUsage::default_value]};
        }
        NOA_CORE_ERROR("the \"{}\" option is not known. Did you give the longname?", a_longname);
    }
}
