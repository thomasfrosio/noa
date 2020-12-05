#include "InputManager.h"


void Noa::InputManager::printCommand() const {
    auto it = m_registered_commands.cbegin(), end = m_registered_commands.cend();
    if (it == end) {
        NOA_LOG_ERROR("no command has been registered. Register commands with setCommand()");
    }
    fmt::print("{}\n\nCommands:\n", m_usage_header);
    for (; it < end; it += 2)
        fmt::print("     {:<{}} {}\n", *it, 15, *(it + 1));
    fmt::print(m_usage_footer);
}


void Noa::InputManager::printOption() const {
    auto it = m_registered_options.cbegin(), end = m_registered_options.cend();
    if (it == end) {
        NOA_LOG_ERROR("no option has been registered. Register options with setOption()");
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
            type = formatType_(*(it + OptionUsage::type));
        else
            type = fmt::format("{} = {}",
                               formatType_(*(it + OptionUsage::type)),
                               *(it + OptionUsage::default_value));

        fmt::print("{:<{}} ({:<{}}) {}\n",
                   option_names, option_names_padding,
                   type, 25,
                   *(it + OptionUsage::docstring));
    }
    fmt::print(m_usage_footer);
}


[[nodiscard]] bool Noa::InputManager::parse() {
    if (m_registered_options.empty()) {
        NOA_LOG_ERROR("no option has been registered. Register options with setOption()");
    }
    parseCommandLine_();
    if (!m_parsing_is_complete)
        return m_parsing_is_complete;
    parseParameterFile_();

    return m_parsing_is_complete;
}


std::string Noa::InputManager::formatType_(const std::string& usage_type) {
    if (usage_type.size() != 2) {
        NOA_LOG_ERROR("usage type \"{}\" is not recognized. It should be a 2 characters string",
                      usage_type);
    }

    const char* type_name;
    switch (usage_type[1]) {
        case 'I':
            type_name = "integer";
            break;
        case 'U':
            type_name = "unsigned integer";
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
            NOA_LOG_ERROR("usage type \"{}\" is not recognized. The second character should be "
                          "I, U, F, S or B (in upper case)", usage_type);
        }
    }

    if (usage_type[0] == '0')
        return fmt::format("n {}(s)", type_name);
    else if (usage_type[0] > 48 && usage_type[0] < 58)
        return fmt::format("{} {}", usage_type[0], type_name);
    else {
        NOA_LOG_ERROR("usage type \"{}\" is not recognized. The first character should be "
                      "a number from 0 to 9", usage_type);
    }
}


void Noa::InputManager::parseCommandLine_() {
    std::string opt, value;

    auto add_pair = [this, &opt, &value]() {
        if (value.empty()) {
            if (opt.empty())
                return;
            else
                NOA_LOG_ERROR_FUNC("parseCommandLine_", "the option \"{}\" is missing a value",
                                   opt);
        }
        if (!isOption_(opt))
            NOA_LOG_ERROR_FUNC("parseCommandLine_", "the option \"{}\" is not known.", opt);

        auto[p, ok] = m_options_cmdline.emplace(std::move(opt), std::move(value));
        if (!ok) {
            NOA_LOG_ERROR_FUNC("parseCommandLine_",
                               "the option \"{}\" is entered twice in the command line",
                               p->first);
        }
    };

    for (size_t i{2}; i < m_cmdline.size(); ++i) /* exclude executable and command */ {
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

        if (str.rfind("--", 0) == 0) /* long_name */ {
            add_pair();
            opt = m_cmdline[i].data() + 2; // remove the --|-
            value.clear();
        } else if (str[0] == '-' && !std::isdigit(str[1])) /* short_name */ {
            add_pair();
            opt = m_cmdline[i].data() + 1; // remove the --|-
            value.clear();
        } else /* value or parameter file*/ {
            if (i == 2) {
                m_parameter_filename = std::move(str);
            } else if (opt.empty()) {
                NOA_LOG_ERROR("the value \"{}\" isn't assigned to any option", str);
            } else if (value.empty()) {
                value = std::move(str);
            } else if (value[value.size() - 1] == ',' || str[0] == ',') {
                value += str;
            } else {
                value += str.insert(0, 1, ',');
            }
        }
    }
    add_pair();
    m_parsing_is_complete = true;
}


void Noa::InputManager::parseParameterFile_() {
    if (m_parameter_filename.empty())
        return;

    TextFile param_file(m_parameter_filename, std::ios::in);
    std::string line;
    while (param_file.getLine(line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;

        // If it doesn't start with the prefix, skip this line.
        if (line.rfind(m_prefix, idx_inc) != idx_inc)
            continue;

        // Get idx range of the right side of the equal sign.
        size_t idx_start = idx_inc + m_prefix.size();
        size_t idx_end = line.find('#', idx_start);
        size_t idx_equal = line.find('=', idx_start);
        if (idx_equal == std::string::npos || idx_equal + 1 >= idx_end ||
            idx_start == idx_equal || std::isspace(line[idx_start]))
            continue;

        // Make sure the value to be parsed isn't only whitespaces.
        std::string_view value{line.data() + idx_equal + 1,
                               (idx_end == std::string::npos) ?
                               line.size() - idx_equal - 1 : idx_end - idx_equal - 1};
        if (value.find_first_not_of(" \t") == std::string::npos)
            continue;

        // Get the [key, value].
        if (!m_options_parameter_file.emplace(
                String::rightTrim(line.substr(idx_start, idx_equal - idx_start)), value).second) {
            NOA_LOG_ERROR("option \"{}\" is specified twice in the parameter file",
                          String::rightTrim(line.substr(idx_start, idx_equal - idx_start)));
        }
    }
    if (param_file.bad()) {
        NOA_LOG_ERROR("\"{}\": error while reading file. {}",
                      m_parameter_filename, std::strerror(errno));
    }
}


std::string* Noa::InputManager::getParsedValue_(const std::string& long_name,
                                                const std::string& short_name) {
    if (m_options_cmdline.count(long_name)) {
        if (m_options_cmdline.count(short_name)) {
            NOA_LOG_ERROR("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                          "same option, thus cannot be both specified in the command line",
                          long_name, short_name);
        }
        return &m_options_cmdline.at(long_name);

    } else if (m_options_cmdline.count(short_name)) {
        return &m_options_cmdline.at(short_name);

    } else if (m_options_parameter_file.count(long_name)) {
        if (m_options_parameter_file.count(short_name)) {
            NOA_LOG_ERROR("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
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
Noa::InputManager::getOption_(const std::string& long_name) const {
    for (size_t i{0}; i < m_registered_options.size(); i += 5) {
        if (m_registered_options[i] == long_name)
            return {&m_registered_options[i + OptionUsage::short_name],
                    &m_registered_options[i + OptionUsage::type],
                    &m_registered_options[i + OptionUsage::default_value]};
    }
    NOA_LOG_ERROR("the \"{}\" option is not registered. Did you give the longname?", long_name);
}


bool Noa::InputManager::isOption_(const std::string& name) const {
    for (size_t i{0}; i < m_registered_options.size(); i += 5) {
        if (m_registered_options[i] == name || m_registered_options[i + 1] == name)
            return true;
    }
    return false;
}


std::string Noa::InputManager::getErrorMessage_(const std::string& l_name,
                                                const std::string* value,
                                                size_t nb,
                                                uint8_t err) const {
    auto[u_short_name, u_type, u_value] = getOption_(l_name);

    if (err == Errno::invalid_argument) {
        if (value->empty())
            return fmt::format("{} ({}) is missing. It should be {}.",
                               l_name, *u_short_name, formatType_(*u_type));
        else if (u_value->empty())
            return fmt::format("{} ({}) contains at least one element that could not "
                               "be converted into the desired type (i.e. {}): \"{}\"",
                               l_name, *u_short_name, formatType_(*u_type), *value);
        else
            return fmt::format("{} ({}) contains at least one element that could not "
                               "be converted into the desired type (i.e. {}): \"{}\", "
                               "with default: \"{}\"",
                               l_name, *u_short_name, formatType_(*u_type), *value, *u_value);
    } else if (err == Errno::out_of_range) {
        return fmt::format("{} ({}) contains at least one element that was out of "
                           "the desired type (i.e. {}) range: \"{}\"",
                           l_name, *u_short_name, formatType_(*u_type), *value);
    } else if (err == Errno::invalid_size) {
        return fmt::format("{} ({}) does not have the expected number of elements: "
                           "{} expected, got {}", l_name, *u_short_name, nb, *value);
    } else {
        return fmt::format("unknown error or reason - please let us know "
                           "that this happened. name: {}, value: {}", l_name, *value);
    }
}
