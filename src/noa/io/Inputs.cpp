#include "noa/io/Inputs.h"
#include "noa/io/files/TextFile.h"

using namespace ::Noa;

void Inputs::printCommand() const {
    auto it = m_registered_commands.cbegin(), end = m_registered_commands.cend();
    if (it == end) {
        NOA_THROW("DEV: commands haven't been registered. Register commands with setCommand()");
    }
    fmt::print("{}\n\nCommands:\n", m_usage_header);
    for (; it < end; it += 2)
        fmt::print("     {:<{}} {}\n", *it, 15, *(it + 1));
    fmt::print(m_usage_footer);
}

void Inputs::printOption() const {
    auto it = m_registered_options.cbegin(), end = m_registered_options.cend();
    if (it == end) {
        NOA_THROW("DEV: options haven't been registered. Register options with setOption()");
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

[[nodiscard]] bool Inputs::parse(const std::string& prefix) {
    if (m_registered_options.empty())
        NOA_THROW("DEV: options have not been registered. Register options with setOption()");

    bool was_completed = parseCommandLine();
    if (!was_completed)
        return was_completed;

    if (!m_parameter_filename.empty())
        parseParameterFile(m_parameter_filename, prefix);

    return was_completed;
}

bool Inputs::parseCommandLine() {
    std::string opt, value;

    auto add_input = [this, &opt, &value]() {
        if (value.empty()) {
            if (opt.empty())
                return;
            else
                NOA_THROW_FUNC("parseCommandLine", "the option \"{}\" is missing a value", opt);
        }
        if (!isOption_(opt))
            NOA_THROW_FUNC("parseCommandLine", "the option \"{}\" is not known.", opt);

        auto[pair, is_inserted] = m_inputs_cmdline.emplace(std::move(opt), std::move(value));
        if (!is_inserted) {
            NOA_THROW_FUNC("parseCommandLine", "the option \"{}\" is entered twice in the command line", pair->first);
        }
    };

    std::string entry;
    for (size_t entry_nb{2};
         entry_nb < m_cmdline.size(); ++entry_nb) /* exclude executable and command */ {
        entry = m_cmdline[entry_nb];

        if (entry == "--" || entry == "-")
            continue;
        else if (isHelp_(entry)) {
            return false;
        }

        if (entry.rfind("--", 0) == 0) /* long_name */ {
            add_input();
            opt = m_cmdline[entry_nb].data() + 2; // remove the --
            value.clear();
        } else if (entry[0] == '-' && !std::isdigit(entry[1])) /* short_name */ {
            add_input();
            opt = m_cmdline[entry_nb].data() + 1; // remove the -
            value.clear();
        } else /* value or parameter file*/ {
            if (entry_nb == 2) {
                m_parameter_filename = std::move(entry);
            } else if (opt.empty()) {
                NOA_THROW("the value \"{}\" isn't assigned to any option", entry);
            } else if (value.empty()) {
                value = std::move(entry);
            } else if (value[value.size() - 1] == ',' || entry[0] == ',') {
                value += entry;
            } else {
                value += entry.insert(0, 1, ',');
            }
        }
    }
    add_input();
    return true;
}

void Inputs::parseParameterFile(const std::string& filename, const std::string& prefix) {
    TextFile<std::ifstream> file(filename, IO::READ);

    std::string line;
    while (file.getLine(line)) {
        size_t idx_inc = line.find_first_not_of(" \t");
        if (idx_inc == std::string::npos)
            continue;

        // If it doesn't start with the prefix, skip this line.
        if (line.rfind(prefix, idx_inc) != idx_inc)
            continue;

        // Get idx range of the right side of the equal sign.
        size_t idx_start = idx_inc + prefix.size();
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

        // Get the [option, value].
        std::string option = String::rightTrim(line.substr(idx_start, idx_equal - idx_start));
        if (isOption_(option)) {
            auto[p, is_inserted] = m_inputs_file.emplace(std::move(option), value);
            if (!is_inserted)
                NOA_THROW("option \"{}\" is specified twice in the parameter file", p->first);
        }
    }
    if (file.bad())
        NOA_THROW("File \"{}\": error while reading file. ERRNO: {}", filename, std::strerror(errno));
}

std::string Inputs::formatType_(const std::string& usage_type) {
    if (usage_type.size() != 2)
        NOA_THROW("DEV: usage type \"{}\" invalid. It should be a 2 char string", usage_type);

    std::string type_name;
    if (usage_type[1] == 'I')
        type_name = "integer";
    else if (usage_type[1] == 'U')
        type_name = "unsigned integer";
    else if (usage_type[1] == 'F')
        type_name = "float";
    else if (usage_type[1] == 'S')
        type_name = "string";
    else if (usage_type[1] == 'B')
        type_name = "bool";
    else
        NOA_THROW("DEV: usage type \"{}\" is not recognized. The second character should be "
                  "I, U, F, S or B (in upper case)", usage_type);

    if (usage_type[0] == '0')
        return String::format("n {}(s)", type_name);
    else if (usage_type[0] > 48 && usage_type[0] < 58)
        return String::format("{} {}", usage_type[0], type_name);
    else {
        NOA_THROW("DEV: usage type \"{}\" is not recognized. The first character should be "
                  "a number from 0 to 9", usage_type);
    }
}

std::string* Inputs::getValue_(const std::string& long_name, const std::string& short_name) {
    if (m_inputs_cmdline.count(long_name)) {
        if (m_inputs_cmdline.count(short_name)) {
            NOA_THROW("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                      "same option, thus cannot be both specified in the command line",
                      long_name, short_name);
        }
        return &m_inputs_cmdline.at(long_name);

    } else if (m_inputs_cmdline.count(short_name)) {
        return &m_inputs_cmdline.at(short_name);

    } else if (m_inputs_file.count(long_name)) {
        if (m_inputs_file.count(short_name)) {
            NOA_THROW("\"{}\" (long-name) and \"{}\" (short-name) are linked to the "
                      "same option, thus cannot be both specified in the parameter file",
                      long_name, short_name);
        }
        return &m_inputs_file.at(long_name);

    } else if (m_inputs_file.count(short_name)) {
        return &m_inputs_file.at(short_name);

    } else {
        return nullptr;
    }
}

std::tuple<const std::string*, const std::string*, const std::string*>
Inputs::getOptionUsage_(const std::string& long_name) const {
    for (size_t i{0}; i < m_registered_options.size(); i += 5) {
        if (m_registered_options[i] == long_name)
            return {&m_registered_options[i + OptionUsage::short_name],
                    &m_registered_options[i + OptionUsage::type],
                    &m_registered_options[i + OptionUsage::default_value]};
    }
    NOA_THROW("DEV: the \"{}\" (long-name) option is not registered", long_name);
}

std::string Inputs::getOptionErrorMessage_(const std::string& l_name, const std::string* value,
                                           size_t nb, Errno err) const {
    auto[u_s_name, u_type, u_value] = getOptionUsage_(l_name);

    if (err == Errno::invalid_argument) {
        if (value->empty())
            return fmt::format("{} ({}) is missing. It should be {}.",
                               l_name, *u_s_name, formatType_(*u_type));
        else if (u_value->empty())
            return fmt::format("{} ({}) contains at least one element that could not "
                               "be converted into the desired type (i.e. {}): \"{}\"",
                               l_name, *u_s_name, formatType_(*u_type), *value);
        else
            return fmt::format("{} ({}) contains at least one element that could not "
                               "be converted into the desired type (i.e. {}): \"{}\", "
                               "with default: \"{}\"",
                               l_name, *u_s_name, formatType_(*u_type), *value, *u_value);

    } else if (err == Errno::out_of_range) {
        return fmt::format("{} ({}) contains at least one element that was out of "
                           "the desired type range (i.e. {}): \"{}\"",
                           l_name, *u_s_name, formatType_(*u_type), *value);

    } else if (err == Errno::invalid_size) {
        return fmt::format("{} ({}) does not have the expected number of elements: "
                           "{} expected, got {}", l_name, *u_s_name, nb, *value);
    } else {
        return fmt::format("unknown error or reason - please let us know "
                           "that this happened. name: {}, value: {}", l_name, *value);
    }
}
