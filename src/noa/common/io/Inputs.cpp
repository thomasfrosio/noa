#include "noa/common/Session.h"
#include "noa/common/Inputs.h"
#include "noa/common/files/TextFile.h"

using namespace ::noa;

std::string Inputs::formatCommands() const {
    size_t commands = m_registered_commands.size() / 2;
    if (commands == 0)
        NOA_THROW("DEV: commands haven't been registered. Register commands with setCommands()");

    std::string buffer;
    buffer.reserve(300 + commands * 150); // this should be enough, if not, buffer will reallocate anyway.
    buffer += string::format(m_usage_header, m_cmdline[0]);
    buffer += "\n\nCommands:\n";
    for (size_t idx = 0; idx < commands; idx += 2)
        buffer += string::format("    {:<{}} {}\n", m_registered_commands[idx], 15, m_registered_commands[idx + 1]);
    buffer += m_usage_footer;
    return buffer;
}

std::string Inputs::formatOptions() const {
    size_t options = m_registered_options.size() / USAGE_ELEMENTS_PER_OPTION;
    if (options == 0)
        NOA_THROW("DEV: options haven't been registered. Register options with setOptions()");

    std::string buffer;
    buffer.reserve(300 + options * 100);
    buffer += string::format("Options for the command: {}\n", m_command);

    for (size_t section = 0; section < m_registered_sections.size(); ++section) {
        const std::string& section_name = m_registered_sections[section];
        buffer += string::format("{}\n{}\n", section_name, std::string(section_name.size(), '='));

        const std::string* tmp = m_registered_options.data();
        if (string::toInt<size_t>(tmp[USAGE_SECTION]) != section)
            continue;
        for (size_t idx = 0; idx < options; idx += USAGE_ELEMENTS_PER_OPTION) {
            tmp += idx;
            buffer += string::format("--{}, -{}, {} (default:{})\n    {}", tmp[USAGE_LONG_NAME], tmp[USAGE_SHORT_NAME],
                                     formatType_(tmp[USAGE_TYPE]), tmp[USAGE_DEFAULT_VALUE], tmp[USAGE_DOCSTRING]);
        }
    }
    return buffer;
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
    TextFile<std::ifstream> file(filename, io::READ);

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
        std::string option = string::rightTrim(line.substr(idx_start, idx_equal - idx_start));
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
        return string::format("n {}(s)", type_name);
    else if (usage_type[0] > 48 && usage_type[0] < 58)
        return string::format("{} {}", usage_type[0], type_name);
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
    for (size_t i{0}; i < m_registered_options.size(); i += USAGE_ELEMENTS_PER_OPTION) {
        if (m_registered_options[i] == long_name)
            return {&m_registered_options[i + USAGE_SHORT_NAME],
                    &m_registered_options[i + USAGE_TYPE],
                    &m_registered_options[i + USAGE_DEFAULT_VALUE]};
    }
    NOA_THROW("DEV: the \"{}\" (long-name) option is not registered", long_name);
}

template<typename T, size_t N>
T Inputs::getOption(const std::string& long_name) {
    using value_t = noa::traits::value_type_t<T>;
    static_assert(N >= 0 && N < 10);
    static_assert(!(noa::traits::is_std_complex_v<value_t> || noa::traits::is_complex_v<value_t>));

    auto[short_name, usage_type, default_string] = getOptionUsage_(long_name);
    assertType_<T, N>(*usage_type); // does T and N match the usage type?
    const std::string* user_string = getValue_(long_name, *short_name); // user_string is nullptr or a non-empty string
    if (user_string == nullptr) // option not entered by the user, so rely only on the default string
        user_string = default_string;

    T output{};
    try {
        if constexpr(N == 0) {
            static_assert(noa::traits::is_std_vector_v<T>);
            // When an unknown number of values is expected (N == 0), values cannot be defaulted
            // based on their position. Thus, let parse() try to convert the raw value.
            output = string::parse<value_t>(*user_string);

        } else if constexpr (noa::traits::is_bool_v<T> ||
                             noa::traits::is_string_v<T> ||
                             noa::traits::is_scalar_v<T>) {
            // If the option is not specified by the user, getValue_ returns a nullptr. Therefore, at this point,
            // user_string is either non-empty and comes from the user, or is the default string (which can be empty).
            // Either way, let parse() try to convert the raw value.
            static_assert(N == 1);
            output = string::parse<T, 1>(*user_string)[0]; // the '1' forces user_string to contain only one field.

        } else if constexpr (noa::traits::is_intX_v<T> || noa::traits::is_floatX_v<T>) {
            // N should be 2, 3 or 4. If the option was not specified by the user and/or if there's no default value,
            // parse user_string, otherwise parse user_string with default_string as backup.
            static_assert(T::size() == N);
            std::array<value_t, N> tmp = default_string == user_string || default_string->empty() ?
                                         string::parse<value_t, N>(*user_string) :
                                         string::parse<value_t, N>(*user_string, *default_string);
            output = tmp.data();

        } else if constexpr (noa::traits::is_std_array_v<T>) {
            // Same as above, but N should just match the std::array size.
            static_assert(std::tuple_size_v<T> == N);
            output = default_string == user_string || default_string->empty() ?
                     string::parse<value_t, N>(*user_string) :
                     string::parse<value_t, N>(*user_string, *default_string);

        } else if constexpr (noa::traits::is_std_vector_v<T>) {
            // Same as above. N should be from 1 to 9. Add explicit size check since parse does not verify in this case.
            output = default_string == user_string || default_string->empty() ?
                     string::parse<value_t>(*user_string) :
                     string::parse<value_t>(*user_string, *default_string);
            if (output.size() != N)
                NOA_THROW("The number of parsed value(s) ({}) does not match the number of "
                          "expected value(s) ({})", output.size(), N);
        } else {
            static_assert(noa::traits::always_false_v<T>);
        }

        // Extra check for strings: empty fields are not allowed.
        if constexpr (noa::traits::is_string_v<T>) {
            if (output.empty())
                NOA_THROW("Empty field detected");
        } else if constexpr (noa::traits::is_std_sequence_string_v<T>) {
            for (size_t idx = 0; idx < output.size(); ++idx)
                if (output[idx].empty())
                    NOA_THROW("The parsed string contains an empty field at index {} (starting from 0)", idx);
        }
    } catch (...) {
        if constexpr (N == 1 || N == 0) {
            // user_string comes from the user (and is not empty) OR it is the default_string (which can be empty).
            if (default_string == user_string) { // option was not entered...
                if (default_string->empty()) // ...and it is not optional
                    NOA_THROW("{} ({}): the option is not specified and is not optional. Should be {}",
                              long_name, *short_name, formatType_(*usage_type));
                else // ...and the default_string could not be parsed
                    NOA_THROW("DEV: {} ({}): the default string is not valid. Got {}",
                              long_name, *short_name, *default_string);
            } else { // option was entered...
                NOA_THROW("{} ({}): the option could not be converted into the desired type ({}). Got: {}",
                          long_name, *short_name, formatType_(*usage_type), *user_string);
            }
        } else {
            if (default_string == user_string) { // option was not entered...
                if (default_string->empty()) // ...and it is not optional
                    NOA_THROW("{} ({}): the option is not specified and is not optional. Should be {}",
                              long_name, *short_name, formatType_(*usage_type));
                else // ...and at least one field is not optional
                    NOA_THROW("{} ({}): the option is not specified and has at least one field that is not optional. "
                              "Should be {}. Default:{}",
                              long_name, *short_name, formatType_(*usage_type), *default_string);
            } else { // option was entered...
                if (default_string->empty()) // ...but with values that could not be parsed
                    NOA_THROW("{} ({}): at least one element could not be converted into the desired type ({}). "
                              "Got: {}", long_name, *short_name, formatType_(*usage_type), *user_string);
                else
                    // ...but one user field could not be converted or was empty and the corresponding default
                    // field was empty or (DEV) was not valid.
                    NOA_THROW("{} ({}): at least one element could not be converted into the desired type ({}). "
                              "Got: {}. Default: {}",
                              long_name, *short_name, formatType_(*usage_type), *user_string, *default_string);
            }
        }
    }

    Session::logger.trace("{} ({}): {}", long_name, *short_name, output);
    return output;
}

#define INSTANTIATE_GET_OPTION_SEQUENCE(T)                                              \
template T Inputs::getOption<T, 1>(const std::string&);                                 \
template std::vector<T> Inputs::getOption<std::vector<T>, 0>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 1>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 2>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 3>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 4>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 5>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 6>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 7>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 8>(const std::string&);       \
template std::vector<T> Inputs::getOption<std::vector<T>, 9>(const std::string&);       \
template std::array<T, 1> Inputs::getOption<std::array<T, 1>, 1>(const std::string&);   \
template std::array<T, 2> Inputs::getOption<std::array<T, 2>, 2>(const std::string&);   \
template std::array<T, 3> Inputs::getOption<std::array<T, 3>, 3>(const std::string&);   \
template std::array<T, 4> Inputs::getOption<std::array<T, 4>, 4>(const std::string&);   \
template std::array<T, 5> Inputs::getOption<std::array<T, 5>, 5>(const std::string&);   \
template std::array<T, 6> Inputs::getOption<std::array<T, 6>, 6>(const std::string&);   \
template std::array<T, 7> Inputs::getOption<std::array<T, 7>, 7>(const std::string&);   \
template std::array<T, 8> Inputs::getOption<std::array<T, 8>, 8>(const std::string&);   \
template std::array<T, 9> Inputs::getOption<std::array<T, 9>, 9>(const std::string&)

INSTANTIATE_GET_OPTION_SEQUENCE(int8_t);
INSTANTIATE_GET_OPTION_SEQUENCE(uint8_t);
INSTANTIATE_GET_OPTION_SEQUENCE(short);
INSTANTIATE_GET_OPTION_SEQUENCE(unsigned short);
INSTANTIATE_GET_OPTION_SEQUENCE(int);
INSTANTIATE_GET_OPTION_SEQUENCE(unsigned int);
INSTANTIATE_GET_OPTION_SEQUENCE(long);
INSTANTIATE_GET_OPTION_SEQUENCE(unsigned long);
INSTANTIATE_GET_OPTION_SEQUENCE(long long);
INSTANTIATE_GET_OPTION_SEQUENCE(unsigned long long);
INSTANTIATE_GET_OPTION_SEQUENCE(bool);
INSTANTIATE_GET_OPTION_SEQUENCE(float);
INSTANTIATE_GET_OPTION_SEQUENCE(double);
INSTANTIATE_GET_OPTION_SEQUENCE(std::string);

// IntX & FloatX
template int2_t Inputs::getOption<int2_t, 2>(const std::string&);
template int3_t Inputs::getOption<int3_t, 3>(const std::string&);
template int4_t Inputs::getOption<int4_t, 4>(const std::string&);
template uint2_t Inputs::getOption<uint2_t, 2>(const std::string&);
template uint3_t Inputs::getOption<uint3_t, 3>(const std::string&);
template uint4_t Inputs::getOption<uint4_t, 4>(const std::string&);
template long2_t Inputs::getOption<long2_t, 2>(const std::string&);
template long3_t Inputs::getOption<long3_t, 3>(const std::string&);
template long4_t Inputs::getOption<long4_t, 4>(const std::string&);
template ulong2_t Inputs::getOption<ulong2_t, 2>(const std::string&);
template ulong3_t Inputs::getOption<ulong3_t, 3>(const std::string&);
template ulong4_t Inputs::getOption<ulong4_t, 4>(const std::string&);
template float2_t Inputs::getOption<float2_t, 2>(const std::string&);
template float3_t Inputs::getOption<float3_t, 3>(const std::string&);
template float4_t Inputs::getOption<float4_t, 4>(const std::string&);
template double2_t Inputs::getOption<double2_t, 2>(const std::string&);
template double3_t Inputs::getOption<double3_t, 3>(const std::string&);
template double4_t Inputs::getOption<double4_t, 4>(const std::string&);
