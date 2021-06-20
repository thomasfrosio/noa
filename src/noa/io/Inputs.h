/// \file Inputs.h
/// \brief Manages all user inputs.
/// \author Thomas - ffyr2w
/// \date 31 Jul 2020

#pragma once

// TODO This class is currently not used and should be a separate project outside or an extension of noa. Maybe noa-parser?
//      ATM it uses some noa types, but this is not really useful and we could only support basic types
//      for parsing. The util/String.h functions, mostly used by this class, could also be removed from noa.

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/util/Traits.h"
#include "noa/util/String.h"

namespace noa {
    /// Parses and formats the inputs of the command line and the parameter file (if any).
    /// \see Inputs() to initialize the input manager.
    /// \see setCommands() to register commands.
    /// \see getCommand() to get the command specified in the command line.
    /// \see setOptions() to register options.
    /// \see parse() to parse the inputs from the command line and parameter file.
    /// \see getOption() to retrieve the formatted inputs.
    class Inputs {
    private:
        std::vector<std::string> m_cmdline;
        std::vector<std::string> m_registered_sections{};
        std::vector<std::string> m_registered_commands{};
        std::vector<std::string> m_registered_options{};

        std::string m_command{};
        std::string m_parameter_filename{};

        // Key = option name, Value(s) = option value(s).
        std::unordered_map<std::string, std::string> m_inputs_cmdline{};
        std::unordered_map<std::string, std::string> m_inputs_file{};

        /// Option usage. See setOption().
        /// \defail  The option vector should be a multiple of 5, such as:
        ///  - 0 \c USAGE_LONG_NAME     : long-name of the option.
        ///  - 1 \c USAGE_SHORT_NAME    : short-name of the option.
        ///  - 2 \c USAGE_TYPE          : expected type of the option. See assertType_()
        ///  - 3 \c USAGE_DEFAULT_VALUE : default value(s). See setOptions()
        ///  - 4 \c USAGE_SECTION       : section number of the option. See setSections()
        ///  - 5 \c USAGE_DOCSTRING     : docstring displayed with the \c "--help" command.
        enum {
            USAGE_LONG_NAME = 0,
            USAGE_SHORT_NAME,
            USAGE_TYPE,
            USAGE_DEFAULT_VALUE,
            USAGE_SECTION,
            USAGE_DOCSTRING,
            USAGE_ELEMENTS_PER_OPTION
        };

        const std::string m_usage_header = "Usage:\n"
                                           "    [./{1}] (-h|v)\n"
                                           "    [./{1}] [command] (-h)\n"
                                           "    [./{1}] [command] (file) ([option1 value1] ...)";

        const std::string m_usage_footer = "\nGlobal options:\n"
                                           "    --help, -h      Show global help.\n"
                                           "    --version, -v   Show the version.\n";

    public:
        /// Stores the command line.
        /// \param[in] argc     How many C-strings are contained in \c argv.
        /// \param[in] argv     Command line arguments. Must be stripped and split C-strings.
        ///
        /// \details: \a argv should be organized as follow:
        ///              [app] (-h|-v)                                   or,
        ///              [app] [command] (-h)                            or,
        ///              [app] [command] (file) ([option1 value1] ...)
        Inputs(const int argc, const char** argv) : m_cmdline(argv, argv + argc) {}

        /** Overload for tests */
        explicit Inputs(std::vector<std::string> args) : m_cmdline(std::move(args)) {}

        /// Registers the commands and sets the actual command.
        /// \tparam T       Mainly used to handle lvalue and rvalue references in one function.
        ///                 Really, it must be a \c std::vector<std::string>.
        /// \param commands Command(s) to register. If commands are already registered, overwrite
        ///                 with these ones. Each command should take two strings in the vector,
        ///                 such as `{"<command>", "<docstring>", ...}`.
        /// \throw          If \a commands isn't of the expected size, i.e. even size.
        ///                 If the actual command is not one of the newly registered commands.
        ///
        /// \note   If the same command is specified twice, everything should work as excepted (printCommand() will
        ///         print both entries though), so there's no duplicate check here.
        /// \note   The "help" and "version" commands are automatically set and handled.
        void setCommands(std::vector<std::string> commands) {
            m_registered_commands = std::move(commands);

            if (m_registered_commands.size() % 2) {
                NOA_THROW("DEV: the size of the command vector should be a multiple of 2, "
                          "got {} element(s)", m_registered_commands.size());
            } else if (m_cmdline.size() < 2) {
                m_command = "help";
                return;
            }

            // Make sure the actual command is one of the newly registered commands.
            const std::string& command = m_cmdline[1];
            for (size_t idx{0}; idx < m_registered_commands.size(); idx += 2) {
                if (m_registered_commands[idx] == command) {
                    m_command = command;
                    return;
                }
            }

            // Specific cases for "-h" and "-v", which are automatically registered.
            if (isHelp_(command))
                m_command = "help";
            else if (isVersion_(command))
                m_command = "version";
            else
                NOA_THROW("DEV: \"{}\" is not a registered command", command);
        }

        /// Gets the actual command, i.e. the command entered at the command line.
        inline const std::string& getCommand() const { return m_command; }

        void setSections(std::vector<std::string>&& sections) {
            m_registered_sections = std::move(sections);
        }

        void setSections(const std::vector<std::string>& sections) {
            m_registered_sections = sections;
        }

        /// Registers the options. These should correspond to the current command (the one returned by getCommand()).
        /// \param[in] options   Option(s) to register. If options are already registered, overwrite
        ///                      with these ones. Each option takes 5 strings in the input vector.
        ///                          - 1: Long-name, without dashes or prefixes.
        ///                          - 2: Short-name, without dashes or prefixes.
        ///                          - 3: The 2 characters usage type.
        ///                          - 4: The default value(s). Fields are separated by commas. The number of fields
        ///                               should match the usage type, with the exception of an empty default string,
        ///                               specifying that the option is non-optional.
        ///                               If more than one field is specified, some fields can be left empty, meaning
        ///                               that these particular fields become non-optional. Note that this is not
        ///                               allowed if the first character of the usage string is 0 (a range is expected).
        ///                          - 5: Docstring of the option. Used by printOptions().
        ///
        /// \note   If there's a duplicate between (long|short)-names, this will likely
        ///         result in an usage type error when retrieving the option with getOption().
        ///         Because this is on the developer's side and not on the user's, there's
        ///         no duplicate check. TLDR: This is the developer's job to make sure
        ///         the input options don't contain duplicates.
        /// \throw  If the command is not set. Set it with setCommand().
        ///         If the size of \c options is not multiple of 5.
        void setOptions(std::vector<std::string>&& options) {
            if (m_command.empty())
                NOA_THROW("DEV: the command is not set. Set it first with setCommand()");
            else if (options.size() % USAGE_ELEMENTS_PER_OPTION)
                NOA_THROW("DEV: the size of the options vector should be a multiple of 5, "
                          "got {} element(s)", options.size());
            m_registered_options = std::move(options);
        }

        void setOptions(const std::vector<std::string>& options) {
            setOptions(std::vector<std::string>(options));
        }

        /// Gets the value(s) of a given option.
        /// \tparam T       Returned type. The original value(s) (i.e. strings) will to be formatted to this type.
        ///                 Supported types:
        ///                     - Base: (u)short, (u)int, (u)long, (u)long long, bool, float, double, string.
        ///                     - Containers: std::vector<X>, std::array<X, N>, where X is any of the base type.
        ///                     - noa types: Int2, Int3, Int4, Float2, Float3, Float4.
        /// \tparam N       Number of expected values.
        ///                     - If \a T is a base type, N must be equal to 1.
        ///                     - If N >= 1, the option should contain \a N entries, both from the user and from
        ///                       the default values. Positional defaulting is allowed.
        ///                     - If N == 0, it indicates that an unknown range of values is to be expected.
        ///                       In this case, \a T must be a std::vector and positional defaulting isn't allowed.
        /// \param option   Long-name of the option (without the dash(es) or prefix).
        /// \return         Formatted value(s).
        ///
        /// \throw          If one value is missing and no default value was found.
        ///                 If one value (i.e. a string) cannot be converted into the returned type \a T or value_type.
        template<typename T, size_t N = 1>
        T getOption(const std::string& option);

        /**
        /// Parses the command line options and the parameter file if there's one.
        /// \param[in] prefix   Prefix of the parameter to consider, e.g. "noa_".
        /// \return             Whether or not the parsing was completed. If not, it means the user
        ///                     has asked for help and that the program should exit.
        ///
        /// \throw If the command line or parameter file don't have a supported format.
        /// \note \a m_inputs_cmdline, \a m_inputs_file and \a m_parameter_file are modified.
         */
        [[nodiscard]] bool parse(const std::string& prefix);

        /// Parses the entries from the cmd line.
        /// \c m_parsing_is_complete will be set to true if the parsing was complete.
        /// \throw If the command line does not have the expected format or if an option is not recognized.
        ///
        /// \details When entered at the command line, options must be prefixed by one or
        ///          two dashes (- or --) and must be followed by a space and a value (options
        ///          without values are not supported). The names are case-sensitive. Options cannot
        ///          be concatenated the way single letter options in Unix programs often can be.
        ///          To specify multiple values for one option, use commas, e.g. --size 100,101,102.
        ///          Commas without values indicates that the default value for that position should
        ///          be taken if possible. For example, "12,," takes the default for the second and
        ///          third values. Positional default values can be used only when a fixed number of
        ///          values are expected. Options can be entered only once in the command line, but
        ///          the same option can be entered in the command line and in the parameter file.
        ///          In this case, the command line takes the precedence over the parameter file.
        ///          All options should be registered with setOption() before calling this function.
        bool parseCommandLine();

        /// Parses a parameter file.
        /// \throw If the parameter file does not have a supported format.
        /// \details Options should start at the beginning of a line and be prefixed by \c m_prefix.
        ///          The values should be specified after a '=', i.e. `[m_prefix][option]=[value]`.
        ///          Spaces are ignored before and after the [option], '=' and [value]. Multiple
        ///          values can be specified like in the cmd line. Inline comments are allowed and
        ///          should start with a '#'. Options can be entered only once in the parameter file,
        ///          but the same option can be entered in the command line and in the parameter file.
        ///          In this case, the command line takes the precedence over the parameter file.
        ///          Options do not have to be registered with setOption(), as opposed to the command
        ///          line options. This allows more generic parameter file.
        void parseParameterFile(const std::string& filename, const std::string& prefix);

        /// Prints the registered commands in a docstring format.
        std::string formatCommands() const;

        /// Prints the registered options in a docstring format.
        std::string formatOptions() const;

    private:
        /// Whether or not the entry corresponds to a "help" command.
        template<typename T, typename = std::enable_if_t<noa::traits::is_string_v<T>>>
        static inline bool isHelp_(T&& str) {
            if (str == "-h" || str == "--help" || str == "help" ||
                str == "h" || str == "-help" || str == "--h")
                return true;
            return false;
        }

        /// Whether or not the entry corresponds to a "version" command.
        template<typename T, typename = std::enable_if_t<noa::traits::is_string_v<T>>>
        static inline bool isVersion_(T&& str) {
            if (str == "-v" || str == "--version" || str == "version" ||
                str == "v" || str == "-version" || str == "--v")
                return true;
            return false;
        }

        /// Converts the usage type into something readable for the user.
        /// \param[in] usage_type   Usage type. See assertType_() for more details.
        /// \return                 Formatted type
        /// \throw                  If the usage type isn't recognized.
        static std::string formatType_(const std::string& usage_type);

        /// Makes sure the usage type matches the excepted type and number of values.
        /// \tparam T               Desired type.
        /// \tparam N               Number of values.
        /// \param[in] usage_type   Usage type. It must be a 2 characters string, such as:
        ///                             -# 1st character: 0 (range), 1, 2, 3, 4, 5, 6, 7, 8, or 9.
        ///                             -# 2nd character: I, U, F, S or B, corresponding to integers,
        ///                                               unsigned integers, floating points, strings
        ///                                               and booleans, respectively.
        /// \throw                  If the usage type doesn't match \c T or \c N.
        /// \note These are checks that could be done at compile time... C++20 hello.
        template<typename T, size_t N>
        static void assertType_(const std::string& usage_type) {
            if (usage_type.size() != 2)
                NOA_THROW("DEV: type usage \"{}\" is not recognized. It should be a "
                          "string with 2 characters", usage_type);

            // Number of values.
            if constexpr(N >= 0 && N < 10) {
                if (usage_type[0] != N + '0')
                    NOA_THROW("DEV: the type usage \"{}\" does not correspond to the desired "
                              "number of values: {}", usage_type, N);
            } else {
                NOA_THROW("DEV: the type usage \"{}\" is not recognized. "
                          "N should be a number from 0 to 9", usage_type);
            }

            // Types.
            if constexpr(noa::traits::is_float_v<T> || noa::traits::is_std_sequence_float_v<T> ||
                         noa::traits::is_floatX_v<T>) {
                if (usage_type[1] != 'F')
                    NOA_THROW("DEV: the type usage \"{}\" does not correspond to the desired "
                              "type (floating point)", usage_type);

            } else if constexpr(noa::traits::is_uint_v<T> || noa::traits::is_std_sequence_uint_v<T> ||
                                noa::traits::is_uintX_v<T>) {
                if (usage_type[1] != 'U')
                    NOA_THROW("DEV: the type usage \"{}\" does not correspond to the desired "
                              "type (unsigned integer)", usage_type);

            } else if constexpr(noa::traits::is_int_v<T> || noa::traits::is_std_sequence_int_v<T> ||
                                noa::traits::is_intX_v<T>) {
                if (usage_type[1] != 'I')
                    NOA_THROW("DEV: the type usage \"{}\" does not correspond to the desired "
                              "type (integer)", usage_type);

            } else if constexpr(noa::traits::is_bool_v<T> || noa::traits::is_std_sequence_bool_v<T>) {
                if (usage_type[1] != 'B')
                    NOA_THROW("DEV: the type usage \"{}\" does not correspond to the desired "
                              "type (boolean)", usage_type);

            } else if constexpr(noa::traits::is_string_v<T> || noa::traits::is_std_sequence_string_v<T>) {
                if (usage_type[1] != 'S')
                    NOA_THROW("DEV: the type usage \"{}\" does not correspond to the desired "
                              "type (string)", usage_type);

            } else {
                NOA_THROW("DEV: the type usage \"{}\" is not recognized.", usage_type);
            }
        }

        /// Extracts the trimmed value of a given option.
        /// \param[in] long_name    Long-name of the option.
        /// \param[in] short_name   Short-name of the option.
        /// \return                 The trimmed value. The string is guaranteed to be not empty.
        ///                         If the option is not found (not registered or not entered), NULL is returned.
        ///
        /// \throw  If \a long_name and \a short_name were both entered in the command
        ///         line or in the parameter file.
        /// \note   Options can be entered only once, but the same option
        ///         can be entered in the command line and in the parameter file.
        ///         In this case, the command line takes the precedence over the
        ///         parameter file.
        std::string* getValue_(const std::string& long_name, const std::string& short_name);

        /// Extracts the option usage for a given long-name.
        /// \param[in] long_name    Long-name of the wanted option.
        /// \return                 {\a short_name, \a usage_type, \a default_values(s)}
        /// \throw If \a long_name is not registered.
        std::tuple<const std::string*, const std::string*, const std::string*>
        getOptionUsage_(const std::string& long_name) const;

        /// Whether or not the \a name was registered as the (long|short)-name of an option.
        inline bool isOption_(const std::string& name) const {
            for (size_t i{0}; i < m_registered_options.size(); i += USAGE_DEFAULT_VALUE)
                if (m_registered_options[i] == name || m_registered_options[i + 1] == name)
                    return true;
            return false;
        }
    };
}
