/**
 * @file InputManager.h
 * @brief Manages all user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/String.h"
#include "noa/files/TextFile.h"


namespace Noa {

    /**
     * Parses and makes available the inputs of the command line and the parameter file (if any).
     * @note This is "high" level in the API, hence most function are directly throwing exception
     *       with proper error messages, as opposed to the "low" level API which returns an Errno.
     *
     * @see InputManager() to initialize the input manager.
     * @see setCommand() to register commands.
     * @see getCommand() to get the actual command.
     * @see setOption() to register options.
     * @see parse() to parse the inputs from the command line and parameter file.
     * @see getOption() to retrieve the formatted inputs.
     */
    class NOA_API InputManager {
    private:
        std::vector<std::string> m_cmdline;

        std::vector<std::string> m_registered_commands{};
        std::vector<std::string> m_registered_options{};

        std::string m_command{};
        std::string m_parameter_filename{};

        // Key = option name, Value(s) = option value(s).
        std::unordered_map<std::string, std::string> m_inputs_cmdline{};
        std::unordered_map<std::string, std::string> m_inputs_file{};

        /**
         * Option usage. See setOption().
         * @defail  The option vector should be a multiple of 5, such as:
         *  - 1 @c long_name     : long-name of the option.
         *  - 2 @c short_name    : short-name of the option.
         *  - 3 @c type          : expected type of the option. See assertType_()
         *  - 4 @c default_value : default value(s). See getOption()
         *  - 5 @c docstring     : docstring displayed with the @c "--help" command.
         */
        struct OptionUsage {
            static constexpr uint32_t long_name{0U};
            static constexpr uint32_t short_name{1U};
            static constexpr uint32_t type{2U};
            static constexpr uint32_t default_value{3U};
            static constexpr uint32_t docstring{4U};
        };

        const std::string m_usage_header = fmt::format(
                FMT_COMPILE("Welcome to NOA.\n"
                            "Version {} - compiled on {}\n"
                            "Website: {}\n\n"
                            "Usage:\n"
                            "     [./noa] (-h|v)\n"
                            "     [./noa] [command] (-h)\n"
                            "     [./noa] [command] (file) ([option1 value1] ...)"),
                NOA_VERSION, __DATE__, NOA_URL
        );

        const std::string m_usage_footer = ("\nGlobal options:\n"
                                            "   --help, -h      Show global help.\n"
                                            "   --version, -v   Show the version.\n");

    public:
        /**
         * Stores the command line.
         * @param[in] argc      How many C-strings are contained in @c argv.
         * @param[in] argv      Command line arguments. Must be stripped and split C-strings.
         *
         * @details: @a argv should be organized as follow:
         *              [app] (-h|-v)                                   or,
         *              [app] [command] (-h)                            or,
         *              [app] [command] (file) ([option1 value1] ...)
         */
        InputManager(const int argc, const char** argv) : m_cmdline(argv, argv + argc) {}


        /** Overload for tests */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<Traits::is_same_v<T, std::vector<std::string>>>>
        explicit InputManager(T&& args) : m_cmdline(std::forward<T>(args)) {}


        /**
         * Registers the commands and sets the actual command.
         *
         * @tparam T            Mainly used to handle lvalue and rvalue references in one function.
         *                      Really, it must be a @c std::vector<std::string>.
         * @param[in] commands  Command(s) to register. If commands are already registered, overwrite
         *                      with these ones. Each command should take two strings in the vector,
         *                      such as `{"<command>", "<docstring>", ...}`.
         * @throw Error         If @a commands isn't of the expected size, i.e. even size.
         *                      If the actual command is not one of the newly registered commands.
         *
         * @note                If the same command is specified twice, everything should work
         *                      as excepted (printCommand() will print both entries though), so
         *                      there's no duplicate check here.
         * @note                The "help" and "version" commands are automatically set and handled.
         */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<Traits::is_same_v<T, std::vector<std::string>>>>
        void setCommand(T&& commands) {
            m_registered_commands = std::forward<T>(commands);

            if (m_registered_commands.size() % 2) {
                NOA_LOG_ERROR("DEV: the size of the command vector should be a multiple of 2, "
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
                NOA_LOG_ERROR("DEV: \"{}\" is not a registered command", command);
        }


        /** Gets the actual command, i.e. the command entered at the command line. */
        inline const std::string& getCommand() const { return m_command; }


        /**
         * Registers the options. These should correspond to the actual command.
         * @tparam T            Mainly used to handle lvalue and rvalue references in one function.
         *                      Really, it must be a @c std::vector<std::string>.
         * @param[in] options   Option(s) to register. If options are already registered, overwrite
         *                      with these ones. Each option takes 5 strings in the input vector.
         *                      See @a OptionUsage.
         *
         * @note                If there's a duplicate between (long|short)-names, this will likely
         *                      result in an usage type error when retrieving the option with getOption().
         *                      Because this is on the programmer side and not on the user, there's
         *                      no duplicate check. TLDR: This is the programmer's job to make sure
         *                      the input options don't contain duplicates.
         *
         * @throw Error         If the command is not set. Set it with setCommand().
         *                      If the size of @c options is not multiple of 5.
         */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<Traits::is_same_v<T, std::vector<std::string>>>>
        void setOption(T&& options) {
            if (m_command.empty())
                NOA_LOG_ERROR("DEV: the command is not set. Set it first with setCommand()");
            else if (options.size() % 5)
                NOA_LOG_ERROR("DEV: the size of the options vector should be a multiple of 5, "
                              "got {} element(s)", options.size());
            m_registered_options = std::forward<T>(options);
        }


        /**
         * Gets the value(s) of a given option.
         * @tparam T            Returned type. The original value(s) (which are strings) will to be
         *                      formatted to this type.
         * @tparam[in] N        Number of expected values. It should be >= 0. If 0, it indicates that
         *                      an unknown range of values are to be expected. In this case, T must
         *                      be a vector and positional defaulting isn't allowed. If >0,
         *                      @a N entries are expected from the user and from the default values,
         *                      and positional defaulting is allowed.
         * @param[in] long_name Long-name of the option (without the dash(es) or prefix).
         * @return              Formatted value(s).
         *
         * @throw Error         If one value is missing and no default value was found.
         *                      If one value (i.e. a string) cannot be converted into the returned type @a T.
         */
        template<typename T, size_t N = 1>
        auto getOption(const std::string& long_name) {
            static_assert(N >= 0 && N < 10);
            static_assert(!(Traits::is_std_sequence_complex_v<T> || Traits::is_complex_v<T>));

            auto[u_short, u_type, u_value] = getOptionUsage_(long_name);
            assertType_<T, N>(*u_type);
            const std::string* value = getValue_(long_name, *u_short);
            if (!value) /* this option was not found, so use default */
                value = u_value;

            T output{};
            errno_t err;
            if constexpr(N == 0) {
                static_assert(Traits::is_std_vector_v<T>);
                // When an unknown number of value is expected (N == 0), values cannot be defaulted
                // based on their position. Thus, let parse() try to convert the raw value.
                err = String::parse(*value, output);

            } else if constexpr (Traits::is_bool_v<T> ||
                                 Traits::is_string_v<T> ||
                                 Traits::is_scalar_v<T>) {
                static_assert(N == 1);
                err = String::parse(*value, &output, 1);

            } else /* N >= 1 */ {
                if constexpr (Traits::is_vector_v<T>) {
                    static_assert(T::size() >= N);
                } else if constexpr (Traits::is_std_array_v<T>) {
                    static_assert(std::tuple_size_v<T> >= N);
                } else if constexpr (Traits::is_std_vector_v<T>) {
                    output.reserve(N);
                }
                // Option is not entered or no defaults.
                // In both cases, we can only rely on what is in "value".
                if (u_value == value || u_value->empty())
                    err = String::parse(*value, output, N);
                else
                    err = String::parse(*value, *u_value, output, N);
            }

            if constexpr (Traits::is_string_v<T>) {
                if (output.empty())
                    err = Errno::invalid_argument;
            } else if constexpr (Traits::is_std_sequence_string_v<T>) {
                for (auto& str: output)
                    if (str.empty())
                        err = Errno::invalid_argument;
            }
            if (err)
                NOA_LOG_ERROR(getOptionErrorMessage_(long_name, value, N, err));

            NOA_LOG_TRACE("{} ({}): {}", long_name, *u_short, output);
            return output;
        }


        /**
         * Parses the command line options and the parameter file if there's one.
         * @param[in] prefix    Prefix of the parameter to consider, e.g. "noa_".
         * @return              Whether or not the parsing was completed. If not, it means the user
         *                      has asked for help and that the program should exit.
         *
         * @throw Error         If the command line or parameter file don't have a supported format.
         * @note @a m_inputs_cmdline, @a m_inputs_file and @a m_parameter_file are modified.
         */
        [[nodiscard]] bool parse(const std::string& prefix);


        /**
         * Parses the entries from the cmd line.
         * @c m_parsing_is_complete will be set to true if the parsing was complete.
         * @throw Error If the command line does not have the expected format or if an option is not recognized.
         *
         * @details When entered at the command line, options must be prefixed by one or
         *          two dashes (- or --) and must be followed by a space and a value (options
         *          without values are not supported). The names are case-sensitive. Options cannot
         *          be concatenated the way single letter options in Unix programs often can be.
         *          To specify multiple values for one option, use commas, e.g. --size 100,101,102.
         *          Commas without values indicates that the default value for that position should
         *          be taken if possible. For example, "12,," takes the default for the second and
         *          third values. Positional default values can be used only when a fixed number of
         *          values are expected. Options can be entered only once in the command line, but
         *          the same option can be entered in the command line and in the parameter file.
         *          In this case, the command line takes the precedence over the parameter file.
         *          All options should be registered with setOption() before calling this function.
         */
        bool parseCommandLine();


        /**
         * Parses a parameter file.
         * @throw Error     If the parameter file does not have a supported format.
         * @details Options should start at the beginning of a line and be prefixed by @c m_prefix.
         *          The values should be specified after a '=', i.e. `[m_prefix][option]=[value]`.
         *          Spaces are ignored before and after the [option], '=' and [value]. Multiple
         *          values can be specified like in the cmd line. Inline comments are allowed and
         *          should start with a '#'. Options can be entered only once in the parameter file,
         *          but the same option can be entered in the command line and in the parameter file.
         *          In this case, the command line takes the precedence over the parameter file.
         *          Options do not have to be registered with setOption(), as opposed to the command
         *          line options. This allows more generic parameter file.
         */
        void parseParameterFile(const std::string& filename, const std::string& prefix);


        /** Prints the registered commands in a docstring format. */
        void printCommand() const;


        /** Prints the NOA version. */
        static inline void printVersion() { fmt::print("{}\n", NOA_VERSION); }


        /** Prints the registered options in a docstring format. */
        void printOption() const;

    private:
        /** Whether or not the entry corresponds to a "help" command. */
        template<typename T, typename = std::enable_if_t<Traits::is_string_v<T>>>
        static inline bool isHelp_(T&& str) {
            if (str == "-h" || str == "--help" || str == "help" ||
                str == "h" || str == "-help" || str == "--h")
                return true;
            return false;
        }


        /** Whether or not the entry corresponds to a "version" command. */
        template<typename T, typename = std::enable_if_t<Traits::is_string_v<T>>>
        static inline bool isVersion_(T&& str) {
            if (str == "-v" || str == "--version" || str == "version" ||
                str == "v" || str == "-version" || str == "--v")
                return true;
            return false;
        }

        /** Gets a meaningful error message for getOption(). */
        std::string getOptionErrorMessage_(const std::string& l_name, const std::string* value,
                                           size_t N, errno_t err) const;


        /**
         * Converts the usage type into something readable for the user.
         * @param[in] usage_type    Usage type. See assertType_() for more details.
         * @return                  Formatted type
         * @throw Error             If the usage type isn't recognized.
         */
        static std::string formatType_(const std::string& usage_type);


        /**
         * Makes sure the usage type matches the excepted type and number of values.
         * @tparam T                Desired type.
         * @tparam N                Number of values.
         * @param[in] usage_type    Usage type. It must be a 2 characters string, such as:
         *                              -# 1st character: 0 (range), 1, 2, 3, 4, 5, 6, 7, 8, or 9.
         *                              -# 2nd character: I, U, F, S or B, corresponding to integers,
         *                                                unsigned integers, floating points, strings
         *                                                and booleans, respectively.
         * @throw Error             If the usage type doesn't match @c T or @c N.
         * @note These are checks that could be done at compile time... C++20 hello.
         */
        template<typename T, size_t N>
        static void assertType_(const std::string& usage_type) {
            if (usage_type.size() != 2)
                NOA_LOG_ERROR("DEV: type usage \"{}\" is not recognized. It should be a "
                              "string with 2 characters", usage_type);

            // Number of values.
            if constexpr(N >= 0 && N < 10) {
                if (usage_type[0] != N + '0')
                    NOA_LOG_ERROR("DEV: the type usage \"{}\" does not correspond to the desired "
                                  "number of values: {}", usage_type, N);
            } else {
                NOA_LOG_ERROR("DEV: the type usage \"{}\" is not recognized. "
                              "N should be a number from  0 to 9", usage_type);
            }

            // Types.
            using namespace Noa::Traits;
            if constexpr(is_float_v<T> || is_std_sequence_float_v<T> || is_vector_float_v<T>) {
                if (usage_type[1] != 'F')
                    NOA_LOG_ERROR("DEV: the type usage \"{}\" does not correspond to the desired "
                                  "type (floating point)", usage_type);

            } else if constexpr(is_int_v<T> || is_std_sequence_int_v<T> || is_vector_int_v<T>) {
                if (usage_type[1] != 'I')
                    NOA_LOG_ERROR("DEV: the type usage \"{}\" does not correspond to the desired "
                                  "type (integer)", usage_type);

            } else if constexpr(is_uint_v<T> || is_std_sequence_uint_v<T> || is_vector_uint_v<T>) {
                if (usage_type[1] != 'U')
                    NOA_LOG_ERROR("DEV: the type usage \"{}\" does not correspond to the desired "
                                  "type (unsigned integer)", usage_type);

            } else if constexpr(is_bool_v<T> || is_std_sequence_bool_v<T>) {
                if (usage_type[1] != 'B')
                    NOA_LOG_ERROR("DEV: the type usage \"{}\" does not correspond to the desired "
                                  "type (boolean)", usage_type);

            } else if constexpr(is_string_v<T> || is_std_sequence_string_v<T>) {
                if (usage_type[1] != 'S')
                    NOA_LOG_ERROR("DEV: the type usage \"{}\" does not correspond to the desired "
                                  "type (string)", usage_type);

            } else {
                NOA_LOG_ERROR("DEV: the type usage \"{}\" is not recognized.", usage_type);
            }
        }


        /**
         * Extracts the trimmed value of a given option.
         * @param[in] long_name     Long-name of the option.
         * @param[in] short_name    Short-name of the option.
         * @return                  The trimmed value. The string is guaranteed to be not empty.
         *                          If the option is not found (not registered or not entered), NULL is returned.
         * @throw Error             If @a long_name and @a short_name were both entered in the command
         *                          line or in the parameter file.
         *
         * @note                    Options can be entered only once, but the same option
         *                          can be entered in the command line and in the parameter file.
         *                          In this case, the command line takes the precedence over the
         *                          parameter file.
         */
        std::string* getValue_(const std::string& long_name, const std::string& short_name);


        /**
         * Extracts the option usage for a given long-name.
         * @param[in] long_name     Long-name of the wanted option.
         * @return                  {@a short_name, @a usage_type, @a default_values(s)}
         * @throw Error             If @a long_name is not registered.
         */
        std::tuple<const std::string*, const std::string*, const std::string*>
        getOptionUsage_(const std::string& long_name) const;


        /** Whether or not the @a name was registered as the (long|short)-name of an option. */
        inline bool isOption_(const std::string& name) const {
            for (size_t i{0}; i < m_registered_options.size(); i += 5)
                if (m_registered_options[i] == name || m_registered_options[i + 1] == name)
                    return true;
            return false;
        }
    };
}
