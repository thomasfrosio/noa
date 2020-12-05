/**
 * @file InputManager.h
 * @brief InputManager manager - Manages user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/util/String.h"
#include "noa/util/Assert.h"
#include "noa/util/Traits.h"
#include "noa/files/TextFile.h"


namespace Noa {

    /**
     * Parse the command line and the parameter file (if any) and access the inputs with get().
     *
     * @details:
     * Supported scenarios:
     *  -1 `[./app] (-h|v)`: setCommand() returns @c "--help" or @c "--version". Parsing the options
     *      with parse() or retrieving values with get() will not be possible.
     *  -2 `[./app] [command] (-h)`: setCommand() returns @c [command] and parse() will return true,
     *      meaning the user asked the help of [command]. Retrieving values with get() will not be possible.
     *  -3 `[./app] [command] (file) ([option1 value1] ...)`: setCommand() returns [command] and the
     *      option/value pairs are parsed and accessible using get(). If an help (i.e. -h, --help)
     *      is found, the parsing is aborted and parse() returns true. If there's a parameter file
     *      (file), it is parsed and its options can be retrieved with get() as well.
     *
     * @see InputManager() to initialize the input manager.
     * @see setCommand() to register allowed commands and get the actual command.
     * @see setOption() to register options.
     * @see parse() to parse the options in the cmd line and parameter file.
     * @see get() to retrieve the formatted inputs.
     */
    class NOA_API InputManager {
    private:
        std::vector<std::string> m_cmdline;
        std::string m_prefix;

        std::vector<std::string> m_registered_commands{};
        std::vector<std::string> m_registered_options{};

        std::string m_command{};
        std::string m_parameter_filename{};

        std::unordered_map<std::string, std::string> m_options_cmdline{};
        std::unordered_map<std::string, std::string> m_options_parameter_file{};

        /**
         * Option usage. See setOption().
         * @defail  The option vector should be a multiple of 5, such as:
         *  - 1 @c long_name     : long-name of the option.
         *  - 2 @c short_name    : short-name of the option.
         *  - 3 @c type          : expected type of the option. See assertType_()
         *  - 4 @c default_value : default value(s). See get()
         *  - 5 @c docstring     : docstring displayed with the @c "--help" command.
         */
        struct OptionUsage {
            static constexpr u_int8_t long_name{0U};
            static constexpr u_int8_t short_name{1U};
            static constexpr u_int8_t type{2U};
            static constexpr u_int8_t default_value{3U};
            static constexpr u_int8_t docstring{4U};
        };

        bool m_parsing_is_complete{false};

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
         * Store the command line.
         * @param[in] argc      How many C-strings are contained in @c argv, i.e. number of arguments.
         *                      Usually comes from main().
         * @param[in] argv      Command line arguments. Must be stripped and split C-strings.
         *                      Usually comes from main().
         * @param[in] prefix    Prefix of the options specified in the parameter file.
         */
        InputManager(const int argc, const char** argv, std::string prefix = "noa_")
                : m_cmdline(argv, argv + argc), m_prefix(std::move(prefix)) {}


        /** Overload for tests */
        template<typename T, typename = std::enable_if_t<Traits::is_same_v<T, std::vector<std::string>>>>
        explicit InputManager(T&& args, std::string prefix = "noa_")
                : m_cmdline(std::forward<T>(args)), m_prefix(std::move(prefix)) {}


        /**
         * Register the allowed commands and return the actual command.
         *
         * @tparam T            Mainly used to handle lvalue and rvalue references in one function.
         *                      Really, it must be a @c std::vector<std::string>.
         * @param[in] commands  Command(s) to register. If commands are already registered, overwrite
         *                      with these ones. Each command should take two strings in the vector,
         *                      such as `{"<command>", "<docstring>", ...}`.
         * @return              The actual command that is registered in the command line
         *                      It can be "version", "help" or any one of the newly registered commands.
         *
         * @note                If the same command is specified twice, everything should work
         *                      as excepted (printCommand() will print both entries though), so
         *                      there's no explicit duplicate check here.
         *
         * @throw ErrorCore     If the actual command is not one of the newly registered commands.
         *                      If the size of @c commands is not multiple of 2.
         */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<::Noa::Traits::is_same_v<T, std::vector<std::string>>>>
        const std::string& setCommand(T&& commands) {
            if (commands.size() % 2)
                NOA_LOG_ERROR("the size of the command vector should be a multiple of 2, "
                              "got {} element(s)", commands.size());

            if (m_cmdline.size() < 2)
                m_command = "help";
            else if (std::find(commands.begin(), commands.end(), m_cmdline[1]) == commands.end()) {
                const std::string& argv1 = m_cmdline[1];
                if (argv1 == "-h" || argv1 == "--help" || argv1 == "help" ||
                    argv1 == "h" || argv1 == "-help" || argv1 == "--h")
                    m_command = "help";
                else if (argv1 == "-v" || argv1 == "--version" || argv1 == "version" ||
                         argv1 == "v" || argv1 == "-version" || argv1 == "--v")
                    m_command = "version";
                else
                    NOA_LOG_ERROR("\"{}\" is not a registered command. "
                                  "Add it with ::Noa::InputManager::setCommand", argv1);
            } else {
                m_command = m_cmdline[1];
            }
            m_registered_commands = std::forward<T>(commands);
            return m_command;
        }


        /**
         * Register the available options that will be used for the parsing. This should correspond
         * to the command returned by setCommand().
         * @tparam T            Mainly used to handle lvalue and rvalue references in one function.
         *                      Really, it must be a @c std::vector<std::string>.
         * @param[in] options   Option(s) to register. If options are already registered, overwrite
         *                      with these ones. Each option takes 5 strings in the input vector.
         *                      See OptionUsage.
         *
         * @note                If there's a duplicate between (long|short)-names, this will likely
         *                      result in an usage type error when retrieving the option with get().
         *                      Because this is on the programmer side and the user has no access to
         *                      this, there's no duplicate check.
         *                      TL;DR: This is the programmer's job to make sure the input options
         *                      don't contain duplicates.
         *
         * @throw ErrorCore     If the command is not set.
         *                      If the size of @c options is not multiple of 5.
         */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<::Noa::Traits::is_same_v<T, std::vector<std::string>>>>
        void setOption(T&& options) {
            if (m_command.empty())
                NOA_LOG_ERROR("the command is not set. Set it first with setCommand()");
            else if (options.size() % 5)
                NOA_LOG_ERROR("the size of the options vector should be a multiple of 5, "
                              "got {} element(s)", options.size());
            m_registered_options = std::forward<T>(options);
        }


        /**
         * Parse the command line options and the parameter file if there's one.
         * @return          Whether or not the parsing was complete. If not, it means the user has
         *                  asked for help and that the program should exit.
         * @throw ErrorCore If the command line or parameter file don't have a supported format.
         */
        [[nodiscard]] bool parse();


        /**
         * Get the option value assigned to a given long-name.
         * @tparam T            Returned type. The original value(s) (which are strings) have to be
         *                      convertible to this type.
         * @tparam[in] N        Number of expected values. It should be >= 0. If 0, it indicates that
         *                      an unknown range of values are to be expected. In this case, T must
         *                      be a vector and positional default values aren't allowed. If >0,
         *                      N entries are expected from the user and from the default values.
         * @param[in] long_name Long-name of the option (without the dash(es) or the prefix).
         * @return              Formatted value(s).
         *
         * @throw ErrorCore     If the parsing wasn't done or completed, if one value is missing or
         *                      if one string cannot be converted into the returned type T.
         */
        template<typename T, size_t N = 1>
        auto get(const std::string& long_name) {
            static_assert(N >= 0 && N < 10);

            if (!m_parsing_is_complete)
                NOA_LOG_ERROR("you cannot retrieve values because the parsing was not completed");

            auto[u_short, u_type, u_value] = getOption_(long_name);
            assertType_<T, N>(*u_type);
            const std::string* value = getParsedValue_(long_name, *u_short);
            if (!value) /* the user didn't enter this option, so use default */
                value = u_value;

            T output;
            if constexpr(N == 0) {
                static_assert(Traits::is_vector_of_bool_v<T> ||
                              Traits::is_vector_of_string_v<T> ||
                              Traits::is_vector_of_scalar_v<T>);
                // When an unknown number of value is expected, values cannot be defaulted
                // based on their position. Thus, let parse() try to convert the raw value.
                if (uint8_t err = String::parse(*value, output))
                    NOA_LOG_ERROR(getErrorMessage_(long_name, value, N, err));

            } else if constexpr (N == 1) {
                static_assert(Traits::is_bool_v<T> ||
                              Traits::is_string_v<T> ||
                              Traits::is_scalar_v<T>);
                std::array<T, 1> tmp;
                if (uint8_t err = String::parse(*value, tmp))
                    NOA_LOG_ERROR(getErrorMessage_(long_name, value, N, err));
                output = std::move(tmp[0]);

            } else /* N > 1 */{
                static_assert(Traits::is_sequence_of_bool_v<T> ||
                              Traits::is_sequence_of_string_v<T> ||
                              Traits::is_sequence_of_scalar_v<T>);
                uint8_t err{0};
                // Option is not entered or no defaults.
                // In both cases, we can only rely on what is in "value".
                if (u_value == value || u_value->empty()) {
                    if constexpr (Traits::is_vector_v<T>) {
                        output.reserve(N);
                        err = String::parse(*value, output);
                        if (!err && output.size() != N)
                            err = Errno::invalid_size;
                    } else /* std::array */ {
                        static_assert(output.size() == N);
                        err = String::parse(*value, output);
                    }
                } else /* Using user values + defaults */ {
                    err = String::parse(*value, *u_value, output);
                }
                if (err)
                    NOA_LOG_ERROR(getErrorMessage_(long_name, value, N, err));
            }
            if constexpr (Traits::is_string_v<T>) {
                if (output.empty())
                    NOA_LOG_ERROR(getErrorMessage_(long_name, value, N, Errno::invalid_argument));
            } else if constexpr (Traits::is_sequence_of_string_v<T>) {
                for (auto& str: output) {
                    if (str.empty())
                        NOA_LOG_ERROR(
                                getErrorMessage_(long_name, value, N, Errno::invalid_argument));
                }
            }
            NOA_LOG_TRACE("{} ({}): {}", long_name, *u_short, output);
            return output;
        }


        /** Prints the registered commands in a docstring format. */
        void printCommand() const;


        /** Prints the NOA version. */
        static inline void printVersion() { fmt::print("{}\n", NOA_VERSION); }


        /** Prints the registered options in a docstring format. */
        void printOption() const;

    private:
        /**
         * Try to print a meaningful error message for get().
         * @param l_name    Long-name.
         * @param value     Parsed value (string).
         * @param N         Number of elements.
         * @param err       Error found by get().
         * @return          Error message.
         */
        std::string getErrorMessage_(const std::string& l_name,
                                     const std::string* value,
                                     size_t nb,
                                     uint8_t err) const;


        /**
         * Convert the usage type into something readable for the user.
         * @param[in] usage_type    Usage type. See assertType_() for more details.
         * @return                  Formatted type
         * @throw ErrorCore         If the usage type isn't recognized.
         */
        static std::string formatType_(const std::string& usage_type);


        /**
         * Make sure the usage type matches the excepted type and number of values.
         * @tparam[in] T        Desired type.
         * @tparam[in] N        Number of values of type @c T.
         * @param usage_type    Usage type. It must be a 2 characters string, such as:
         *                      -# 1st character: 0 (range), 1, 2, 3, 4, 5, 6, 7, 8, or 9.
         *                      -# 2nd character: I, U, F, S or B, corresponding to integers,
         *                                        unsigned integers, floating points, strings
         *                                        and booleans, respectively.
         * @throw ErrorCore     If the usage type doesn't match @c T or @c N.
         * @note These are checks that could be done at compile time. C++20 hello.
         */
        template<typename T, size_t N>
        static void assertType_(const std::string& usage_type) {
            if (usage_type.size() != 2)
                NOA_LOG_ERROR("type usage \"{}\" is not recognized. It should be a "
                              "string with 2 characters", usage_type);


            // Number of values.
            if constexpr(N >= 0 && N < 10) {
                if (usage_type[0] != N + '0')
                    NOA_LOG_ERROR("the type usage \"{}\" does not correspond to the desired "
                                  "number of values: {}", usage_type, N);
            } else {
                NOA_LOG_ERROR("the type usage \"{}\" is not recognized. "
                              "N should be a number from  0 to 9", usage_type);
            }

            // Types.
            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
                if (usage_type[1] != 'F')
                    NOA_LOG_ERROR("the type usage \"{}\" does not correspond to the desired "
                                  "type (floating point)", usage_type);

            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
                if (usage_type[1] != 'I')
                    NOA_LOG_ERROR("the type usage \"{}\" does not correspond to the desired "
                                  "type (integer)", usage_type);

            } else if constexpr(Traits::is_unsigned_v<T> || Traits::is_sequence_of_unsigned_v<T>) {
                if (usage_type[1] != 'U')
                    NOA_LOG_ERROR("the type usage \"{}\" does not correspond to the desired "
                                  "type (unsigned integer)", usage_type);

            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
                if (usage_type[1] != 'B')
                    NOA_LOG_ERROR("the type usage \"{}\" does not correspond to the desired "
                                  "type (boolean)", usage_type);

            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
                if (usage_type[1] != 'S')
                    NOA_LOG_ERROR("the type usage \"{}\" does not correspond to the desired "
                                  "type (string)", usage_type);

            } else {
                NOA_LOG_ERROR("the type usage \"{}\" is not recognized.", usage_type);
            }
        }


        /**
         * Parse the sequence of options and values from the cmd line.
         * @c m_parsing_is_complete will be set to true if the parsing was complete.
         * @throw ErrorCore If the command line does not have the expected format or if an option
         *                  is not recognized.
         *
         * @details When entered at the command line, options must be prefixed by one or
         *          two dashes (- or --) and must be followed by a space _and_ a value (options
         *          without values are no supported). The names are case-sensitive. Options cannot
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
        void parseCommandLine_();


        /**
         * Parse the parameter file (registered by parseCommandLine_()).
         * @throw ErrorCore If the parameter file does not have a supported format.
         *
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
        void parseParameterFile_();


        /**
         * Extract the parsed values of a given option.
         * @param[in] long_name     Long-name of the option.
         * @param[in] short_name    Short-name of the option.
         * @return                  The parsed value(s). If the option is not known (i.e. it wasn't
         *                          registered) or if it was simply not found during the parsing, a
         *                          @c nullprt is returned.
         *
         * @note                    Options can be entered only once, but the same option
         *                          can be entered in the command line and in the parameter file.
         *                          In this case, the command line takes the precedence over the
         *                          parameter file.
         *
         * @throw ErrorCore         If @c long_name and @c short_name were both entered in the command
         *                          line or in the parameter file.
         */
        std::string* getParsedValue_(const std::string& long_name, const std::string& short_name);


        /**
         * Extract the corresponding short-name, usage type and default value(s) for a given long-name.
         * @param long_name[in] Long-name of the wanted option.
         * @return              @c {short_name}, @c {usage_type}, @c {default_values(s)}
         *
         * @throw ErrorCore     If the long-name doesn't match any of the registered options.
         */
        std::tuple<const std::string*, const std::string*, const std::string*>
        getOption_(const std::string& long_name) const;


        /**
         * @param[in] name  (long|short)-name to test.
         * @return          Whether or not the @c name was registered as the (long|short)-name
         *                  of an option with @c setOption().
         */
        inline bool isOption_(const std::string& name) const;
    };
}
