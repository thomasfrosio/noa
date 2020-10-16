/**
 * @file input.h
 * @brief Input manager - Manages user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */
#pragma once

#include "noa/Base.h"
#include "noa/utils/String.h"
#include "noa/utils/Assert.h"
#include "noa/utils/Helper.h"

#include <cstring>  // std::strerror
#include <cerrno>   // errno
#include <utility>


namespace Noa {

    /**
     * @brief       Input manager
     * @details     Parse the command line and the parameter file (if any) and make
     *              the inputs accessible using this->get().
     *
     * Supported scenarios:
     *      - 1 `[./app] (-h|v)`
     *      - 2 `[./app] [command] (-h)`
     *      - 3 `[./app] [command] (file) ([option1 value1] ...)`
     *
     * - Scenario 1: `setCommand()` returns "--help" or "--version". Parsing the options with
     *               `parse()` or retrieving values with `get()` will not be possible.
     * - Scenario 2: `setCommand()` returns [command] and `parse()` will return true, meaning the user
     *               asked the help of [command]. Retrieving values with `get()` will not be possible.
     * - Scenario 3: `setCommand()` returns [command] and the option/value pairs are parsed and
     *               accessible using `get()`. If an help (i.e. -h, --help) is found, the parsing
     *               is aborted and `parse()` returns true. If there's a parameter file (file),
     *               it is parsed and its options can be retrieved with `get()` as well.
     *
     * @see         `InputManager()` to initialize the input manager.
     * @see         `setCommand()` to register allowed commands and get the actual command.
     * @see         `setOption()` to register options.
     * @see         `parse()` to parse the options in the cmd line and parameter file.
     * @see         `get()` to retrieve the formatted inputs.
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
         * Option usage. See `::Noa::InputManager::setOption`.
         * @defail  The option vector should be a multiple of 5, such as:
         *          1. `long_name`:       long-name of the option.
         *          2. `short_name`:      short-name of the option.
         *          3. `type`:            expected type of the option. See assertType_()
         *          4. `default_value`:   default value(s). See get()
         *          5. `help`:            docstring displayed with the "--help" command.
         */
        struct OptionUsage {
            static constexpr u_int8_t long_name{0};
            static constexpr u_int8_t short_name{1};
            static constexpr u_int8_t type{2};
            static constexpr u_int8_t default_value{3};
            static constexpr u_int8_t help{4};
        };

        bool m_parsing_is_complete{false};

        const std::string m_usage_header = fmt::format(
                FMT_COMPILE("Welcome to noa.\n"
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
         * @brief               Store the command line.
         * @param[in] argc      How many C-strings are contained in argv, i.e. number of arguments.
         *                      Usually comes from main().
         * @param[in] argv      Command line arguments, i.e. stripped and split C-strings.
         *                      Usually comes from main().
         * @param[in] prefix    Prefix of the options specified in the parameter file.
         */
        InputManager(const int argc, const char** argv, std::string prefix = "noa_")
                : m_cmdline(argv, argv + argc), m_prefix(std::move(prefix)) {}


        /// Overload for tests
        template<typename T,
                typename = std::enable_if_t<::Noa::Traits::is_same_v<T, std::vector<std::string>>>>
        explicit InputManager(T&& args, std::string prefix = "noa_")
                : m_cmdline(std::forward<T>(args)), m_prefix(std::move(prefix)) {}


        /**
         * @brief                   Register the allowed commands and set the actual command.
         *
         * @tparam[in] T            Mainly used to handle lvalue and rvalue references in one
         *                          "method". It should be a std::vector<std::string>.
         * @param[in] commands      Command(s) to register. If commands are already registered,
         *                          overwrite with these ones. Can be empty. Each command should take two
         *                          strings in the vector, such as {"<command>", "<docstring>", ...}.
         * @return                  The actual command that is registered in the command line
         *                          It can be "version", "help" or any one of the registered commands.
         *
         * @note                    If the same command is specified twice, everything will work
         *                          as excepted (printCommand() will print both entries though), so
         *                          there's no explicit duplicate check here.
         *
         * @throw ::Noa::ErrorCore  The actual command has to be registered (i.e. it must be
         *                          one of the commands vector). Moreover, the size of the
         *                          commands vector must be a multiple of 2. Otherwise throw
         *                          the exception.
         */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<::Noa::Traits::is_same_v<T, std::vector<std::string>>>>
        const std::string& setCommand(T&& commands) {
            if (commands.size() % 2) {
                NOA_CORE_ERROR("the size of the command vector should "
                               "be a multiple of 2, got {} element(s)", commands.size());
            }
            m_registered_commands = std::forward<T>(commands);
            parseCommand_();
            return m_command;
        }


        /** @brief Prints the registered commands in a docstring format. */
        void printCommand() const;


        /** @brief Prints the registered commands in a docstring format. */
        static inline void printVersion() { fmt::print("{}\n", NOA_VERSION); }


        /**
         * @brief                   Register the available options that will be used for the parsing.
         *                          This should correspond to the command returned by setCommand().
         * @tparam[in] T            Mainly used to handle lvalue and rvalue references in one
         *                          "method". It should be a std::vector<std::string>.
         * @param[in] options       Option(s) to register. If options are already registered,
         *                          overwrite with these ones. Can be empty. Each option takes 5 strings
         *                          in the input vector: see `::Noa::InputManager::OptionUsage`.
         *
         * @note                    If there's a duplicate between (long|short)-names, this will
         *                          likely result in an usage type error when retrieving the option
         *                          with get(). On the other end, this is on the programmer side and
         *                          the user has no access to this, so there's no duplicate check.
         *                          TL;DR: This is the programmer's job to make sure the input options
         *                          don't contain duplicates.
         *
         * @throw ::Noa::ErrorCore  The command has to be set already. Moreover, the size of
         *                          the options vector must be a multiple of 5 and no duplicates
         *                          should be found between the (long|short)-names. Otherwise throw
         *                          an exception.
         */
        template<typename T = std::vector<std::string>,
                typename = std::enable_if_t<::Noa::Traits::is_same_v<T, std::vector<std::string>>>>
        void setOption(T&& options) {
            if (m_command.empty()) {
                NOA_CORE_ERROR("the command is not set. "
                               "Set it first with ::Noa::InputManager::setCommand");
            } else if (options.size() % 5) {
                NOA_CORE_ERROR("the size of the options vector should "
                               "be a multiple of 5, got {} element(s)", options.size());
            }
            m_registered_options = std::forward<T>(options);
        }


        /** @brief Prints the registered commands in a docstring format. */
        void printOption() const;


        /**
         * Parse the command line options and the parameter file if there's one.
         * @return                  Whether or not the parsing was completed. If not, it means the
         *                          user has asked for help and that the program should exit.
         * @throw ::Noa::ErrorCore  If the command line or parameter file don't have the expected format.
         */
        [[nodiscard]] bool parse();


        /**
         * @brief                   Get the option value assigned to a given long-name.
         * @tparam[in,out] T        Returned type. The original value(s) (which are strings) have
         *                          to be convertible to this type.
         * @tparam[in] N            Number of expected values. It should be >= 0.
         *                          If 0, it indicates that an unknown range of integers are to be
         *                          expected. The option can be optional but positional default
         *                          values aren't allowed.
         *                          If >0, N values are expected, either from the user, from the
         *                          default values or a mix of both.
         * @param[in] long_name     Long-name of the option (without the dash(es) or the prefix).
         * @return                  Formatted value(s).
         *
         * @throw ::Noa::ErrorCore  Many things can cause this to throw an exception. Briefly,
         *                          an error will be raised if the parsing wasn't done or completed,
         *                          if one value is missing or if one value cannot be converted into
         *                          the returned type T.
         */
        template<typename T, size_t N = 1>
        auto get(const std::string& long_name) {
            static_assert(N >= 0 && N < 10);

            if (!m_parsing_is_complete) {
                NOA_CORE_ERROR("you cannot retrieve values because the parsing is not completed. "
                               "See InputManager::parse() for more details.");
            }
            uint8_t status{0};

            // Get usage and the value(s).
            auto[u_short, u_type, u_value] = getOption_(long_name);
            assertType_<T, N>(*u_type);
            const std::string* value = getParsedValue_(long_name, *u_short);
            if (!value)
                value = u_value;

            T output;
            if constexpr(N == 0) {
                static_assert(Traits::is_vector_of_bool_v<T> ||
                              Traits::is_vector_of_string_v<T> ||
                              Traits::is_vector_of_scalar_v<T>);
                // When an unknown number of value is expected, values cannot be defaulted
                // based on their position. Thus, let parse try to convert the raw value.
                status = String::parse(value, output);
                if (status)
                    NOA_CORE_ERROR(getErrorMessage_(long_name, *u_short,
                                                    *u_type, *value, status));

            } else if constexpr (N == 1) {
                static_assert(Traits::is_bool_v<T> ||
                              Traits::is_string_v<T> ||
                              Traits::is_scalar_v<T>);
                std::array<T, 1> tmp;
                auto[status, count] = ::Noa::String::parse(value, tmp);
                if (status)
                    NOA_CORE_ERROR(getErrorMessage_(long_name, *u_short,
                                                    *u_type, *value, status));
                output = tmp[0]; // maybe move if string?

            } else /* N > 1 */{
                static_assert(Traits::is_sequence_of_bool_v<T> ||
                              Traits::is_sequence_of_string_v<T> ||
                              Traits::is_sequence_of_scalar_v<T>);
                if (u_value == value) /* Using the default value. */ {
                    if constexpr (Traits::is_vector_v<T>) {
                        output.reserve(N);
                        status = String::parse(u_value, output);
                        if (output.size() != N)
                            status = 1;
                    } else {
                        std::array<T, N> tmp;
                        auto[status, count] = ::Noa::String::parse(value, tmp);
                        if (count != N)
                            status = 1;
                    }
                    if (status) {
                        NOA_CORE_ERROR(getErrorMessage_(long_name, *u_short,
                                                        *u_type, *value, status));
                    }

                } else /* Using user value + default. */ {
                    std::vector<std::string> tmp_value_default;
                    std::vector<std::string> tmp_value;
                    Noa::String::parse(*u_value, tmp_value_default);
                    Noa::String::parse(*value, tmp_value);

                    if (tmp_value.size() != N) {
                        NOA_CORE_ERROR("{} ({}): {} values are expected, got {}",
                                       long_name, *u_short, N, tmp_value.size());
                    } else if (tmp_value_default.size() != N) {
                        NOA_CORE_ERROR("{} ({}): {} default values are expected, got {}",
                                       long_name, *u_short, N, tmp_value_default.size());
                    }

                    // If user didn't enter value, take the default.
                    // If the default it empty, let parse() report error.
                    for (size_t i{0}; i < N; ++i) {
                        if (tmp_value[i].empty()) {
                            tmp_value[i] = std::move(tmp_value_default[i]);
                        }
                    }
                    if constexpr (Traits::is_vector_v<T>)
                        output.reserve();
                    status = String::parse(value, output);
                    if (status)
                        NOA_CORE_ERROR(getErrorMessage_(long_name, *u_short,
                                                        *u_type, *value, status));
                }
            }
            NOA_CORE_TRACE("{} ({}): {}", long_name, *u_short, output);
            return output;
        }

    private:
        static std::string getErrorMessage_(const std::string& l_name,
                                            const std::string& s_name,
                                            const std::string& u_type,
                                            const std::string& value,
                                            uint8_t status) {
            if (status == 1) {
                return fmt::format("{} ({}) contains one element that couldn't be converted "
                                   "into the desired type (i.e. {}): \"{}\"",
                                   l_name, s_name, formatType_(u_type), value);
            } else if (status == 2) {
                return fmt::format("{} ({}) contains one element that was out of range of "
                                   "the desired type (i.e. {}): \"{}\"",
                                   l_name, s_name, formatType_(u_type), value);
            } else {
                return fmt::format("unknown status or reason - please let us know "
                                   "that this happened. name: {}, value: {}",
                                   l_name, value);
            }
        }

        /**
         * @brief                   Convert the usage type into something readable for the user.
         * @param[in] usage_type    Usage type. See `::Noa::InputManager::assertType_` for more details.
         * @return                  Formatted type
         *
         * @throw ::Noa::ErrorCore  If the usage type isn't recognized.
         */
        static std::string formatType_(const std::string& usage_type);


        /**
         * Make sure the usage type matches the excepted type and number of values.
         * @tparam[in] T            Expected type.
         * @tparam[in] N            Number of values.
         * @param usage_type        Usage type. It must be a 2 characters string, such as:
         *                          - 1st character: R (range), 1, 2, 3, 4, 5, 6, 7, 8, or 9.
         *                          - 2nd character: I, F, S or B, corresponding to integers, floating
         *                                           points, strings and booleans, respectively.
         *
         * @throw ::Noa::ErrorCore  If the usage type doesn't match the expected type and number of values.
         *
         * @example
         * @code
         * // These usage types...
         * std::vector<std::string> ut{"1S", "1F", "RI", "3B"};
         * // ... correspond to:
         * ::Noa::InputManager::assertType_<std::string, 1>(ut[0]);
         * ::Noa::InputManager::assertType_<std::vector<float>, 1>(ut[1]);
         * ::Noa::InputManager::assertType_<std::vector<long>, 5>(ut[2]);
         * ::Noa::InputManager::assertType_<std::array<bool, 3>, 3>(ut[3]);
         * @endcode
         */
        template<typename T, size_t N>
        static void assertType_(const std::string& usage_type) {
            if (usage_type.size() != 2) {
                NOA_CORE_ERROR(
                        "type usage \"{}\" not recognized. It should be a string with 2 characters",
                        usage_type);
            }

            // Number of values.
            if constexpr(N >= 0 && N < 10) {
                if (usage_type[0] != N + '0') {
                    NOA_CORE_ERROR("the type usage \"{}\" does not correspond to the expected "
                                   "number of values should be {}", usage_type, N);
                }
            } else {
                NOA_CORE_ERROR("the type usage \"{}\" isn't recognized. "
                               "N should be a number from  0 to 9", usage_type);
            }

            // Types.
            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
                if (usage_type[1] != 'F') {
                    NOA_CORE_ERROR("the type usage \"{}\" does not correspond to the expected "
                                   "type (floating point)",
                                   usage_type);
                }
            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
                if (usage_type[1] != 'I') {
                    NOA_CORE_ERROR("the type usage \"{}\" does not correspond to the expected "
                                   "type (integer)",
                                   usage_type);
                }
            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
                if (usage_type[1] != 'B') {
                    NOA_CORE_ERROR("the type usage \"{}\" does not correspond to the expected "
                                   "type (boolean)",
                                   usage_type);
                }
            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
                if (usage_type[1] != 'S') {
                    NOA_CORE_ERROR("the type usage \"{}\" does not correspond to the expected "
                                   "type (string)",
                                   usage_type);
                }
            } else {
                NOA_CORE_ERROR("the type usage \"{}\" isn't recognized.", usage_type);
            }
        }

        /**
         * @brief                   Parse the command from the command line.
         * @throw ::Noa::ErrorCore  If the command is not registered as an available command.
         */
        void parseCommand_();


        /**
         * Parse the sequence of options and values from the cmd line.
         * m_parsing_is_complete will be set to true if the parsing was complete.
         *
         * @throw ::Noa::ErrorCore  If the command line doesn't have the expected format or if an
         *                          option isn't recognized.
         *
         * @details When entered at the command line, options must be prefixed by one or
         *          two dashes (- or --) and must be followed by a space _and_ a value (options
         *          without values are no supported). The names are case-sensitive. Options cannot
         *          be concatenated the way single letter options in Unix programs often can be.
         *          To specify multiple values for one option, use commas, e.g. --size 100,100,100.
         *          Commas without values indicates that the default value for that position should
         *          be taken if possible. For example, "12,," takes the default for the second and
         *          third values. Position default values can be used only when a fixed number of
         *          values are expected. Options can be entered only once in the command line, but
         *          the same option can be entered in the command line and in the parameter file.
         *          In this case, the command line takes the precedence over the parameter file.
         *          All options should be registered before parsing with setOption().
         */
        void parseCommandLine_();


        /**
         * Parse the registered parameter file.
         * Uses m_parameter_filename.
         *
         * @throw ::Noa::ErrorCore  If the parameter file doesn't have the expected format.
         *
         * @details Options should start at the beginning of a line and be prefixed by `m_prefix`.
         *          The values should be specified after a '=', i.e. <m_prefix><option>=<value>.
         *          Spaces are ignored before and after the <option>, '=' and <value>. Multiple
         *          values can be specified like in the cmd line. Inline comments are allowed and
         *          should start with a '#'. Options can be entered only once in the parameter file,
         *          but the same option can be entered in the command line and in the parameter file.
         *          In this case, the command line takes the precedence over the parameter file.
         *          Options do not have to be registered with setOption(), as opposed to the command
         *          line. This allows more generic parameter file.
         */
        void parseParameterFile_();


        /**
         * @brief                   Extract the parsed values of a given option.
         * @param[in] longname      Long-name of the option.
         * @param[in] shortname       Short-name of the option.
         * @return                  The parsed value(s).
         *                          If the option isn't known (i.e. it wasn't registered) or if
         *                          it was simply not found during the parsing, a nullprt is returned.
         *
         * @note                    Options can be entered only than once, but the same option
         *                          can be entered in the command line and in the parameter file.
         *                          In this case, the command line takes the precedence over the
         *                          parameter file.
         *
         * @throw ::Noa::ErrorCore  If the long-name and the short-name were both found during the
         *                          parsing.
         */
        std::string* getParsedValue_(const std::string& long_name, const std::string& short_name);


        /**
         * @brief                   Extract for the registered options the short-name, usage type
         *                          and default value(s) for a given long-name.
         * @param long_name         Long-name of the wanted option.
         * @return                  {"<short-name>", "<usage type>", "<default values(s)>"}
         *
         * @throw ::Noa::ErrorCore  If the long-name doesn't match any of the registered options.
         */
        std::tuple<const std::string*, const std::string*, const std::string*>
        getOption_(const std::string& long_name) const;


        /**
         * @param[in] name  (long|short)-name to test.
         * @return          Whether or not the input name was registered as the (long|short)-name
         *                  of an option with setOption().
         */
        inline bool isOption_(const std::string& name) const;
    };
}
