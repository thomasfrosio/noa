/**
 * @file input.h
 * @brief Input manager - Manages user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */

#pragma once

#include "../Base.h"
#include "../utils/String.h"
#include "../utils/Assert.h"
#include "../utils/Helper.h"

#include <cstring>  // std::strerror
#include <cerrno>   // errno


namespace Noa {

    /**
     * @brief       Input manager
     * @details     Parse the command line and the parameter file (if any) and makes
     *              the inputs accessible for the application.
     *
     * Supported scenarios:
     *      - 1 `[./app]` or `[./app] [-h|v]`
     *      - 2 `[./app] [command]` or `[./app] [command] [-h]`
     *      - 3 `[./app] [command] [--options]`
     *      - 4 `[./app] [command] [file] [--options]`
     *
     * - Scenario 1: `setCommand()` returns "--help" or "--version". Parsing the options with
     *               `parse()` or retrieving values with `get()` will not be possible.
     * - Scenario 2: `setCommand()` returns [command] and `parse()` returns true (i.e. help
     *               was asked for this [command]). Retrieving values with `get()` will not be possible.
     * - Scenario 3: `setCommand()` returns [command] and the [--options] are parsed and
     *               accessible using `get()`. If a [-h] is found within [--options], the parsing
     *               stops and `parse()` returns true.
     * - Scenario 4: [file] is parsed and its options can be accessed with `get()` as well.
     *               Otherwise, it is like scenario 3.
     *
     * @see         `InputManager()` to initialize the manager.
     * @see         `setCommand()` to register commands and get the actual command.
     * @see         `setOption()` to register options.
     * @see         `parse()` to parse the options in the cmd line and parameter file.
     * @see         `get()` to retrieve the formatted inputs.
     */
    class InputManager {
    private:
        const int m_argc;
        const char** m_argv;
        std::string m_prefix;

        std::vector<std::string> m_available_commands{};
        std::vector<std::string> m_available_options{};

        std::string m_command{};
        std::string m_parameter_file{};

        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline{};
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file{};

        /**
         * @brief   Options usage. See `::Noa::InputManager::setOption`.
         * @defail  The option vector should be a multiple of 5, such as:
         *          1. `e_long_name`:       long-name of the option.
         *          2. `e_short_name`:      short-name of the option.
         *          3. `e_type`:            expected type of the option.
         *          4. `e_default_value`:   default value(s). See `::Noa::InputManager::get`.
         *          5. `e_help`:            docstring displayed with the "--help" command.
         */
        enum m_option_usage : unsigned int {
            e_long_name, e_short_name, e_type, e_default_value, e_help
        };

        bool m_is_parsed{false};

        const std::string m_usage_header = fmt::format(
                FMT_COMPILE("Welcome to noa.\n"
                            "Version {} - compiled on {}\n"
                            "Website: {}\n\n"
                            "Usage:\n"
                            "     noa [global options]\n"
                            "     noa command [command options...]\n"
                            "     noa command parameter_file [command options...]\n\n"),
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
                : m_argc(argc), m_argv(argv), m_prefix(std::move(prefix)) {}


        /**
         * @brief                   Register the allowed commands and set the actual command.
         *
         * @tparam[in] T            Mainly used to handle lvalue and rvalue references in one
         *                          "method". It should be a std::vector<std::string>.
         * @param[in] commands      Command(s) to register. Each command takes two strings:
         *                          {"<command name>", "<docstring>", ...}
         * @return                  Const ref of the actual command that is registered in the save
         *                          command line. It is either "--version", "--help" or one of the
         *                          commands in the commands vector.
         *
         * @throw ::Noa::ErrorCore  The actual command has to be registered (i.e. it must be
         *                          one of the commands vector). Moreover, the size of the
         *                          commands vector must be a multiple of 2. Otherwise throw
         *                          the exception.
         */
        template<typename T = std::vector<std::string>>
        const std::string& setCommand(T&& commands) {
            static_assert(std::is_same_v<std::remove_reference_t<T>, std::vector<std::string>>);
            if (commands.size() % 2) {
                NOA_CORE_ERROR("the size of the command vector should "
                               "be a multiple of 2, got {} element(s)", commands.size());
            }
            m_available_commands = std::forward<T>(commands);
            parseCommand();
            return m_command;
        }


        /** @brief Prints the registered commands in a docstring format. */
        void printCommand() const;


        /** @brief Prints the registered commands in a docstring format. */
        static inline void printVersion() { fmt::print(NOA_VERSION); }


        /**
         * @brief                   Register the available options that will be used for the parsing.
         *                          This should correspond to the actual command.
         * @tparam[in] T            Mainly used to handle lvalue and rvalue references in one
         *                          "method". It should be a std::vector<std::string>.
         * @param[in] options       Option(s) to register. Each option takes 5 strings:
         *                          See `::Noa::InputManager::m_option_usage`.
         *
         * @throw ::Noa::ErrorCore  The actual command has to be set already. Moreover, the size of
         *                          the options vector must be a multiple of 5. Otherwise throw
         *                          the exception.
         */
        template<typename T = std::vector<std::string>>
        void setOption(T&& options) {
            static_assert(std::is_same_v<std::remove_reference_t<T>, std::vector<std::string>>);
            if (m_command.empty()) {
                NOA_CORE_ERROR("the command is not set. "
                               "Set it first with ::Noa::InputManager::registerCommand");
            } else if (options.size() % 5) {
                NOA_CORE_ERROR("the size of the options vector should "
                               "be a multiple of 5, got {} element(s)", options.size());
            }
            m_available_options = std::forward<T>(options);
        }


        /** @brief Prints the registered commands in a docstring format. */
        void printOption() const;


        /**
         * @brief                   Parse the command line options and the parameter file if there's one.
         * @return                  Whether or not the user has asked for help, i.e. "--help" or
         *                          a variant of it ("-h", etc.) was found in the command line.
         *
         * @warning                 If the returned value is true, it means that the parsing was
         *                          likely interrupted and the parameter file was not parsed at all.
         *                          As such, the input manager will not validate the parsing and
         *                          getting values with the get() method will not be possible.
         *                          TL;DR: if the return value is true, the program should print
         *                                 the help and exit.
         *
         * @throw ::Noa::ErrorCore  If the command line or parameter file don't have the expected format.
         */
        [[nodiscard]] bool parse();


        /**
         * @brief                   Get the option value assigned to a given long-name.
         * @tparam[in,out] T        Returned type. The original value(s) (which are strings) have
         *                          to be convertible to this type.
         * @tparam[in] N            Number of expected values. It should be a positive int, or
         *                          -1 which would indicates that an unknown range of integers
         *                          are to be expected. 0 is not allowed.
         * @tparam[in] verbose      Whether or not the output key should be stdout.
         * @param[in] long_name     Long-name of the option (without the dash(es) or the prefix).
         * @return                  Formatted value(s).
         *
         * @throw ::Noa::ErrorCore  Many things can cause this to throw an exception.
         *                          Cf. source code.
         */
        template<typename T, int N = 1, bool verbose = true>
        auto get(const std::string& long_name) {
            NOA_CORE_DEBUG(__PRETTY_FUNCTION__);
            static_assert(N != 0);
            static_assert(Traits::is_sequence_v<T> ||
                          (Traits::is_arith_v<T> && N == 1) ||
                          (Traits::is_bool_v<T> && N == 1) ||
                          (Traits::is_string_v<T> && N == 1));

            if (!m_is_parsed) {
                NOA_CORE_ERROR("the inputs are not parsed yet. Parse them "
                               "by calling InputManager::parse()");
            }

            // Get usage and the value(s).
            auto[usage_short, usage_type, usage_value] = getOption(long_name);
            assertType<T, N>(usage_type);
            std::vector<std::string>* value = getParsedValue(long_name, usage_short);

            // Parse the default value.
            std::vector<std::string> default_value = String::parse(usage_value);
            if (N != -1 && default_value.size() != N) {
                NOA_CORE_ERROR("Number of default value(s) ({}) doesn't match the desired "
                               "number of value(s) ({})", default_value.size(), N);
            }

            // If option not registered or left empty, replace with the default.
            if (!value || value->empty()) {
                if (usage_value.empty()) {
                    NOA_CORE_ERROR("No value available for option {} ({})", long_name,
                                   usage_short);
                }
                value = &default_value;
            }

            std::remove_reference_t<std::remove_cv_t<T>> output;
            if constexpr(N == -1) {
                // When an unknown number of value is expected, values cannot be defaulted
                // based on their position. Thus, empty strings are not allowed here.
                if constexpr (Traits::is_sequence_of_bool_v<T>) {
                    output = String::toBool<T>(*value);
                } else if constexpr (Traits::is_sequence_of_int_v<T>) {
                    output = String::toInt<T>(*value);
                } else if constexpr (Traits::is_sequence_of_float_v<T>) {
                    output = String::toFloat<T>(*value);
                } else if constexpr (Traits::is_sequence_of_string_v<T>) {
                    output = *value;
                }
            } else if constexpr (N == 1) {
                // If empty or empty string, take default. Otherwise try to convert.
                if (value->size() != 1) {
                    NOA_CORE_ERROR("{} ({}): only 1 value is expected, got {}",
                                   long_name, usage_short, value->size());
                }
                auto& chosen_value = ((*value)[0].empty()) ?
                                     default_value[0] : (*value)[0];
                if constexpr (Traits::is_bool_v<T>) {
                    output = String::toBool(chosen_value);
                } else if constexpr (Traits::is_int_v<T>) {
                    output = String::toInt(chosen_value);
                } else if constexpr (Traits::is_float_v<T>) {
                    output = String::toFloat(chosen_value);
                } else if constexpr (Traits::is_string_v<T>) {
                    output = chosen_value;
                }
            } else {
                // Fixed range.
                if (value->size() != N) {
                    NOA_CORE_ERROR("{} ({}): {} values are expected, got {}",
                                   long_name, usage_short, N, value->size());
                }

                if constexpr (Traits::is_vector_v<T>)
                    output.reserve(value->size());
                for (size_t i{0}; i < value->size(); ++i) {
                    auto& chosen_value = ((*value)[i].empty()) ?
                                         default_value[i] : (*value)[i];
                    if constexpr (Traits::is_sequence_of_bool_v<T>) {
                        Helper::sequenceAssign(output, String::toBool(chosen_value), i);
                    } else if constexpr (Traits::is_sequence_of_int_v<T>) {
                        Helper::sequenceAssign(output, String::toInt(chosen_value), i);
                    } else if constexpr (Traits::is_sequence_of_float_v<T>) {
                        Helper::sequenceAssign(output, String::toFloat(chosen_value), i);
                    } else if constexpr (Traits::is_sequence_of_string_v<T>) {
                        Helper::sequenceAssign(output, chosen_value, i);
                    }
                }
            }

            if constexpr (verbose)
                NOA_CORE_TRACE("{} ({}): {}", long_name, usage_short, output);
            return output;
        }

    private:
        /**
         * @brief                   Convert the usage type into something readable for the user.
         * @param[in] usage_type    Usage type. See `::Noa::InputManager::assertType` for more details.
         * @return                  Formatted type
         *
         * @throw ::Noa::ErrorCore  If the usage type isn't recognized.
         */
        static std::string formatType(const std::string& usage_type);


        /**
         * @brief                   Make sure the usage type matches the excepted type and number
         *                          of values.
         * @tparam[in] T            Expected type.
         * @tparam[in] N            Number of values.
         * @param usage_type        Usage type. It must be a 2 characters string, such as:
         *                          - 1st character: A, S, P or T, corresponding to an array, a single,
         *                                           a pair or a trio of values, respectively.
         *                          - 2nd character: I, F, S or B, corresponding to integers, floating
         *                                           points, strings and booleans, respectively.
         *
         * @note                    The expected type has to be a sequence (std::vector or std::array)
         *                          if the first character is A, P or T, but single values (i.e. S) can
         *                          also be assigned to a sequence.
         *
         * @throw ::Noa::ErrorCore  If the usage type doesn't match the expected type and number of values.
         * @see                     `::Noa::Traits::is_*` type traits.
         *
         * @example
         * @code
         * // These usage types...
         * std::vector<std::string> ut{"SS", "SF", "AI", "TB"};
         * // ... correspond to:
         * ::Noa::InputManager::assertType<std::string, 1>(ut[0]);
         * ::Noa::InputManager::assertType<std::vector<float>, 1>(ut[1]);
         * ::Noa::InputManager::assertType<std::vector<long>, 5>(ut[2]);
         * ::Noa::InputManager::assertType<std::array<bool, 3>, 3>(ut[3]);
         * @endcode
         */
        template<typename T, int N>
        static void assertType(const std::string& usage_type) {
            if (usage_type.size() != 2) {
                NOA_CORE_ERROR(
                        "type usage ({}) not recognized. It should be a string with 2 characters",
                        usage_type);
            }

            // Number of values.
            if constexpr(N == -1) {
                if (usage_type[0] != 'A') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "number of values. For an array (A), N should be -1",
                                   usage_type);
                }
            } else if constexpr(N == 1) {
                if (usage_type[0] != 'S') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "number of values. For a single value (S), N should be 1",
                                   usage_type);
                }
            } else if constexpr(N == 2) {
                if (usage_type[0] != 'P') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "number of values. For a pair of values (P), N should be 2",
                                   usage_type);
                }
            } else if constexpr(N == 3) {
                if (usage_type[0] != 'T') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "number of values. For a trio of values (T), N should be 3",
                                   usage_type);
                }
            }

            // Types.
            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
                if (usage_type[1] != 'F') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "type (floating point)",
                                   usage_type);
                }
            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
                if (usage_type[1] != 'I') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "type (integer)",
                                   usage_type);
                }
            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
                if (usage_type[1] != 'B') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "type (boolean)",
                                   usage_type);
                }
            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
                if (usage_type[1] != 'S') {
                    NOA_CORE_ERROR("the type usage ({}) does not correspond to the expected "
                                   "type (string)",
                                   usage_type);
                }
            }
        }

        /**
         * @brief                   Parse the command from the command line.
         * @throw ::Noa::ErrorCore  If the command is not registered as an available command.
         */
        void parseCommand();


        /**
         * @brief                   Parse the sequence of options and values from the cmd line.
         *
         * @return                  Whether or not the user has asked for help, i.e. "--help" or
         *                          a variant of it ("-h", etc.) was found in the command line.
         *
         * @throw ::Noa::ErrorCore  If the command line doesn't have the expected format.
         *
         * @details When entered at the command line, options must be prefixed by one or
         *          two dashes (- or --) and must be followed by a space _and_ a value. One
         *          dash specifies an option with a short-name and two dashes specify an option
         *          with a long-name. Options cannot be concatenated the way single letter
         *          options in Unix programs often can be. Options without values are not supported.
         *          Arguments that are not prefixed with "--" or '-', but are prefixed
         *          with an option. If the value contains embedded blanks it must be enclosed
         *          in quotes. To specify multiple values for one option, use a comma or
         *          a whitespace, e.g. --size 100,100,100. Commas without values indicates that
         *          the default value for that position should be taken. For example, "12,,15,"
         *          takes the default for the second and fourth values. Defaults can be used
         *          only when a fixed number of values are expected.
         */
        bool parseCommandLine();


        /**
         * @brief                   Parse the registered parameter file.
         *
         * @throw ::Noa::ErrorCore  If the parameter file doesn't have the expected format.
         *
         * @details Options should start at the beginning of a line and be prefixed by `m_prefix`.
         *          The values should be specified after an '=', i.e. option=value. Whitespaces
         *          are ignored before and after the option, '=' and value. Multiple values can
         *          be specified like in the cmd line. Inline comments are allowed and starts
         *          with a '#'.
         */
        void parseParameterFile();


        /**
         * @brief                   Extract the parsed values of a given option.
         * @param[in] longname      Long-name of the option.
         * @param a_shortname       Short-name of the option.
         * @return                  The parsed value(s). These are formatable.
         *                          If the option isn't known (i.e. it wasn't registered) or if
         *                          it was simply not found during the parsing, a nullprt is returned.
         *
         * @note                    The command line takes precedence over the parameter file.
         *                          In other words, if an option is specified in both the command
         *                          line and the parameter file, the values from the parameter file
         *                          are ignored.
         *
         * @throw ::Noa::ErrorCore  If the long-name and the short-name were both found during the
         *                          parsing.
         */
        std::vector<std::string>* getParsedValue(const std::string& long_name,
                                                 const std::string& short_name);


        /**
         * @brief                   Extract for the registered options the short-name, usage type
         *                          and default value(s) for a given long-name.
         * @param long_name         Long-name of the wanted option.
         * @return                  {"<short-name>", "<usage type>", "<default values(s)>"}
         *
         * @throw ::Noa::ErrorCore  If the long-name doesn't match any of the registered options.
         */
        std::tuple<const std::string&, const std::string&, const std::string&>
        getOption(const std::string& long_name) const;
    };
}
