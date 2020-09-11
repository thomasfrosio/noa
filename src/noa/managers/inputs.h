/**
 * @file InputManager.h
 * @brief Input manager - Manages all user inputs.
 * @author Thomas - ffyr2w
 * @date 31 Jul 2020
 */

#pragma once

#include "noa/noa.h"
#include "noa/utils/String.h"
#include "noa/utils/Assert.h"
#include "noa/utils/Helper.h"


namespace Noa {
    class InputManager {
    private:
        const int m_argc;
        const char** m_argv;

        std::vector<std::string> m_available_commands{};
        std::vector<std::string> m_available_options{};

        std::string command{};
        std::string parameter_file{};

        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline{};
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file{};

        bool is_parsed{false};

        enum Usage : unsigned int {
            u_long_name, u_short_name, u_type, u_default_value, u_help
        };

        const std::string m_usage_header = fmt::format(
                FMT_COMPILE("Welcome to noa.\n"
                            "Version {}\n"
                            "Website: {}\n\n"
                            "Usage:\n"
                            "     noa [global options]\n"
                            "     noa command [command options...]\n"
                            "     noa command parameter_file [command options...]\n\n"),
                getVersion(),
                NOA_URL);

        const std::string m_usage_footer = fmt::format("\nGlobal options:\n"
                                                       "   --help, -h      Show global help.\n"
                                                       "   --version, -v   Show the version.\n");

    public:
        InputManager(const int argc, const char** argv) : m_argc(argc), m_argv(argv) {}

        template<typename T = std::vector<std::string>>
        const std::string& setCommand(T&& a_programs) {
            static_assert(std::is_same_v<std::remove_reference_t<T>, std::vector<std::string>>);
            if (a_programs.size() % 2) {
                NOA_LOG_ERROR("the size of the command vector should "
                          "be a multiple of 2, got {} element(s)", a_programs.size());
            }
            m_available_commands = std::forward<T>(a_programs);
            parseCommand();
            return command;
        }

        void printCommand() const;

        static inline std::string getVersion();

        static inline void printVersion();

        template<typename T = std::vector<std::string>>
        void setOption(T&& a_option) {
            static_assert(std::is_same_v<std::remove_reference_t<T>, std::vector<std::string>>);
            if (command.empty()) {
                NOA_LOG_ERROR("the command is not set. "
                          "Set it first with InputManager::setCommand");
            } else if (a_option.size() % 5) {
                NOA_LOG_ERROR("the size of the option vector should be a "
                          "multiple of 5, got {} element(s)", a_option.size());
            }
            m_available_options = std::forward<T>(a_option);
        }

        void printOption() const;

        [[nodiscard]] bool parse();

        template<typename T, int N = 1>
        auto get(const std::string& a_long_name) {
            NOA_LOG_DEBUG(__PRETTY_FUNCTION__);
            static_assert(N != 0);
            static_assert(Traits::is_sequence_v<T> ||
                          (Traits::is_arith_v<T> && N == 1) ||
                          (Traits::is_bool_v<T> && N == 1) ||
                          (Traits::is_string_v<T> && N == 1));

            if (!is_parsed) {
                NOA_LOG_ERROR("the inputs are not parsed yet. Parse them "
                          "by calling InputManager::parse()");
            }

            // Get usage and the value(s).
            auto[usage_short, usage_type, usage_value] = getOption(a_long_name);
            assertType<T, N>(usage_type);
            std::vector<std::string>* value = getParsedValue(a_long_name, usage_short);

            // Parse the default value.
            std::vector<std::string> default_value = String::parse(usage_value);
            if (N != -1 && default_value.size() != N) {
                NOA_LOG_ERROR("Number of default value(s) ({}) doesn't match the desired "
                          "number of value(s) ({})", default_value.size(), N);
            }

            // If option not registered or left empty, replace with the default.
            if (!value || value->empty()) {
                if (usage_value.empty()) {
                    NOA_LOG_ERROR("No value available for option {} ({})", a_long_name, usage_short);
                }
                value = &default_value;
            }

            std::remove_reference_t<T> output;
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
                    NOA_LOG_ERROR("{} ({}): only 1 value is expected, got {}",
                                  a_long_name, usage_short, value->size());
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
                    NOA_LOG_ERROR("{} ({}): {} values are expected, got {}",
                                  a_long_name, usage_short, N, value->size());
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

            NOA_LOG_TRACE("{} ({}): {}", a_long_name, usage_short, output);
            return output;
        }

    private:
        static std::string formatType(const std::string& a_type);

        template<typename T, int N>
        static void assertType(const std::string& a_usage_type) {
            if (a_usage_type.size() != 2) {
                NOA_LOG_ERROR("type usage ({}) not recognized. It should be a string with 2 characters",
                              a_usage_type);
            }

            // Number of values.
            if constexpr(N == -1) {
                if (a_usage_type[0] != 'A') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "number of values. For an array (A), N should be -1",
                                  a_usage_type);
                }
            } else if constexpr(N == 1) {
                if (a_usage_type[0] != 'S') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "number of values. For a single value (S), N should be 1",
                                  a_usage_type);
                }
            } else if constexpr(N == 2) {
                if (a_usage_type[0] != 'P') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "number of values. For a pair of values (P), N should be 2",
                                  a_usage_type);
                }
            } else if constexpr(N == 3) {
                if (a_usage_type[0] != 'T') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "number of values. For a trio of values (T), N should be 3",
                                  a_usage_type);
                }
            }

            // Types.
            if constexpr(Traits::is_float_v<T> || Traits::is_sequence_of_float_v<T>) {
                if (a_usage_type[1] != 'F') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "type (floating point)",
                                  a_usage_type);
                }
            } else if constexpr(Traits::is_int_v<T> || Traits::is_sequence_of_int_v<T>) {
                if (a_usage_type[1] != 'I') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "type (integer)",
                                  a_usage_type);
                }
            } else if constexpr(Traits::is_bool_v<T> || Traits::is_sequence_of_bool_v<T>) {
                if (a_usage_type[1] != 'B') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "type (boolean)",
                                  a_usage_type);
                }
            } else if constexpr(Traits::is_string_v<T> || Traits::is_sequence_of_string_v<T>) {
                if (a_usage_type[1] != 'S') {
                    NOA_LOG_ERROR("the type usage ({}) does not correspond to the expected "
                              "type (string)",
                                  a_usage_type);
                }
            }
        }

        /**
         *
         */
        void parseCommand();


        /**
         *
         * @return
         */
        bool parseCommandLine();


        /**
         *
         */
        void parseParameterFile();


        /**
         *
         * @param a_longname
         * @param a_shortname
         * @return
         */
        std::vector<std::string>* getParsedValue(const std::string& a_longname,
                                                 const std::string& a_shortname);

        /**
         *
         * @param a_longname
         * @return
         */
        std::tuple<const std::string&, const std::string&, const std::string&>
        getOption(const std::string& a_longname) const;
    };
}

#ifdef NOA_HEADER_ONLY
#include "inputs-inl.h"
#endif
