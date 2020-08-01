/**
 * @file Usage.h
 * @brief Store and format the usage help strings.
 * @author Thomas - ffyr2w
 * @date 30 Jul 2020
 */

#pragma once

#include "Core.h"

namespace Noa {
    class Usage {
    private:
        const size_t m_usage_description_padding = 15;

        const std::string m_usage_header = fmt::format(
                "Welcome to noa.\n"
                "Version {}\n"
                "Website: {}\n\n"
                "Usage:\n"
                "     noa program [global options]\n"
                "     noa program [program options...]\n"
                "     noa program parameter_file [program options...]\n\n",
                NOA_VERSION_LONG, NOA_WEBSITE);

        const std::string m_usage_footer = fmt::format(
                "\nGlobal options:\n"
                "   --help, -h      Show global or program help.\n"
                "   --version, -v   Show the version.\n");

        std::string m_usage_programs = "Programs:\n";

    public:
        void setPrograms(std::vector<std::string> a_prog_usage) {
            if (a_prog_usage.size() % 2) {
                NOA_CORE_ERROR("The program usage should have a multiple of 2 elements, got {}",
                               a_prog_usage.size());
            }
            for (unsigned int i{0}; i < a_prog_usage.size(); i += 2) {
                m_usage_programs += fmt::format("     {:<{}} {}\n",
                                                a_prog_usage[i],
                                                m_usage_description_padding,
                                                a_prog_usage[i + 1]);
            }
        }

        void printGlobalHelp() const {
            fmt::print(m_usage_header);
            fmt::print(m_usage_programs);
            fmt::print(m_usage_footer);
        }

        /**
         *
         * @param a_usage   longname, shortname, type, default, description.
         * type = SI, PI, TI, AI - SF, PF, TF, AF - SB, PB, TB, AB - SS, PS, TS, AS
         *
         * --{}, -{}   (3 floats = 30.0,30,30)   Description of this options.\n
         * --{}, -{}   (1 integer)               Description of this options.\n
         * "{:<{}}"
         */
        void printProgramHelp(const std::string& a_program,
                              const std::vector<std::string>& a_usage) const {
            fmt::print(m_usage_header);
            fmt::print("{} options:\n", a_program);

            // Get the first necessary padding.
            size_t option_names_padding{0};
            for (unsigned int i = 0; i < a_usage.size(); i += 5) {
                size_t current_size = a_usage[i].size() + a_usage[i + 1].size();
                if (current_size > option_names_padding)
                    option_names_padding = current_size;
            }
            option_names_padding += 10;

            std::string type;
            for (unsigned int i = 0; i < a_usage.size(); i += 5) {
                std::string option_names = fmt::format("   --{}, -{}", a_usage[i], a_usage[i + 1]);
                if (a_usage[i + 3].empty())
                    type = fmt::format("({})", formatType(a_usage[i + 2]));
                else
                    type = fmt::format("({} = {})", formatType(a_usage[i + 2]), a_usage[i + 3]);

                fmt::print("{:<{}} {:<{}} {}\n",
                           option_names, option_names_padding, type, 25, a_usage[i + 4]);
            }
            fmt::print(m_usage_footer);
        }

    private:
        static std::string formatType(const std::string& a_type) {
            auto getType = [&]() {
                switch (a_type[1]) {
                    case 'I':
                        return "integer";
                    case 'F':
                        return "float";
                    case 'S':
                        return "string";
                    case 'B':
                        return "bool";
                    default: {
                        NOA_CORE_ERROR("Usage::formatType: usage type ({}) not recognized", a_type);
                    }
                }
            };

            switch (a_type[0]) {
                case 'S':
                    return fmt::format("1 {}", getType());
                case 'P':
                    return fmt::format("2 {}s", getType());
                case 'T':
                    return fmt::format("3 {}s", getType());
                case 'A':
                    return fmt::format("n {}(s)", getType());
                default: {
                    NOA_CORE_ERROR("Usage::formatType: usage type ({}) not recognized", a_type);
                }
            }
        }
    };
}
