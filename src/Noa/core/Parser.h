#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cctype>
#include <unordered_map>
#include <memory>

#include "Options.h"
#include "String.h"

namespace Noa {

    class Parser {

        // METHODS
    public:
        // Parse the command line and store options in m_options_cmdline.
        explicit Parser(int argc, char* argv[]);

        // Parse the parameter file and store options in m_options_parameter_file.
        void parseParameterFile(const char* a_path);

        // Parse line from a parameter file: noa_* = value1, value2, etc...
        std::pair<std::string, std::vector<std::string>> parseParameterLine(const std::string& a_str);


        // ATTRIBUTES
    public:
        std::unordered_map<std::string, std::vector<std::string>> m_options_cmdline;
        std::unordered_map<std::string, std::vector<std::string>> m_options_parameter_file;

    public:
        std::string program;
        bool has_parameter_file{false};
        bool has_asked_help{false};
    };

}
