//
//
//

#include "Parser.h"
#include "String.h"


int main(int argc, char** argv) {

    Noa::Parser parser(argc, argv);

   std::cout << "Command Line: \n";
    for (auto& x : parser.m_options_cmdline) {
        std::cout << x.first << " : ";
        for (auto& y : x.second) {
            std::cout << y << " , ";
        }
        std::cout << '\n';
    }

    std::cout << "Parameter File: \n";
    std::cout << "param:" << parser.has_parameter_file << '\n';
    for (auto& x : parser.m_options_cmdline) {
        std::cout << x.first << " : ";
        for (auto& y : x.second) {
            std::cout << y << " , ";
        }
        std::cout << '\n';
    }


    // end
    return 0;
}
