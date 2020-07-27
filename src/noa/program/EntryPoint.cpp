/**
 * @file EntryPoint.cpp
 * @brief Entry point for various programs.
 * @author Thomas - ffyr2w
 * @date 19 Jul 2020
 */

#include "../include/Parser.h"


int main(int argc, char** argv) {

    try {
        Noa::Log::Init();
        Noa::Parser parser(argc, argv);

        int a01 = parser.getInteger("parameter1", "param1", 10, 0, 1000);
        std::cout << a01 << '\n';

    } catch (Noa::ReturnMain&) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
