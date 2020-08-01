/**
 * @file EntryPoint.cpp
 * @brief Entry point for various programs.
 * @author Thomas - ffyr2w
 * @date 19 Jul 2020
 */

#include "../include/InputManager.h"


int main(const int argc, const char** argv) {

    try {
        Noa::Log::Init();
        Noa::InputManager input_manager(argc, argv);
        auto& command = input_manager.setCommand(
                {"command1", "Description of the first command.",
                 "command2", "Description of the second command."});

        if (command == "command1") {
            // run command1;
        } else if (command == "command2") {
            // run command2;
        } else if (command == "--help") {
            input_manager.printCommand();
        } else if (command == "--version") {
            fmt::print(FMT_STRING("{}\n"), NOA_VERSION);
        }

    } catch (Noa::ReturnMain&) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
