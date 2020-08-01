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
        input_manager.setAvailable({"program1", "Description of the first program.",
                                    "program2", "Description of the second program."});
        const std::string& program = input_manager.setProgram();

        if (program == "program1") {
            // run program1;
        } else if (program == "program2") {
            // run program2;
        } else if (program == "--help") {
            input_manager.printAvailable();
        } else if (program == "--version") {
            fmt::print(FMT_STRING("{}\n"), NOA_VERSION);
        }

    } catch (Noa::ReturnMain&) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
