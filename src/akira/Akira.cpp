//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    ::Noa::Log::Init("akira.log", "AKIRA", true);
    ::Noa::InputManager input_manager(argc, argv, "ak_");

    const std::string& cmd = input_manager.setCommand(
            {"fft", "Fast Fourier Transform related methods",
             "transform", "Linear transformation (scale, rotate, translate) related methods",
             "dimension", "Pad and/ crop a volume or an image"}
    );

    try {
        if (cmd == "fft") {
            // do something...
        } else if (cmd == "transform") {

        } else if (cmd == "dimension") {

        } else if (cmd == "--help") {
            input_manager.printCommand();
        } else if (cmd == "--version") {
            ::Noa::InputManager::printVersion();
        }
        return EXIT_SUCCESS;

    } catch (const ::Noa::Error& e) {
        return EXIT_FAILURE;
    }
}
