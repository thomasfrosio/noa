//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    ::Noa::Log::Init("akira.log", "AKIRA", ::Noa::Log::level::basic);

    try {
        std::vector<std::string> cmdline{
                "./exe", "command1", "../../../tests/noa/fixtures/TestInputs_parameter_file.txt"
        };
        std::vector<std::string> options{
                "option1", "opt1", "3S", "", "doc...",
                "option2", "opt2", "4S", "", "doc...",
                "option3", "opt3", "0I", "", "doc...",
                "option4", "opt4", "1S", "", "doc...",
                "option5", "opt5", "1B", "", "doc...",
                "option6", "opt6", "3I", "", "doc...",
                "option7", "opt7", "1B", "", "doc...",
                "option8", "opt8", "1S", "", "doc...",
                "option9", "opt9", "1F", "", "doc...",
                "option10", "opt10", "3F", "", "doc...",
                "option11", "opt11", "1S", "", "doc...",
                "option12", "opt12", "0S", "", "doc...",
                "option13", "opt13", "9F", "", "doc...",

                "option21", "opt21", "3S", "value1, value2, value3", "doc...",
                "option22", "opt22", "4S", "d1,d2,d3,d4", "doc...",
                "option23", "opt23", "3I", "-10,,", "doc...",
                "option24", "opt24", "6B", "1,1,1,1,1,1", "doc...",
                "option25", "opt25", "0S", "file1.txt, file2.txt", "doc...",
                "option26", "opt26", "3F", "123,123.f, -0.234e-3", "doc...",

                "option_with_long_name", "opt_withln", "0F", ",,,.5,.5", "doc...",
                "option_unknown", "opt_unknown", "3B", "n,n,n", "doc..."
        };

        Noa::InputManager im(cmdline);
        auto& cmd1 = im.setCommand({"command1", "doc1"});
        im.setOption(std::move(options));

        if (!im.parse()) {
            im.printOption();
        }
        if (!::Noa::Log::setLevel(im.get<int>("verbosity"))) {
            NOA_APP_ERROR("verbosity should be 0, 1, 2, or 3, got {}",
                          im.get<bool>("verbosity"));
        }

        im.get<std::vector<int>, 0>("option3");
    } catch (const ::Noa::Error& e) {
        return EXIT_FAILURE;
    }



//
//    const std::string& cmd = input_manager.setCommand(
//            {"fft", "Fast Fourier Transform related methods",
//             "transform", "Linear transformation (scale, rotate, translate) related methods",
//             "dimension", "Pad and/ crop a volume or an image"}
//    );
//
//    try {
//        if (cmd == "fft") {
//            // do something...
//        } else if (cmd == "transform") {
//
//        } else if (cmd == "dimension") {
//
//        } else if (cmd == "--help") {
//            input_manager.printCommand();
//        } else if (cmd == "--version") {
//            ::Noa::InputManager::printVersion();
//        }
//        return EXIT_SUCCESS;
//
//    } catch (const ::Noa::Error& e) {
//        return EXIT_FAILURE;
//    }
}
