//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    ::Noa::Log::Init("akira.log", "AKIRA");

    std::vector<std::string> cmdline{
            "./exe", "command1",
            "--option1", "value1",
            "-opt2", "value1,", "value2,",
            "--option3", "-1", "12.34", "-234,123,12",
            "-opt4", "true,false,1,", "0",
            "-opt5", "123,12,", "2",
            "-option23", "-1",
            "-opt13", "v1,v2,v3,v4,v5",
            "-option43", "false"
    };

    ::Noa::InputManager input_manager(cmdline, "ak_");
    auto& cmd1 = input_manager.setCommand({"command1", "doc1"});

    try {
        input_manager.setOption({"option1", "opt1", "SS", "", "doc1",
                                 "option2", "opt2", "TS", "v1,v2,v3", "doc2",
                                 "option3", "opt3", "AF", "", "doc3",
                                 "option4", "opt4", "AB", "", "doc4",
                                 "option5", "opt5", "TI", "1,1,1", "doc5",
                                 "option23", "opt23", "AI", "", "doc...",
                                 "option13", "opt13", "AS", "", "doc...",
                                 "option43", "opt43", "AB", "", "doc..."
                                });
        if (!input_manager.parse())
            NOA_APP_ERROR("parsing did not complete");

        auto v1 = input_manager.get<std::string>("option1");
        auto v2 = input_manager.get<std::vector<std::string>, 3>("option2");
        auto v3 = input_manager.get<std::vector<float>, -1>("option3");
        auto v4 = input_manager.get<std::vector<bool>, -1>("option4");
        auto v5 = input_manager.get<std::array<int, 3>, 3>("option5");
        auto v6 = input_manager.get<std::array<long long, 1>, -1>("option23");
        auto v7 = input_manager.get<std::vector<std::string>, -1>("option13");
        auto v8 = input_manager.get<std::vector<bool>, -1>("option43");

        int a;
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
