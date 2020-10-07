//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    ::Noa::Log::Init("akira.log", "AKIRA");

    try {
        std::vector<std::string> cmdline{
                "./exe", "command1",
                "--option10", "value1",
                "--option11", "value1,", "value2,value3,value4", ",value5",
                "--option12", "value1,value2,value3",
                "--option13", "v1,", "v2", ",v3,", "v4,v5",

                "--option20", "-1",
                "--option21", "21,21,-1,2",
                "--option22", "-10,10,2",
                "--option23", "-1",

                "--option30", "0.3423", ",", "-0.23", ",", ".13", ",", "0.2e10", ",-.5,0.2",
                "--option31", "2323", "231",
                "--option32", "5.,.5", ".9e8",
                "--option33", "-1,-0.5,0.555,1.5e-9", ",23,", "-232.12",

                "--option40", "0,n,0,y,1,t,1,1,f",
                "--option41", "1,", "true", "1,", "false,0",
                "--option42", "0,0,TRUE",
                "--option43", "False",
        };
        Noa::InputManager im(cmdline);
        auto& cmd1 = im.setCommand({"command1", "doc1"});
        im.setOption({"option10", "opt10", "1S", "", "doc...",
                      "option11", "opt11", "5S", "", "doc...",
                      "option12", "opt12", "3S", "", "doc...",
                      "option13", "opt13", "RS", "", "doc...",

                      "option20", "opt20", "1I", "", "doc...",
                      "option21", "opt21", "4I", "", "doc...",
                      "option22", "opt22", "3I", "", "doc...",
                      "option23", "opt23", "RI", "", "doc...",

                      "option30", "opt30", "6F", "", "doc...",
                      "option31", "opt31", "2F", "", "doc...",
                      "option32", "opt32", "3F", "", "doc...",
                      "option33", "opt33", "RF", "", "doc...",

                      "option40", "opt40", "9B", "", "doc...",
                      "option41", "opt41", "5B", "", "doc...",
                      "option42", "opt42", "3B", "", "doc...",
                      "option43", "opt43", "RB", "", "doc...",
                     });
        im.parse();
        im.get<std::vector<bool>, 9>("option40");

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
