//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    using namespace Noa;
    Log::Init("akira.log", "AKIRA", ::Noa::Log::level::basic);

    std::vector<std::string> cmdline{
            "./exe", "my_command_test",
            "-opt10", "v1",
            "--opt11", "v1,",
            "-opt12", "my file.txt", ",", ",", "something else.txt",
            "-opt13", "v1,v2,v3,v4,v5",

            "-opt21", ",21",
            "--opt22", "-10,10,2",
            "-opt23", "-1", ",1,2,-2,-10,1",

            "-opt30", "0.3423",
            "-opt31", "2323,",
            "--opt32", "5.,", ".9e8", ",",

            "-opt40", "0",
            "-opt42", "0,,",
    };
    Manager:: Input im(cmdline);
    im.setCommand({"my_command_test", "doc_test..."});
    im.setOption({"option10", "opt10", "1S", "d1", "doc...",
                  "option11", "opt11", "2S", "d1,d2", "doc...",
                  "option12", "opt12", "3S", "file1.txt, file2.txt,file3.txt", "doc...",
                  "option13", "opt13", "0S", "d1,d2,d3,d4,d5,d6,d7", "doc...",

                  "option20", "opt20", "1I", "-100", "doc...",
                  "option21", "opt21", "2I", "-101,", "doc...",
                  "option22", "opt22", "3I", "1,1,1", "doc...",
                  "option23", "opt23", "0I", "123,123,123", "doc...",

                  "option30", "opt30", "1F", ".55", "doc...",
                  "option31", "opt31", "2F", ",1001", "doc...",
                  "option32", "opt32", "3F", ",1.3,1.4", "doc...",
                  "option33", "opt33", "0F", ".1,.2,.3", "doc...",

                  "option40", "opt40", "1B", "true", "doc...",
                  "option41", "opt41", "2B", "true,true", "doc...",
                  "option42", "opt42", "3B", "1,0,1", "doc...",
                  "option43", "opt43", "0B", "true", "doc...",
                 });
    im.parse();

    im.get<bool>("option40");
    im.get<std::array<bool, 2>, 2>("option41");
    im.get<std::vector<bool>, 3>("option42");
    im.get<std::vector<bool>, 0>("option43");
}
