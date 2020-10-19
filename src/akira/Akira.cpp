//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    using namespace Noa;
    Log::Init("akira.log", "AKIRA", ::Noa::Log::level::verbose);

    Manager::Input im(std::vector<std::string>{"./noa", "cmd1",
                                      "--opt3", "1,",
                                      "-opt2", "1,,3",
                                      "-opt7", "1.,2.,3.,",
                                      "--option9", "1,1,1,,,"});
    im.setCommand({"cmd0", "doc0", "cmd1", "doc1"});
    im.setOption({"option1", "opt1", "1I", "", "doc...",
                  "option2", "opt2", "3I", "", "doc...",
                  "option3", "opt3", "0F", "", "doc...",
                  "option4", "opt4", "1S", "", "doc...",
                  "option5", "opt5", "2F", "5,", "doc...",
                  "option6", "opt6", "3B", "", "doc...",
                  "option7", "opt7", "0F", ".1,.2f,-.03,12.4", "doc...",
                  "option8", "opt8", "5S", "v1,v2,,v4,v5", "doc...",
                  "option9", "opt9", "6B", "1,1,false,false,,", "doc...",
                  "option10", "opt10", "7I", "-1,-2,-3,-4,,,", "doc..."});
    im.parse();
    try {
        auto b = im.get<std::array<bool, 6>, 6>("option9");
    } catch (const Noa::Error& e) {
        return EXIT_FAILURE;
    }


}
