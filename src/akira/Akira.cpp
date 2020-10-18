//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Inputs.h"


int main(int argc, const char** argv) {
    using namespace Noa;
    Log::Init("akira.log", "AKIRA", ::Noa::Log::level::basic);

    std::string test1 = "123,,12, 0, \t,, 8";
    std::string test2 = ",1,2,3,4,5,";
    std::vector<double> vec;
    uint8_t err = String::parse(test1, test2, vec);
    for (auto& e: vec)
        std::cout << e << '\n';
    std::cout << static_cast<uint>(err) << '\n';
}
