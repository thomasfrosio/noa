//
// Created by thomas on 11/09/2020.
//

#include "noa/Base.h"
#include "noa/managers/Input.h"
#include "noa/files/Project.h"


int main(int argc, const char** argv) {
    using namespace Noa;
    Log::Init("akira.log", "AKIRA", ::Noa::Log::level::verbose);

    try {
        std::filesystem::path a = "hello.rxr";
        fmt::print(a);

    } catch (const Noa::Error& e) {
        return EXIT_FAILURE;
    }

}
