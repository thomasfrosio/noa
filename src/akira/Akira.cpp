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
        std::cout << sizeof(int64_t);

    } catch (const Noa::Error& e) {
        e.print();
        return EXIT_FAILURE;
    }

}
