//
// Created by thomas on 06/12/2020.
//

#include <noa/Base.h>
#include <noa/util/Vectors.h>


int main(int argc, const char** argv) {
    using namespace Noa;

    try {
        Log::init("sandbox.log", Log::Level::alert);
        fmt::print("{}", Int3<int>{1,2,3});

        return EXIT_SUCCESS;
    } catch (const Noa::Error& e) {
        e.print();
        return EXIT_FAILURE;
    }
}
