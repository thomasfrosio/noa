//
// Created by thomas on 06/12/2020.
//

#include <noa/Base.h>
#include <noa/util/Traits.h>


int main(int argc, const char** argv) {
    using namespace Noa;

    try {
        Log::init("sandbox.log", Log::Level::alert);


    } catch (const Noa::Error& e) {
        e.print();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
