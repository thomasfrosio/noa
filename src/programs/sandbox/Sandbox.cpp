//
// Created by thomas on 06/12/2020.
//

#include <noa/Base.h>
#include <noa/structures/Vectors.h>

int main(int argc, const char** argv) {
    using namespace Noa;

    try {
        Log::init("sandbox.log", Log::Level::alert);

        Int2<int> a(1);

    } catch (const Noa::Error& e) {
        e.print();
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
