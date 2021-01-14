//
// Created by thomas on 06/12/2020.
//

#include <noa/Base.h>
#include <noa/util/files/MRCFile.h>


int main(int argc, const char** argv) {
    using namespace Noa;

    try {
        Log::init("sandbox.log", Log::Level::alert);



        return EXIT_SUCCESS;
    } catch (const Noa::Exception& e) {
        return EXIT_FAILURE;
    }
}
