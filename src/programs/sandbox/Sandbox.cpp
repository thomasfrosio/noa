//
// Created by thomas on 06/12/2020.
//


#include <iostream>
#include "noa/gpu/cuda/Device.h"

int main(int argc, const char** argv) {
    using namespace Noa;

    try {
        Log::init("sandbox.log");
        std::cout << GPU::Device::getSummary(GPU::Device::getCurrent());

        return EXIT_SUCCESS;
    } catch (const Noa::Exception& e) {
        return EXIT_FAILURE;
    }
}
