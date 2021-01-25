//
// Created by thomas on 06/12/2020.
//


#include <iostream>
#include "noa/util/String.h"

int main(int argc, const char** argv) {
    using namespace Noa;

    try {
       std::cout << __cplusplus << '\n';
       std::cout << String::format("blabla");

        return EXIT_SUCCESS;
    } catch (const Noa::Exception& e) {
        return EXIT_FAILURE;
    }
}
