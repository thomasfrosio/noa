// This is the entry point to ALL tests.

//#include "noa/unified/Session.hpp"
#include <noa/core/Types.hpp>
#include <noa/core/io/IO.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

namespace test {
    noa::Path NOA_DATA_PATH;
}

int main(int argc, char* argv[]) {
    Catch::Session catch_session; // There must be exactly one instance

    const char* path = std::getenv("NOA_DATA_PATH");
    if (path == nullptr) {
        std::cerr << "The environmental variable \"NOA_DATA_PATH\" is empty. "
                     "Set it to the path of the noa-data repository and try again.\n";
        return EXIT_FAILURE;
    }
    test::NOA_DATA_PATH = path;
    test::NOA_DATA_PATH /= "assets";

    const int returnCode = catch_session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

//    noa::Session::set_cuda_lazy_loading();
//    noa::Session::set_thread_limit(8);

    const int numFailed = catch_session.run();
    return numFailed;
}
