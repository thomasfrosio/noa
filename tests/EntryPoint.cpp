// This is the entry point to ALL tests.

#include <noa/Session.h>
#include <noa/common/Types.h>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

namespace test {
    noa::path_t NOA_DATA_PATH;
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

    int returnCode = catch_session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    noa::Session noa_session("tests", {}, noa::Logger::SILENT);
    noa::Session::threads(std::min(noa::Session::threads(), size_t{8}));
    int numFailed = catch_session.run();
    return numFailed;
}
