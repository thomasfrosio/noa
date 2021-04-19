// This is the entry point to ALL tests.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "noa/Session.h"

namespace Test {
    Noa::path_t path_archive;
}

// The first
int main(int argc, char* argv[]) {
    Catch::Session catch_session; // There must be exactly one instance

    Test::path_archive = argv[1];
    Test::path_archive /= "archive";

    int returnCode = catch_session.applyCommandLine(argc - 1, argv + 1); // Catch ignores the first parameter.
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    Noa::Session noa_session("tests", "tests.log", Noa::Logger::SILENT);
    int numFailed = catch_session.run();
    std::filesystem::remove("tests.log");
    return numFailed;
}
