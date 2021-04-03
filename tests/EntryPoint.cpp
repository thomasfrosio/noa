// This is the entry point to ALL tests.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "noa/Session.h"

int main(int argc, char* argv[]) {
    Catch::Session catch_session; // There must be exactly one instance

    int returnCode = catch_session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    Noa::Session noa_session("tests", "tests.log", Noa::Logger::SILENT);
    int numFailed = catch_session.run();
    std::filesystem::remove("tests.log");
    return numFailed;
}
