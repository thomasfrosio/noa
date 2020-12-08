// This is the entry point to ALL tests.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>
//#include "Helpers.h"

#include "noa/Base.h"


int main(int argc, char* argv[]) {
    Catch::Session session; // There must be exactly one instance

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    // Initialize the logger here, since it doesn't let me do it within the test cases...
    Noa::Log::init("tests.log", Noa::Log::Level::silent);

    int numFailed = session.run();
    return numFailed;
}
