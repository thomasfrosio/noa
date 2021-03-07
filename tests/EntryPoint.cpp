// This is the entry point to ALL tests.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "noa/Log.h"

int main(int argc, char* argv[]) {
    Catch::Session session; // There must be exactly one instance

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    std::string logfile = "tests.log";
    Noa::Log::init(logfile, Noa::Log::SILENT);

    int numFailed = session.run();
    std::filesystem::remove(logfile);
    return numFailed;
}
