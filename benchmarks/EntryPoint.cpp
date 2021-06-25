// This is the entry point to ALL tests.

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include <noa/Session.h>

int main(int argc, char* argv[]) {
    Catch::Session session; // There must be exactly one instance

    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    noa::Session bench_session("BENCH", "benchmarks.log", noa::Logger::VERBOSE);

    int numFailed = session.run();
    return numFailed;
}
