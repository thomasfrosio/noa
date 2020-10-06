//
// Created by thomas on 15/09/2020.
//

#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"
#include "noa/Base.h"


int main( int argc, char* argv[] )
{
    Catch::Session session; // There must be exactly one instance

    // writing to session.configData() here sets defaults
    // this is the preferred way to set them

    int returnCode = session.applyCommandLine( argc, argv );
    if( returnCode != 0 ) // Indicates a command line error
        return returnCode;

    // writing to session.configData() or session.Config() here
    // overrides command line args
    // only do this if you know you need to

    // Initialize the Noa logger here, since it doesn't let me do it within the test cases...
    Noa::Log::Init("tests.log", "TESTS", Noa::Log::level::silent);  // silent

    int numFailed = session.run();

    // numFailed is clamped to 255 as some unices only use the lower 8 bits.
    // This clamping has already been applied, so just return it here
    // You can also do any post run clean-up here
    return numFailed;
}
