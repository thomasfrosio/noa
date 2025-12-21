// The entry point to ALL tests.

#include <noa/runtime/Session.hpp>

#include <catch2/catch_session.hpp>
#include <filesystem>
#include <iostream>

namespace test {
    std::filesystem::path NOA_DATA_PATH;
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
    // TODO If path not found, remove tests that have the [asset] tag and run the rest?

    const int err = catch_session.applyCommandLine(argc, argv);
    if (err != 0) // Indicates a command line error
        return err;

    noa::Session::set_gpu_lazy_loading();
    noa::Session::set_thread_limit(6);

    return catch_session.run();
}
