#include <noa/runtime/Session.hpp>
#include <catch2/catch_session.hpp>

int main(int argc, char* argv[]) {
    Catch::Session catch_session; // There must be exactly one instance
    const int err = catch_session.applyCommandLine(argc, argv);
    if (err != 0) // Indicates a command line error
        return err;

    noa::Session::set_gpu_lazy_loading();
    noa::Session::set_thread_limit(6);

    return catch_session.run();
}
