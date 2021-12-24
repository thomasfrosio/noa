// This is the entry point to ALL benchmarks.
// Use --benchmark_filter=<regex> to run specific benchmarks.

#include <iostream>

#include <noa/Session.h>
#include <benchmark/benchmark.h>

namespace benchmark {
    noa::path_t PATH_NOA_DATA;
}

int main(int argc, char** argv) {
    const char* path = std::getenv("PATH_NOA_DATA");
    if (path == nullptr) {
        std::cerr << "The environmental variable \"PATH_NOA_DATA\" is empty. "
                     "Set it to the path of the noa-data repository and try again.\n";
        return EXIT_FAILURE;
    }
    ::benchmark::PATH_NOA_DATA = path;
    ::benchmark::PATH_NOA_DATA /= "assets";

    ::noa::Session bench_session("BM", {}, ::noa::Logger::SILENT);
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
