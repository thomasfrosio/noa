// This is the entry point to ALL benchmarks.
// Use --benchmark_filter=<regex> to run specific benchmarks.

#include <iostream>

#include <noa/Session.h>
#include <benchmark/benchmark.h>

namespace benchmark {
    noa::path_t NOA_DATA_PATH;
}

int main(int argc, char** argv) {
    const char* path = std::getenv("NOA_DATA_PATH");
    if (path == nullptr) {
        std::cerr << "The environmental variable \"NOA_DATA_PATH\" is empty. "
                     "Set it to the path of the noa-data repository and try again.\n";
        return EXIT_FAILURE;
    }
    ::benchmark::NOA_DATA_PATH = path;
    ::benchmark::NOA_DATA_PATH /= "assets";

    ::noa::Session bench_session("BM", {}, ::noa::Logger::SILENT);
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
