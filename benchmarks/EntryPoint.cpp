// This is the entry point to ALL benchmarks.
// Use --benchmark_filter=<regex> to run specific benchmarks.

#include <noa/Session.h>
#include <benchmark/benchmark.h>

int main(int argc, char** argv) {
    ::noa::Session bench_session("BM", {}, ::noa::Logger::SILENT);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
