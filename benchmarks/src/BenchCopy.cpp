#include <benchmark/benchmark.h>

#include <noa/Array.hpp>
#include <noa/unified/Random.hpp>

using namespace ::noa::types;

namespace {
    constexpr Shape<i64, 4> shapes[]{
        {1, 1, 512, 512},
        {1, 1, 4096, 4096},
        {1, 256, 256, 256},
        {1, 512, 512, 512},
    };

    template<typename T>
    void bench000_memcpy(benchmark::State& state) {
        const auto shape = shapes[state.range(0)];
        StreamGuard stream{Device{}, Stream::DEFAULT};

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape);
        Array dst = noa::like(src);

        for (auto _: state) {
            std::copy_n(src.get(), shape.n_elements(), dst.get());
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void bench000_cpu_copy(benchmark::State& state) {
        const auto shape = shapes[state.range(0)];
        StreamGuard stream{Device{}, Stream::DEFAULT};
        stream.set_thread_limit(1);

        Array src = noa::random<T>(noa::Uniform<T>{-5, 5}, shape);
        Array dst = noa::like(src);

        for (auto _: state) {
            src.to(dst);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(bench000_memcpy, f32)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(bench000_cpu_copy, f32)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
/*
2024-09-29T08:32:36+02:00
Running ./noa_benchmarks
Run on (32 X 4761.23 MHz CPU s)
CPU Caches:
L1 Data 32 KiB (x16)
L1 Instruction 32 KiB (x16)
L2 Unified 512 KiB (x16)
L3 Unified 16384 KiB (x4)
Load Average: 2.61, 3.86, 2.99
***WARNING*** CPU scaling is enabled, the benchmark real time measurements may be noisy and will incur extra overhead.
-----------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations
-----------------------------------------------------------------------------
bench000_memcpy<f32>/0/real_time        0.022 ms        0.022 ms        32949
bench000_memcpy<f32>/1/real_time         3.60 ms         3.60 ms          187
bench000_memcpy<f32>/2/real_time         3.62 ms         3.62 ms          187
bench000_memcpy<f32>/3/real_time         46.4 ms         46.4 ms           12
bench000_cpu_copy<f32>/0/real_time      0.020 ms        0.020 ms        34287
bench000_cpu_copy<f32>/1/real_time       3.62 ms         3.62 ms          189
bench000_cpu_copy<f32>/2/real_time       3.64 ms         3.64 ms          187
bench000_cpu_copy<f32>/3/real_time       46.7 ms         46.6 ms           12
*/
