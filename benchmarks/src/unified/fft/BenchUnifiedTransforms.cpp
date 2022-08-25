#include <benchmark/benchmark.h>

#include <noa/FFT.h>
#include <noa/Math.h>

using namespace ::noa;

namespace {
    constexpr size4_t shapes[] = {
            {1, 1, 512, 512},
            {1, 1, 4096, 4096},
            {1, 256, 256, 256},
            {1, 512, 512, 512},
    };

    template<typename T>
    void unified_cpu_r2c(benchmark::State& state) {
        using complex_t = noa::Complex<T>;

        const size4_t shape = shapes[state.range(0)];
        StreamGuard stream{Device{}, Stream::DEFAULT}; // just make sure it is not async

        Array<T> src = math::random<T>(math::uniform_t{}, shape, -5, 5);
        Array<complex_t> dst{shape.fft()};

        for (auto _: state) {
            fft::r2c(src, dst, fft::NORM_NONE);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(unified_cpu_r2c, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(unified_cpu_r2c, double)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
