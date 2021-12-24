#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Filters.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_lowpass(benchmark::State& state) {
        path_t path_base = benchmark::PATH_NOA_DATA / "filter";
        YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["lowpass"][state.range(0)];

        auto batches = benchmarks["batches"].as<size_t>();
        auto shape = benchmarks["shape"].as<size3_t>();
        float cutoff = 0.5f;
        float width = 0.4f;

        size_t elements = noa::elementsFFT(shape);
        size3_t pitch = noa::shapeFFT(shape);
        cpu::memory::PtrHost<T> input_result(elements * batches);

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(input_result.get(), input_result.elements(), randomizer);

        cpu::Stream stream;
        stream.threads(2);
        for (auto _: state) {
            // Test on-the-fly, in-place.
            cpu::fft::lowpass<fft::H2H>(input_result.get(), pitch, input_result.get(), pitch, shape, batches,
                                        cutoff, width, stream);
            ::benchmark::DoNotOptimize(input_result.get());
            ::benchmark::ClobberMemory();
        }
    }
}

BENCHMARK_TEMPLATE(CPU_lowpass, float)->DenseRange(0, 7)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_lowpass, double)->DenseRange(0, 7)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_lowpass, cfloat_t)->DenseRange(0, 7)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_lowpass, cdouble_t)->DenseRange(0, 7)->Unit(benchmark::kMillisecond)->UseRealTime();
