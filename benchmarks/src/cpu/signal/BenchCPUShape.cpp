#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Shape.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_filter_sphere3D(benchmark::State& state) {
        const size4_t shape = {4, 256, 256, 256};
        const size4_t strides = shape.strides();

        cpu::memory::PtrHost<T> src(shape.elements());
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream(cpu::Stream::DEFAULT);
        stream.threads(16);
        for (auto _: state) {
            cpu::signal::sphere(src.share(), strides, dst.share(), strides, shape,
                                float3_t(dim3_t(shape.get(1)) / 2), 75, 10, float22_t{}, false, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// half_t is almost x2 slower...
BENCHMARK_TEMPLATE(CPU_filter_sphere3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_sphere3D, double)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_sphere3D, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
