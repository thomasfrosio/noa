#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Transpose.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_memory_transpose1(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::transpose(src.get(), shape, shape, dst.get(), shape, {0, 2, 1}, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_memory_transpose1, float)->Unit(benchmark::kMillisecond)->UseRealTime();
