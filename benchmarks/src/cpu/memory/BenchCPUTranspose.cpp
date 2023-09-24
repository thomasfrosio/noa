#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.hpp>
#include <noa/cpu/memory/AllocatorHeap.hpp>
#include <noa/cpu/memory/Permute.hpp>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_memory_transpose1(benchmark::State& state) {
        const size4_t shape{128, 128, 128, 128};
        const size4_t strides = shape.strides();

        cpu::memory::PtrHost<T> src(shape.elements());
        cpu::memory::PtrHost<T> dst(shape.elements());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream(cpu::Stream::DEFAULT);
        for (auto _: state) {
            cpu::memory::permute(src.share(), strides, shape, dst.share(), strides, {0, 1, 2, 3}, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_memory_transpose1, float)->Unit(benchmark::kMillisecond)->UseRealTime();
