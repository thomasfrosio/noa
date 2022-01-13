#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Cast.h>
#include <noa/cpu/memory/Set.h>
#include <noa/cpu/memory/Copy.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T, typename U>
    void CPU_memory_cast1(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};
        const auto clamp = static_cast<bool>(state.range(0));

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<U> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::cast(src.get(), dst.get(), src.elements(), clamp, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T, typename U>
    void CPU_memory_cast2(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};
        const auto clamp = static_cast<bool>(state.range(0));

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<U> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::cast(src.get(), shape, dst.get(), shape, shape, 1, clamp, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_memory_copy1(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::copy(src.get(), dst.get(), src.elements(), stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_memory_copy2(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::copy(src.get(), shape, dst.get(), shape, shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_memory_set1(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        cpu::memory::PtrHost<T> dst(elements(shape));

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::set(dst.get(), dst.elements(), 3.f, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_memory_set2(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::memory::set(dst.get(), shape, shape, 1, 3.f, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_memory_copy1, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_memory_copy2, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_memory_set1, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_memory_set2, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_memory_cast1, int, long)->Unit(benchmark::kMillisecond)->UseRealTime()->Arg(0);
BENCHMARK_TEMPLATE(CPU_memory_cast1, int, long)->Unit(benchmark::kMillisecond)->UseRealTime()->Arg(1);
BENCHMARK_TEMPLATE(CPU_memory_cast2, float, short)->Unit(benchmark::kMillisecond)->UseRealTime()->Arg(0);
