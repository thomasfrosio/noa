#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/signal/Shape.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_filter_sphere3D(benchmark::State& state) {
        const size3_t shape = {256, 256, 256};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::signal::sphere(src.get(), shape, dst.get(), shape, shape, 1, float3_t(shape / 2), 75, 10, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_sphere2D(benchmark::State& state) {
        const size3_t shape = {4096, 4096, 1};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::signal::sphere(src.get(), shape, dst.get(), shape, shape, 1, float3_t(shape / 2), 75, 10, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_rectangle3D(benchmark::State& state) {
        const size3_t shape = {256,256,256};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::signal::rectangle(src.get(), shape, dst.get(), shape, shape, 1,
                                   float3_t(shape / 2), {75, 50, 50}, 10, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_rectangle2D(benchmark::State& state) {
        const size3_t shape = {4096, 4096, 1};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::signal::rectangle(src.get(), shape, dst.get(), shape, shape, 1,
                                   float3_t(shape / 2), {75, 50, 1}, 10, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_filter_sphere3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_sphere2D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_rectangle3D, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_rectangle2D, float)->Unit(benchmark::kMillisecond)->UseRealTime();

// half_t is almost x2 slower...
BENCHMARK_TEMPLATE(CPU_filter_sphere3D, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_rectangle3D, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
