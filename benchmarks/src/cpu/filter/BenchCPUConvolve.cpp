#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Convolve.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    constexpr size3_t g_shapes[] = {
            {512,  512,  1},
            {4096, 4096, 1},
            {256,  256,  256},
            {512,  512,  512},
    };

    template<typename T>
    void CPU_filter_convolve1(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9, 11};

        const size3_t shape = g_shapes[state.range(0)];
        const size_t filter_size = filter1_size[state.range(1)];

        cpu::memory::PtrHost<T> filter(filter_size);
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);
        test::randomize(filter.get(), filter.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::convolve1(src.get(), shape, dst.get(), shape, shape, 1, filter.get(), filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_convolve2(benchmark::State& state) {
        constexpr size2_t filter2_size[] = {{3, 3},
                                            {5, 5}};

        const size3_t shape = g_shapes[state.range(0)];
        const size2_t filter_size = filter2_size[state.range(1)];

        cpu::memory::PtrHost<T> filter(elements(filter_size));
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);
        test::randomize(filter.get(), filter.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::convolve2(src.get(), shape, dst.get(), shape, shape, 1, filter.get(), filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_convolve3(benchmark::State& state) {
        constexpr size3_t filter3_size[] = {{3, 3, 3},
                                            {5, 5, 5}};

        const size3_t shape = g_shapes[state.range(0)];
        const size3_t filter_size = filter3_size[state.range(1)];

        cpu::memory::PtrHost<T> filter(elements(filter_size));
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);
        test::randomize(filter.get(), filter.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::convolve3(src.get(), shape, dst.get(), shape, shape, 1, filter.get(), filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_convolve3_separable(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9, 11};

        const size3_t shape = g_shapes[state.range(0)];
        const size_t filter_size = filter1_size[state.range(1)];

        cpu::memory::PtrHost<T> filter(filter_size);
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);
        test::randomize(filter.get(), filter.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::convolve(src.get(), shape, dst.get(), shape, shape, 1,
                                  filter.get(), filter_size,
                                  filter.get(), filter_size,
                                  filter.get(), filter_size,
                                  stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_filter_convolve1, half_t)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 3, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_convolve1, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 3, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_convolve2, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 1, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_convolve3, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 1, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

// As expected, this is much faster and can deal with large kernels quite easily.
BENCHMARK_TEMPLATE(CPU_filter_convolve3_separable, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 3, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
