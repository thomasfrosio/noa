#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/filter/Median.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    constexpr size3_t g_shapes[] = {
            {4096, 4096, 1},
            {256,  256,  256},
    };

    template<typename T>
    void CPU_filter_median1(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9};
        constexpr noa::BorderMode border_modes[] = {noa::BORDER_REFLECT, noa::BORDER_ZERO};

        const size3_t shape = g_shapes[state.range(0)];
        const size_t filter_size = filter1_size[state.range(1)];
        const noa::BorderMode border_mode = border_modes[state.range(2)];

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::median1(src.get(), shape, dst.get(), shape, shape, 1, border_mode, filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_median2(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9};
        constexpr noa::BorderMode border_modes[] = {noa::BORDER_REFLECT, noa::BORDER_ZERO};

        const size3_t shape = g_shapes[state.range(0)];
        const size_t filter_size = filter1_size[state.range(1)];
        const noa::BorderMode border_mode = border_modes[state.range(2)];

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::median2(src.get(), shape, dst.get(), shape, shape, 1, border_mode, filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_filter_median3(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9};
        constexpr noa::BorderMode border_modes[] = {noa::BORDER_REFLECT, noa::BORDER_ZERO};

        const size3_t shape = g_shapes[state.range(0)];
        const size_t filter_size = filter1_size[state.range(1)];
        const noa::BorderMode border_mode = border_modes[state.range(2)];

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::filter::median3(src.get(), shape, dst.get(), shape, shape, 1, border_mode, filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// half_t is almost as fast as float.
// The border modes are almost equivalent.
// Going from size=3 to 5 almost doubles the runtime.
// In median3, going from 3 to 9 increases the runtime by ~10.
BENCHMARK_TEMPLATE(CPU_filter_median1, half_t)
        ->ArgsProduct({{0, 1}, {0, 1, 2}, {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_median1, float)
        ->ArgsProduct({{0, 1}, {0, 1, 2}, {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_median2, float)
        ->ArgsProduct({{0, 1}, {0, 1, 2}, {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_filter_median3, float)
        ->ArgsProduct({{1}, {0, 1, 2}, {0, 1}})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
