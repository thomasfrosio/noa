#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/math/Reductions.h>

#include "Helpers.h"

#include <iostream>

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_math_test1(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::math::ewise(src.get(), shape, dst.get(), shape, shape, 1, [](const T& a) { return -a; }, stream);
            cpu::math::ewise(src.get(), shape, dst.get(), shape, shape, 1, math::inverse_t{}, stream);
//            std::transform(src.get(), src.get() + elements(shape), dst.get(), [](const T& a) { return -a; });
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_math_test2(benchmark::State& state) {
        const size3_t shape{512, 512, 512};
        const size2_t pitch{shape.x, shape.y};

        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<int> dst(src.size());

        test::Randomizer<T> randomizer(0, 1);
        test::randomize(src.get(), src.elements(), randomizer);
        double a(3.);

        cpu::Stream stream;
        for (auto _: state) {
//            cpu::math::reduce(src.get(), shape, shape, dst.get(), 1, math::min_t{}, 19.f, stream);
//            cpu::math::reduce(src.get(), shape, shape, dst.get() + 1, 1, math::dist2_t{}, 0.f, stream);
//            cpu::math::min(src.get(), shape, shape, dst.get(), 1, stream);
//            cpu::math::max(src.get(), shape, shape, dst.get()+1, 1, stream);
//            cpu::math::mean(src.get(), shape, shape, dst.get()+2, 1, stream);
//            cpu::math::variance(src.get(), shape, shape, dst.get()+3, 1, stream);
//            cpu::math::minMaxSumMean(src.get(), dst.get(), dst.get() + 1, dst.get() + 2, dst.get() + 3, src.size(), 1);

            cpu::math::ewise(src.get(), shape, a, dst.get(), shape, shape, 1, math::less_equal_t<int>{}, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_math_test1, float)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_math_test2, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_math_test2, float)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_math_test2, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
//BENCHMARK_TEMPLATE(CPU_math_test2, int)->Unit(benchmark::kMillisecond)->UseRealTime();
