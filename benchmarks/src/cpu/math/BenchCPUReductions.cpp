#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/math/Reduce.h>
#include <noa/cpu/memory/PtrHost.h>

using namespace ::noa;

namespace {
    constexpr size4_t shapes[] = {
            {1, 1,   512,  512},
            {1, 1,   4096, 4096},
            {1, 128, 128,  256},
            {1, 256, 256,  256},
            {1, 256, 256,  512}
    };

    template<typename T>
    void CPU_reduce_sum_contiguous(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        cpu::Stream stream{cpu::Stream::DEFAULT};

        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            T sum = cpu::math::sum(src.share(), shape.stride(), shape, stream);
            ::benchmark::DoNotOptimize(sum);
        }
    }
}

BENCHMARK_TEMPLATE(CPU_reduce_sum_contiguous, float)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
