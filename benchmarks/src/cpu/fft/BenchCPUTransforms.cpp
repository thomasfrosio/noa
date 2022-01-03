#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Transforms.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    constexpr size3_t shapes[] = {
            {512, 512, 1},
            {4096, 4096, 1},
            {128, 128, 128},
            {256, 256, 256},
            {512, 512, 512},
    };

    template<typename T>
    void CPU_fft_r2c(benchmark::State& state) {
        const size3_t shape = shapes[state.range(0)];
        const size3_t shape_fft = noa::shapeFFT(shape);

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<complex_t> dst(elements(shape_fft));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::r2c(src.get(), dst.get(), shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_r2c_inplace(benchmark::State& state) {
        const size3_t shape = shapes[state.range(0)];

        cpu::memory::PtrHost<T> src(elements(shape + 2));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::r2c(src.get(), shape, 1, stream);
            ::benchmark::DoNotOptimize(src.get());
        }
    }

    template<typename T>
    void CPU_fft_r2c_save_plan(benchmark::State& state) {
        const size3_t shape = shapes[state.range(0)];
        const size3_t shape_fft = noa::shapeFFT(shape);

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<T> src(elements(shape));
        cpu::memory::PtrHost<complex_t> dst(elements(shape_fft));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        // Use same flag for plan creation to see if FFTw3 is able to same the plan in a cache or something...
        cpu::fft::Plan plan(src.get(), dst.get(), shape, 1, noa::cpu::fft::ESTIMATE, stream);
        for (auto _: state) {
            cpu::fft::r2c(src.get(), dst.get(), shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_c2r(benchmark::State& state) {
        const size3_t shape = shapes[state.range(0)];
        const size3_t shape_fft = noa::shapeFFT(shape);

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<complex_t> src(elements(shape_fft));
        cpu::memory::PtrHost<T> dst(elements(shape));

        test::Randomizer<complex_t> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::c2r(src.get(), dst.get(), shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// Saving the plan doesn't have much performance benefits since FFTW3 apparently keeps track of old plans.
// The planning-rigor flag is useful and fft::PATIENT is giving a 70% performance increase.
// In-place seems to be as fast as out-of place.
// Multiple threads are quickly very beneficial (small 2D images can be done in 1 or 2 threads though).
// Double precision is almost as fast as single precision.
BENCHMARK_TEMPLATE(CPU_fft_r2c, float)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c, double)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c_inplace, float)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c_save_plan, float)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_c2r, float)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_c2r, double)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
