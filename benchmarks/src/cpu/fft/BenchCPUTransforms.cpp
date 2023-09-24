#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.hpp>
#include <noa/cpu/math/Random.hpp>
#include <noa/cpu/memory/AllocatorHeap.hpp>
#include <noa/cpu/fft/Transforms.hpp>

using namespace ::noa;

namespace {
    constexpr size4_t shapes[] = {
            {1, 1, 512, 512},
            {1, 1, 4096, 4096},
            {1, 256, 256, 256},
            {1, 512, 512, 512},
    };

    template<typename T>
    void CPU_fft_r2c(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<complex_t> dst{shape.fft().elements()};

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cpu::fft::r2c(src.share(), dst.share(), shape, cpu::fft::ESTIMATE, fft::NORM_NONE, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_r2c_inplace(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<complex_t> dst{shape.fft().elements()};
        const auto& src = std::reinterpret_pointer_cast<T[]>(dst.share());

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, dst.share(), dst.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cpu::fft::r2c(src, dst.share(), shape, cpu::fft::ESTIMATE, fft::NORM_NONE, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_r2c_cache_plan(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};

        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<complex_t> dst{shape.fft().elements()};
        cpu::fft::Plan plan{src.share(), dst.share(), shape, cpu::fft::MEASURE, stream};

        cpu::memory::PtrHost<T> new_src{src.elements()};
        cpu::memory::PtrHost<complex_t> new_dst{dst.elements()};
        cpu::math::randomize(math::uniform_t{}, new_src.share(), new_src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cpu::fft::r2c(new_src.share(), new_dst.share(), plan, stream);
            ::benchmark::DoNotOptimize(new_dst.get());
        }
    }

    template<typename T>
    void CPU_fft_r2c_measure(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<complex_t> dst{shape.fft().elements()};

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        cpu::fft::r2c(src.share(), dst.share(), shape, cpu::fft::MEASURE, fft::NORM_NONE, stream);
        for (auto _: state) {
            cpu::fft::r2c(src.share(), dst.share(), shape, cpu::fft::MEASURE, fft::NORM_NONE, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_c2r(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cpu::memory::PtrHost<complex_t> src{shape.fft().elements()};
        cpu::memory::PtrHost<T> dst{shape.elements()};

        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cpu::fft::c2r(src.share(), dst.share(), shape, cpu::fft::ESTIMATE, fft::NORM_NONE, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// Saving the plan doesn't have much performance benefits since FFTW3 apparently keeps track of old plans.
// The planning-rigor flag is useful and fft::PATIENT is giving a 70% performance increase.
// In-place seems to be as fast as out-of place.
// Multiple threads are quickly very beneficial (small 2D images can be done in 1 or 2 threads though).
// Double precision is almost as fast as single precision.
BENCHMARK_TEMPLATE(CPU_fft_r2c, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c, double)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c_inplace, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c_cache_plan, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_r2c_measure, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_fft_c2r, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_c2r, double)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
