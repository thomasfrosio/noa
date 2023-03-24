#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.hpp>
#include <noa/gpu/cuda/Stream.hpp>
#include <noa/gpu/cuda/math/Random.hpp>
#include <noa/gpu/cuda/memory/PtrManaged.hpp>
#include <noa/gpu/cuda/fft/Transforms.hpp>

using namespace ::noa;

namespace {
    constexpr size4_t shapes[] = {
            {1, 1, 512, 512},
            {1, 1, 4096, 4096},
            {1, 256, 256, 256},
            {1, 512, 512, 512},
    };

    template<typename T>
    void CUDA_fft_r2c(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrManaged<T> src{shape.elements(), stream};
        cuda::memory::PtrManaged<complex_t> dst{shape.fft().elements(), stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::fft::r2c(src.share(), dst.share(), shape, fft::NORM_NONE, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_fft_r2c_inplace(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrManaged<complex_t> dst{shape.fft().elements(), stream};
        const auto& src = std::reinterpret_pointer_cast<T[]>(dst.share());
        cuda::math::randomize(math::uniform_t{}, dst.share(), dst.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::fft::r2c(src, dst.share(), shape, fft::NORM_NONE, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_fft_r2c_cache_plan(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrManaged<T> src{shape.elements(), stream};
        cuda::memory::PtrManaged<complex_t> dst{shape.fft().elements(), stream};
        cuda::fft::Plan<T> plan{cuda::fft::R2C, shape};

        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::fft::r2c(src.share(), dst.share(), plan, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_fft_c2r(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        using complex_t = noa::Complex<T>;
        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrManaged<complex_t> src{shape.fft().elements(), stream};
        cuda::memory::PtrManaged<T> dst{shape.elements(), stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::fft::c2r(src.share(), dst.share(), shape, fft::NORM_NONE, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// The caching mechanism seems to work since it is as fast as manually reusing the plan.
// In-place seems to be ~20% slower than out-of-place.
// Double precision is ~3x slower than single precision, although this should be very GPU dependant.
BENCHMARK_TEMPLATE(CUDA_fft_r2c, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_r2c, double)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_r2c_inplace, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_r2c_cache_plan, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();

// Same as r2c in terms of performance.
BENCHMARK_TEMPLATE(CUDA_fft_c2r, float)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_fft_c2r, double)->DenseRange(0, 3)->Unit(benchmark::kMillisecond)->UseRealTime();
