#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.hpp>
#include <noa/gpu/cuda/Stream.hpp>
#include <noa/gpu/cuda/math/Random.hpp>
#include <noa/gpu/cuda/math/Reduce.hpp>
#include <noa/gpu/cuda/memory/AllocatorDevice.hpp>
#include <noa/gpu/cuda/memory/AllocatorDevicePadded.hpp>

using namespace ::noa;

namespace {
    constexpr size4_t shapes[] = {
            {1, 1, 1, 2},
            {1, 1, 1, 256},
            {1, 1, 1, 512},
            {1, 1, 1, 8192},
            {1, 1, 1, 16384},
            {1, 1, 1, 32768},
            {1, 1, 1, 65536},
            {1, 1, 1, 1048576},
            {1, 1, 1, 4194304},
            {1, 1, 1, 16777216},
            {1, 1, 1, 33554432}
    };

    constexpr size4_t shapes2[] = {
            {1, 1,   512,  512},
            {1, 1,   4096, 4096},
            {1, 128, 128,  256},
            {1, 256, 256,  256},
            {1, 256, 256,  512}
    };

    template<typename T>
    void CUDA_reduce_sum_contiguous(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrDevice<T> src{shape.elements(), stream};
        cuda::memory::PtrDevice<T> dst{1, stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::math::sum(src.share(), shape.strides(), shape, dst.share(), size4_t{}, size4_t{1}, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_reduce_sum_pitch(benchmark::State& state) {
        const size4_t shape = shapes2[state.range(0)];

        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrDevicePadded<T> src{shape};
        cuda::memory::PtrDevice<T> dst{1, stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.pitches().elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::math::sum(src.share(), src.strides(), shape, dst.share(), size4_t{}, size4_t{1}, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// Compared to the CUDA 11.4 samples (the best of their kernels, kernel 7), our implementation is very similar
// in terms of performance. For very small arrays (<=512 elements), we perform worst (0.006ms vs 0.030ms), probably
// because we launch more threads (we have a fixed size block of 512 threads). For medium-sized arrays (8192 to 1M),
// we are faster (0.12ms vs 0.025-0.07ms). For large arrays (millions of elements), it is very similar. It seems like
// we have slightly better performance for single precision, but slightly worse for double precision. In any case,
// I would say we are comparable to the CUDA samples.
BENCHMARK_TEMPLATE(CUDA_reduce_sum_contiguous, float)->DenseRange(0, 10)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_reduce_sum_contiguous, double)->DenseRange(0, 10)->Unit(benchmark::kMillisecond)->UseRealTime();

// Non-contiguous arrays (pitched in this case) have basically the same performance characteristics than contiguous ones,
// which is great. Pitched arrays can still be vectorized. In fact, in this case, they have the same vectorization as
// contiguous arrays (see reduceLarge4D_ kernel).
BENCHMARK_TEMPLATE(CUDA_reduce_sum_pitch, float)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_reduce_sum_pitch, double)->DenseRange(0, 4)->Unit(benchmark::kMillisecond)->UseRealTime();
