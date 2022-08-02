#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.h>
#include <noa/gpu/cuda/Stream.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/memory/PtrDevice.h>

using namespace ::noa;

namespace {
    constexpr size4_t shapes[] = {
            {1, 1, 1, 2},
            {1, 1, 1, 256},
            {1, 1, 1, 8192},
            {1, 1, 1, 16384},
            {1, 1, 1, 65536},
            {1, 1, 1, 1048576},
            {1, 1, 1, 16777216},
            {1, 1, 1, 33554432}
    };

    template<typename T>
    void CUDA_copy_device_device_contiguous(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];

        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrDevice<T> src{shape.elements(), stream};
        cuda::memory::PtrDevice<T> dst{src.elements(), stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::memory::copy(src.share(), src.strides(), dst.share(), dst.strides(), shape, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_copy_device_device_strided(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];
        const size4_t strides = shape.strides() * 2;

        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrDevice<T> src{strides[0], stream};
        cuda::memory::PtrDevice<T> dst{strides[0], stream};
        cuda::math::randomize(math::uniform_t{}, src.share(), src.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::memory::copy(src.share(), strides, dst.share(), strides, shape, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CUDA_copy_device_device_contiguous, float)->DenseRange(0, 7)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_copy_device_device_strided, float)->DenseRange(0, 7)->Unit(benchmark::kMillisecond)->UseRealTime();
