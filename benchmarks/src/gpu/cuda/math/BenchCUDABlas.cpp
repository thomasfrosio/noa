#include <benchmark/benchmark.h>

#include <noa/gpu/cuda/Event.hpp>
#include <noa/gpu/cuda/Stream.hpp>
#include <noa/gpu/cuda/math/Blas.hpp>
#include <noa/gpu/cuda/math/Ewise.h>
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

    constexpr size_t sizes[] = {10, 2048, 4096, 32768, 262144, 4194304};

    template<typename T>
    void CUDA_dot(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];
        const size4_t stride = shape.strides();

        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrDevice<T> lhs{shape.elements(), stream};
        cuda::memory::PtrDevice<T> rhs{shape.elements(), stream};
        cuda::memory::PtrDevice<T> dst{1, stream};
        cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), T{-5}, T{5}, stream);
        cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::math::dot(lhs.share(), stride, shape,
                            rhs.share(), stride, shape,
                            dst.share(), stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_dot_ewise_sum(benchmark::State& state) {
        const size4_t shape = shapes[state.range(0)];
        const size4_t reduced_shape{1, 1, 1, 1};

        cuda::Stream stream{cuda::Stream::DEFAULT};

        cuda::memory::PtrDevice<T> lhs{shape.elements(), stream};
        cuda::memory::PtrDevice<T> rhs{shape.elements(), stream};
        cuda::memory::PtrDevice<T> tmp{shape.elements(), stream};
        cuda::memory::PtrDevice<T> dst{1, stream};
        cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), T{-5}, T{5}, stream);
        cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), T{-5}, T{5}, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::math::ewise(lhs.share(), shape.strides(),
                              rhs.share(), shape.strides(),
                              tmp.share(), shape.strides(),
                              shape, math::multiply_t{}, stream);
            cuda::math::sum(tmp.share(), shape.strides(), shape,
                            dst.share(), reduced_shape.strides(), reduced_shape, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CUDA_dot_matmul(benchmark::State& state) {
        using real_t = traits::value_type_t<T>;
        const size_t size = sizes[state.range(0)];
        const size4_t lhs_shape{1, 1, 1, size};
        const size4_t rhs_shape{1, 1, size, 1};
        const size4_t out_shape{1, 1, 1, 1};
        const size4_t lhs_stride = lhs_shape.strides();
        const size4_t rhs_stride = rhs_shape.strides();
        const size4_t out_stride = out_shape.strides();

        cuda::Stream stream(cuda::Stream::DEFAULT);

        cuda::memory::PtrDevice<T> lhs(size, stream);
        cuda::memory::PtrDevice<T> rhs(size, stream);
        cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
        cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
        cuda::memory::PtrDevice<T> out(1, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::math::matmul(lhs.share(), lhs_stride, lhs_shape,
                               rhs.share(), rhs_stride, rhs_shape,
                               out.share(), out_stride, out_shape, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(out.get());
        }
    }

    template<typename T>
    void CUDA_matmul(benchmark::State& state) {
        using real_t = traits::value_type_t<T>;
        const size_t size = sizes[state.range(0)];
        const size4_t lhs_shape{1, 1, size, size};
        const size4_t rhs_shape{1, 1, size, size};
        const size4_t out_shape{1, 1, size, size};
        const size4_t lhs_stride = lhs_shape.strides();
        const size4_t rhs_stride = rhs_shape.strides();
        const size4_t out_stride = out_shape.strides();

        cuda::Stream stream(cuda::Stream::DEFAULT);

        cuda::memory::PtrDevice<T> lhs(size * size, stream);
        cuda::memory::PtrDevice<T> rhs(size * size, stream);
        cuda::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
        cuda::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
        cuda::memory::PtrDevice<T> out(size * size, stream);
        stream.synchronize();

        for (auto _: state) {
            cuda::Event start, end;
            start.record(stream);

            cuda::math::matmul(lhs.share(), lhs_stride, lhs_shape,
                              rhs.share(), rhs_stride, rhs_shape,
                              out.share(), out_stride, out_shape, stream);

            end.record(stream);
            end.synchronize();
            state.SetIterationTime(cuda::Event::elapsed(start, end));
            ::benchmark::DoNotOptimize(out.get());
        }
    }
}


BENCHMARK_TEMPLATE(CUDA_dot, float)->DenseRange(0, 10)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_dot_ewise_sum, float)->DenseRange(0, 10)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_dot_matmul, float)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CUDA_matmul, float)->DenseRange(0, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_matmul, double)->DenseRange(0, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CUDA_matmul, cfloat_t)->DenseRange(0, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
