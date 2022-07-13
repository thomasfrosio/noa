#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/math/Blas.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/memory/PtrHost.h>

using namespace ::noa;

namespace {
    constexpr size_t sizes[] = {10, 2048, 4096, 32768, 262144, 4194304};

    template<typename T>
    void CPU_dot_dot(benchmark::State& state) {
        using real_t = traits::value_type_t<T>;
        const size_t size = sizes[state.range(0)];
        const size4_t shape{1, 1, 1, size};
        const size4_t stride = shape.stride();

        cpu::Stream stream(cpu::Stream::DEFAULT);

        cpu::memory::PtrHost<T> lhs(size);
        cpu::memory::PtrHost<T> rhs(size);
        cpu::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            T out_dot = cpu::math::dot(lhs.share(), stride, shape, rhs.share(), stride, shape, stream);
            ::benchmark::DoNotOptimize(out_dot);
        }
    }

    template<typename T>
    void CPU_dot_matmul(benchmark::State& state) {
        using real_t = traits::value_type_t<T>;
        const size_t size = sizes[state.range(0)];
        const size4_t lhs_shape{1, 1, 1, size};
        const size4_t rhs_shape{1, 1, size, 1};
        const size4_t out_shape{1, 1, 1, 1};
        const size4_t lhs_stride = lhs_shape.stride();
        const size4_t rhs_stride = rhs_shape.stride();
        const size4_t out_stride = out_shape.stride();

        cpu::Stream stream(cpu::Stream::DEFAULT);

        cpu::memory::PtrHost<T> lhs(size);
        cpu::memory::PtrHost<T> rhs(size);
        cpu::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
        cpu::memory::PtrHost<T> out(1);

        for (auto _: state) {
            cpu::math::matmul(lhs.share(), lhs_stride, lhs_shape,
                              rhs.share(), rhs_stride, rhs_shape,
                              out.share(), out_stride, out_shape, stream);
            ::benchmark::DoNotOptimize(out.get());
        }
    }

    template<typename T>
    void CPU_matmul(benchmark::State& state) {
        using real_t = traits::value_type_t<T>;
        const size_t size = sizes[state.range(0)];
        const size4_t lhs_shape{1, 1, size, size};
        const size4_t rhs_shape{1, 1, size, size};
        const size4_t out_shape{1, 1, size, size};
        const size4_t lhs_stride = lhs_shape.stride();
        const size4_t rhs_stride = rhs_shape.stride();
        const size4_t out_stride = out_shape.stride();

        cpu::Stream stream(cpu::Stream::DEFAULT);

        cpu::memory::PtrHost<T> lhs(size * size);
        cpu::memory::PtrHost<T> rhs(size * size);
        cpu::math::randomize(math::uniform_t{}, lhs.share(), lhs.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, rhs.share(), rhs.elements(), real_t{-5}, real_t{5}, stream);
        cpu::memory::PtrHost<T> out(size * size);

        for (auto _: state) {
            cpu::math::matmul(lhs.share(), lhs_stride, lhs_shape,
                              rhs.share(), rhs_stride, rhs_shape,
                              out.share(), out_stride, out_shape, stream);
            ::benchmark::DoNotOptimize(out.get());
        }
    }
}

// Internally, matmul checks if the matrix-matrix product is a dot product since dot implementations are ~4x faster.
// With BLAS, times are very similar to NumPy's (which is also multithreaded), or slightly faster.
// Without BLAS, it is often 2 or 3 times slower than NumPy.
BENCHMARK_TEMPLATE(CPU_dot_dot, float)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_dot_dot, double)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_dot_dot, cfloat_t)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_dot_matmul, float)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_dot_matmul, double)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_dot_matmul, cfloat_t)->DenseRange(0, 5)->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_matmul, float)->DenseRange(0, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_matmul, double)->DenseRange(0, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_matmul, cfloat_t)->DenseRange(0, 2)->Unit(benchmark::kMillisecond)->UseRealTime();
