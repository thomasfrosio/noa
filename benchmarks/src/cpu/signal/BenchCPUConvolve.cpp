#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.hpp>
#include <noa/cpu/math/Random.hpp>
#include <noa/cpu/memory/PtrHost.hpp>
#include <noa/cpu/signal/Convolve.hpp>

using namespace ::noa;

namespace {
    constexpr size4_t g_shapes[] = {
            {1, 1, 512,  512},
            {1, 1, 4096, 4096},
            {1, 256,  256,  256},
            {1, 512,  512,  512},
    };

    template<typename T>
    void CPU_signal_convolve1(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9, 11};

        const size4_t shape = g_shapes[state.range(0)];
        const size4_t stride = shape.strides();
        const size_t filter_size = filter1_size[state.range(1)];

        cpu::memory::PtrHost<T> filter{filter_size};
        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<T> dst{src.elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, filter.share(), filter.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::signal::convolve1(src.share(), stride, dst.share(), stride, shape, filter.share(), filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_signal_convolve2(benchmark::State& state) {
        constexpr size2_t filter2_size[] = {{3, 3},
                                            {5, 5}};

        const size4_t shape = g_shapes[state.range(0)];
        const size4_t stride = shape.strides();
        const size2_t filter_size = filter2_size[state.range(1)];

        cpu::memory::PtrHost<T> filter{filter_size.elements()};
        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<T> dst{src.elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, filter.share(), filter.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::signal::convolve2(src.share(), stride, dst.share(), stride, shape, filter.share(), filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_signal_convolve3(benchmark::State& state) {
        constexpr size3_t filter3_size[] = {{3, 3, 3},
                                            {5, 5, 5}};

        const size4_t shape = g_shapes[state.range(0)];
        const size4_t stride = shape.strides();
        const size3_t filter_size = filter3_size[state.range(1)];

        cpu::memory::PtrHost<T> filter{filter_size.elements()};
        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<T> dst{src.elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, filter.share(), filter.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::signal::convolve3(src.share(), stride, dst.share(), stride, shape, filter.share(), filter_size, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_signal_convolve3_separable(benchmark::State& state) {
        constexpr size_t filter1_size[] = {3, 5, 9, 11};

        const size4_t shape = g_shapes[state.range(0)];
        const size4_t stride = shape.strides();
        const size_t filter_size = filter1_size[state.range(1)];

        cpu::memory::PtrHost<T> filter{filter_size};
        cpu::memory::PtrHost<T> src{shape.elements()};
        cpu::memory::PtrHost<T> dst{src.elements()};

        using real_t = traits::value_type_t<T>;
        cpu::Stream stream{cpu::Stream::DEFAULT};
        cpu::math::randomize(math::uniform_t{}, src.share(), src.elements(), real_t{-5}, real_t{5}, stream);
        cpu::math::randomize(math::uniform_t{}, filter.share(), filter.elements(), real_t{-5}, real_t{5}, stream);

        for (auto _: state) {
            cpu::signal::convolve(src.share(), stride, dst.share(), stride, shape,
                                  filter.share(), filter_size,
                                  filter.share(), filter_size,
                                  filter.share(), filter_size,
                                  stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

// half_t is ~2x slower than single precision, even if the convolution itself is done in single precision.
// So basically, this overhead comes from the half_t <-> float conversion.
// As expected, the filter size considerably affects the runtime.
// convolve3 with window of 5 is ~4x slower than window of 3.
// Multithreading scales quite well.
BENCHMARK_TEMPLATE(CPU_signal_convolve1, half_t)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 3, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_signal_convolve1, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 3, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_signal_convolve2, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 1, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_signal_convolve3, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 1, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();

// As expected, this is much faster and can deal with large kernels quite easily.
// window 3 is ~2.6x faster, window 5 is ~7x to 10x faster.
BENCHMARK_TEMPLATE(CPU_signal_convolve3_separable, float)
        ->ArgsProduct({benchmark::CreateDenseRange(0, 3, 1), benchmark::CreateDenseRange(0, 3, 1)})
        ->Unit(benchmark::kMillisecond)->UseRealTime();
