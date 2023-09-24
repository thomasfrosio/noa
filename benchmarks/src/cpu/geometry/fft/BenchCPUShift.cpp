#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.hpp>
#include <noa/cpu/memory/AllocatorHeap.hpp>
#include <noa/cpu/transform/fft/Shift.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_transform_fft_shift2D(benchmark::State& state) {
        const size3_t shape{4096, 4096, 1};

        const size2_t shape_2d{shape.x, shape.y};
        const size2_t pitch{shape.x / 2 + 1, shape.y};
        cpu::memory::PtrHost<T> src(elementsFFT(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        stream.threads(2);
        for (auto _: state) {
            cpu::transform::fft::shift2D<fft::HC2HC>(
                    src.get(), pitch, dst.get(), pitch, shape_2d, {3.f, 2.f}, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_transform_fft_shift3D(benchmark::State& state) {
        const size3_t shape{256,  256,  256};

        const size3_t pitch{shape.x / 2 + 1, shape.y, shape.z};
        cpu::memory::PtrHost<T> src(elementsFFT(shape));
        cpu::memory::PtrHost<T> dst(src.size());

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        stream.threads(2);
        for (auto _: state) {
            cpu::transform::fft::shift3D<fft::HC2HC>(
                    src.get(), pitch, dst.get(), pitch, shape, {3.f, 2.f, -34.f}, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }
}

BENCHMARK_TEMPLATE(CPU_transform_fft_shift2D, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_transform_fft_shift3D, cfloat_t)->Unit(benchmark::kMillisecond)->UseRealTime();
