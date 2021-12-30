#include <benchmark/benchmark.h>

#include <noa/cpu/Stream.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>

#include "Helpers.h"

using namespace ::noa;

namespace {
    template<typename T>
    void CPU_fft_remap(benchmark::State& state) {
        path_t path_base = benchmark::PATH_NOA_DATA / "filter";
        YAML::Node benchmarks = YAML::LoadFile(path_base / "benchmarks.yaml")["remap"][state.range(0)];

        auto shape = benchmarks["shape"].as<size3_t>();
        auto permutation = benchmarks["permutations"][state.range(1)].as<fft::Remap>();

        cpu::memory::PtrHost<T> src;
        cpu::memory::PtrHost<T> dst;
        size3_t src_pitch, dst_pitch;
        if (permutation & (fft::SRC_FULL | fft::SRC_FULL_CENTERED)) {
            src_pitch = shape;
            src.reset(elements(shape));
        } else {
            src_pitch = {shape.x / 2 + 1, shape.y, shape.z};
            src.reset(elementsFFT(shape));
        }

        if (permutation & (fft::DST_FULL | fft::DST_FULL_CENTERED)) {
            dst_pitch = shape;
            dst.reset(elements(shape));
        } else {
            dst_pitch = {shape.x / 2 + 1, shape.y, shape.z};
            dst.reset(elementsFFT(shape));
        }

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::remap(permutation, src.get(), src_pitch, dst.get(), dst_pitch, shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_remap_h2hc(benchmark::State& state) {
        size3_t shape = {256, 256, 256};
        size3_t pitch = {shape.x / 2 + 1, shape.y, shape.z};
        fft::Remap permutation = fft::H2HC;

        cpu::memory::PtrHost<T> src(elementsFFT(shape));
        cpu::memory::PtrHost<T> dst(elementsFFT(shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::remap(permutation, src.get(), pitch, dst.get(), pitch, shape, 1, stream);
            ::benchmark::DoNotOptimize(dst.get());
        }
    }

    template<typename T>
    void CPU_fft_remap_h2hc_inplace(benchmark::State& state) {
        size3_t shape = {256, 256, 256};
        size3_t pitch = {shape.x / 2 + 1, shape.y, shape.z};
        fft::Remap permutation = fft::H2HC;

        cpu::memory::PtrHost<T> src(elementsFFT(shape));

        test::Randomizer<T> randomizer(-5, 5);
        test::randomize(src.get(), src.elements(), randomizer);

        cpu::Stream stream;
        for (auto _: state) {
            cpu::fft::remap(permutation, src.get(), pitch, src.get(), pitch, shape, 1, stream);
            ::benchmark::DoNotOptimize(src.get());
        }
    }

    void customArguments(benchmark::internal::Benchmark* b) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 10; ++j) // all permutations
                b->Args({i, j});
    }
}

// As expected, this is memory bound. The larger the type, the slower it gets.
BENCHMARK_TEMPLATE(CPU_fft_remap, float)->Apply(customArguments)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap, cfloat_t)->Apply(customArguments)->Unit(benchmark::kMillisecond)->UseRealTime();

// In-place is ~50% faster
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc_inplace, half_t)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc, float)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_fft_remap_h2hc_inplace, float)->Unit(benchmark::kMillisecond)->UseRealTime();
